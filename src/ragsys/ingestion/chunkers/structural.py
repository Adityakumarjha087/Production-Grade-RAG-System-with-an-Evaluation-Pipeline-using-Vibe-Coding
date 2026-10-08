"""Document-structure-aware chunking.

Rules, in priority order:

1. Never split an element (paragraph, list item, table) unless it alone exceeds
   ``chunk_size``; oversized tables are split by row groups with the header row
   repeated, oversized paragraphs fall back to recursive splitting.
2. Start a new chunk at a section boundary, unless the current chunk is still
   smaller than ``min_chunk_size`` (avoids heading-only chunks).
3. Prefix each chunk's *text* with its section breadcrumb, so the embedding and
   the LLM see "which part of which circular" this is. The source span stays exact.
"""

from __future__ import annotations

from ragsys.ingestion.chunkers.base import make_chunk
from ragsys.ingestion.chunkers.recursive import RecursiveChunker
from ragsys.models import Chunk, Document, Element, ElementType

Span = tuple[int, int]


def breadcrumb(section_path: list[str]) -> str:
    """Render a section path as ``A > B > C``."""
    return " > ".join(section_path)


def _table_row_groups(table: str, size: int) -> list[tuple[str, Span]]:
    """Split a markdown table into (header-prefixed text, row span relative to table)."""
    lines = table.split("\n")
    header_lines = (
        2 if len(lines) > 2 and set(lines[1].replace("|", "").strip()) <= set("-: ") else 1
    )
    header = "\n".join(lines[:header_lines])
    offsets, pos = [], 0
    for line in lines:
        offsets.append((pos, pos + len(line)))
        pos += len(line) + 1
    groups: list[tuple[str, Span]] = []
    first = header_lines
    for i in range(header_lines, len(lines) + 1):
        body = "\n".join(lines[first:i])
        too_big = i < len(lines) and len(header) + len(body) + len(lines[i]) + 2 > size
        if (i == len(lines) or too_big) and i > first:
            groups.append((f"{header}\n{body}", (offsets[first][0], offsets[i - 1][1])))
            first = i
    return groups


class StructuralChunker:
    """See module docstring."""

    name = "structural"

    def __init__(self, chunk_size: int, chunk_overlap: int, min_chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self._fallback = RecursiveChunker(chunk_size, chunk_overlap)
        self._params = f"{chunk_size}-{chunk_overlap}-{min_chunk_size}"

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` along its structure."""
        out: list[tuple[Span, str | None, list[str]]] = []
        buffer: list[Element] = []
        for el in doc.elements:
            if len(el.text) > self.chunk_size:
                self._flush(buffer, out)
                out.extend(self._split_oversized(doc, el))
                continue
            if buffer and self._should_flush(buffer, el):
                self._flush(buffer, out)
            buffer.append(el)
        self._flush(buffer, out)
        return [
            self._to_chunk(doc, i, span, text, path) for i, (span, text, path) in enumerate(out)
        ]

    def _should_flush(self, buffer: list[Element], el: Element) -> bool:
        length = el.char_end - buffer[0].char_start
        if length > self.chunk_size:
            return True
        section_changed = (
            el.type == ElementType.HEADING or el.section_path != buffer[-1].section_path
        )
        current = buffer[-1].char_end - buffer[0].char_start
        return section_changed and current >= self.min_chunk_size

    @staticmethod
    def _flush(buffer: list[Element], out: list[tuple[Span, str | None, list[str]]]) -> None:
        if buffer:
            path = next((e.section_path for e in buffer if e.type != ElementType.HEADING), None)
            span = (buffer[0].char_start, buffer[-1].char_end)
            out.append((span, None, path or buffer[0].section_path))
            buffer.clear()

    def _split_oversized(
        self, doc: Document, el: Element
    ) -> list[tuple[Span, str | None, list[str]]]:
        if el.type == ElementType.TABLE:
            return [
                ((el.char_start + s, el.char_start + e), text, el.section_path)
                for text, (s, e) in _table_row_groups(el.text, self.chunk_size)
            ]
        spans = self._fallback.spans(doc.text, el.char_start, el.char_end)
        return [(span, None, el.section_path) for span in spans]

    def _to_chunk(
        self, doc: Document, ordinal: int, span: Span, text: str | None, path: list[str]
    ) -> Chunk:
        body = text if text is not None else doc.text[span[0] : span[1]]
        crumb = breadcrumb([doc.title, *[p for p in path if p != doc.title]])
        return make_chunk(
            doc,
            strategy=self.name,
            params=self._params,
            ordinal=ordinal,
            start=span[0],
            end=span[1],
            text=f"{crumb}\n\n{body}" if crumb else body,
            section_path=path,
        )
