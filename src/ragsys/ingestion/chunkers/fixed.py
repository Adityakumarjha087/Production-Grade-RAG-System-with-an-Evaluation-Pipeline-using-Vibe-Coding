"""Baseline: fixed-size character windows with overlap.

Ignores document structure entirely (it will happily cut a table in half). It
exists as the control group the other strategies are compared against.
"""

from __future__ import annotations

from ragsys.ingestion.chunkers.base import make_chunk, strip_span
from ragsys.models import Chunk, Document


def _snap_end(text: str, start: int, end: int, size: int) -> int:
    """Move ``end`` back to the last whitespace in the second half of the window."""
    if end >= len(text):
        return len(text)
    cut = max(text.rfind(" ", start + size // 2, end), text.rfind("\n", start + size // 2, end))
    return cut if cut > start else end


def _snap_start(text: str, pos: int, limit: int) -> int:
    """Move ``pos`` forward to the start of the next word (not past ``limit``)."""
    if pos == 0 or text[pos - 1].isspace():
        return pos
    for i in range(pos, limit):
        if text[i].isspace():
            return i + 1
    return pos


def window_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Word-aligned ``(start, end)`` windows of at most ``size`` chars."""
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = _snap_end(text, start, min(start + size, len(text)), size)
        s, e = strip_span(text, start, end)
        if e > s:
            spans.append((s, e))
        if end >= len(text):
            break
        start = max(_snap_start(text, end - overlap, end), start + 1)
    return spans


class FixedSizeChunker:
    """See module docstring."""

    name = "fixed"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._params = f"{chunk_size}-{chunk_overlap}"

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` into fixed windows."""
        spans = window_spans(doc.text, self.chunk_size, self.chunk_overlap)
        return [
            make_chunk(doc, strategy=self.name, params=self._params, ordinal=i, start=s, end=e)
            for i, (s, e) in enumerate(spans)
        ]
