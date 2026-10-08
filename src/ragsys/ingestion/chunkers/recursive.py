"""Recursive splitting on a hierarchy of separators (paragraph > line > sentence > word).

Same idea as LangChain's ``RecursiveCharacterTextSplitter`` but implemented on
character *spans* so every chunk keeps an exact source offset.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragsys.ingestion.chunkers.base import make_chunk, strip_span
from ragsys.ingestion.chunkers.fixed import window_spans
from ragsys.models import Chunk, Document

Span = tuple[int, int]
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", " ")


def _split_on(text: str, start: int, end: int, sep: str) -> list[Span]:
    """Split ``[start, end)`` after each occurrence of ``sep`` (separator kept left)."""
    spans: list[Span] = []
    cursor = start
    while True:
        idx = text.find(sep, cursor, end)
        if idx == -1:
            break
        spans.append((cursor, idx + len(sep)))
        cursor = idx + len(sep)
    if cursor < end:
        spans.append((cursor, end))
    return spans


def split_recursive(
    text: str, start: int, end: int, size: int, separators: Sequence[str]
) -> list[Span]:
    """Split a span into contiguous pieces no longer than ``size``."""
    if end - start <= size:
        return [(start, end)]
    if not separators:
        return [(start + s, start + e) for s, e in window_spans(text[start:end], size, 0)]
    pieces = _split_on(text, start, end, separators[0])
    if len(pieces) == 1:
        return split_recursive(text, start, end, size, separators[1:])
    result: list[Span] = []
    for s, e in pieces:
        result.extend(split_recursive(text, s, e, size, separators[1:]))
    return result


def merge_pieces(pieces: Sequence[Span], size: int, overlap: int) -> list[Span]:
    """Greedily merge contiguous pieces into windows, carrying ``overlap`` chars back."""
    windows: list[Span] = []
    current: list[Span] = []
    for piece in pieces:
        if current and piece[1] - current[0][0] > size:
            windows.append((current[0][0], current[-1][1]))
            while current and (
                current[-1][1] - current[0][0] > overlap or piece[1] - current[0][0] > size
            ):
                current.pop(0)
        current.append(piece)
    if current:
        windows.append((current[0][0], current[-1][1]))
    return windows


class RecursiveChunker:
    """See module docstring."""

    name = "recursive"

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators)
        self._params = f"{chunk_size}-{chunk_overlap}"

    def spans(self, text: str, start: int = 0, end: int | None = None) -> list[Span]:
        """Chunk spans for ``text[start:end]`` (whitespace-trimmed, non-empty)."""
        end = len(text) if end is None else end
        pieces = split_recursive(text, start, end, self.chunk_size, self.separators)
        merged = merge_pieces(pieces, self.chunk_size, self.chunk_overlap)
        trimmed = [strip_span(text, s, e) for s, e in merged]
        return [(s, e) for s, e in trimmed if e > s]

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` recursively."""
        return [
            make_chunk(doc, strategy=self.name, params=self._params, ordinal=i, start=s, end=e)
            for i, (s, e) in enumerate(self.spans(doc.text))
        ]
