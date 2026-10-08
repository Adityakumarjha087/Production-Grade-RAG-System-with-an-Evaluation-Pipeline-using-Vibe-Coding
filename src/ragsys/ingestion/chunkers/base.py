"""Chunker interface and helpers shared by all strategies.

All chunkers produce chunks whose ``[char_start, char_end)`` is a span of
``Document.text``. Page numbers and section path are derived from the elements
the span overlaps, so every strategy gets accurate citations for free.
"""

from __future__ import annotations

import bisect
import hashlib
from typing import Protocol

from ragsys.models import Chunk, Document, Element


class Chunker(Protocol):
    """Splits a document into chunks."""

    name: str

    def chunk(self, doc: Document) -> list[Chunk]:
        """Return chunks in document order."""
        ...


def chunk_id(doc_id: str, strategy: str, params: str, ordinal: int) -> str:
    """Deterministic id -> re-ingesting the same doc upserts instead of duplicating."""
    raw = f"{doc_id}|{strategy}|{params}|{ordinal}".encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink ``[start, end)`` so it doesn't begin or end with whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def overlapping_elements(doc: Document, start: int, end: int) -> list[Element]:
    """Elements whose span intersects ``[start, end)``."""
    starts = [e.char_start for e in doc.elements]
    lo = max(bisect.bisect_right(starts, start) - 1, 0)
    hi = bisect.bisect_left(starts, end)
    return [e for e in doc.elements[lo:hi] if e.char_end > start and e.char_start < end]


def make_chunk(
    doc: Document,
    *,
    strategy: str,
    params: str,
    ordinal: int,
    start: int,
    end: int,
    text: str | None = None,
    section_path: list[str] | None = None,
) -> Chunk:
    """Build a Chunk for span ``[start, end)``; ``text`` defaults to the span text."""
    elements = overlapping_elements(doc, start, end)
    pages = [e.page for e in elements] or [1]
    if section_path is None:
        section_path = elements[0].section_path if elements else []
    return Chunk(
        chunk_id=chunk_id(doc.doc_id, strategy, params, ordinal),
        doc_id=doc.doc_id,
        doc_title=doc.title,
        text=text if text is not None else doc.text[start:end],
        strategy=strategy,
        ordinal=ordinal,
        char_start=start,
        char_end=end,
        page_start=min(pages),
        page_end=max(pages),
        section_path=list(section_path),
        source_path=doc.source_path,
        source_url=doc.metadata.get("url", ""),
        doc_date=doc.metadata.get("date", ""),
    )
