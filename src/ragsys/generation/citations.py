"""Parse ``[n]`` citation markers and validate them against the context.

The model is never trusted to cite correctly: a marker is only accepted if it
refers to a passage that was actually in the prompt. Invalid markers are reported
(and counted in metrics) so hallucinated citations are visible, not silent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragsys.models import Citation, ScoredChunk

_MARKER = re.compile(r"\[(\d{1,3}(?:\s*[,–-]\s*\d{1,3})*)\]")
_RANGE = re.compile(r"(\d+)\s*[–-]\s*(\d+)")
MAX_RANGE = 20
SNIPPET_CHARS = 240


def _expand(group: str) -> list[int]:
    numbers: list[int] = []
    for part in group.split(","):
        part = part.strip()
        if m := _RANGE.fullmatch(part):
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo <= hi and hi - lo <= MAX_RANGE:
                numbers.extend(range(lo, hi + 1))
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def extract_markers(text: str) -> list[int]:
    """Citation numbers in order of first appearance (``[1]``, ``[1, 2]``, ``[2-4]``)."""
    seen: dict[int, None] = {}
    for match in _MARKER.finditer(text):
        for n in _expand(match.group(1)):
            seen.setdefault(n, None)
    return list(seen)


def to_citation(marker: int, hit: ScoredChunk) -> Citation:
    """Citation record for passage ``marker`` (1-based)."""
    c = hit.chunk
    return Citation(
        marker=marker,
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        doc_title=c.doc_title,
        page_start=c.page_start,
        page_end=c.page_end,
        section=" > ".join(c.section_path),
        source_path=c.source_path,
        source_url=c.source_url,
        doc_date=c.doc_date,
        snippet=c.text[:SNIPPET_CHARS],
    )


def resolve_citations(
    text: str, context: Sequence[ScoredChunk]
) -> tuple[list[Citation], list[int]]:
    """Split markers in ``text`` into valid citations and invalid marker numbers."""
    valid: list[Citation] = []
    invalid: list[int] = []
    for marker in extract_markers(text):
        if 1 <= marker <= len(context):
            valid.append(to_citation(marker, context[marker - 1]))
        else:
            invalid.append(marker)
    return valid, invalid
