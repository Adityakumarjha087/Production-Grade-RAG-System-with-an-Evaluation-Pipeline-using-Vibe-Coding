"""Retrieval metrics against chunking-independent gold spans.

Relevance: a chunk is relevant to a gold span if they are in the same document and

    overlap(chunk, span) / min(|chunk|, |span|) >= threshold

Using ``min`` makes the rule fair to both small chunks (fully inside a long span)
and long chunks (fully containing a short span).

* ``recall@k`` = fraction of an item's gold spans hit by at least one top-k chunk
  (multi-hop items have several spans).
* ``MRR`` = mean of 1 / rank of the first relevant chunk (0 if none retrieved).
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

from ragsys.evaluation.testset import EvidenceSpan
from ragsys.models import Chunk


def span_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Length of the intersection of two half-open intervals."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def is_relevant(chunk: Chunk, span: EvidenceSpan, threshold: float = 0.5) -> bool:
    """See module docstring."""
    if chunk.doc_id != span.doc_id:
        return False
    inter = span_overlap(chunk.char_start, chunk.char_end, span.char_start, span.char_end)
    shorter = min(chunk.char_end - chunk.char_start, span.char_end - span.char_start)
    return shorter > 0 and inter / shorter >= threshold


def recall_at_k(
    chunks: Sequence[Chunk], spans: Sequence[EvidenceSpan], k: int, threshold: float = 0.5
) -> float:
    """Fraction of gold spans covered by the top-``k`` chunks."""
    if not spans:
        raise ValueError("recall is undefined for items without gold evidence")
    top = chunks[:k]
    hit = sum(1 for s in spans if any(is_relevant(c, s, threshold) for c in top))
    return hit / len(spans)


def reciprocal_rank(
    chunks: Sequence[Chunk], spans: Sequence[EvidenceSpan], threshold: float = 0.5
) -> float:
    """1 / rank of the first chunk relevant to any gold span (0 if none)."""
    for rank, chunk in enumerate(chunks, start=1):
        if any(is_relevant(chunk, s, threshold) for s in spans):
            return 1.0 / rank
    return 0.0


def aggregate_retrieval(
    runs: Sequence[tuple[Sequence[Chunk], Sequence[EvidenceSpan]]],
    ks: Sequence[int],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Mean recall@k for each k and MRR over items that have gold evidence."""
    scored = [(chunks, spans) for chunks, spans in runs if spans]
    if not scored:
        return {}
    metrics = {f"recall@{k}": mean(recall_at_k(c, s, k, threshold) for c, s in scored) for k in ks}
    metrics["mrr"] = mean(reciprocal_rank(c, s, threshold) for c, s in scored)
    return metrics
