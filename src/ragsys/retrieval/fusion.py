"""Reciprocal Rank Fusion (Cormack et al., 2009).

    score(d) = sum over lists L containing d of  1 / (k + rank_L(d))

RRF only uses ranks, so it needs no score normalisation between BM25 (unbounded)
and cosine (bounded): that is the reason to prefer it over weighted score sums.
``k = 60`` is the value from the paper and dampens the dominance of top ranks.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragsys.models import ScoredChunk


def rrf_fuse(
    ranked_lists: Sequence[Sequence[ScoredChunk]], k: int = 60, top_n: int | None = None
) -> list[ScoredChunk]:
    """Fuse ranked lists into one list ordered by RRF score (ties: first-seen order)."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = {}
    chunks: dict[str, ScoredChunk] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            cid = item.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(cid, item)
    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    fused = [ScoredChunk(chunk=chunks[cid].chunk, score=scores[cid], stage="rrf") for cid in order]
    return fused[:top_n] if top_n is not None else fused
