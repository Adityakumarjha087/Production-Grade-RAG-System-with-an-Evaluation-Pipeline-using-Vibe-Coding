"""Cross-encoder reranking.

A cross-encoder reads (query, passage) *together*, so it is far more accurate
than bi-encoder similarity but too slow to run over the whole corpus — hence it
only rescores the top-N fused candidates. Its sigmoid score is also the most
calibrated relevance signal we have, which is why the abstention gate uses it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from ragsys.indexing.bm25 import tokenize
from ragsys.models import ScoredChunk

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder


def sigmoid(x: float) -> float:
    """Numerically safe logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class Reranker(Protocol):
    """Rescores candidates for a query; returned scores are in [0, 1]."""

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Return candidates sorted by descending relevance."""
        ...


def _sorted(candidates: Sequence[ScoredChunk], scores: Sequence[float]) -> list[ScoredChunk]:
    rescored = [
        ScoredChunk(chunk=c.chunk, score=s, stage="rerank")
        for c, s in zip(candidates, scores, strict=True)
    ]
    return sorted(rescored, key=lambda sc: sc.score, reverse=True)


class FastEmbedReranker:
    """ONNX cross-encoder via fastembed (default ``Xenova/ms-marco-MiniLM-L-12-v2``)."""

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: TextCrossEncoder | None = None

    def _get_model(self) -> TextCrossEncoder:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def warmup(self) -> None:
        """Load the model eagerly."""
        list(self._get_model().rerank("warmup", ["warmup"]))

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Score each (query, chunk text) pair with the cross-encoder."""
        if not candidates:
            return []
        logits = self._get_model().rerank(
            query, [c.chunk.text for c in candidates], batch_size=self.batch_size
        )
        return _sorted(candidates, [sigmoid(float(x)) for x in logits])


class LexicalOverlapReranker:
    """Deterministic stand-in for tests/offline: fraction of query terms in the chunk."""

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Score = |query terms ∩ chunk terms| / |query terms|."""
        terms = set(tokenize(query))
        scores = [
            len(terms & set(tokenize(c.chunk.text))) / len(terms) if terms else 0.0
            for c in candidates
        ]
        return _sorted(candidates, scores)
