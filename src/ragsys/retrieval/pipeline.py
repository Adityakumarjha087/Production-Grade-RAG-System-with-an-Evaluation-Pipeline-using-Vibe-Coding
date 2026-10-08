"""Configurable retrieval: dense? + sparse? -> RRF (if both) -> rerank? -> top-k.

Each stage's output is kept on the result so evaluation can measure recall
*per stage* (does the reranker actually help?) and tracing can show it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from ragsys.config import RetrievalConfig
from ragsys.models import ScoredChunk
from ragsys.observability.tracing import NoopSpan, Span
from ragsys.retrieval.fusion import rrf_fuse
from ragsys.retrieval.reranker import Reranker
from ragsys.retrieval.retrievers import DenseRetriever, SparseRetriever


@dataclass
class RetrievalResult:
    """Final chunks plus per-stage diagnostics."""

    query: str
    chunks: list[ScoredChunk]
    stages: dict[str, list[ScoredChunk]] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    confidence_source: str | None = None
    signals: dict[str, float] = field(default_factory=dict)


@contextmanager
def _timed(timings: dict[str, float], name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round((time.perf_counter() - start) * 1000, 2)


def _summary(hits: list[ScoredChunk], n: int = 10) -> list[dict[str, object]]:
    """Compact representation of hits for traces."""
    return [
        {"chunk_id": h.chunk.chunk_id, "doc": h.chunk.doc_title[:60], "score": round(h.score, 4)}
        for h in hits[:n]
    ]


class RetrievalPipeline:
    """See module docstring. Stages are toggled by :class:`RetrievalConfig`."""

    def __init__(
        self,
        cfg: RetrievalConfig,
        dense: DenseRetriever | None,
        sparse: SparseRetriever | None,
        reranker: Reranker | None,
    ) -> None:
        if cfg.dense.enabled and dense is None:
            raise ValueError("dense retrieval enabled but no DenseRetriever given")
        if cfg.sparse.enabled and sparse is None:
            raise ValueError("sparse retrieval enabled but no SparseRetriever given")
        if cfg.rerank.enabled and reranker is None:
            raise ValueError("rerank enabled but no Reranker given")
        self.cfg = cfg
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker

    def retrieve(self, query: str, span: Span | None = None) -> RetrievalResult:
        """Run the enabled stages for ``query``."""
        parent = span or NoopSpan()
        result = RetrievalResult(query=query, chunks=[])
        candidates = self._first_stage(query, parent, result)
        if self.cfg.rerank.enabled and self.reranker is not None:
            reranker, pool = self.reranker, candidates[: self.cfg.rerank.candidates]
            candidates = self._run_stage(
                parent, result, "rerank", lambda: reranker.rerank(query, pool)
            )
        result.chunks = candidates[: self.cfg.final_k]
        result.signals = self._signals(result)
        result.confidence, result.confidence_source = self._confidence(result)
        return result

    def _first_stage(self, query: str, parent: Span, result: RetrievalResult) -> list[ScoredChunk]:
        lists: list[list[ScoredChunk]] = []
        if self.cfg.dense.enabled and self.dense is not None:
            dense, k_dense = self.dense, self.cfg.dense.top_k
            lists.append(
                self._run_stage(parent, result, "dense", lambda: dense.retrieve(query, k_dense))
            )
        if self.cfg.sparse.enabled and self.sparse is not None:
            sparse, k_sparse = self.sparse, self.cfg.sparse.top_k
            lists.append(
                self._run_stage(parent, result, "sparse", lambda: sparse.retrieve(query, k_sparse))
            )
        if len(lists) == 1:
            return lists[0]
        return self._run_stage(
            parent, result, "rrf", lambda: rrf_fuse(lists, k=self.cfg.fusion.rrf_k)
        )

    @staticmethod
    def _run_stage(
        parent: Span, result: RetrievalResult, name: str, fn: Callable[[], list[ScoredChunk]]
    ) -> list[ScoredChunk]:
        with parent.child(f"retrieval.{name}", as_type="retriever", input=result.query) as span:
            with _timed(result.timings_ms, name):
                hits = fn()
            span.update(output=_summary(hits), metadata={"n": len(hits)})
        result.stages[name] = hits
        return hits

    @staticmethod
    def _signals(result: RetrievalResult) -> dict[str, float]:
        """Calibrated relevance signals: top cross-encoder score and max cosine.

        BM25 and RRF scores are not comparable across queries, so they are not signals.
        """
        signals: dict[str, float] = {}
        if "rerank" in result.stages:
            signals["rerank"] = max((h.score for h in result.stages["rerank"]), default=0.0)
        if "dense" in result.stages:
            signals["dense"] = max((h.score for h in result.stages["dense"]), default=0.0)
        return signals

    @staticmethod
    def _confidence(result: RetrievalResult) -> tuple[float | None, str | None]:
        """Primary signal for display: the cross-encoder if present, else cosine."""
        for source in ("rerank", "dense"):
            if source in result.signals:
                return result.signals[source], source
        return None, None
