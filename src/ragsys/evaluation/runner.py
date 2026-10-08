"""Run every configuration in an experiment matrix and collect comparable metrics.

Matrix = chunkers × retrieval variants (``configs/experiments/matrix.yaml``).
Indexes are built once per chunker (variants sharing a chunker share an index);
models are loaded once and shared across variants; LLM calls go through a disk
cache so re-runs are free.
"""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import structlog
from qdrant_client import QdrantClient

from ragsys.cache import NullCache
from ragsys.config import PipelineConfig, Settings, deep_merge, load_yaml
from ragsys.embeddings import Embedder, get_embedder
from ragsys.evaluation.answer_metrics import (
    AnswerJudge,
    AnswerSample,
    abstention_metrics,
    aggregate_answer_scores,
)
from ragsys.evaluation.retrieval_metrics import aggregate_retrieval
from ragsys.evaluation.testset import EvidenceSpan, TestItem, reanchor_evidence
from ragsys.factory import Components, build_components, build_reranker
from ragsys.generation.llm import LLMClient
from ragsys.indexing.bm25 import BM25Encoder
from ragsys.indexing.qdrant_store import point_id
from ragsys.ingestion.chunkers import get_chunker
from ragsys.ingestion.pipeline import chunk_corpus, index_chunks
from ragsys.models import Chunk, Document
from ragsys.observability.tracing import NoopTracer
from ragsys.retrieval.reranker import Reranker

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Variant:
    """One cell of the matrix."""

    name: str
    chunker: str
    retrieval: str
    overlay: dict[str, Any]


@dataclass
class VariantResult:
    """Metrics for one variant."""

    name: str
    chunker: str
    retrieval: str
    n_retrieval: int
    metrics: dict[str, float] = field(default_factory=dict)
    stage_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    n_answers: int = 0
    cost_usd: float = 0.0


def expand_matrix(matrix: dict[str, Any]) -> list[Variant]:
    """Cartesian product of ``chunkers`` and ``retrieval`` overlays."""
    chunkers: dict[str, Any] = matrix.get("chunkers", {})
    retrievals: dict[str, Any] = matrix.get("retrieval", {})
    if not chunkers or not retrievals:
        raise ValueError("matrix needs non-empty 'chunkers' and 'retrieval' sections")
    return [
        Variant(f"{c}+{r}", c, r, deep_merge(chunkers[c] or {}, retrievals[r] or {}))
        for c, r in product(chunkers, retrievals)
    ]


def load_matrix(path: Path) -> list[Variant]:
    """Read and expand a matrix YAML file."""
    return expand_matrix(load_yaml(path))


def answer_subset(items: Sequence[TestItem], n: int, seed: int) -> list[TestItem]:
    """Deterministic sample for (expensive) answer evaluation."""
    pool = list(items)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


class EvalRunner:
    """See module docstring."""

    def __init__(
        self,
        base: PipelineConfig,
        settings: Settings,
        items: Sequence[TestItem],
        docs: Sequence[Document],
        *,
        client: QdrantClient,
        llm: LLMClient | None,
        judge: AnswerJudge | None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.base = base
        self.settings = settings
        self.docs = list(docs)
        self.items, self.dropped = reanchor_evidence(items, self.docs)
        self.client = client
        self.llm = llm
        self.judge = judge
        self.embedder = embedder or get_embedder(base.embedding)
        self._fixed_reranker = reranker  # injected (tests): used for every variant
        self._rerankers: dict[tuple[str, str], Reranker | None] = {}
        self._indexed: set[str] = set()

    def _reranker_for(self, cfg: PipelineConfig) -> Reranker | None:
        """One loaded model per (provider, model) — variants may compare rerankers."""
        rr = cfg.retrieval.rerank
        if not rr.enabled:
            return None
        if self._fixed_reranker is not None:
            return self._fixed_reranker
        key = (rr.provider, rr.model)
        if key not in self._rerankers:
            self._rerankers[key] = build_reranker(cfg)
        return self._rerankers[key]

    def _components(self, cfg: PipelineConfig) -> Components:
        return build_components(
            cfg,
            self.settings,
            client=self.client,
            embedder=self.embedder,
            reranker=self._reranker_for(cfg),
            llm=self.llm or _NoLLM(),
            cache=NullCache(),
            tracer=NoopTracer(),
        )

    def ensure_index(self, cfg: PipelineConfig) -> None:
        """Build the collection for this chunker once per run (no-op if up to date)."""
        alias = cfg.collection_name()
        if alias in self._indexed:
            return
        chunks = chunk_corpus(self.docs, get_chunker(cfg.chunking))
        bm25 = BM25Encoder(cfg.retrieval.sparse.k1, cfg.retrieval.sparse.b)
        index_chunks(
            chunks,
            alias=alias,
            client=self.client,
            embedder=self.embedder,
            bm25=bm25,
            batch_size=cfg.index.upsert_batch_size,
        )
        self._indexed.add(alias)

    def evaluate_retrieval(self, comps: Components, result: VariantResult) -> None:
        """Recall@k/MRR for the final list and per stage, plus retrieval latency."""
        ks, thr = self.base.eval.ks, self.base.eval.relevance_overlap
        answerable = [i for i in self.items if i.answerable and i.evidence]
        final_runs: list[tuple[list[Chunk], list[EvidenceSpan]]] = []
        stage_runs: dict[str, list[tuple[list[Chunk], list[EvidenceSpan]]]] = {}
        latencies: list[float] = []
        for item in answerable:
            start = time.perf_counter()
            res = comps.service.retrieval.retrieve(item.question)
            latencies.append((time.perf_counter() - start) * 1000)
            final_runs.append(([h.chunk for h in res.chunks], item.evidence))
            for stage, hits in res.stages.items():
                stage_runs.setdefault(stage, []).append(([h.chunk for h in hits], item.evidence))
        result.n_retrieval = len(answerable)
        result.metrics.update(aggregate_retrieval(final_runs, ks, thr))
        result.metrics["retrieval_p50_ms"] = _percentile(latencies, 50)
        result.metrics["retrieval_p95_ms"] = _percentile(latencies, 95)
        result.stage_metrics = {s: aggregate_retrieval(r, ks, thr) for s, r in stage_runs.items()}

    async def evaluate_answers(self, comps: Components, result: VariantResult) -> None:
        """Generate answers on a fixed subset and judge them."""
        if self.llm is None or self.judge is None:
            return
        subset = answer_subset(self.items, self.base.eval.answer_subset, self.base.eval.seed)
        samples = []
        for item in subset:
            answer = await comps.service.answer(item.question)
            result.cost_usd += answer.cost_usd
            samples.append(
                AnswerSample(
                    question=item.question,
                    answer=answer.text,
                    contexts=context_texts(comps, [src.chunk_id for src in answer.sources]),
                    reference=item.reference_answer,
                    abstained=answer.abstained,
                    answerable=item.answerable,
                )
            )
        scores = await self.judge.score(samples)
        result.n_answers = len(samples)
        result.metrics.update(aggregate_answer_scores(scores))
        result.metrics.update(abstention_metrics(samples))

    async def run_variant(self, variant: Variant, *, answers: bool = True) -> VariantResult:
        """Evaluate one variant."""
        cfg = PipelineConfig.model_validate(
            deep_merge(self.base.model_dump(mode="json"), {**variant.overlay, "name": variant.name})
        )
        self.ensure_index(cfg)
        comps = self._components(cfg)
        result = VariantResult(variant.name, variant.chunker, variant.retrieval, 0)
        self.evaluate_retrieval(comps, result)
        if answers:
            await self.evaluate_answers(comps, result)
        log.info("eval.variant_done", variant=variant.name, **result.metrics)
        return result

    async def run(
        self, variants: Sequence[Variant], *, answers: bool = True
    ) -> list[VariantResult]:
        """Evaluate all variants sequentially (models are shared, so this is CPU-bound)."""
        return [await self.run_variant(v, answers=answers) for v in variants]


def context_texts(comps: Components, chunk_ids: Sequence[str]) -> list[str]:
    """Full text of the context chunks (``Answer.sources`` only carries snippets)."""
    if not chunk_ids:
        return []
    points = comps.client.retrieve(
        comps.store.collection, ids=[point_id(c) for c in chunk_ids], with_payload=True
    )
    by_id = {str(p.payload["chunk_id"]): str(p.payload["text"]) for p in points if p.payload}
    return [by_id.get(c, "") for c in chunk_ids]


class _NoLLM:
    """Placeholder LLM for retrieval-only runs; generation must not be called."""

    model = "none"

    async def complete(self, **_: Any) -> Any:
        raise RuntimeError("no LLM configured for this evaluation run")

    def stream(self, **_: Any) -> Any:
        raise RuntimeError("no LLM configured for this evaluation run")
