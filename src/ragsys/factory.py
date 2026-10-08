"""Composition root: the only place that turns config into concrete objects.

Every other module depends on protocols. Tests (and the evaluation harness) build
the same graph with fakes by passing overrides to :func:`build_service`.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from ragsys.cache import MemoryCache, NullCache, RedisCache, ResponseCache
from ragsys.config import PipelineConfig, Settings
from ragsys.embeddings import Embedder, get_embedder
from ragsys.generation.abstention import AbstentionPolicy
from ragsys.generation.generator import AnswerGenerator
from ragsys.generation.llm import LLMClient, get_llm
from ragsys.indexing.bm25 import BM25Encoder
from ragsys.indexing.qdrant_store import QdrantStore, make_client
from ragsys.observability.tracing import Tracer, make_tracer
from ragsys.retrieval.pipeline import RetrievalPipeline
from ragsys.retrieval.reranker import FastEmbedReranker, LexicalOverlapReranker, Reranker
from ragsys.retrieval.retrievers import DenseRetriever, SparseRetriever
from ragsys.service import IndexVersion, RAGService


@dataclass
class Components:
    """Everything a running service holds (exposed for health checks/warmup)."""

    cfg: PipelineConfig
    client: QdrantClient
    store: QdrantStore
    embedder: Embedder
    reranker: Reranker | None
    llm: LLMClient
    cache: ResponseCache
    tracer: Tracer
    service: RAGService


def build_reranker(cfg: PipelineConfig) -> Reranker | None:
    """Cross-encoder if reranking is enabled."""
    rr = cfg.retrieval.rerank
    if not rr.enabled:
        return None
    if rr.provider == "lexical":
        return LexicalOverlapReranker()
    return FastEmbedReranker(rr.model)


def build_cache(cfg: PipelineConfig, settings: Settings) -> ResponseCache:
    """Response cache backend from config (redis requires REDIS_URL)."""
    if cfg.cache.backend == "redis":
        if not settings.redis_url:
            raise ValueError("cache.backend=redis but REDIS_URL is not set")
        return RedisCache(settings.redis_url)
    if cfg.cache.backend == "memory":
        return MemoryCache(cfg.cache.max_entries)
    return NullCache()


def build_retrieval(
    cfg: PipelineConfig, store: QdrantStore, embedder: Embedder, reranker: Reranker | None
) -> RetrievalPipeline:
    """Retrieval pipeline with only the enabled stages wired."""
    r = cfg.retrieval
    dense = DenseRetriever(store, embedder) if r.dense.enabled else None
    sparse = (
        SparseRetriever(store, BM25Encoder(r.sparse.k1, r.sparse.b)) if r.sparse.enabled else None
    )
    return RetrievalPipeline(r, dense, sparse, reranker if r.rerank.enabled else None)


def build_components(
    cfg: PipelineConfig,
    settings: Settings,
    *,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    llm: LLMClient | None = None,
    cache: ResponseCache | None = None,
    tracer: Tracer | None = None,
) -> Components:
    """Build the full object graph; any argument overrides the config-driven default."""
    client = client or make_client(settings.qdrant_url, settings.qdrant_api_key)
    store = QdrantStore(client, cfg.collection_name())
    embedder = embedder or get_embedder(cfg.embedding)
    if reranker is None and cfg.retrieval.rerank.enabled:
        reranker = build_reranker(cfg)
    llm = llm or get_llm(cfg.generation, settings.anthropic_api_key)
    cache = cache or build_cache(cfg, settings)
    tracer = tracer or make_tracer(
        cfg.tracing.enabled,
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
        settings.langfuse_host,
        cfg.tracing.sample_rate,
    )
    policy = AbstentionPolicy(cfg.generation.abstention)
    service = RAGService(
        cfg=cfg,
        retrieval=build_retrieval(cfg, store, embedder, reranker),
        generator=AnswerGenerator(llm, policy, cfg.generation.max_tokens),
        cache=cache,
        tracer=tracer,
        index_version=IndexVersion(lambda: str(store.metadata().get("index_version", "none"))),
    )
    return Components(cfg, client, store, embedder, reranker, llm, cache, tracer, service)
