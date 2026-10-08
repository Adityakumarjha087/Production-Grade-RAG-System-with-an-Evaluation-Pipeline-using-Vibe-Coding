"""Configuration: secrets from the environment, pipeline behaviour from YAML.

Two layers on purpose:

* ``Settings`` (pydantic-settings) reads *secrets and endpoints* from env / ``.env``.
* ``PipelineConfig`` is loaded from ``configs/base.yaml`` and optionally deep-merged
  with an experiment overlay. Everything that changes system *behaviour* lives here,
  so an evaluation run can snapshot it and be reproduced exactly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")


class Settings(BaseSettings):
    """Secrets and service endpoints (never committed)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str | None = None
    qdrant_url: str = ":memory:"
    qdrant_api_key: str | None = None
    redis_url: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    rag_config: Path = DEFAULT_CONFIG_PATH
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator(
        "anthropic_api_key",
        "qdrant_api_key",
        "redis_url",
        "langfuse_public_key",
        "langfuse_secret_key",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """``KEY=`` in .env means "not set" (so SDK credential discovery still works)."""
        return None if isinstance(value, str) and not value.strip() else value


class PathsConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    parsed_dir: Path = Path("data/parsed")
    testset_path: Path = Path("data/testsets/testset.jsonl")
    results_dir: Path = Path("eval/results")
    llm_cache_dir: Path = Path("data/cache/llm")


class ParsingConfig(BaseModel):
    parser: Literal["docling", "pymupdf"] = "docling"
    fallback_parser: Literal["pymupdf"] | None = "pymupdf"


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "recursive", "structural"] = "structural"
    chunk_size: int = Field(1200, gt=50, description="Max characters per chunk.")
    chunk_overlap: int = Field(150, ge=0)
    min_chunk_size: int = Field(200, ge=0, description="Structural: merge smaller sections.")


class EmbeddingConfig(BaseModel):
    provider: Literal["fastembed", "hash"] = "fastembed"
    model: str = "BAAI/bge-base-en-v1.5"
    dim: int = 768
    batch_size: int = 32


class IndexConfig(BaseModel):
    collection_prefix: str = "sebi"
    upsert_batch_size: int = 128


class StageConfig(BaseModel):
    enabled: bool = True
    top_k: int = 30


class SparseConfig(StageConfig):
    k1: float = 1.2
    b: float = 0.75


class FusionConfig(BaseModel):
    rrf_k: int = 60


class RerankConfig(BaseModel):
    enabled: bool = True
    provider: Literal["fastembed", "lexical"] = "fastembed"
    model: str = "Xenova/ms-marco-MiniLM-L-12-v2"
    candidates: int = 20


class RetrievalConfig(BaseModel):
    dense: StageConfig = Field(default_factory=StageConfig)
    sparse: SparseConfig = Field(default_factory=SparseConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    final_k: int = Field(6, gt=0)

    @model_validator(mode="after")
    def _needs_first_stage(self) -> RetrievalConfig:
        if not (self.dense.enabled or self.sparse.enabled):
            raise ValueError("at least one of retrieval.dense / retrieval.sparse must be enabled")
        return self


class AbstentionConfig(BaseModel):
    """Thresholds below which the system refuses to answer.

    ``min_rerank_score`` applies to the sigmoid of the cross-encoder logit;
    ``min_dense_score`` (cosine) is the fallback when the reranker is disabled.
    """

    enabled: bool = True
    min_rerank_score: float = 0.15
    min_dense_score: float = 0.55
    require_citation: bool = True


class GenerationConfig(BaseModel):
    provider: Literal["anthropic", "extractive"] = "anthropic"
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_tokens: int = 4096
    use_fallbacks: bool = True
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)


class CacheConfig(BaseModel):
    backend: Literal["redis", "memory", "none"] = "memory"
    ttl_seconds: int = 24 * 3600
    max_entries: int = 1024


class ModelPrice(BaseModel):
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


class TracingConfig(BaseModel):
    enabled: bool = True
    sample_rate: float = 1.0


class EvalConfig(BaseModel):
    judge_backend: Literal["ragas", "lexical"] = "ragas"
    judge_model: str = "claude-sonnet-5-5"
    testgen_model: str = "claude-opus-5-5"
    answer_subset: int = 30
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    relevance_overlap: float = 0.5
    seed: int = 13


class PipelineConfig(BaseModel):
    """Everything that defines system behaviour; hashed into cache keys."""

    name: str = "default"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    def collection_name(self) -> str:
        """Qdrant collection for this config's chunking strategy + embedding model."""
        model_slug = self.embedding.model.split("/")[-1].replace(".", "-").lower()
        return f"{self.index.collection_prefix}_{self.chunking.strategy}_{model_slug}"

    def fingerprint(self) -> str:
        """Stable hash of the behaviour-relevant config (paths/pricing excluded)."""
        payload = self.model_dump(mode="json", exclude={"name", "paths", "pricing", "eval"})
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base`` (overlay wins)."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Read a YAML mapping; an ``extends: other.yaml`` key inherits from that file.

    ``extends`` is resolved relative to the including file and may be chained.
    """
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular 'extends' in config: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    parent = data.pop("extends", None)
    if parent:
        data = deep_merge(load_yaml(path.parent / parent, _seen | {path}), data)
    return data


def load_config(path: Path | None = None, overlay: dict[str, Any] | None = None) -> PipelineConfig:
    """Load ``base.yaml`` (or ``path``) and apply an optional overlay dict."""
    data = load_yaml(path or DEFAULT_CONFIG_PATH)
    if overlay:
        data = deep_merge(data, overlay)
    return PipelineConfig.model_validate(data)
