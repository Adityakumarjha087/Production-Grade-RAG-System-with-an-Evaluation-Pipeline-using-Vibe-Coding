"""Shared fixtures.

Default tests are hermetic: no network, no model downloads, no API keys. They use
the hashing embedder, lexical reranker, extractive LLM and in-memory Qdrant —
real implementations of the same protocols, so behaviour (not mocks) is tested.
Tests needing downloads/services are marked ``slow`` / ``integration``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from ragsys.config import PipelineConfig, Settings, load_config
from ragsys.devtools.sample_corpus import build_sample_corpus, build_sample_testset
from ragsys.embeddings import HashEmbedder
from ragsys.evaluation.testset import TestItem
from ragsys.factory import Components, build_components
from ragsys.generation.llm import Completion, LLMEvent, TextDelta
from ragsys.indexing.bm25 import BM25Encoder
from ragsys.ingestion.chunkers import get_chunker
from ragsys.ingestion.parsers import PyMuPDFParser, RawElement, build_document
from ragsys.ingestion.pipeline import chunk_corpus, index_chunks
from ragsys.models import Document, ElementType, Usage
from ragsys.observability.tracing import NoopTracer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_env_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not pick up a developer's real keys or endpoints."""
    for var in (
        "ANTHROPIC_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "REDIS_URL",
        "QDRANT_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(ROOT)


def make_doc(blocks: list[tuple[str, str]], name: str = "doc") -> Document:
    """Build a Document from (kind, text) pairs without a PDF.

    kind: h1 | h2 | p | li | table. Four elements per page.
    """
    kinds = {
        "h1": (ElementType.HEADING, 1),
        "h2": (ElementType.HEADING, 2),
        "p": (ElementType.PARAGRAPH, None),
        "li": (ElementType.LIST_ITEM, None),
        "table": (ElementType.TABLE, None),
    }
    raw = [
        RawElement(kinds[k][0], text, page=1 + i // 4, level=kinds[k][1])
        for i, (k, text) in enumerate(blocks)
    ]
    pages = raw[-1].page if raw else 1
    return build_document(Path(f"{name}.pdf"), raw, "test", pages, doc_id=f"doc-{name}")


@pytest.fixture
def doc_factory() -> Any:
    return make_doc


@pytest.fixture(scope="session")
def sample_pdfs(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    return build_sample_corpus(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def sample_docs(sample_pdfs: list[Path]) -> list[Document]:
    parser = PyMuPDFParser()
    return [parser.parse(p) for p in sample_pdfs]


@pytest.fixture(scope="session")
def sample_testset(sample_docs: list[Document]) -> list[TestItem]:
    return build_sample_testset(sample_docs)


def offline_config(**overlay: Any) -> PipelineConfig:
    """Base config switched to hermetic components."""
    from ragsys.config import deep_merge

    base = {
        "name": "test",
        "parsing": {"parser": "pymupdf", "fallback_parser": None},
        "chunking": {
            "strategy": "structural",
            "chunk_size": 600,
            "chunk_overlap": 80,
            "min_chunk_size": 120,
        },
        "embedding": {"provider": "hash", "model": "hash", "dim": 256},
        "index": {"collection_prefix": "test"},
        "retrieval": {
            "dense": {"enabled": True, "top_k": 10},
            "sparse": {"enabled": True, "top_k": 10},
            "rerank": {"enabled": True, "provider": "lexical", "candidates": 10},
            "final_k": 4,
        },
        "generation": {
            "provider": "extractive",
            # measured for the hashing embedder + lexical reranker on the sample corpus
            "abstention": {"min_rerank_score": 0.3, "min_dense_score": 0.22},
        },
        "cache": {"backend": "memory"},
        "eval": {"judge_backend": "lexical", "answer_subset": 20, "ks": [1, 3, 5]},
    }
    return load_config(ROOT / "configs/base.yaml", deep_merge(base, overlay))


@pytest.fixture
def cfg() -> PipelineConfig:
    return offline_config()


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def qdrant() -> Iterator[QdrantClient]:
    client = QdrantClient(location=":memory:")
    yield client
    client.close()


def index_docs(docs: list[Document], cfg: PipelineConfig, client: QdrantClient) -> None:
    chunks = chunk_corpus(docs, get_chunker(cfg.chunking))
    index_chunks(
        chunks,
        alias=cfg.collection_name(),
        client=client,
        embedder=HashEmbedder(cfg.embedding.dim),
        bm25=BM25Encoder(cfg.retrieval.sparse.k1, cfg.retrieval.sparse.b),
    )


@pytest.fixture
def components(
    cfg: PipelineConfig, settings: Settings, qdrant: QdrantClient, sample_docs: list[Document]
) -> Components:
    index_docs(sample_docs, cfg, qdrant)
    return build_components(cfg, settings, client=qdrant, tracer=NoopTracer())


class ScriptedLLM:
    """LLM double returning a fixed answer; records calls."""

    def __init__(
        self, text: str, model: str = "claude-opus-5-5", stop_reason: str = "end_turn"
    ) -> None:
        self.text = text
        self.model = model
        self.stop_reason = stop_reason
        self.calls: list[dict[str, Any]] = []

    def _completion(self) -> Completion:
        return Completion(
            self.text, Usage(input_tokens=1000, output_tokens=100), self.model, self.stop_reason
        )

    async def stream(self, *, system: str, prompt: str, max_tokens: int) -> AsyncIterator[LLMEvent]:
        self.calls.append({"system": system, "prompt": prompt})
        for word in self.text.split(" "):
            yield TextDelta(word + " ")
        yield self._completion()

    async def complete(
        self, *, system: str, prompt: str, max_tokens: int, json_schema: Any = None
    ) -> Completion:
        self.calls.append({"system": system, "prompt": prompt, "schema": json_schema})
        return self._completion()


class FailingLLM:
    model = "claude-opus-5-5"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def stream(self, **_: Any) -> AsyncIterator[LLMEvent]:
        raise self.exc
        yield  # pragma: no cover

    async def complete(self, **_: Any) -> Completion:
        raise self.exc
