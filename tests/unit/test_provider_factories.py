"""Provider factories pick the right backend and reject unknown names with a clear message."""

from __future__ import annotations

import pytest

from ragsys.cache import MemoryCache, NullCache
from ragsys.config import EmbeddingConfig, GenerationConfig, Settings
from ragsys.embeddings import HashEmbedder, get_embedder
from ragsys.factory import build_cache, build_reranker
from ragsys.generation.llm import ExtractiveLLM, get_llm
from ragsys.retrieval.reranker import LexicalOverlapReranker
from tests.conftest import offline_config


def test_hash_embedder_takes_its_dimension_from_config() -> None:
    embedder = get_embedder(EmbeddingConfig(provider="hash", dim=64))
    assert isinstance(embedder, HashEmbedder)
    assert embedder.dim == 64


def test_unknown_embedding_provider_is_named_in_the_error() -> None:
    cfg = EmbeddingConfig.model_construct(provider="word2vec")
    with pytest.raises(ValueError, match="word2vec"):
        get_embedder(cfg)


def test_extractive_llm_needs_no_api_key() -> None:
    assert isinstance(get_llm(GenerationConfig(provider="extractive"), None), ExtractiveLLM)


def test_unknown_llm_provider_is_named_in_the_error() -> None:
    cfg = GenerationConfig.model_construct(provider="gpt")
    with pytest.raises(ValueError, match="gpt"):
        get_llm(cfg, None)


@pytest.mark.parametrize(("backend", "expected"), [("memory", MemoryCache), ("none", NullCache)])
def test_build_cache_picks_the_configured_backend(backend: str, expected: type) -> None:
    cache = build_cache(offline_config(cache={"backend": backend}), Settings(_env_file=None))  # type: ignore[call-arg]
    assert isinstance(cache, expected)


def test_disabled_reranking_builds_no_reranker() -> None:
    cfg = offline_config(retrieval={"rerank": {"enabled": False}})
    assert build_reranker(cfg) is None


def test_lexical_reranker_needs_no_model_download() -> None:
    cfg = offline_config(retrieval={"rerank": {"enabled": True, "provider": "lexical"}})
    assert isinstance(build_reranker(cfg), LexicalOverlapReranker)
