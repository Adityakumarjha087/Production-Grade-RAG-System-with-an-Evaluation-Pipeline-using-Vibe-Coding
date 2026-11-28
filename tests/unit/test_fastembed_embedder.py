"""FastEmbedEmbedder converts model output to plain float lists (model stubbed, no download)."""

from __future__ import annotations

import numpy as np

from ragsys.embeddings.fastembed_embedder import FastEmbedEmbedder


class StubModel:
    def __init__(self) -> None:
        self.passage_calls: list[tuple[list[str], int]] = []
        self.query_calls: list[str] = []

    def passage_embed(self, texts: list[str], batch_size: int):
        self.passage_calls.append((texts, batch_size))
        return (np.full(3, float(i), dtype=np.float32) for i, _ in enumerate(texts))

    def query_embed(self, text: str):
        self.query_calls.append(text)
        return iter([np.array([0.5, 0.25, 0.125], dtype=np.float32)])


def embedder_with(model: StubModel, batch_size: int = 8) -> FastEmbedEmbedder:
    embedder = FastEmbedEmbedder("stub-model", 3, batch_size)
    embedder._model = model  # type: ignore[assignment]
    return embedder


def test_documents_are_embedded_in_batches_as_plain_float_lists() -> None:
    model = StubModel()
    vectors = embedder_with(model, batch_size=2).embed_documents(["a", "b", "c"])
    assert vectors == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    assert all(type(x) is float for v in vectors for x in v)
    assert model.passage_calls == [(["a", "b", "c"], 2)]


def test_query_uses_the_query_path_and_returns_one_vector() -> None:
    model = StubModel()
    assert embedder_with(model).embed_query("what is T+1?") == [0.5, 0.25, 0.125]
    assert model.query_calls == ["what is T+1?"]
    assert model.passage_calls == []


def test_warmup_loads_the_model_through_a_query() -> None:
    model = StubModel()
    embedder_with(model).warmup()
    assert model.query_calls == ["warmup"]
