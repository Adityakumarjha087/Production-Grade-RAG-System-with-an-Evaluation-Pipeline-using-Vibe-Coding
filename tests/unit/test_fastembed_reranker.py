"""FastEmbedReranker turns cross-encoder logits into sorted [0, 1] scores (model stubbed, no download)."""

from __future__ import annotations

import math

from ragsys.retrieval.reranker import FastEmbedReranker, sigmoid
from tests.unit.test_retrieval import sc


class StubModel:
    def __init__(self, logits: list[float]) -> None:
        self.logits = logits
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, texts: list[str], batch_size: int) -> list[float]:
        self.calls.append((query, texts, batch_size))
        return self.logits


def reranker_with(model: StubModel, batch_size: int = 16) -> FastEmbedReranker:
    reranker = FastEmbedReranker("stub-model", batch_size=batch_size)
    reranker._model = model  # type: ignore[assignment]
    return reranker


def test_candidates_come_back_sorted_by_sigmoid_of_the_logit() -> None:
    model = StubModel([-2.0, 3.0, 0.0])
    out = reranker_with(model, batch_size=4).rerank(
        "q", [sc("a", text="A"), sc("b", text="B"), sc("c", text="C")]
    )
    assert [s.chunk.chunk_id for s in out] == ["b", "c", "a"]
    assert out[0].score == sigmoid(3.0)
    assert out[1].score == 0.5
    assert all(0.0 < s.score < 1.0 and s.stage == "rerank" for s in out)
    assert model.calls == [("q", ["A", "B", "C"], 4)]


def test_no_candidates_never_loads_the_model() -> None:
    reranker = FastEmbedReranker("not-a-real-model")
    assert reranker.rerank("q", []) == []
    assert reranker._model is None


def test_sigmoid_is_safe_for_extreme_logits() -> None:
    assert sigmoid(-1000.0) == 0.0
    assert sigmoid(1000.0) == 1.0
    assert math.isclose(sigmoid(2.0) + sigmoid(-2.0), 1.0)
