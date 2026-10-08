from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragsys.config import RetrievalConfig
from ragsys.factory import Components, build_retrieval
from ragsys.models import Chunk, ScoredChunk
from ragsys.retrieval.fusion import rrf_fuse
from ragsys.retrieval.pipeline import RetrievalPipeline
from ragsys.retrieval.reranker import LexicalOverlapReranker, sigmoid
from tests.conftest import offline_config


def sc(cid: str, score: float = 0.0, stage: str = "x", text: str = "") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=cid,
        doc_id="d",
        doc_title="T",
        text=text or cid,
        strategy="s",
        ordinal=0,
        char_start=0,
        char_end=1,
        page_start=1,
        page_end=1,
    )
    return ScoredChunk(chunk=chunk, score=score, stage=stage)


class TestRRF:
    def test_scores_follow_formula(self) -> None:
        fused = rrf_fuse([[sc("a"), sc("b")], [sc("b"), sc("c")]], k=60)
        scores = {f.chunk.chunk_id: f.score for f in fused}
        assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
        assert scores["a"] == pytest.approx(1 / 61)
        assert [f.chunk.chunk_id for f in fused] == ["b", "a", "c"]
        assert all(f.stage == "rrf" for f in fused)

    def test_agreement_beats_single_top_rank(self) -> None:
        # "c" is 2nd in both lists; "a" is 1st in one list only -> c wins
        fused = rrf_fuse([[sc("a"), sc("c")], [sc("b"), sc("c")]])
        assert fused[0].chunk.chunk_id == "c"

    def test_top_n_and_validation(self) -> None:
        assert len(rrf_fuse([[sc("a"), sc("b"), sc("c")]], top_n=2)) == 2
        with pytest.raises(ValueError):
            rrf_fuse([], k=0)

    def test_empty(self) -> None:
        assert rrf_fuse([[], []]) == []


def test_sigmoid_bounds() -> None:
    assert sigmoid(0) == 0.5
    assert sigmoid(-1000) == pytest.approx(0.0)
    assert sigmoid(1000) == pytest.approx(1.0)


def test_lexical_reranker_orders_by_overlap() -> None:
    hits = [sc("a", text="unrelated words"), sc("b", text="KYC records retention period")]
    ranked = LexicalOverlapReranker().rerank("KYC retention period", hits)
    assert [h.chunk.chunk_id for h in ranked] == ["b", "a"]
    assert ranked[0].score == pytest.approx(1.0) and ranked[0].stage == "rerank"


def test_config_requires_a_first_stage() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        RetrievalConfig.model_validate({"dense": {"enabled": False}, "sparse": {"enabled": False}})


def test_pipeline_rejects_missing_components() -> None:
    with pytest.raises(ValueError, match="dense"):
        RetrievalPipeline(RetrievalConfig(), None, None, None)


OFF = {"enabled": False}
ON = {"enabled": True}
# name: (overlay, expected stages, expected primary confidence source, expected signals)
VARIANTS = {
    "dense": ({"dense": ON, "sparse": OFF, "rerank": OFF}, {"dense"}, "dense", {"dense"}),
    "bm25": ({"dense": OFF, "sparse": ON, "rerank": OFF}, {"sparse"}, None, set()),
    "hybrid": ({"rerank": OFF}, {"dense", "sparse", "rrf"}, "dense", {"dense"}),
    "hybrid_rerank": ({}, {"dense", "sparse", "rrf", "rerank"}, "rerank", {"rerank", "dense"}),
}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_stage_toggles(name: str, components: Components) -> None:
    overlay, stages, conf_source, signals = VARIANTS[name]
    cfg = offline_config(retrieval=overlay)
    from ragsys.retrieval.reranker import LexicalOverlapReranker as Lex

    pipeline = build_retrieval(cfg, components.store, components.embedder, Lex())
    result = pipeline.retrieve("How long must KYC records be preserved?")
    assert set(result.stages) == stages
    assert result.confidence_source == conf_source
    assert set(result.signals) == signals
    assert 0 < len(result.chunks) <= cfg.retrieval.final_k
    assert set(result.timings_ms) == stages
    assert "KYC" in result.chunks[0].chunk.doc_title


@pytest.mark.parametrize(
    ("query", "title_word"),
    [
        ("pay-out of funds and securities deadline T+1", "Settlement"),
        ("maximum TER first 500 crore", "Expense"),
        ("complaint resolution 21 calendar days", "Grievance"),
        ("SAMPLE/HO/MIRSD/2024/014", "KYC"),  # exact circular number -> BM25 strength
    ],
)
def test_hybrid_finds_right_document(query: str, title_word: str, components: Components) -> None:
    result = components.service.retrieval.retrieve(query)
    assert title_word in result.chunks[0].chunk.doc_title
