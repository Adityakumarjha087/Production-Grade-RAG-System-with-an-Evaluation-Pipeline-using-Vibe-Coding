from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from ragsys.evaluation.answer_metrics import (
    AnswerSample,
    AnswerScore,
    LexicalJudge,
    RagasJudge,
    abstention_metrics,
    aggregate_answer_scores,
    make_judge,
)
from ragsys.evaluation.report import columns_for, to_markdown, write_report
from ragsys.evaluation.retrieval_metrics import (
    aggregate_retrieval,
    is_relevant,
    recall_at_k,
    reciprocal_rank,
)
from ragsys.evaluation.review import apply_decision, review_stats, run_review
from ragsys.evaluation.runner import EvalRunner, VariantResult, answer_subset, expand_matrix
from ragsys.evaluation.testset import (
    EvidenceSpan,
    QType,
    ReviewStatus,
    TestItem,
    TestsetGenerator,
    doc_passages,
    hash_testset,
    load_testset,
    locate_quote,
    parse_item,
    reanchor_evidence,
    sample_passages,
    save_testset,
)
from ragsys.generation.llm import ExtractiveLLM
from ragsys.generation.prompts import ABSTAIN_TEXT
from ragsys.models import Chunk, Document
from tests.conftest import ScriptedLLM, offline_config


def chunk(start: int, end: int, doc: str = "d") -> Chunk:
    return Chunk(
        chunk_id=f"{doc}{start}",
        doc_id=doc,
        doc_title="T",
        text="x",
        strategy="s",
        ordinal=0,
        char_start=start,
        char_end=end,
        page_start=1,
        page_end=1,
    )


def span(start: int, end: int, doc: str = "d") -> EvidenceSpan:
    return EvidenceSpan(doc_id=doc, char_start=start, char_end=end, text="t")


class TestRetrievalMetrics:
    def test_relevance_rule(self) -> None:
        assert is_relevant(chunk(0, 100), span(10, 50))  # chunk contains span
        assert is_relevant(chunk(40, 60), span(0, 100))  # chunk inside long span
        assert not is_relevant(chunk(0, 100), span(90, 200))  # 10/100 overlap
        assert not is_relevant(chunk(0, 100, "other"), span(10, 50))

    def test_recall_and_mrr(self) -> None:
        ranked = [chunk(500, 600), chunk(0, 100), chunk(200, 300)]
        spans = [span(10, 50), span(210, 260)]
        assert recall_at_k(ranked, spans, 1) == 0.0
        assert recall_at_k(ranked, spans, 2) == 0.5
        assert recall_at_k(ranked, spans, 3) == 1.0
        assert reciprocal_rank(ranked, spans) == 0.5
        assert reciprocal_rank([chunk(900, 950)], spans) == 0.0

    def test_recall_needs_spans(self) -> None:
        with pytest.raises(ValueError):
            recall_at_k([], [], 1)

    def test_aggregate_skips_items_without_evidence(self) -> None:
        runs = [([chunk(0, 100)], [span(0, 50)]), ([chunk(0, 100)], [])]
        assert aggregate_retrieval(runs, [1]) == {"recall@1": 1.0, "mrr": 1.0}
        assert aggregate_retrieval([], [1]) == {}


class TestTestset:
    def test_locate_quote(self) -> None:
        text = "Records shall be preserved\nfor eight   years. Other text."
        assert locate_quote(text, "for eight years.") is not None
        s, e = locate_quote(text, '"preserved for EIGHT years"')  # type: ignore[misc]
        assert text[s:e].startswith("preserved")
        assert locate_quote(text, "ten years") is None
        assert locate_quote(text, "abc") is None  # too short to be meaningful

    def test_parse_item_validates_quotes(self, sample_docs: list[Document]) -> None:
        passage = next(p for p in doc_passages(sample_docs[1]) if "eight years" in p.text)
        good = {
            "question": "How long are KYC records kept?",
            "answer": "Eight years.",
            "evidence_quotes": ["for a period of eight years"],
        }
        item = parse_item(good, QType.FACTUAL, [passage], "m")
        assert item is not None and item.status == ReviewStatus.PENDING
        ev = item.evidence[0]
        assert sample_docs[1].text[ev.char_start : ev.char_end] == ev.text
        hallucinated = {**good, "evidence_quotes": ["for a period of nine years"]}
        assert parse_item(hallucinated, QType.FACTUAL, [passage], "m") is None
        assert parse_item({**good, "evidence_quotes": []}, QType.FACTUAL, [passage], "m") is None

    def test_unanswerable_item(self, sample_docs: list[Document]) -> None:
        passage = doc_passages(sample_docs[0])[0]
        item = parse_item(
            {"question": "What is the penalty for X?", "answer": "n/a", "evidence_quotes": []},
            QType.UNANSWERABLE,
            [passage],
            "m",
        )
        assert item is not None and not item.answerable and item.reference_answer == ABSTAIN_TEXT

    def test_passages_cover_docs_and_tables(self, sample_docs: list[Document]) -> None:
        picked = sample_passages(sample_docs, 4, random.Random(0))
        assert len({p.doc.doc_id for p in picked}) == 4  # round-robin across docs
        tables = sample_passages(sample_docs, 5, random.Random(0), kind="table")
        assert tables and all(p.text.startswith("|") for p in tables)

    async def test_generator_with_scripted_llm(self, sample_docs: list[Document]) -> None:
        reply = json.dumps(
            {
                "question": "How long must KYC records be preserved?",
                "answer": "Eight years.",
                "evidence_quotes": ["for a period of eight years"],
            }
        )
        llm = ScriptedLLM(reply)
        gen = TestsetGenerator(llm, seed=1)
        items = await gen.generate([sample_docs[1]], 3, {QType.FACTUAL: 1.0})
        assert items and llm.calls[0]["schema"]["required"] == [
            "question",
            "answer",
            "evidence_quotes",
        ]

    async def test_generator_drops_failures(self, sample_docs: list[Document]) -> None:
        gen = TestsetGenerator(ExtractiveLLM())  # raises on json_schema -> item dropped
        assert await gen.generate(sample_docs, 2, {QType.FACTUAL: 1.0}) == []

    def test_save_load_hash(self, tmp_path: Path, sample_testset: list[TestItem]) -> None:
        path = tmp_path / "t.jsonl"
        save_testset(path, sample_testset)
        loaded = load_testset(path)
        assert loaded == sample_testset
        assert hash_testset(loaded) == hash_testset(sample_testset)

    def test_reanchor_after_offset_shift(
        self, sample_testset: list[TestItem], sample_docs: list[Document]
    ) -> None:
        shifted = [d.model_copy(update={"text": "PREFIX " + d.text}) for d in sample_docs]
        kept, dropped = reanchor_evidence(sample_testset, shifted)
        assert dropped == [] and len(kept) == len(sample_testset)
        ev = next(i for i in kept if i.evidence).evidence[0]
        doc = next(d for d in shifted if d.doc_id == ev.doc_id)
        assert doc.text[ev.char_start : ev.char_end] == ev.text

    def test_reanchor_drops_missing_evidence(self, sample_testset: list[TestItem]) -> None:
        kept, dropped = reanchor_evidence(sample_testset, [])
        assert len(kept) == sum(1 for i in sample_testset if not i.evidence) and dropped


class TestReview:
    def _item(self) -> TestItem:
        return TestItem(id="1", question="Q?", reference_answer="A", qtype=QType.FACTUAL)

    def test_decisions(self) -> None:
        assert apply_decision(self._item(), "approve").status == ReviewStatus.APPROVED
        rejected = apply_decision(self._item(), "reject", note="ambiguous")
        assert rejected.status == ReviewStatus.REJECTED and rejected.notes == "ambiguous"
        edited = apply_decision(self._item(), "edit", question=" New Q? ")
        assert edited.question == "New Q?" and edited.reference_answer == "A"
        assert edited.status == ReviewStatus.APPROVED
        with pytest.raises(ValueError):
            apply_decision(self._item(), "maybe")

    def test_interactive_loop_saves_progress(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"
        items = [self._item().model_copy(update={"id": str(i)}) for i in range(4)]
        save_testset(path, items)
        answers = iter(["a", "r", "bad reason", "e", "Better Q?", "", "q"])
        shown: list[str] = []
        stats = run_review(path, ask=lambda _: next(answers), show=shown.append)
        saved = load_testset(path)
        assert [i.status for i in saved] == [
            ReviewStatus.APPROVED,
            ReviewStatus.REJECTED,
            ReviewStatus.APPROVED,
            ReviewStatus.PENDING,
        ]
        assert saved[2].question == "Better Q?"
        assert stats["status:pending"] == 1 and stats["total"] == 4
        assert review_stats(saved)["type:factual"] == 4


class TestAnswerMetrics:
    SAMPLES = [
        AnswerSample(
            "q1",
            "records kept eight years [1]",
            ["KYC records kept for eight years"],
            "eight years",
            False,
            True,
        ),
        AnswerSample("q2", ABSTAIN_TEXT, [], "x", True, True),  # false abstention
        AnswerSample("q3", ABSTAIN_TEXT, [], ABSTAIN_TEXT, True, False),  # correct abstention
        AnswerSample(
            "q4", "made up [1]", ["ctx"], ABSTAIN_TEXT, False, False
        ),  # should have abstained
    ]

    async def test_lexical_judge(self) -> None:
        scores = await LexicalJudge().score(self.SAMPLES)
        assert scores[0].faithfulness == pytest.approx(1.0) and scores[0].correctness > 0.5
        assert scores[1] == AnswerScore(None, 0.0)  # abstained on answerable: wrong
        assert scores[2] == AnswerScore(None, None)

    def test_abstention_metrics(self) -> None:
        m = abstention_metrics(self.SAMPLES)
        assert m == {"abstain_precision": 0.5, "abstain_recall": 0.5, "false_abstain_rate": 0.5}

    def test_aggregate(self) -> None:
        agg = aggregate_answer_scores(
            [AnswerScore(1.0, 0.5), AnswerScore(None, 0.0), AnswerScore(0.5, None)]
        )
        assert agg == {"faithfulness": 0.75, "correctness": 0.25}

    def test_ragas_judge_builds_claude_llm_without_sampling_params(self) -> None:
        judge = make_judge("ragas", "claude-sonnet-5-5", "test-key")
        assert isinstance(judge, RagasJudge)
        wrapper = judge._llm()
        assert wrapper.bypass_temperature and wrapper.bypass_n
        from langchain_core.messages import HumanMessage

        payload = wrapper.langchain_llm._get_request_payload([HumanMessage("hi")])
        assert payload["model"] == "claude-sonnet-5-5"
        assert "temperature" not in payload and "top_p" not in payload

    async def test_ragas_skips_llm_when_nothing_judgeable(self) -> None:
        scores = await RagasJudge("m").score(self.SAMPLES[1:3])
        assert scores == [AnswerScore(None, 0.0), AnswerScore(None, None)]


def test_answer_subset_deterministic(sample_testset: list[TestItem]) -> None:
    a = answer_subset(sample_testset, 5, seed=1)
    assert a == answer_subset(sample_testset, 5, seed=1) and len(a) == 5


def test_runner_loads_one_reranker_per_model(qdrant, sample_docs, settings) -> None:
    runner = EvalRunner(
        offline_config(), settings, [], sample_docs, client=qdrant, llm=None, judge=None
    )
    lexical = offline_config()
    other = offline_config(retrieval={"rerank": {"provider": "fastembed", "model": "m2"}})
    assert runner._reranker_for(lexical) is runner._reranker_for(lexical)
    assert runner._reranker_for(other) is not runner._reranker_for(lexical)
    assert runner._reranker_for(offline_config(retrieval={"rerank": {"enabled": False}})) is None


def test_expand_matrix_requires_sections() -> None:
    with pytest.raises(ValueError):
        expand_matrix({"chunkers": {"a": {}}})


async def test_runner_end_to_end(
    tmp_path: Path, qdrant, sample_docs, sample_testset, settings
) -> None:
    cfg = offline_config(paths={"results_dir": str(tmp_path)})
    matrix = expand_matrix(
        {
            "chunkers": {"fixed": {"chunking": {"strategy": "fixed"}}, "structural": {}},
            "retrieval": {
                "bm25": {"retrieval": {"dense": {"enabled": False}, "rerank": {"enabled": False}}},
                "hybrid_rerank": {},
            },
        }
    )
    from ragsys.retrieval.reranker import LexicalOverlapReranker

    runner = EvalRunner(
        cfg,
        settings,
        sample_testset,
        sample_docs,
        client=qdrant,
        llm=ExtractiveLLM(),
        judge=LexicalJudge(),
        reranker=LexicalOverlapReranker(),
    )
    results = await runner.run(matrix)
    assert [r.name for r in results] == [
        "fixed+bm25",
        "fixed+hybrid_rerank",
        "structural+bm25",
        "structural+hybrid_rerank",
    ]
    for r in results:
        assert r.n_retrieval == 9 and r.n_answers == len(sample_testset)
        assert {
            "recall@1",
            "recall@5",
            "mrr",
            "faithfulness",
            "correctness",
            "abstain_recall",
        } <= set(r.metrics)
        assert r.metrics["recall@5"] >= 0.6  # the small corpus is easy; guards regressions
    best = results[-1]
    assert set(best.stage_metrics) == {"dense", "sparse", "rrf", "rerank"}
    assert len(runner._indexed) == 2  # one index per chunker, shared by retrieval variants

    out = write_report(results, tmp_path, {"n_items": 11}, cfg.model_dump(mode="json"))
    assert {p.name for p in out.iterdir()} == {
        "report.md",
        "results.csv",
        "results.json",
        "config.yaml",
    }
    report = (out / "report.md").read_text()
    assert "| structural+hybrid_rerank |" in report and "**" in report
    assert "Recall by retrieval stage" in report


def test_markdown_table_formats() -> None:
    rows = [
        VariantResult("a", "c", "r", 1, {"recall@1": 0.5, "retrieval_p50_ms": 12.3}),
        VariantResult("b", "c", "r", 1, {"recall@1": 0.7, "retrieval_p50_ms": 20.0}),
    ]
    md = to_markdown(rows, columns_for(rows))
    assert "| a | 0.500 | **12** |" in md and "| b | **0.700** | 20 |" in md


class TestSpecs:
    def test_build_from_specs(self, sample_docs: list[Document], tmp_path: Path) -> None:
        from ragsys.evaluation.testset import TestSpec, build_from_specs, load_specs

        spec_file = tmp_path / "s.yaml"
        spec_file.write_text(
            "items:\n"
            "  - {qtype: factual, doc: kyc_records, question: 'How long are KYC records kept?',\n"
            "     answer: Eight years., quotes: ['for a period of eight years']}\n"
            "  - {qtype: factual, doc: kyc_records, question: 'Wrong doc for this quote?',\n"
            "     answer: x, quotes: ['shall resolve the complaint within 21 calendar days']}\n"
            "  - {qtype: factual, doc: nope, question: 'Unknown document?', answer: x, quotes: [abc def]}\n"
            "  - {qtype: unanswerable, question: 'What is the penalty for X?'}\n"
        )
        specs = load_specs(spec_file)
        assert all(isinstance(s, TestSpec) for s in specs)
        items, errors = build_from_specs(specs, sample_docs)
        assert [i.question for i in items] == [
            "How long are KYC records kept?",
            "What is the penalty for X?",
        ]
        assert all(i.status == ReviewStatus.PENDING for i in items)
        assert items[1].reference_answer == ABSTAIN_TEXT and not items[1].evidence
        ev = items[0].evidence[0]
        doc = next(d for d in sample_docs if d.doc_id == ev.doc_id)
        assert "kyc_records" in doc.source_path and doc.text[ev.char_start : ev.char_end] == ev.text
        assert len(errors) == 2 and "quote not found" in errors[0] and "not in corpus" in errors[1]


def test_repo_seed_specs_are_valid() -> None:
    from ragsys.evaluation.testset import load_specs

    specs = load_specs(Path("configs/testsets/sebi_seed.yaml"))
    assert len(specs) >= 25
    assert {s.qtype for s in specs} >= {QType.FACTUAL, QType.MULTI_HOP, QType.UNANSWERABLE}
    assert all(s.quotes for s in specs if s.qtype != QType.UNANSWERABLE)
