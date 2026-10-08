from __future__ import annotations

import pytest

from ragsys.config import AbstentionConfig
from ragsys.generation.abstention import (
    LOW_CONFIDENCE,
    MODEL_ABSTAINED,
    NO_CITATIONS,
    NO_RESULTS,
    REFUSED,
    AbstentionPolicy,
    is_abstention,
)
from ragsys.generation.citations import extract_markers, resolve_citations
from ragsys.generation.generator import REFUSAL_TEXT, AnswerGenerator, GenerationOutcome
from ragsys.generation.llm import Completion, ExtractiveLLM, TextDelta
from ragsys.generation.llm.disk_cache import DiskCachedLLM
from ragsys.generation.llm.extractive import extract_answer
from ragsys.generation.prompts import ABSTAIN_TEXT, SYSTEM_PROMPT, build_user_prompt
from ragsys.models import Usage
from tests.conftest import ScriptedLLM
from tests.unit.test_retrieval import sc

CONTEXT = [
    sc("a", text="KYC records must be preserved for eight years after the relationship ends."),
    sc("b", text="Pay-out of funds happens on T+1 by 1:30 PM."),
]


class TestCitations:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Fact [1]. Other [2].", [1, 2]),
            ("Both [1, 2] and again [1].", [1, 2]),
            ("Range [2-4] and [3–5].", [2, 3, 4, 5]),
            ("Adjacent [3][1].", [3, 1]),
            ("No markers, [a] or [] here.", []),
        ],
    )
    def test_extract_markers(self, text: str, expected: list[int]) -> None:
        assert extract_markers(text) == expected

    def test_huge_ranges_ignored(self) -> None:
        assert extract_markers("[1-500]") == []

    def test_resolve_separates_invalid(self) -> None:
        valid, invalid = resolve_citations("A [1]. B [7]. C [0].", CONTEXT)
        assert [c.marker for c in valid] == [1]
        assert valid[0].chunk_id == "a"
        assert invalid == [7, 0]


class TestAbstention:
    @pytest.mark.parametrize(
        "text",
        [
            ABSTAIN_TEXT,
            "i don’t know based on the provided documents",
            "I don't know based on the provided documents. Sorry.",
        ],
    )
    def test_detects_abstention_variants(self, text: str) -> None:
        assert is_abstention(text)

    def test_normal_answer_not_abstention(self) -> None:
        assert not is_abstention("Records are kept for eight years [1].")

    @pytest.mark.parametrize(
        ("signals", "expected"),
        [
            ({"rerank": 0.2}, LOW_CONFIDENCE),
            ({"rerank": 0.4}, None),
            ({"dense": 0.4}, LOW_CONFIDENCE),
            ({"rerank": 0.001, "dense": 0.4}, LOW_CONFIDENCE),  # both agree: off-topic
            ({"rerank": 0.001, "dense": 0.7}, None),  # paraphrase: dense still matches
            ({"rerank": 0.9, "dense": 0.3}, None),
            ({}, None),  # BM25-only: no calibrated signal -> leave it to the LLM
        ],
    )
    def test_pre_check_requires_all_signals_low(self, signals: dict, expected: str | None) -> None:
        policy = AbstentionPolicy(AbstentionConfig(min_rerank_score=0.3, min_dense_score=0.5))
        assert policy.pre_check(signals, 3) == expected

    def test_no_results(self) -> None:
        assert AbstentionPolicy(AbstentionConfig()).pre_check({"rerank": 1.0}, 0) == NO_RESULTS

    def test_disabled_gate_still_requires_results(self) -> None:
        policy = AbstentionPolicy(AbstentionConfig(enabled=False))
        assert policy.pre_check({"rerank": 0.0}, 3) is None
        assert policy.pre_check({"rerank": 0.0}, 0) == NO_RESULTS

    def test_post_check(self) -> None:
        policy = AbstentionPolicy(AbstentionConfig())
        valid, _ = resolve_citations("x [1]", CONTEXT)
        assert policy.post_check(ABSTAIN_TEXT, []) == MODEL_ABSTAINED
        assert policy.post_check("uncited claim", []) == NO_CITATIONS
        assert policy.post_check("cited [1]", valid) is None


def test_prompt_numbers_passages_and_puts_question_last() -> None:
    prompt = build_user_prompt("  How long?  ", CONTEXT)
    assert '<passage id="1">\n[1] T — p. 1\nKYC records' in prompt
    assert '<passage id="2">' in prompt
    assert prompt.endswith("<question>\nHow long?\n</question>")
    assert "{" not in SYSTEM_PROMPT and ABSTAIN_TEXT in SYSTEM_PROMPT


class TestExtractiveLLM:
    def test_answers_with_citation(self) -> None:
        answer = extract_answer(build_user_prompt("How long are KYC records preserved?", CONTEXT))
        assert answer.startswith("KYC records must be preserved for eight years")
        assert "[1]" in answer

    def test_abstains_when_unrelated(self) -> None:
        assert (
            extract_answer(build_user_prompt("What is the penalty for insider trading?", CONTEXT))
            == ABSTAIN_TEXT
        )

    async def test_stream_then_completion(self) -> None:
        llm = ExtractiveLLM()
        events = [
            e
            async for e in llm.stream(
                system="s", prompt=build_user_prompt("pay-out T+1 time", CONTEXT), max_tokens=10
            )
        ]
        assert all(isinstance(e, TextDelta) for e in events[:-1])
        final = events[-1]
        assert isinstance(final, Completion)
        assert "".join(e.text for e in events[:-1]) == final.text
        assert final.usage.input_tokens > 0

    async def test_structured_output_unsupported(self) -> None:
        with pytest.raises(NotImplementedError):
            await ExtractiveLLM().complete(system="", prompt="", max_tokens=1, json_schema={})


class TestGenerator:
    def _gen(self, text: str, stop: str = "end_turn") -> AnswerGenerator:
        return AnswerGenerator(
            ScriptedLLM(text, stop_reason=stop), AbstentionPolicy(AbstentionConfig()), 100
        )

    async def _run(self, gen: AnswerGenerator) -> tuple[str, GenerationOutcome]:
        streamed, outcome = "", None
        async for ev in gen.stream("q", CONTEXT):
            if isinstance(ev, TextDelta):
                streamed += ev.text
            else:
                outcome = ev
        assert outcome is not None
        return streamed, outcome

    async def test_valid_answer(self) -> None:
        streamed, out = await self._run(self._gen("Eight years [1]. Also [9]."))
        assert out.text == "Eight years [1]. Also [9]."
        assert [c.marker for c in out.citations] == [1]
        assert out.invalid_citations == [9]
        assert out.abstain_reason is None
        assert streamed.strip() == out.text

    async def test_uncited_answer_becomes_abstention(self) -> None:
        _, out = await self._run(self._gen("Probably eight years."))
        assert (
            out.text == ABSTAIN_TEXT and out.abstain_reason == NO_CITATIONS and out.citations == []
        )
        assert out.raw_text == "Probably eight years."

    async def test_refusal(self) -> None:
        _, out = await self._run(self._gen("", stop="refusal"))
        assert out.text == REFUSAL_TEXT and out.abstain_reason == REFUSED


async def test_disk_cached_llm(tmp_path) -> None:
    inner = ScriptedLLM("cached answer [1]")
    llm = DiskCachedLLM(inner, tmp_path)
    first = await llm.complete(system="s", prompt="p", max_tokens=5)
    second = await llm.complete(system="s", prompt="p", max_tokens=5)
    assert first.text == second.text and len(inner.calls) == 1
    assert second.usage == Usage()  # nothing spent on a hit
    streamed = [e async for e in llm.stream(system="s", prompt="p2", max_tokens=5)]
    replay = [e async for e in llm.stream(system="s", prompt="p2", max_tokens=5)]
    assert len(inner.calls) == 2 and replay[-1].text == streamed[-1].text
    assert (llm.hits, llm.misses) == (2, 2)


async def test_disk_cache_skips_refusals(tmp_path) -> None:
    inner = ScriptedLLM("", stop_reason="refusal")
    llm = DiskCachedLLM(inner, tmp_path)
    await llm.complete(system="s", prompt="p", max_tokens=5)
    await llm.complete(system="s", prompt="p", max_tokens=5)
    assert len(inner.calls) == 2
