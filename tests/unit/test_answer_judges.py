"""RAGAS judge result mapping and judge factory, with the LLM call stubbed out."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ragsys.evaluation.answer_metrics import (
    AnswerSample,
    AnswerScore,
    LexicalJudge,
    RagasJudge,
    _num,
    make_judge,
)


def sample(question: str, *, abstained: bool = False, answerable: bool = True) -> AnswerSample:
    return AnswerSample(question, "an answer", ["a context"], "a reference", abstained, answerable)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.5, 0.5), ("0.25", 0.25), (1, 1.0), (float("nan"), None), (None, None), ("n/a", None)],
)
def test_num_coerces_judge_values(value: object, expected: float | None) -> None:
    assert _num(value) == expected


async def test_ragas_scores_only_judgeable_samples_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def fake_evaluate(self: RagasJudge, samples: Sequence[AnswerSample]) -> list[dict[str, float]]:
        seen.append([s.question for s in samples])
        return [
            {"faithfulness": 0.9, "answer_correctness": 0.7},
            {"faithfulness": float("nan"), "answer_correctness": 0.4},
        ]

    monkeypatch.setattr(RagasJudge, "_evaluate", fake_evaluate)
    samples = [
        sample("first"),
        sample("abstained", abstained=True),  # answerable but abstained: scored 0 without a call
        sample("unanswerable", answerable=False),  # not applicable
        sample("last"),
    ]
    scores = await RagasJudge("m").score(samples)
    assert seen == [["first", "last"]]  # the LLM only sees judgeable samples
    assert scores == [
        AnswerScore(0.9, 0.7),
        AnswerScore(None, 0.0),
        AnswerScore(None, None),
        AnswerScore(None, 0.4),  # NaN faithfulness becomes "not applicable"
    ]


async def test_ragas_missing_metric_columns_become_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RagasJudge, "_evaluate", lambda self, samples: [{}])
    assert await RagasJudge("m").score([sample("q")]) == [AnswerScore(None, None)]


def test_make_judge_selects_backend() -> None:
    assert isinstance(make_judge("lexical", "m", None), LexicalJudge)
    assert isinstance(make_judge("ragas", "m", None), RagasJudge)


def test_make_judge_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown judge backend: gpt"):
        make_judge("gpt", "m", None)
