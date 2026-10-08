"""Answer quality: faithfulness & correctness (LLM-as-judge), plus abstention metrics.

* **Faithfulness** – are the answer's claims supported by the retrieved contexts?
* **Correctness** – does the answer match the reference answer (factual F1 of
  claims; RAGAS ``AnswerCorrectness`` with the embedding-similarity weight set to 0
  so the score is purely LLM-judged)?

Judge model ≠ generator model (configurable) to avoid self-preference bias.
Abstentions are not judged: they are scored by the abstention metrics instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any, Protocol

import anyio

from ragsys.indexing.bm25 import tokenize


@dataclass(frozen=True)
class AnswerSample:
    """One generated answer to be judged."""

    question: str
    answer: str
    contexts: list[str]
    reference: str
    abstained: bool
    answerable: bool


@dataclass(frozen=True)
class AnswerScore:
    """Per-sample scores (None = not applicable)."""

    faithfulness: float | None
    correctness: float | None


class AnswerJudge(Protocol):
    """Scores answers."""

    name: str

    async def score(self, samples: Sequence[AnswerSample]) -> list[AnswerScore]:
        """One score per sample, in order."""
        ...


def _judgeable(s: AnswerSample) -> bool:
    return s.answerable and not s.abstained


def _f1(pred: set[str], gold: set[str]) -> float:
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(pred), tp / len(gold)
    return 2 * precision * recall / (precision + recall)


class LexicalJudge:
    """Offline proxy (no LLM): token support for faithfulness, token F1 for correctness.

    Useful for CI and for sanity-checking the harness; it is NOT a substitute for an
    LLM judge (it can't detect paraphrased or contradicted claims).
    """

    name = "lexical"

    async def score(self, samples: Sequence[AnswerSample]) -> list[AnswerScore]:
        out = []
        for s in samples:
            if not _judgeable(s):
                out.append(AnswerScore(None, 0.0 if s.answerable else None))
                continue
            answer = set(tokenize(s.answer))
            context = set(tokenize(" ".join(s.contexts)))
            faith = len(answer & context) / len(answer) if answer else 0.0
            out.append(AnswerScore(faith, _f1(answer, set(tokenize(s.reference)))))
        return out


class RagasJudge:
    """RAGAS ``Faithfulness`` + ``AnswerCorrectness`` with Claude as the judge.

    Uses the LangChain wrapper with ``bypass_temperature``/``bypass_n`` because
    current Claude models reject non-default sampling parameters.
    """

    name = "ragas"

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 4096) -> None:
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens

    def _llm(self) -> Any:
        from langchain_anthropic import ChatAnthropic
        from ragas.llms import LangchainLLMWrapper

        chat_kwargs: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens}
        if self.api_key:
            chat_kwargs["api_key"] = self.api_key
        chat = ChatAnthropic(**chat_kwargs)
        return LangchainLLMWrapper(chat, bypass_temperature=True, bypass_n=True)

    def _evaluate(self, samples: Sequence[AnswerSample]) -> list[dict[str, float]]:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import AnswerCorrectness, Faithfulness

        llm = self._llm()
        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=s.question,
                    response=s.answer,
                    retrieved_contexts=s.contexts,
                    reference=s.reference,
                )
                for s in samples
            ]
        )
        metrics = [Faithfulness(llm=llm), AnswerCorrectness(llm=llm, weights=[1.0, 0.0])]
        result: Any = evaluate(
            dataset, metrics=metrics, llm=llm, show_progress=False, raise_exceptions=False
        )
        return list(result.to_pandas().to_dict("records"))

    async def score(self, samples: Sequence[AnswerSample]) -> list[AnswerScore]:
        idx = [i for i, s in enumerate(samples) if _judgeable(s)]
        scores = [AnswerScore(None, 0.0 if s.answerable else None) for s in samples]
        if not idx:
            return scores
        # RAGAS runs its own event loop; keep it off ours.
        records = await anyio.to_thread.run_sync(self._evaluate, [samples[i] for i in idx])
        for i, rec in zip(idx, records, strict=True):
            scores[i] = AnswerScore(
                _num(rec.get("faithfulness")), _num(rec.get("answer_correctness"))
            )
        return scores


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN -> None


def abstention_metrics(samples: Sequence[AnswerSample]) -> dict[str, float]:
    """Precision/recall of abstaining on unanswerable questions + false-abstention rate."""
    tp = sum(1 for s in samples if s.abstained and not s.answerable)
    fp = sum(1 for s in samples if s.abstained and s.answerable)
    fn = sum(1 for s in samples if not s.abstained and not s.answerable)
    answerable = sum(1 for s in samples if s.answerable)
    out: dict[str, float] = {}
    if tp + fp:
        out["abstain_precision"] = tp / (tp + fp)
    if tp + fn:
        out["abstain_recall"] = tp / (tp + fn)
    if answerable:
        out["false_abstain_rate"] = fp / answerable
    return out


def aggregate_answer_scores(scores: Sequence[AnswerScore]) -> dict[str, float]:
    """Mean faithfulness/correctness over samples where they are defined."""
    out: dict[str, float] = {}
    faith = [s.faithfulness for s in scores if s.faithfulness is not None]
    corr = [s.correctness for s in scores if s.correctness is not None]
    if faith:
        out["faithfulness"] = mean(faith)
    if corr:
        out["correctness"] = mean(corr)
    return out


def make_judge(backend: str, model: str, api_key: str | None) -> AnswerJudge:
    """Judge from config."""
    if backend == "lexical":
        return LexicalJudge()
    if backend == "ragas":
        return RagasJudge(model, api_key)
    raise ValueError(f"Unknown judge backend: {backend}")
