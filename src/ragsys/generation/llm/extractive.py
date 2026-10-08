"""Deterministic, offline "LLM" that answers by extracting cited sentences.

Purpose: run the whole system (API, streaming, citations, abstention, evaluation)
without an API key or network — in CI, in Docker smoke tests, and on a laptop.
It parses the prompt format produced by :mod:`ragsys.generation.prompts`, picks the
sentences with the highest query-term overlap and cites their passage.
It is a test double with real behaviour, not a production answer generator.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

from ragsys.generation.llm.base import Completion, LLMEvent, TextDelta
from ragsys.generation.prompts import ABSTAIN_TEXT
from ragsys.indexing.bm25 import tokenize
from ragsys.models import Usage

_PASSAGE = re.compile(r'<passage id="(\d+)">\n[^\n]*\n(.*?)\n</passage>', re.DOTALL)
_QUESTION = re.compile(r"<question>\n(.*?)\n</question>", re.DOTALL)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_OVERLAP = 0.34


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _candidates(prompt: str) -> list[tuple[int, str]]:
    """(passage id, sentence) pairs from the prompt's context block."""
    pairs: list[tuple[int, str]] = []
    for pid, body in _PASSAGE.findall(prompt):
        for sentence in _SENTENCE.split(body):
            sentence = sentence.strip(" |-")
            if len(sentence) > 20 and not sentence.startswith("|"):
                pairs.append((int(pid), sentence))
    return pairs


def extract_answer(prompt: str, max_sentences: int = 2) -> str:
    """Best-overlap sentences with citations, or the abstention text."""
    match = _QUESTION.search(prompt)
    terms = set(tokenize(match.group(1))) if match else set()
    if not terms:
        return ABSTAIN_TEXT
    scored = []
    for pid, sentence in _candidates(prompt):
        overlap = len(terms & set(tokenize(sentence))) / len(terms)
        scored.append((overlap, -pid, sentence))
    scored.sort(reverse=True)
    best = [(o, -npid, s) for o, npid, s in scored[:max_sentences] if o >= MIN_OVERLAP]
    if not best:
        return ABSTAIN_TEXT
    parts = [f"{s.rstrip('.')} [{pid}]." for _, pid, s in best]
    return " ".join(parts)


class ExtractiveLLM:
    """See module docstring."""

    def __init__(self, model: str = "extractive") -> None:
        self.model = model

    async def stream(self, *, system: str, prompt: str, max_tokens: int) -> AsyncIterator[LLMEvent]:
        """Stream the extracted answer word by word."""
        text = extract_answer(prompt)
        for i, word in enumerate(text.split(" ")):
            yield TextDelta(word if i == 0 else f" {word}")
        yield self._completion(system, prompt, text)

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        """Non-streaming variant. Structured output is not supported offline."""
        if json_schema is not None:
            raise NotImplementedError("ExtractiveLLM cannot produce structured output")
        return self._completion(system, prompt, extract_answer(prompt))

    def _completion(self, system: str, prompt: str, text: str) -> Completion:
        usage = Usage(
            input_tokens=_approx_tokens(system + prompt), output_tokens=_approx_tokens(text)
        )
        return Completion(text=text, usage=usage, model=self.model, stop_reason="end_turn")
