"""Grounded answer generation: prompt -> stream LLM -> validate citations -> abstain?"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from ragsys.generation.abstention import REFUSED, AbstentionPolicy
from ragsys.generation.citations import resolve_citations
from ragsys.generation.llm import Completion, LLMClient, TextDelta
from ragsys.generation.prompts import ABSTAIN_TEXT, SYSTEM_PROMPT, build_user_prompt
from ragsys.models import Citation, ScoredChunk
from ragsys.observability.tracing import NoopSpan, Span

REFUSAL_TEXT = "I can't help with that request."


@dataclass
class GenerationOutcome:
    """Post-processed result of one generation."""

    text: str
    raw_text: str
    completion: Completion
    citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    abstain_reason: str | None = None


class AnswerGenerator:
    """See module docstring."""

    def __init__(self, llm: LLMClient, policy: AbstentionPolicy, max_tokens: int) -> None:
        self.llm = llm
        self.policy = policy
        self.max_tokens = max_tokens

    def postprocess(
        self, completion: Completion, context: Sequence[ScoredChunk]
    ) -> GenerationOutcome:
        """Validate citations and apply the post-generation abstention check."""
        raw = completion.text.strip()
        if completion.refused:
            return GenerationOutcome(REFUSAL_TEXT, raw, completion, abstain_reason=REFUSED)
        citations, invalid = resolve_citations(raw, context)
        reason = self.policy.post_check(raw, citations)
        text = ABSTAIN_TEXT if reason else raw
        return GenerationOutcome(
            text, raw, completion, [] if reason else citations, invalid, reason
        )

    async def stream(
        self, question: str, context: Sequence[ScoredChunk], span: Span | None = None
    ) -> AsyncIterator[TextDelta | GenerationOutcome]:
        """Yield text deltas, then one :class:`GenerationOutcome`."""
        prompt = build_user_prompt(question, context)
        parent = span or NoopSpan()
        with parent.child(
            "llm.generate",
            as_type="generation",
            model=self.llm.model,
            input={"system": SYSTEM_PROMPT, "user": prompt},
        ) as gen_span:
            completion: Completion | None = None
            async for event in self.llm.stream(
                system=SYSTEM_PROMPT, prompt=prompt, max_tokens=self.max_tokens
            ):
                if isinstance(event, TextDelta):
                    yield event
                else:
                    completion = event
            if completion is None:
                raise RuntimeError("LLM stream ended without a final completion")
            outcome = self.postprocess(completion, context)
            gen_span.update(
                output=outcome.raw_text,
                model=completion.model,
                usage_details={
                    "input": completion.usage.input_tokens,
                    "output": completion.usage.output_tokens,
                    "cache_read_input_tokens": completion.usage.cache_read_tokens,
                },
                metadata={
                    "stop_reason": completion.stop_reason,
                    "abstain_reason": outcome.abstain_reason,
                    "invalid_citations": outcome.invalid_citations,
                },
            )
        yield outcome
