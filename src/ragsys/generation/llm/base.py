"""Provider-agnostic LLM interface.

Kept deliberately tiny (one system prompt + one user prompt) because that is all
RAG generation, test-set generation and judging need. A thin interface we own is
easier to test and reason about than a framework abstraction.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from ragsys.models import Usage


@dataclass(frozen=True)
class TextDelta:
    """A streamed piece of answer text."""

    text: str


@dataclass(frozen=True)
class Completion:
    """The final result of a call (always the last streamed event)."""

    text: str
    usage: Usage
    model: str
    stop_reason: str | None = None

    @property
    def refused(self) -> bool:
        """True when the provider's safety layer declined the request."""
        return self.stop_reason == "refusal"


LLMEvent = TextDelta | Completion


class LLMClient(Protocol):
    """Minimal async LLM client."""

    model: str

    def stream(self, *, system: str, prompt: str, max_tokens: int) -> AsyncIterator[LLMEvent]:
        """Yield ``TextDelta``s and finally exactly one ``Completion``."""
        ...

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        """Non-streaming call; ``json_schema`` requests structured JSON output."""
        ...
