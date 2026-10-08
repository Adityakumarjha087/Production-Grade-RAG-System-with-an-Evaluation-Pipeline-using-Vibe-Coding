"""Claude via the official Anthropic SDK.

* Streaming uses ``messages.stream`` and ``get_final_message()`` for usage.
* ``output_config.effort`` controls thinking depth (grounded QA runs fine at ``low``).
* Server-side refusal fallback (``fallbacks="default"``) is enabled by default:
  if the model's safety classifier declines, the API retries on Anthropic's
  recommended fallback model inside the same call. Costs are computed from the
  model that actually served the response (``final.model``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Usage as SDKUsage

from ragsys.generation.llm.base import Completion, LLMEvent, TextDelta
from ragsys.models import Usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def to_usage(usage: SDKUsage | Any) -> Usage:
    """Map SDK usage (fields may be None) to our Usage model."""
    return Usage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
    )


def final_text(message: Any) -> str:
    """Concatenate text blocks (thinking/fallback blocks are skipped)."""
    return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


class AnthropicLLM:
    """See module docstring."""

    def __init__(
        self,
        model: str,
        *,
        effort: str = "low",
        use_fallbacks: bool = True,
        api_key: str | None = None,
        client: AsyncAnthropic | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.effort = effort
        self.use_fallbacks = use_fallbacks
        self._client = client or AsyncAnthropic(
            api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    def _params(
        self, system: str, prompt: str, max_tokens: int, json_schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        output_config: dict[str, Any] = {"effort": self.effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        if self.use_fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def _api(self) -> Any:
        return self._client.beta.messages if self.use_fallbacks else self._client.messages

    @staticmethod
    def _completion(message: Any) -> Completion:
        return Completion(
            text=final_text(message),
            usage=to_usage(message.usage),
            model=str(message.model),
            stop_reason=message.stop_reason,
        )

    async def stream(self, *, system: str, prompt: str, max_tokens: int) -> AsyncIterator[LLMEvent]:
        """Stream answer text, then yield the final Completion."""
        params = self._params(system, prompt, max_tokens, None)
        async with self._api().stream(**params) as stream:
            async for text in stream.text_stream:
                yield TextDelta(text)
            message = await stream.get_final_message()
        yield self._completion(message)

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        """Run a call to completion (streamed under the hood to avoid HTTP timeouts)."""
        params = self._params(system, prompt, max_tokens, json_schema)
        async with self._api().stream(**params) as stream:
            message = await stream.get_final_message()
        return self._completion(message)
