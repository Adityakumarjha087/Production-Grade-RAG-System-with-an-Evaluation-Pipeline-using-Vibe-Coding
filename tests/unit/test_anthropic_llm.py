"""AnthropicLLM against the *real* SDK with a mocked HTTP transport.

This verifies the request body we send (model, effort, fallbacks beta, structured
output schema) is accepted by the SDK, and that we parse the SSE stream, usage
and stop reasons correctly — without network or an API key.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from ragsys.generation.llm import TextDelta
from ragsys.generation.llm.anthropic_llm import FALLBACK_BETA, AnthropicLLM


def sse(events: list[tuple[str, dict[str, Any]]]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def message_stream(
    text_parts: list[str], stop_reason: str = "end_turn", model: str = "claude-opus-5-5"
) -> bytes:
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 1200,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 100,
                        "cache_creation_input_tokens": 0,
                    },
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    ]
    events += [
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": t}},
        )
        for t in text_parts
    ]
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 42},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return sse(events)


class Recorder:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=self.body
        )

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def make_llm(recorder: Recorder, **kwargs: Any) -> AnthropicLLM:
    client = AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(recorder)),
    )
    return AnthropicLLM("claude-opus-5-5", client=client, **kwargs)


async def test_stream_parses_text_usage_and_skips_thinking() -> None:
    rec = Recorder(message_stream(["Eight ", "years [1]."]))
    llm = make_llm(rec)
    events = [e async for e in llm.stream(system="sys", prompt="question", max_tokens=512)]
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    final = events[-1]
    assert deltas == ["Eight ", "years [1]."]
    assert final.text == "Eight years [1]."
    assert final.usage.input_tokens == 1200 and final.usage.output_tokens == 42
    assert final.usage.cache_read_tokens == 100
    assert final.model == "claude-opus-5-5" and not final.refused


async def test_request_body_has_effort_and_fallbacks() -> None:
    rec = Recorder(message_stream(["ok"]))
    await make_llm(rec, effort="low").complete(system="sys", prompt="q", max_tokens=256)
    body, request = rec.payload, rec.requests[-1]
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 256
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "q"}]
    assert body["output_config"] == {"effort": "low"}
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    assert body["stream"] is True
    assert "temperature" not in body and "thinking" not in body


async def test_structured_output_and_no_fallbacks() -> None:
    rec = Recorder(message_stream(['{"a": 1}']))
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    out = await make_llm(rec, use_fallbacks=False).complete(
        system="s", prompt="q", max_tokens=64, json_schema=schema
    )
    body = rec.payload
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": schema}
    assert "fallbacks" not in body and "anthropic-beta" not in rec.requests[-1].headers
    assert json.loads(out.text) == {"a": 1}


async def test_refusal_and_fallback_model_reported() -> None:
    rec = Recorder(message_stream([], stop_reason="refusal", model="claude-opus-4-8"))
    out = await make_llm(rec).complete(system="s", prompt="q", max_tokens=64)
    assert out.refused and out.model == "claude-opus-4-8"


async def test_http_errors_surface_as_sdk_errors() -> None:
    import anthropic

    def fail(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}
        )

    client = AsyncAnthropic(
        api_key="k",
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(fail)),
    )
    with pytest.raises(anthropic.RateLimitError):
        await AnthropicLLM("claude-opus-5-5", client=client).complete(
            system="s", prompt="q", max_tokens=8
        )
