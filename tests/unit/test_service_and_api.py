from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from ragsys.api.app import create_app
from ragsys.config import Settings
from ragsys.factory import Components, build_components
from ragsys.generation.abstention import LOW_CONFIDENCE, REFUSED
from ragsys.generation.prompts import ABSTAIN_TEXT
from ragsys.observability.tracing import NoopSpan
from ragsys.service import DoneEvent, SourcesEvent, TokenEvent
from tests.conftest import FailingLLM, ScriptedLLM, index_docs, offline_config

QUESTION = "How long must intermediaries preserve KYC records?"


class RecordingTracer:
    """Captures the span tree to assert what gets traced."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.updates: list[dict[str, Any]] = []
        self.ended = 0

    def start_trace(self, name: str, **_: Any) -> RecordingSpan:
        self.names.append(name)
        return RecordingSpan(self)

    def flush(self) -> None:
        pass


class RecordingSpan(NoopSpan):
    trace_id = "trace-123"

    def __init__(self, tracer: RecordingTracer) -> None:
        self.tracer = tracer

    def child(self, name: str, **_: Any) -> RecordingSpan:
        self.tracer.names.append(name)
        return RecordingSpan(self.tracer)

    def update(self, **fields: Any) -> None:
        self.tracer.updates.append(fields)

    def end(self) -> None:
        self.tracer.ended += 1


def build(qdrant, sample_docs, **kw: Any) -> Components:
    cfg = kw.pop("cfg", None) or offline_config()
    index_docs(sample_docs, cfg, qdrant)
    return build_components(cfg, Settings(_env_file=None), client=qdrant, **kw)  # type: ignore[call-arg]


class TestService:
    async def test_answer_with_citations_cost_and_latency(self, qdrant, sample_docs) -> None:
        tracer = RecordingTracer()
        llm = ScriptedLLM("Eight years after the relationship ends [1].")
        comps = build(qdrant, sample_docs, llm=llm, tracer=tracer)
        events = [e async for e in comps.service.stream(QUESTION, "req-1")]
        assert isinstance(events[0], SourcesEvent) and events[0].sources
        assert isinstance(events[-1], DoneEvent)
        answer = events[-1].answer
        assert answer.text == "Eight years after the relationship ends [1]."
        assert [c.marker for c in answer.citations] == [1]
        assert "KYC" in answer.citations[0].doc_title
        assert answer.cost_usd == pytest.approx((1000 * 4 + 100 * 20) / 1e6)
        assert {"retrieval", "generation", "ttft", "total", "retrieval.rerank"} <= set(
            answer.latency_ms
        )
        assert answer.request_id == "req-1" and answer.trace_id == "trace-123"
        assert "".join(e.text for e in events if isinstance(e, TokenEvent)).strip() == answer.text
        assert tracer.names == [
            "rag.query",
            "retrieval.dense",
            "retrieval.sparse",
            "retrieval.rrf",
            "retrieval.rerank",
            "llm.generate",
        ]
        assert tracer.ended == 6  # every span closed

    async def test_second_request_served_from_cache(self, qdrant, sample_docs) -> None:
        llm = ScriptedLLM("Eight years [1].")
        comps = build(qdrant, sample_docs, llm=llm)
        first = await comps.service.answer(QUESTION)
        second = await comps.service.answer("  how long must intermediaries preserve KYC records ")
        assert len(llm.calls) == 1
        assert second.cached and not first.cached
        assert second.text == first.text and second.cost_usd == 0.0
        assert second.citations == first.citations

    async def test_low_confidence_skips_llm(self, qdrant, sample_docs) -> None:
        llm = ScriptedLLM("should not be called [1]")
        comps = build(qdrant, sample_docs, llm=llm)
        answer = await comps.service.answer("What is the weather on Mars today?")
        assert answer.abstained and answer.abstain_reason == LOW_CONFIDENCE
        assert answer.text == ABSTAIN_TEXT and llm.calls == []
        assert answer.cost_usd == 0.0

    async def test_refusals_are_not_cached(self, qdrant, sample_docs) -> None:
        llm = ScriptedLLM("", stop_reason="refusal")
        comps = build(qdrant, sample_docs, llm=llm)
        a = await comps.service.answer(QUESTION)
        await comps.service.answer(QUESTION)
        assert a.abstain_reason == REFUSED and len(llm.calls) == 2

    async def test_reindex_invalidates_cache(self, qdrant, sample_docs) -> None:
        llm = ScriptedLLM("Eight years [1].")
        comps = build(qdrant, sample_docs, llm=llm)
        await comps.service.answer(QUESTION)
        index_docs(sample_docs[:3], comps.cfg, qdrant)  # new index version
        comps.service.index_version._expires = 0  # skip the 30s TTL in the test
        await comps.service.answer(QUESTION)
        assert len(llm.calls) == 2

    async def test_errors_propagate_and_trace_ends(self, qdrant, sample_docs) -> None:
        tracer = RecordingTracer()
        comps = build(qdrant, sample_docs, llm=FailingLLM(RuntimeError("boom")), tracer=tracer)
        with pytest.raises(RuntimeError, match="boom"):
            await comps.service.answer(QUESTION)
        assert tracer.ended >= 1


def sse_events(text: str) -> list[tuple[str, Any]]:
    events = []
    for block in text.strip().split("\r\n\r\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


@pytest.fixture
def api(qdrant, sample_docs):
    def make(**kw: Any) -> TestClient:
        app = create_app(build(qdrant, sample_docs, **kw), Settings(_env_file=None))  # type: ignore[call-arg]
        return TestClient(app, raise_server_exceptions=False)

    return make


class TestAPI:
    def test_health_and_readiness(self, api) -> None:
        with api() as client:
            assert client.get("/healthz").json() == {"status": "ok", "checks": {}}
            ready = client.get("/readyz")
            assert ready.status_code == 200
            assert ready.json()["checks"] == {"models_loaded": True, "index": True, "cache": True}

    def test_not_ready_without_index(self, qdrant) -> None:
        comps = build_components(offline_config(), Settings(_env_file=None), client=qdrant)  # type: ignore[call-arg]
        with TestClient(create_app(comps)) as client:
            resp = client.get("/readyz")
        assert resp.status_code == 503 and resp.json()["checks"]["index"] is False

    def test_query_json(self, api) -> None:
        with api() as client:
            resp = client.post(
                "/v1/query", json={"query": QUESTION}, headers={"X-Request-ID": "abc"}
            )
        body = resp.json()
        assert resp.status_code == 200 and resp.headers["x-request-id"] == "abc"
        assert body["request_id"] == "abc"
        assert body["citations"] and not body["abstained"]
        assert "eight years" in body["text"].lower()  # extractive LLM quotes the source

    def test_generates_request_id(self, api) -> None:
        with api() as client:
            resp = client.post("/v1/query", json={"query": QUESTION})
        assert len(resp.headers["x-request-id"]) == 32

    @pytest.mark.parametrize(
        "payload", [{}, {"query": "  "}, {"query": "x" * 2001}, {"q": "hi there"}]
    )
    def test_validation(self, api, payload: dict) -> None:
        with api() as client:
            assert client.post("/v1/query", json=payload).status_code == 422

    def test_stream_event_order(self, api) -> None:
        with api() as client:
            resp = client.post("/v1/query/stream", json={"query": QUESTION})
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = sse_events(resp.text)
        kinds = [k for k, _ in events]
        assert kinds[0] == "sources" and kinds[-1] == "done"
        assert set(kinds[1:-1]) == {"token"}
        done = events[-1][1]
        streamed = "".join(d["text"] for k, d in events if k == "token")
        assert streamed == done["text"] and done["citations"]

    def test_upstream_rate_limit_maps_to_503(self, api) -> None:
        err = anthropic.RateLimitError(
            "slow down",
            response=httpx2.Response(429, request=httpx2.Request("POST", "http://x")),
            body=None,
        )
        with api(llm=FailingLLM(err)) as client:
            resp = client.post(
                "/v1/query", json={"query": QUESTION}, headers={"X-Request-ID": "r1"}
            )
        assert resp.status_code == 503
        assert resp.json() == {
            "error": "llm_rate_limited",
            "message": "request failed",
            "request_id": "r1",
        }

    def test_internal_error_is_500_without_leaking_details(self, api) -> None:
        with api(llm=FailingLLM(RuntimeError("secret internals"))) as client:
            resp = client.post("/v1/query", json={"query": QUESTION})
        assert resp.status_code == 500 and "secret" not in resp.text

    def test_stream_error_event(self, api) -> None:
        with api(llm=FailingLLM(RuntimeError("boom"))) as client:
            resp = client.post(
                "/v1/query/stream", json={"query": QUESTION}, headers={"X-Request-ID": "s1"}
            )
        events = sse_events(resp.text)
        assert [k for k, _ in events] == ["sources", "error"]
        assert events[-1][1]["request_id"] == "s1"
