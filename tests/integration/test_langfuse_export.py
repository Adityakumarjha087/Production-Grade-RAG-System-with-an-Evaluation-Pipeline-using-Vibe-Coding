"""The real Langfuse SDK exports our spans (captured by a local fake endpoint).

No Langfuse account needed: we point ``host`` at a local HTTP server and assert
the OTLP export contains the expected span names and generation usage.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ragsys.observability.tracing import LangfuseTracer
from tests.conftest import ScriptedLLM, index_docs, offline_config

pytestmark = pytest.mark.integration


class _Capture(BaseHTTPRequestHandler):
    bodies: list[tuple[str, bytes]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", 0))
        _Capture.bodies.append((self.path, self.rfile.read(length)))
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_GET(self) -> None:
        self.do_POST()

    def log_message(self, *_: object) -> None:
        pass


@pytest.fixture
def endpoint() -> Iterator[str]:
    _Capture.bodies = []
    server = HTTPServer(("127.0.0.1", 0), _Capture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


async def test_request_trace_is_exported(endpoint: str, qdrant, sample_docs, settings) -> None:
    from ragsys.factory import build_components

    cfg = offline_config()
    index_docs(sample_docs, cfg, qdrant)
    tracer = LangfuseTracer("pk-lf-test", "sk-lf-test", endpoint)
    comps = build_components(
        cfg, settings, client=qdrant, tracer=tracer, llm=ScriptedLLM("Eight years [1].")
    )
    answer = await comps.service.answer("How long must intermediaries preserve KYC records?")
    tracer.flush()

    assert answer.trace_id and len(answer.trace_id) == 32  # OTel trace id surfaced to clients
    exports = [body for path, body in _Capture.bodies if path.endswith("/v1/traces")]
    assert exports, f"no OTLP export received: {[p for p, _ in _Capture.bodies]}"
    blob = b"".join(exports)
    for span in (
        b"rag.query",
        b"retrieval.dense",
        b"retrieval.sparse",
        b"retrieval.rrf",
        b"retrieval.rerank",
        b"llm.generate",
    ):
        assert span in blob, span
    assert answer.trace_id.encode() in blob or bytes.fromhex(answer.trace_id) in blob
    assert b"claude-opus-5-5" in blob  # generation model recorded
