"""Request-context middleware (pure ASGI, so it does not buffer streaming bodies).

Assigns/propagates ``X-Request-ID``, binds it to structlog's contextvars so every
log line in the request carries it, and logs one access line with the latency
measured until the *last* body chunk (i.e. the full SSE stream).
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = structlog.get_logger("ragsys.http")
HEADER = b"x-request-id"


def _incoming_id(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name == HEADER:
            candidate = str(value.decode("latin-1")).strip()
            if 0 < len(candidate) <= 128 and candidate.isprintable():
                return candidate
    return None


class RequestContextMiddleware:
    """See module docstring."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = _incoming_id(scope) or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((HEADER, request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            log.info(
                "http.request",
                method=scope.get("method"),
                path=scope.get("path"),
                status=status,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            structlog.contextvars.clear_contextvars()
