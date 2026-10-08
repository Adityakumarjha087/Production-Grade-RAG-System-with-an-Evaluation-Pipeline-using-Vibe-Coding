"""FastAPI application factory.

Endpoints
---------
``POST /v1/query``         JSON answer.
``POST /v1/query/stream``  Server-Sent Events: ``sources`` -> ``token``* -> ``done`` (or ``error``).
``GET  /healthz``          Liveness (process is up).
``GET  /readyz``           Readiness (index reachable, cache reachable, models loaded).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ragsys import __version__
from ragsys.api.middleware import RequestContextMiddleware
from ragsys.api.schemas import ErrorResponse, HealthResponse, QueryRequest
from ragsys.config import Settings, load_config
from ragsys.factory import Components, build_components
from ragsys.models import Answer
from ragsys.observability.logging import configure_logging
from ragsys.service import DoneEvent, SourcesEvent, TokenEvent

log = structlog.get_logger(__name__)


def _warmup(components: Components) -> None:
    """Load ONNX models before accepting traffic so the first request isn't slow."""
    for model in (components.embedder, components.reranker):
        warm = getattr(model, "warmup", None)
        if callable(warm):
            warm()


def _request_id(request: Request) -> str:
    return str(request.scope.get("state", {}).get("request_id", ""))


def _error_status(exc: Exception) -> tuple[int, str]:
    """Map exceptions to HTTP status: upstream LLM problems are 502/503, not 500."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return 500, "internal_error"
    if isinstance(exc, anthropic.RateLimitError):
        return 503, "llm_rate_limited"
    if isinstance(exc, anthropic.APIError):
        return 502, "llm_upstream_error"
    return 500, "internal_error"


def create_app(components: Components | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Pass ``components`` to inject fakes (tests)."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if components is None:
            configure_logging(settings.log_level, settings.log_json)
            cfg = load_config(settings.rag_config)
            app.state.components = build_components(cfg, settings)
        else:
            app.state.components = components
        await anyio.to_thread.run_sync(_warmup, app.state.components)
        app.state.ready = True
        log.info(
            "api.started",
            config=app.state.components.cfg.name,
            collection=app.state.components.store.collection,
        )
        yield
        app.state.components.tracer.flush()
        close = getattr(app.state.components.cache, "close", None)
        if callable(close):
            await close()

    app = FastAPI(title="ragsys", version=__version__, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        status, code = _error_status(exc)
        log.error("api.error", error=str(exc), error_type=type(exc).__name__, status=status)
        body = ErrorResponse(error=code, message="request failed", request_id=_request_id(request))
        return JSONResponse(status_code=status, content=body.model_dump())

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=HealthResponse)
    async def readyz(request: Request) -> JSONResponse:
        comps: Components = request.app.state.components
        checks = {
            "models_loaded": bool(getattr(request.app.state, "ready", False)),
            "index": await anyio.to_thread.run_sync(comps.store.exists),
            "cache": await comps.cache.ping(),
        }
        ok = all(checks.values())
        body = HealthResponse(status="ready" if ok else "not_ready", checks=checks)
        return JSONResponse(status_code=200 if ok else 503, content=body.model_dump())

    @app.post("/v1/query", response_model=Answer)
    async def query(body: QueryRequest, request: Request) -> Answer:
        comps: Components = request.app.state.components
        return await comps.service.answer(body.query, _request_id(request))

    @app.post("/v1/query/stream")
    async def query_stream(body: QueryRequest, request: Request) -> EventSourceResponse:
        comps: Components = request.app.state.components
        request_id = _request_id(request)
        return EventSourceResponse(_sse_events(comps, body.query, request_id), ping=15)


async def _sse_events(
    comps: Components, query: str, request_id: str
) -> AsyncIterator[dict[str, Any]]:
    """Translate service events to SSE frames; errors become an ``error`` event."""
    try:
        async for event in comps.service.stream(query, request_id):
            if isinstance(event, SourcesEvent):
                data = json.dumps([s.model_dump() for s in event.sources])
                yield {"event": "sources", "data": data}
            elif isinstance(event, TokenEvent):
                yield {"event": "token", "data": json.dumps({"text": event.text})}
            elif isinstance(event, DoneEvent):
                yield {"event": "done", "data": event.answer.model_dump_json()}
    except Exception as exc:
        _, code = _error_status(exc)
        log.error("api.stream_error", error=str(exc), error_type=type(exc).__name__)
        err = ErrorResponse(error=code, message="request failed", request_id=request_id)
        yield {"event": "error", "data": err.model_dump_json()}


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn ragsys.api.app:app_factory --factory``."""
    return create_app()
