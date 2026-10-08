"""Tracing facade with a Langfuse backend and a no-op backend.

We pass span handles *explicitly* (``parent.child(...)``) instead of relying on
OpenTelemetry's implicit current-span context. Implicit context does not survive
being suspended inside an async generator that Starlette resumes from a different
task (exactly what SSE streaming does), and explicit parents make the trace shape
obvious in code.

Tracing must never break a request: every Langfuse call is wrapped, and failures
are logged and swallowed.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Literal, Protocol, Self

import structlog

log = structlog.get_logger(__name__)

SpanType = Literal["span", "generation", "retriever", "embedding", "chain"]


class Span(Protocol):
    """A unit of traced work. Usable as a context manager (ends on exit)."""

    trace_id: str | None

    def child(
        self,
        name: str,
        *,
        as_type: SpanType = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Span:
        """Start a nested span."""
        ...

    def update(self, **fields: Any) -> None:
        """Attach output / metadata / usage_details / cost_details / level."""
        ...

    def end(self) -> None:
        """Finish the span."""
        ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class Tracer(Protocol):
    """Creates root spans (one per request = one trace)."""

    def start_trace(
        self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None
    ) -> Span:
        """Start a new trace and return its root span."""
        ...

    def flush(self) -> None:
        """Send buffered events (call on shutdown)."""
        ...


class NoopSpan:
    """Span that records nothing."""

    trace_id: str | None = None

    def child(self, name: str, **_: Any) -> NoopSpan:
        return self

    def update(self, **fields: Any) -> None:
        return None

    def end(self) -> None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.end()


class NoopTracer:
    """Tracer used when Langfuse is not configured."""

    def start_trace(self, name: str, **_: Any) -> NoopSpan:
        return NoopSpan()

    def flush(self) -> None:
        return None


class LangfuseSpan:
    """Adapter from our Span protocol to a Langfuse observation."""

    def __init__(self, observation: Any) -> None:
        self._obs = observation
        self.trace_id: str | None = getattr(observation, "trace_id", None)

    def child(
        self,
        name: str,
        *,
        as_type: SpanType = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Span:
        try:
            kwargs: dict[str, Any] = {"name": name, "as_type": as_type, "input": input}
            if metadata:
                kwargs["metadata"] = metadata
            if model:
                kwargs["model"] = model
            return LangfuseSpan(self._obs.start_observation(**kwargs))
        except Exception as exc:
            log.warning("tracing.child_failed", error=str(exc))
            return NoopSpan()

    def update(self, **fields: Any) -> None:
        try:
            self._obs.update(**fields)
        except Exception as exc:
            log.warning("tracing.update_failed", error=str(exc))

    def end(self) -> None:
        try:
            self._obs.end()
        except Exception as exc:
            log.warning("tracing.end_failed", error=str(exc))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is not None:
            name = exc_type.__name__ if exc_type else "Error"
            self.update(level="ERROR", status_message=f"{name}: {exc}")
        self.end()


class LangfuseTracer:
    """Creates one Langfuse trace per request."""

    def __init__(
        self, public_key: str, secret_key: str, host: str, sample_rate: float = 1.0
    ) -> None:
        from langfuse import Langfuse

        self._client = Langfuse(
            public_key=public_key, secret_key=secret_key, base_url=host, sample_rate=sample_rate
        )

    def start_trace(
        self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None
    ) -> Span:
        try:
            obs = self._client.start_observation(
                name=name, as_type="chain", input=input, metadata=metadata
            )
            # Langfuse v4 derives trace input/output from the root observation.
            return LangfuseSpan(obs)
        except Exception as exc:
            log.warning("tracing.start_failed", error=str(exc))
            return NoopSpan()

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:
            log.warning("tracing.flush_failed", error=str(exc))


def make_tracer(
    enabled: bool,
    public_key: str | None,
    secret_key: str | None,
    host: str,
    sample_rate: float = 1.0,
) -> Tracer:
    """Langfuse tracer when enabled and keys are present, otherwise no-op."""
    if not (enabled and public_key and secret_key):
        return NoopTracer()
    return LangfuseTracer(public_key, secret_key, host, sample_rate)
