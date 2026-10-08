"""End-to-end RAG request orchestration (used by the API, the CLI and evaluation).

    cache lookup -> retrieval (thread pool) -> abstention gate -> streamed generation
    -> citation validation -> cost -> cache store -> structured log + trace

The service yields typed events so transports (SSE, JSON, CLI) stay thin.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from functools import partial

import anyio
import structlog

from ragsys.cache import ResponseCache, cache_key
from ragsys.config import PipelineConfig
from ragsys.generation.abstention import REFUSED, AbstentionPolicy
from ragsys.generation.citations import to_citation
from ragsys.generation.generator import AnswerGenerator, GenerationOutcome
from ragsys.generation.llm import TextDelta
from ragsys.generation.prompts import ABSTAIN_TEXT
from ragsys.models import Answer, Citation, Usage
from ragsys.observability.cost import compute_cost
from ragsys.observability.tracing import Span, Tracer
from ragsys.retrieval.pipeline import RetrievalPipeline, RetrievalResult

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SourcesEvent:
    """Context passages the answer may cite (sent before any token)."""

    sources: list[Citation]


@dataclass(frozen=True)
class TokenEvent:
    """A piece of streamed answer text."""

    text: str


@dataclass(frozen=True)
class DoneEvent:
    """Final, validated answer. Its ``text`` is authoritative over streamed tokens."""

    answer: Answer


ServiceEvent = SourcesEvent | TokenEvent | DoneEvent


class IndexVersion:
    """Reads the index version from collection metadata, cached for ``ttl`` seconds."""

    def __init__(self, fetch: Callable[[], str], ttl: float = 30.0) -> None:
        self._fetch = fetch
        self._ttl = ttl
        self._value = ""
        self._expires = 0.0

    def __call__(self) -> str:
        now = time.monotonic()
        if now >= self._expires:
            try:
                self._value = self._fetch()
            except Exception as exc:
                log.warning("index_version.fetch_failed", error=str(exc))
                self._value = self._value or "unknown"
            self._expires = now + self._ttl
        return self._value


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class RAGService:
    """See module docstring."""

    def __init__(
        self,
        *,
        cfg: PipelineConfig,
        retrieval: RetrievalPipeline,
        generator: AnswerGenerator,
        cache: ResponseCache,
        tracer: Tracer,
        index_version: Callable[[], str],
    ) -> None:
        self.cfg = cfg
        self.retrieval = retrieval
        self.generator = generator
        self.policy: AbstentionPolicy = generator.policy
        self.cache = cache
        self.tracer = tracer
        self.index_version = index_version
        self._fingerprint = cfg.fingerprint()

    async def answer(self, query: str, request_id: str = "") -> Answer:
        """Non-streaming convenience wrapper around :meth:`stream`."""
        final: Answer | None = None
        async for event in self.stream(query, request_id):
            if isinstance(event, DoneEvent):
                final = event.answer
        assert final is not None
        return final

    async def stream(self, query: str, request_id: str = "") -> AsyncIterator[ServiceEvent]:
        """Run one request, yielding events. Always ends the trace."""
        started = time.perf_counter()
        trace = self.tracer.start_trace(
            "rag.query",
            input=query,
            metadata={
                "request_id": request_id,
                "config": self.cfg.name,
                "fingerprint": self._fingerprint,
            },
        )
        try:
            async for event in self._run(query, request_id, trace, started):
                if isinstance(event, DoneEvent):
                    self._finish(event.answer, trace)
                yield event
        except Exception as exc:
            log.exception("rag.request_failed", request_id=request_id, error=str(exc))
            raise
        finally:
            trace.end()

    async def _run(
        self, query: str, request_id: str, trace: Span, started: float
    ) -> AsyncIterator[ServiceEvent]:
        key = cache_key(query, self._fingerprint, self.index_version())
        lookup = time.perf_counter()
        cached = await self.cache.get(key)
        if cached is not None:
            answer = cached.model_copy(
                update={
                    "cached": True,
                    "request_id": request_id,
                    "trace_id": trace.trace_id,
                    "usage": Usage(),
                    "cost_usd": 0.0,
                    "latency_ms": {"cache_lookup": _ms(lookup), "total": _ms(started)},
                }
            )
            yield SourcesEvent(answer.sources)
            yield TokenEvent(answer.text)
            yield DoneEvent(answer)
            return
        latency: dict[str, float] = {"cache_lookup": _ms(lookup)}
        retrieved = await self._retrieve(query, trace, latency)
        sources = [to_citation(i, h) for i, h in enumerate(retrieved.chunks, start=1)]
        reason = self.policy.pre_check(retrieved.signals, len(retrieved.chunks))
        if reason is not None:
            answer = self._abstain(query, request_id, trace, retrieved, reason, latency, started)
            yield SourcesEvent([])
            yield TokenEvent(answer.text)
        else:
            yield SourcesEvent(sources)
            outcome: GenerationOutcome | None = None
            gen_start, first_token = time.perf_counter(), None
            async for event in self.generator.stream(query, retrieved.chunks, trace):
                if isinstance(event, TextDelta):
                    first_token = first_token or _ms(gen_start)
                    yield TokenEvent(event.text)
                else:
                    outcome = event
            assert outcome is not None
            latency.update(generation=_ms(gen_start), ttft=first_token or _ms(gen_start))
            answer = self._build_answer(
                query, request_id, trace, retrieved, sources, outcome, latency, started
            )
        if answer.abstain_reason != REFUSED:
            await self.cache.set(key, answer, self.cfg.cache.ttl_seconds)
        yield DoneEvent(answer)

    async def _retrieve(
        self, query: str, trace: Span, latency: dict[str, float]
    ) -> RetrievalResult:
        """Retrieval is CPU-bound (embedding, cross-encoder): run it off the event loop."""
        start = time.perf_counter()
        result = await anyio.to_thread.run_sync(partial(self.retrieval.retrieve, query, trace))
        latency.update({f"retrieval.{k}": v for k, v in result.timings_ms.items()})
        latency["retrieval"] = _ms(start)
        return result

    def _abstain(
        self,
        query: str,
        request_id: str,
        trace: Span,
        retrieved: RetrievalResult,
        reason: str,
        latency: dict[str, float],
        started: float,
    ) -> Answer:
        latency["total"] = _ms(started)
        return Answer(
            query=query,
            text=ABSTAIN_TEXT,
            abstained=True,
            abstain_reason=reason,
            confidence=retrieved.confidence,
            model="",
            latency_ms=latency,
            request_id=request_id,
            trace_id=trace.trace_id,
        )

    def _build_answer(
        self,
        query: str,
        request_id: str,
        trace: Span,
        retrieved: RetrievalResult,
        sources: list[Citation],
        outcome: GenerationOutcome,
        latency: dict[str, float],
        started: float,
    ) -> Answer:
        completion = outcome.completion
        latency["total"] = _ms(started)
        return Answer(
            query=query,
            text=outcome.text,
            citations=outcome.citations,
            sources=sources,
            invalid_citations=outcome.invalid_citations,
            abstained=outcome.abstain_reason is not None,
            abstain_reason=outcome.abstain_reason,
            confidence=retrieved.confidence,
            model=completion.model,
            usage=completion.usage,
            cost_usd=compute_cost(completion.usage, completion.model, self.cfg.pricing),
            latency_ms=latency,
            request_id=request_id,
            trace_id=trace.trace_id,
        )

    def _finish(self, answer: Answer, trace: Span) -> None:
        """Per-request structured log line + trace output."""
        summary = {
            "abstained": answer.abstained,
            "abstain_reason": answer.abstain_reason,
            "cached": answer.cached,
            "confidence": answer.confidence,
            "model": answer.model,
            "citations": [c.marker for c in answer.citations],
            "invalid_citations": answer.invalid_citations,
            "input_tokens": answer.usage.input_tokens,
            "output_tokens": answer.usage.output_tokens,
            "cost_usd": answer.cost_usd,
            "latency_ms": answer.latency_ms,
        }
        trace.update(output=answer.text, metadata=summary)
        log.info("rag.request", request_id=answer.request_id, trace_id=answer.trace_id, **summary)
