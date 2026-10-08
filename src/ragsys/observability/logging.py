"""Structured (JSON) logging with request-scoped context via contextvars.

The API logs JSON to stdout (container convention); the CLI logs human-readable
lines to stderr so command output on stdout stays pipeable. The output stream is
resolved at call time, not import time, so redirected/replaced streams work.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog


class _LazyStreamLogger:
    """Minimal structlog logger that looks up ``sys.stdout``/``sys.stderr`` per call."""

    def __init__(self, stream: Literal["stdout", "stderr"]) -> None:
        self._stream = stream

    def msg(self, message: str) -> None:
        print(message, file=getattr(sys, self._stream), flush=True)

    log = debug = info = warning = warn = error = critical = exception = fatal = msg


def configure_logging(
    level: str = "INFO", json_output: bool = True, stream: Literal["stdout", "stderr"] = "stdout"
) -> None:
    """Configure structlog + stdlib logging once at process start."""
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )

    def factory(*_: Any) -> _LazyStreamLogger:
        return _LazyStreamLogger(stream)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=factory,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=level.upper(), stream=getattr(sys, stream), format="%(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
