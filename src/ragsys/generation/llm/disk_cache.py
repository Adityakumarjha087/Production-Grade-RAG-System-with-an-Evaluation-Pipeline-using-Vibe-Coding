"""Disk-memoised LLM wrapper used by the evaluation harness.

Re-running an evaluation (e.g. after changing only the report format) should not
re-spend money. Calls are keyed by a hash of everything that affects the output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ragsys.generation.llm.base import Completion, LLMClient, LLMEvent, TextDelta
from ragsys.models import Usage


class DiskCachedLLM:
    """Wraps any :class:`LLMClient`; cache hits report zero usage (nothing was spent)."""

    def __init__(self, inner: LLMClient, cache_dir: Path) -> None:
        self.inner = inner
        self.model = inner.model
        self.cache_dir = cache_dir
        self.hits = 0
        self.misses = 0

    def _key(self, **parts: Any) -> Path:
        blob = json.dumps({"model": self.model, **parts}, sort_keys=True).encode()
        digest = hashlib.sha256(blob).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _load(self, path: Path) -> Completion | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text("utf-8"))
        self.hits += 1
        return Completion(
            text=data["text"],
            usage=Usage(),
            model=data["model"],
            stop_reason=data.get("stop_reason"),
        )

    def _save(self, path: Path, completion: Completion) -> None:
        self.misses += 1
        if completion.refused:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": completion.text,
            "model": completion.model,
            "stop_reason": completion.stop_reason,
        }
        path.write_text(json.dumps(payload), "utf-8")

    async def stream(self, *, system: str, prompt: str, max_tokens: int) -> AsyncIterator[LLMEvent]:
        """Replay from cache or stream through and record."""
        path = self._key(system=system, prompt=prompt, max_tokens=max_tokens, schema=None)
        cached = self._load(path)
        if cached is not None:
            yield TextDelta(cached.text)
            yield cached
            return
        async for event in self.inner.stream(system=system, prompt=prompt, max_tokens=max_tokens):
            if isinstance(event, Completion):
                self._save(path, event)
            yield event

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        """Return cached completion or call through and record."""
        path = self._key(system=system, prompt=prompt, max_tokens=max_tokens, schema=json_schema)
        cached = self._load(path)
        if cached is not None:
            return cached
        completion = await self.inner.complete(
            system=system, prompt=prompt, max_tokens=max_tokens, json_schema=json_schema
        )
        self._save(path, completion)
        return completion
