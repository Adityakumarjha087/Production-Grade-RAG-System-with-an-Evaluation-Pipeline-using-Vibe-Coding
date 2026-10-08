"""Exact-match response cache.

Key = hash(normalised query, pipeline config fingerprint, index version), so a
config change or a re-index can never serve a stale answer. Semantic caching
(embedding-similar queries share answers) was rejected on purpose: in a
regulatory domain "similar" questions can have different correct answers.

A cache failure must never fail a request: Redis errors degrade to a miss.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from collections import OrderedDict
from typing import Protocol

import structlog

from ragsys.models import Answer

log = structlog.get_logger(__name__)
_WS = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Case/whitespace/trailing-punctuation insensitive form of a query."""
    q = unicodedata.normalize("NFKC", query).lower().strip()
    return _WS.sub(" ", q).rstrip("?.! ")


def cache_key(query: str, config_fingerprint: str, index_version: str) -> str:
    """Namespaced cache key."""
    raw = f"{normalize_query(query)}|{config_fingerprint}|{index_version}".encode()
    return "rag:answer:" + hashlib.sha256(raw).hexdigest()


class ResponseCache(Protocol):
    """Async key -> Answer cache."""

    async def get(self, key: str) -> Answer | None:
        """Cached answer or None."""
        ...

    async def set(self, key: str, answer: Answer, ttl_seconds: int) -> None:
        """Store an answer."""
        ...

    async def ping(self) -> bool:
        """Health check."""
        ...


class NullCache:
    """Caching disabled."""

    async def get(self, key: str) -> Answer | None:
        return None

    async def set(self, key: str, answer: Answer, ttl_seconds: int) -> None:
        return None

    async def ping(self) -> bool:
        return True


class MemoryCache:
    """In-process LRU with TTL (single replica / tests)."""

    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max_entries
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()

    async def get(self, key: str) -> Answer | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, payload = item
        if expires < time.monotonic():
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return Answer.model_validate_json(payload)

    async def set(self, key: str, answer: Answer, ttl_seconds: int) -> None:
        self._data[key] = (time.monotonic() + ttl_seconds, answer.model_dump_json())
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    async def ping(self) -> bool:
        return True


class RedisCache:
    """Shared cache across API replicas."""

    def __init__(self, url: str) -> None:
        from redis.asyncio import Redis

        self._redis = Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)

    async def get(self, key: str) -> Answer | None:
        try:
            payload = await self._redis.get(key)
        except Exception as exc:
            log.warning("cache.get_failed", error=str(exc))
            return None
        return Answer.model_validate_json(payload) if payload else None

    async def set(self, key: str, answer: Answer, ttl_seconds: int) -> None:
        try:
            await self._redis.set(key, answer.model_dump_json(), ex=ttl_seconds)
        except Exception as exc:
            log.warning("cache.set_failed", error=str(exc))

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:
            return False

    async def close(self) -> None:
        """Close the connection pool."""
        await self._redis.aclose()
