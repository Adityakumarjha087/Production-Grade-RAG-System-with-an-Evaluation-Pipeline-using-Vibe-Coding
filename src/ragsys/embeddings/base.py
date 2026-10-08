"""Embedding interface plus a deterministic hashing embedder for tests/offline use."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Dense text embedder. Query and document encodings may differ (e.g. bge prefix)."""

    model_name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...


class HashEmbedder:
    """Feature-hashed bag-of-words + bigrams, L2-normalised.

    Not semantically smart, but deterministic, instant and lexically meaningful:
    good enough to make retrieval tests assert real behaviour without downloads.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model_name = f"hash-{dim}"

    def _vector(self, text: str) -> list[float]:
        words = _WORD.findall(text.lower())
        vec = [0.0] * self.dim
        for feature in [*words, *(f"{a}_{b}" for a, b in pairwise(words))]:
            h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages."""
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query (same space as documents)."""
        return self._vector(text)
