"""BM25 as Qdrant sparse vectors.

Qdrant's ``Modifier.IDF`` computes the IDF term server-side from collection
statistics. We supply the *term-frequency* half of BM25 per document:

    w(t, d) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

and weight 1 per unique query term, so the dot product Qdrant computes equals
classic BM25. ``avgdl`` is computed over the corpus at indexing time and stored in
the collection metadata.

The tokenizer is domain-aware: regulatory identifiers like
``SEBI/HO/MRD/2023/12`` or ``Regulation 23(1)(b)`` are kept as whole tokens (and
also split into parts), because exact identifier matching is where lexical search
beats embeddings.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import snowballstemmer

_COMPOUND = re.compile(r"[a-z0-9]+(?:[/._\-][a-z0-9]+)+")
_CLAUSE_REF = re.compile(r"\b\d+[a-z]?(?:\([0-9a-z]{1,4}\))+")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "shall",
        "which",
        "such",
        "any",
        "all",
        "may",
        "not",
        "no",
        "been",
        "being",
        "these",
        "those",
        "their",
        "there",
        "than",
        "into",
        "under",
        "upon",
        "also",
        "other",
        "its",
        "such",
    ]
)
_STEMMER = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    """Lowercase, keep identifiers whole, drop stopwords, stem plain words."""
    lowered = text.lower()
    tokens: list[str] = []
    tokens.extend(_COMPOUND.findall(lowered))
    tokens.extend(_CLAUSE_REF.findall(lowered))
    for word in _WORD.findall(lowered):
        if word in _STOPWORDS or len(word) == 1:
            continue
        tokens.append(word if word.isdigit() else _STEMMER.stemWord(word))
    return tokens


def token_index(token: str) -> int:
    """Stable 31-bit hash for a token (Qdrant sparse indices are uint32)."""
    return zlib.crc32(token.encode()) & 0x7FFFFFFF


@dataclass(frozen=True)
class SparseVec:
    """Sorted sparse vector (indices unique)."""

    indices: list[int]
    values: list[float]


def _to_sparse(weights: dict[int, float]) -> SparseVec:
    items = sorted(weights.items())
    return SparseVec([i for i, _ in items], [v for _, v in items])


class BM25Encoder:
    """Produces BM25 document/query sparse vectors (see module docstring)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, avgdl: float | None = None) -> None:
        self.k1 = k1
        self.b = b
        self.avgdl = avgdl

    def fit(self, texts: Iterable[str]) -> float:
        """Compute and store average document length (in tokens)."""
        lengths = [len(tokenize(t)) for t in texts]
        self.avgdl = (sum(lengths) / len(lengths)) if lengths else 1.0
        return self.avgdl

    def encode_document(self, text: str) -> SparseVec:
        """TF-saturated weights for one document."""
        if self.avgdl is None:
            raise RuntimeError("BM25Encoder.fit() must be called before encoding documents")
        tokens = tokenize(text)
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / max(self.avgdl, 1e-9))
        weights: dict[int, float] = {}
        for token, tf in Counter(tokens).items():
            idx = token_index(token)
            weights[idx] = weights.get(idx, 0.0) + tf * (self.k1 + 1) / (tf + norm)
        return _to_sparse(weights)

    def encode_query(self, text: str) -> SparseVec:
        """Weight 1.0 per unique query term."""
        return _to_sparse({token_index(t): 1.0 for t in set(tokenize(text))})
