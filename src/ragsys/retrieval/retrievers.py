"""First-stage retrievers: dense (embeddings) and sparse (BM25)."""

from __future__ import annotations

from ragsys.embeddings import Embedder
from ragsys.indexing.bm25 import BM25Encoder
from ragsys.indexing.qdrant_store import QdrantStore
from ragsys.models import ScoredChunk


class DenseRetriever:
    """Cosine similarity over chunk embeddings."""

    def __init__(self, store: QdrantStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Top-``k`` chunks by cosine similarity."""
        return self.store.search_dense(self.embedder.embed_query(query), k)


class SparseRetriever:
    """BM25 via Qdrant sparse vectors with the IDF modifier."""

    def __init__(self, store: QdrantStore, encoder: BM25Encoder) -> None:
        self.store = store
        self.encoder = encoder

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Top-``k`` chunks by BM25."""
        return self.store.search_sparse(self.encoder.encode_query(query), k)
