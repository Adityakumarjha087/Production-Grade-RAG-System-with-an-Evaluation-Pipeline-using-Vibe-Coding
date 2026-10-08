"""Qdrant collection holding both the dense and BM25 sparse vectors of each chunk.

One collection per (chunking strategy, embedding model) so configurations can be
compared side by side. Collection metadata stores ``avgdl`` (for BM25) and an
``index_version`` hash that the API puts into cache keys: re-ingesting changes the
version and so invalidates stale cached answers automatically.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from ragsys.indexing.bm25 import SparseVec
from ragsys.models import Chunk, ScoredChunk

DENSE = "dense"
SPARSE = "bm25"


def point_id(chunk_id: str) -> str:
    """Qdrant needs UUID/int ids; our chunk ids are 32 hex chars = a UUID."""
    return str(uuid.UUID(hex=chunk_id))


def make_client(url: str, api_key: str | None = None) -> QdrantClient:
    """``:memory:`` -> in-process (tests); ``http(s)://`` -> server; else a local path.

    Local path mode persists to disk without a server (handy for laptop dev/CLI);
    production uses the server.
    """
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url.startswith(("http://", "https://")):
        return QdrantClient(url=url, api_key=api_key, timeout=30)
    return QdrantClient(path=url)


def physical_name(alias: str, version: str) -> str:
    """Name of the versioned collection behind ``alias``."""
    return f"{alias}__{version}"


def alias_target(client: QdrantClient, alias: str) -> str | None:
    """Collection currently behind ``alias`` (None if the alias doesn't exist)."""
    for a in client.get_aliases().aliases:
        if a.alias_name == alias:
            return str(a.collection_name)
    return None


def swap_alias(client: QdrantClient, alias: str, target: str) -> list[str]:
    """Atomically point ``alias`` at ``target`` and drop the collection it replaced.

    Readers using the alias never see a half-built index (blue/green deploy).
    Only the *previous* target is deleted — not every other version — so a
    concurrently running ingestion job never has its in-progress build removed.
    Returns the names of deleted collections.
    """
    previous = alias_target(client, alias)
    ops: list[Any] = []
    if previous is not None:
        ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    ops.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=target, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=ops)
    if previous is None or previous == target:
        return []
    client.delete_collection(previous)
    return [previous]


class QdrantStore:
    """Thin, typed wrapper around one collection."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    def exists(self) -> bool:
        """Whether the collection exists."""
        return bool(self.client.collection_exists(self.collection))

    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None:
        """Create the collection (dropping it first when ``recreate``)."""
        if recreate and self.exists():
            self.client.delete_collection(self.collection)
        if self.exists():
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """Merge ``metadata`` into the collection metadata."""
        self.client.update_collection(self.collection, metadata=metadata)

    def metadata(self) -> dict[str, Any]:
        """Collection metadata (empty if unset)."""
        info = self.client.get_collection(self.collection)
        return dict(info.config.metadata or {})

    def count(self) -> int:
        """Number of points."""
        return int(self.client.count(self.collection, exact=True).count)

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVec],
    ) -> None:
        """Insert or overwrite chunks with their vectors."""
        if not len(chunks) == len(dense) == len(sparse):
            raise ValueError("chunks, dense and sparse must have the same length")
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector={
                    DENSE: list(d),
                    SPARSE: models.SparseVector(indices=s.indices, values=s.values),
                },
                payload=c.model_dump(mode="json"),
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def search_dense(self, vector: Sequence[float], k: int) -> list[ScoredChunk]:
        """Cosine nearest neighbours."""
        return self._query(list(vector), DENSE, k, "dense")

    def search_sparse(self, vector: SparseVec, k: int) -> list[ScoredChunk]:
        """BM25 search (empty query -> no results)."""
        if not vector.indices:
            return []
        query = models.SparseVector(indices=vector.indices, values=vector.values)
        return self._query(query, SPARSE, k, "sparse")

    def _query(self, query: Any, using: str, k: int, stage: str) -> list[ScoredChunk]:
        response = self.client.query_points(
            self.collection, query=query, using=using, limit=k, with_payload=True
        )
        return [
            ScoredChunk(chunk=Chunk.model_validate(p.payload), score=float(p.score), stage=stage)
            for p in response.points
        ]
