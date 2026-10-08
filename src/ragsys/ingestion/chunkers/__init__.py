"""Chunking strategies behind a common :class:`Chunker` protocol."""

from __future__ import annotations

from ragsys.config import ChunkingConfig
from ragsys.ingestion.chunkers.base import Chunker
from ragsys.ingestion.chunkers.fixed import FixedSizeChunker
from ragsys.ingestion.chunkers.recursive import RecursiveChunker
from ragsys.ingestion.chunkers.structural import StructuralChunker

STRATEGIES = ("fixed", "recursive", "structural")


def get_chunker(cfg: ChunkingConfig) -> Chunker:
    """Build the chunker named by ``cfg.strategy``."""
    if cfg.strategy == "fixed":
        return FixedSizeChunker(cfg.chunk_size, cfg.chunk_overlap)
    if cfg.strategy == "recursive":
        return RecursiveChunker(cfg.chunk_size, cfg.chunk_overlap)
    if cfg.strategy == "structural":
        return StructuralChunker(cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_size)
    raise ValueError(f"Unknown chunking strategy: {cfg.strategy}")


__all__ = [
    "STRATEGIES",
    "Chunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "StructuralChunker",
    "get_chunker",
]
