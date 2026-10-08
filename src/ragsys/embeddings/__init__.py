"""Dense embedders."""

from __future__ import annotations

from ragsys.config import EmbeddingConfig
from ragsys.embeddings.base import Embedder, HashEmbedder


def get_embedder(cfg: EmbeddingConfig) -> Embedder:
    """Build the embedder named by ``cfg.provider``."""
    if cfg.provider == "hash":
        return HashEmbedder(cfg.dim)
    if cfg.provider == "fastembed":
        from ragsys.embeddings.fastembed_embedder import FastEmbedEmbedder

        return FastEmbedEmbedder(cfg.model, cfg.dim, cfg.batch_size)
    raise ValueError(f"Unknown embedding provider: {cfg.provider}")


__all__ = ["Embedder", "HashEmbedder", "get_embedder"]
