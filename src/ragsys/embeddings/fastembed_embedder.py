"""ONNX embeddings via fastembed: no torch, CPU-friendly, small Docker image."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import TextEmbedding


class FastEmbedEmbedder:
    """Wraps ``fastembed.TextEmbedding``; the model is loaded on first use."""

    def __init__(self, model_name: str, dim: int, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self._model: TextEmbedding | None = None

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def warmup(self) -> None:
        """Load (and if needed download) the model now rather than on first request."""
        self.embed_query("warmup")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages (bge: no instruction prefix for passages)."""
        model = self._get_model()
        return [v.tolist() for v in model.passage_embed(list(texts), batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query (fastembed adds the bge query instruction)."""
        vectors = list(self._get_model().query_embed(text))
        return [float(x) for x in vectors[0]]
