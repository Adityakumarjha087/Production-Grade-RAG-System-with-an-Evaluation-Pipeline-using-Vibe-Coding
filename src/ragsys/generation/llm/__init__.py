"""LLM clients."""

from __future__ import annotations

from ragsys.config import GenerationConfig
from ragsys.generation.llm.base import Completion, LLMClient, LLMEvent, TextDelta
from ragsys.generation.llm.extractive import ExtractiveLLM


def get_llm(cfg: GenerationConfig, api_key: str | None) -> LLMClient:
    """Build the generation LLM named by ``cfg.provider``."""
    if cfg.provider == "extractive":
        return ExtractiveLLM()
    if cfg.provider == "anthropic":
        from ragsys.generation.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(
            cfg.model, effort=cfg.effort, use_fallbacks=cfg.use_fallbacks, api_key=api_key
        )
    raise ValueError(f"Unknown LLM provider: {cfg.provider}")


__all__ = ["Completion", "ExtractiveLLM", "LLMClient", "LLMEvent", "TextDelta", "get_llm"]
