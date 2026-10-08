"""Token cost accounting from the pricing table in config."""

from __future__ import annotations

from collections.abc import Mapping

import structlog

from ragsys.config import ModelPrice
from ragsys.models import Usage

log = structlog.get_logger(__name__)
_warned: set[str] = set()


def find_price(model: str, pricing: Mapping[str, ModelPrice]) -> ModelPrice | None:
    """Exact match first, then the longest configured name that prefixes ``model``."""
    if model in pricing:
        return pricing[model]
    matches = [name for name in pricing if model.startswith(name)]
    return pricing[max(matches, key=len)] if matches else None


def compute_cost(usage: Usage, model: str, pricing: Mapping[str, ModelPrice]) -> float:
    """USD cost of ``usage`` on ``model`` (0.0 and a one-time warning if unpriced)."""
    price = find_price(model, pricing)
    if price is None:
        if model not in _warned:
            _warned.add(model)
            log.warning("cost.unknown_model", model=model)
        return 0.0
    total = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    )
    return round(total / 1_000_000, 8)
