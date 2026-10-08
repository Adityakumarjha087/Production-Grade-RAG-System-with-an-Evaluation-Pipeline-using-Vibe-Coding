"""When to say "I don't know".

Three layers, cheapest first:

1. **Pre-generation gate** – if *every* calibrated retrieval signal (cross-encoder
   score, dense cosine) is below its threshold, skip the LLM entirely (saves cost,
   prevents guessing). Requiring agreement avoids false abstentions: cross-encoders
   under-score paraphrases that dense retrieval still matches (see DECISIONS.md).
2. **Instructed abstention** – the prompt tells the model to emit a fixed sentence
   when the context is insufficient; we detect it.
3. **Post-generation grounding check** – an answer with zero valid citations is
   treated as ungrounded and replaced by an abstention.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from ragsys.config import AbstentionConfig
from ragsys.generation.prompts import ABSTAIN_TEXT
from ragsys.models import Citation

NO_RESULTS = "no_retrieval_results"
LOW_CONFIDENCE = "low_retrieval_confidence"
MODEL_ABSTAINED = "model_abstained"
NO_CITATIONS = "no_valid_citations"
REFUSED = "provider_refusal"

_NORMALISE = re.compile(r"[^a-z ]+")
_ABSTAIN_KEY = _NORMALISE.sub("", ABSTAIN_TEXT.lower().replace("'", "")).strip()


def is_abstention(text: str) -> bool:
    """True if ``text`` is (or starts with) the instructed abstention sentence."""
    normalised = _NORMALISE.sub("", text.lower().replace("'", "").replace("’", "")).strip()
    return normalised.startswith(_ABSTAIN_KEY)


class AbstentionPolicy:
    """Applies :class:`AbstentionConfig` thresholds."""

    def __init__(self, cfg: AbstentionConfig) -> None:
        self.cfg = cfg

    def threshold(self, source: str | None) -> float | None:
        """Threshold for a confidence source (``None`` = no gate available)."""
        if source == "rerank":
            return self.cfg.min_rerank_score
        if source == "dense":
            return self.cfg.min_dense_score
        return None

    def pre_check(self, signals: Mapping[str, float], num_hits: int) -> str | None:
        """Reason to abstain before calling the LLM, or ``None`` to proceed.

        Abstains only when all available signals are below their thresholds; with
        no calibrated signal (BM25-only) the decision is left to the LLM.
        """
        if num_hits == 0:
            return NO_RESULTS
        checks = [
            score < limit
            for name, score in signals.items()
            if (limit := self.threshold(name)) is not None
        ]
        if self.cfg.enabled and checks and all(checks):
            return LOW_CONFIDENCE
        return None

    def post_check(self, text: str, citations: list[Citation]) -> str | None:
        """Reason to treat a generated answer as an abstention, or ``None``."""
        if is_abstention(text):
            return MODEL_ABSTAINED
        if self.cfg.enabled and self.cfg.require_citation and not citations:
            return NO_CITATIONS
        return None
