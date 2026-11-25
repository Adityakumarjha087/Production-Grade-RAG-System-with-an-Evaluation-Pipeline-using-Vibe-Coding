"""Report tables, git SHA fallback and latency percentile helper."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ragsys.evaluation import report
from ragsys.evaluation.report import stage_markdown, to_markdown
from ragsys.evaluation.runner import VariantResult, _percentile


def result(name: str, **metrics: float) -> VariantResult:
    return VariantResult(
        name=name, chunker="fixed", retrieval="bm25", n_retrieval=3, metrics=metrics
    )


def test_best_value_is_bold_and_lower_is_better_for_latency_and_false_abstains() -> None:
    a = result("a", mrr=0.5, retrieval_p50_ms=120.0, false_abstain_rate=0.2)
    b = result("b", mrr=0.7, retrieval_p50_ms=80.0, false_abstain_rate=0.1)
    table = to_markdown([a, b], ["mrr", "retrieval_p50_ms", "false_abstain_rate"])
    row_a, row_b = table.splitlines()[2:]
    assert row_a == "| a | 0.500 | 120 | 0.200 |"
    assert row_b == "| b | **0.700** | **80** | **0.100** |"


def test_single_result_and_missing_metric_are_not_bolded() -> None:
    table = to_markdown([result("only", mrr=0.5)], ["mrr", "faithfulness"])
    assert table.splitlines()[2] == "| only | 0.500 | – |"


def test_stage_markdown_shows_dash_for_missing_stages() -> None:
    r = result("a")
    r.stage_metrics = {"dense": {"recall@5": 0.25}}
    row = stage_markdown([r], k=5).splitlines()[2]
    assert row == "| a | 0.250 | – | – | – |"


def test_percentile_edge_cases() -> None:
    assert _percentile([], 50) == 0.0
    assert _percentile([7.0], 95) == 7.0
    assert _percentile([float(i) for i in range(1, 101)], 50) == pytest.approx(50.5)


def test_git_sha_is_unknown_without_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert report.git_sha() == "unknown"
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)  # not a repository
    assert report.git_sha() == "unknown"
