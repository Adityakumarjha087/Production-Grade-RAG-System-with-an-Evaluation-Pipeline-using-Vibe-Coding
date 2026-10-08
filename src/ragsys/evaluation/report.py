"""Comparison table + reproducibility metadata for an evaluation run."""

from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ragsys.evaluation.runner import VariantResult

PRIMARY_COLUMNS = [
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
    "faithfulness",
    "correctness",
    "abstain_recall",
    "false_abstain_rate",
    "retrieval_p50_ms",
]


def git_sha() -> str:
    """Current commit (``unknown`` outside a git checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fmt(value: float | None, column: str) -> str:
    if value is None:
        return "–"
    if column.endswith("_ms"):
        return f"{value:.0f}"
    return f"{value:.3f}"


def columns_for(results: Sequence[VariantResult]) -> list[str]:
    """Primary columns that at least one result has."""
    present = {k for r in results for k in r.metrics}
    return [c for c in PRIMARY_COLUMNS if c in present]


def to_markdown(results: Sequence[VariantResult], columns: Sequence[str]) -> str:
    """GitHub-flavoured markdown table; best value per column in bold."""
    lower_is_better = {"false_abstain_rate", "retrieval_p50_ms"}
    best: dict[str, float] = {}
    for col in columns:
        values = [r.metrics[col] for r in results if col in r.metrics]
        if values:
            best[col] = min(values) if col in lower_is_better else max(values)
    header = "| config | " + " | ".join(columns) + " |"
    sep = "|---|" + "|".join("---:" for _ in columns) + "|"
    rows = []
    for r in results:
        cells = []
        for col in columns:
            text = _fmt(r.metrics.get(col), col)
            if col in best and r.metrics.get(col) == best[col] and len(results) > 1:
                text = f"**{text}**"
            cells.append(text)
        rows.append(f"| {r.name} | " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def stage_markdown(results: Sequence[VariantResult], k: int = 5) -> str:
    """Recall@k at each retrieval stage: shows what fusion and reranking add."""
    stages = ["dense", "sparse", "rrf", "rerank"]
    header = "| config | " + " | ".join(f"{s} R@{k}" for s in stages) + " |"
    sep = "|---|" + "|".join("---:" for _ in stages) + "|"
    rows = [
        f"| {r.name} | "
        + " | ".join(_fmt(r.stage_metrics.get(s, {}).get(f"recall@{k}"), "x") for s in stages)
        + " |"
        for r in results
    ]
    return "\n".join([header, sep, *rows])


def write_report(
    results: Sequence[VariantResult], out_root: Path, meta: dict[str, Any], config: dict[str, Any]
) -> Path:
    """Write report.md, results.csv, results.json and config.yaml into a timestamped dir."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out_root / stamp
    out.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "timestamp": stamp, "git_sha": git_sha()}
    columns = columns_for(results)
    (out / "results.json").write_text(
        json.dumps({"meta": meta, "results": [asdict(r) for r in results]}, indent=2), "utf-8"
    )
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        all_cols = sorted({k for r in results for k in r.metrics})
        writer = csv.writer(fh)
        writer.writerow(
            ["config", "chunker", "retrieval", "n_retrieval", "n_answers", "cost_usd", *all_cols]
        )
        for r in results:
            writer.writerow(
                [
                    r.name,
                    r.chunker,
                    r.retrieval,
                    r.n_retrieval,
                    r.n_answers,
                    f"{r.cost_usd:.4f}",
                    *[r.metrics.get(c, "") for c in all_cols],
                ]
            )
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), "utf-8")
    meta_lines = "\n".join(f"- **{k}**: {v}" for k, v in meta.items())
    report = (
        f"# Evaluation report\n\n{meta_lines}\n\n"
        f"## Comparison\n\n{to_markdown(results, columns)}\n\n"
        f"## Recall by retrieval stage\n\n{stage_markdown(results)}\n"
    )
    (out / "report.md").write_text(report, "utf-8")
    return out
