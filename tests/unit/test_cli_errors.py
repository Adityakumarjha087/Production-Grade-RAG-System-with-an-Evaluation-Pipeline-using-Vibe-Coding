"""CLI commands fail with a readable message when their inputs are missing (no models, no network)."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from ragsys.cli import app
from ragsys.config import PipelineConfig
from ragsys.evaluation.testset import save_testset
from tests.conftest import offline_config

runner = CliRunner()


def config_file(tmp_path: Path) -> Path:
    cfg: PipelineConfig = offline_config(
        paths={
            "raw_dir": str(tmp_path / "raw"),
            "parsed_dir": str(tmp_path / "parsed"),
            "testset_path": str(tmp_path / "testset.jsonl"),
            "results_dir": str(tmp_path / "results"),
            "llm_cache_dir": str(tmp_path / "llm"),
        }
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    return path


def test_eval_without_approved_items_explains_how_to_proceed(tmp_path: Path) -> None:
    cfg = config_file(tmp_path)
    save_testset(tmp_path / "testset.jsonl", [])
    result = runner.invoke(app, ["eval", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "no approved test items" in result.output


def test_ingest_without_pdfs_names_the_directory(tmp_path: Path) -> None:
    cfg = config_file(tmp_path)
    (tmp_path / "raw").mkdir()
    result = runner.invoke(app, ["ingest", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "no PDFs found" in result.output


def test_testset_stats_on_an_empty_set_reports_zero(tmp_path: Path) -> None:
    cfg = config_file(tmp_path)
    save_testset(tmp_path / "testset.jsonl", [])
    result = runner.invoke(app, ["testset", "stats", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "'total': 0" in result.output
