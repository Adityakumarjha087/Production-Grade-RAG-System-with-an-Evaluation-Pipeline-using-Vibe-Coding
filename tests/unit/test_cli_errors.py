"""CLI commands fail with a readable message when their inputs are missing (no models, no network)."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from ragsys.cli import app
from ragsys.config import PipelineConfig
from ragsys.evaluation.testset import ReviewStatus, load_testset, save_testset
from tests.conftest import offline_config

runner = CliRunner()

KYC_SPEC = (
    "items:\n"
    "  - {qtype: factual, doc: kyc_records, question: 'How long are KYC records kept?',\n"
    "     answer: Eight years., quotes: ['for a period of eight years']}\n"
    "  - {qtype: factual, doc: kyc_records, question: 'Bad quote?', answer: x, quotes: ['no such text here']}\n"
)


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


def spec_workspace(tmp_path: Path, sample_pdfs: list[Path]) -> tuple[Path, Path]:
    """A config whose raw dir holds the sample PDFs, plus a spec file with one good and one bad item."""
    cfg = config_file(tmp_path)
    (tmp_path / "raw").mkdir()
    for pdf in sample_pdfs:
        (tmp_path / "raw" / pdf.name).write_bytes(pdf.read_bytes())
    specs = tmp_path / "specs.yaml"
    specs.write_text(KYC_SPEC, encoding="utf-8")
    return cfg, specs


def test_from_specs_writes_pending_items_and_reports_skipped_ones(
    tmp_path: Path, sample_pdfs: list[Path]
) -> None:
    cfg, specs = spec_workspace(tmp_path, sample_pdfs)
    result = runner.invoke(app, ["testset", "from-specs", str(specs), "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "skipped: quote not found" in result.output
    assert "wrote 1 pending items" in result.output and "(1 skipped)" in result.output
    items = load_testset(tmp_path / "testset.jsonl")
    assert [i.status for i in items] == [ReviewStatus.PENDING]


def test_from_specs_approved_flag_skips_review(tmp_path: Path, sample_pdfs: list[Path]) -> None:
    cfg, specs = spec_workspace(tmp_path, sample_pdfs)
    out = tmp_path / "approved.jsonl"
    result = runner.invoke(
        app,
        [
            "testset",
            "from-specs",
            str(specs),
            "--config",
            str(cfg),
            "--approved",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "wrote 1 approved items" in result.output
    assert [i.status for i in load_testset(out)] == [ReviewStatus.APPROVED]


def test_review_approves_a_pending_item_and_saves_it(
    tmp_path: Path, sample_pdfs: list[Path]
) -> None:
    cfg, specs = spec_workspace(tmp_path, sample_pdfs)
    runner.invoke(app, ["testset", "from-specs", str(specs), "--config", str(cfg)])
    result = runner.invoke(app, ["testset", "review", "--config", str(cfg)], input="a\n")
    assert result.exit_code == 0, result.output
    assert "How long are KYC records kept?" in result.output
    assert [i.status for i in load_testset(tmp_path / "testset.jsonl")] == [ReviewStatus.APPROVED]
