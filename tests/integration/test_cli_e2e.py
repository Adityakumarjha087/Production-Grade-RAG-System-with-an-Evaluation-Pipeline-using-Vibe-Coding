"""Full CLI workflow on the sample corpus, hermetic (local-path Qdrant, offline models)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragsys.cli import app
from tests.conftest import ROOT

pytestmark = pytest.mark.integration


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
extends: {ROOT / "configs/base.yaml"}
name: cli-test
paths:
  raw_dir: {tmp_path / "raw"}
  parsed_dir: {tmp_path / "parsed"}
  testset_path: {tmp_path / "testset.jsonl"}
  results_dir: {tmp_path / "results"}
  llm_cache_dir: {tmp_path / "llm_cache"}
parsing: {{parser: pymupdf, fallback_parser: null}}
chunking: {{chunk_size: 600, chunk_overlap: 80, min_chunk_size: 120}}
embedding: {{provider: hash, model: hash, dim: 256}}
retrieval:
  rerank: {{provider: lexical}}
generation:
  provider: extractive
  abstention: {{min_rerank_score: 0.3}}
cache: {{backend: memory}}
eval: {{judge_backend: lexical, answer_subset: 11}}
""")
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("""
chunkers: {fixed: {chunking: {strategy: fixed}}, structural: {chunking: {strategy: structural}}}
retrieval:
  bm25: {retrieval: {dense: {enabled: false}, rerank: {enabled: false}}}
  hybrid_rerank: {}
""")
    monkeypatch.setenv("QDRANT_URL", str(tmp_path / "qdrant"))
    monkeypatch.setenv("RAG_CONFIG", str(cfg))
    return tmp_path


def run(*args: str) -> str:
    result = CliRunner().invoke(app, list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


def test_full_workflow(workspace: Path) -> None:
    out = run("sample-data", "--out", str(workspace / "raw"))
    assert "wrote 4 sample PDFs" in out and "wrote 11 approved items" in out

    out = run("ingest", "--strategy", "all")
    assert out.count("chunks, built") == 3
    assert out.count("up to date") == 0
    assert run("ingest").count("up to date") == 1  # idempotent re-run

    out = run("query", "How long must intermediaries preserve KYC records?", "--no-stream")
    assert "eight years" in out.lower() and "[1]" in out and "abstained=False" in out

    out = run("query", "What is the weather on Mars?", "--no-stream")
    assert "abstained=True" in out

    out = run("testset", "stats")
    assert "'status:approved': 11" in out

    out = run("eval", "--matrix", str(workspace / "matrix.yaml"))
    assert "| structural+hybrid_rerank |" in out
    reports = list((workspace / "results").glob("*/report.md"))
    assert len(reports) == 1
    assert "testset_hash" in reports[0].read_text()


def test_eval_without_approved_items_fails_cleanly(workspace: Path) -> None:
    (workspace / "testset.jsonl").write_text("")
    result = CliRunner().invoke(app, ["eval", "--no-answers"])
    assert result.exit_code != 0 and "no approved test items" in result.output
