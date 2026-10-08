from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from ragsys.embeddings import HashEmbedder
from ragsys.indexing.bm25 import BM25Encoder
from ragsys.indexing.qdrant_store import QdrantStore, alias_target
from ragsys.ingestion import pipeline
from ragsys.ingestion.chunkers import StructuralChunker
from ragsys.ingestion.parsers import PyMuPDFParser
from ragsys.ingestion.pipeline import (
    PARSE_SCHEMA_VERSION,
    chunk_corpus,
    find_pdfs,
    index_chunks,
    parse_corpus,
)
from ragsys.models import Document


class ExplodingParser:
    name = "exploding"

    def parse(self, path: Path) -> Document:
        raise RuntimeError("corrupt pdf")


def test_find_pdfs_recursive_and_sorted(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    for p in (tmp_path / "b" / "z.PDF", tmp_path / "a.pdf", tmp_path / "notes.txt"):
        p.write_bytes(b"x")
    assert [p.name for p in find_pdfs(tmp_path)] == ["a.pdf", "z.PDF"]


def test_parse_corpus_caches(sample_pdfs: list[Path], tmp_path: Path) -> None:
    first = parse_corpus(sample_pdfs, tmp_path, PyMuPDFParser())
    second = parse_corpus(sample_pdfs, tmp_path, PyMuPDFParser())
    assert first.from_cache == 0 and second.from_cache == len(sample_pdfs)
    assert [d.text for d in first.documents] == [d.text for d in second.documents]


def test_parse_cache_is_invalidated_by_a_schema_version_bump(
    sample_pdfs: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parse_corpus(sample_pdfs, tmp_path, PyMuPDFParser())
    monkeypatch.setattr(pipeline, "PARSE_SCHEMA_VERSION", PARSE_SCHEMA_VERSION + 1)
    assert parse_corpus(sample_pdfs, tmp_path, PyMuPDFParser()).from_cache == 0


def test_parse_corpus_uses_fallback(sample_pdfs: list[Path], tmp_path: Path) -> None:
    report = parse_corpus(sample_pdfs[:1], tmp_path, ExplodingParser(), PyMuPDFParser())
    assert report.used_fallback == [str(sample_pdfs[0])]
    assert report.documents[0].parser == "pymupdf"


def test_parse_corpus_records_failures_without_aborting(
    sample_pdfs: list[Path], tmp_path: Path
) -> None:
    report = parse_corpus(sample_pdfs[:2], tmp_path, ExplodingParser())
    assert len(report.failed) == 2 and report.documents == []


def _index(docs: list[Document], client: QdrantClient, force: bool = False):
    chunks = chunk_corpus(docs, StructuralChunker(600, 80, 120))
    return index_chunks(
        chunks,
        alias="idx",
        client=client,
        embedder=HashEmbedder(64),
        bm25=BM25Encoder(),
        force=force,
    )


def test_index_is_idempotent_and_versioned(
    sample_docs: list[Document], qdrant: QdrantClient
) -> None:
    first = _index(sample_docs, qdrant)
    again = _index(sample_docs, qdrant)
    assert not first.skipped and again.skipped
    assert first.version == again.version
    store = QdrantStore(qdrant, "idx")
    assert store.count() == first.num_chunks
    meta = store.metadata()
    assert meta["index_version"] == first.version and meta["avgdl"] > 0


def test_reindex_with_new_content_swaps_alias(
    sample_docs: list[Document], qdrant: QdrantClient
) -> None:
    v1 = _index(sample_docs[:2], qdrant)
    v2 = _index(sample_docs, qdrant)
    assert v1.version != v2.version
    assert alias_target(qdrant, "idx") == f"idx__{v2.version}"
    names = {c.name for c in qdrant.get_collections().collections}
    assert f"idx__{v1.version}" not in names  # old version garbage-collected


def test_force_rebuilds(sample_docs: list[Document], qdrant: QdrantClient) -> None:
    _index(sample_docs, qdrant)
    assert not _index(sample_docs, qdrant, force=True).skipped


@pytest.mark.parametrize("bad", [[]])
def test_empty_corpus_indexes_nothing(bad: list, qdrant: QdrantClient) -> None:
    report = _index(bad, qdrant)
    assert report.num_chunks == 0
