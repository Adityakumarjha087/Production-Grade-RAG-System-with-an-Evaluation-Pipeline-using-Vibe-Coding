"""Ingestion: parse (cached) -> chunk -> embed + BM25 -> index (blue/green).

Idempotency: the index version is a hash of the chunk ids + embedding model. If a
collection with that version already exists the run is a no-op; otherwise a new
versioned collection is built and the alias is swapped atomically.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from qdrant_client import QdrantClient

from ragsys.embeddings import Embedder
from ragsys.indexing.bm25 import BM25Encoder
from ragsys.indexing.qdrant_store import QdrantStore, physical_name, swap_alias
from ragsys.ingestion.chunkers import Chunker
from ragsys.ingestion.parsers import Parser
from ragsys.models import Chunk, Document

log = structlog.get_logger(__name__)


@dataclass
class ParseReport:
    """Outcome of parsing a corpus."""

    documents: list[Document] = field(default_factory=list)
    from_cache: int = 0
    used_fallback: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexReport:
    """Outcome of indexing one chunking strategy."""

    collection: str
    version: str
    num_documents: int
    num_chunks: int
    skipped: bool
    seconds: float


def find_pdfs(raw_dir: Path) -> list[Path]:
    """All PDFs under ``raw_dir`` in a stable order."""
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() == ".pdf")


def _cache_path(parsed_dir: Path, parser: str, path: Path) -> Path:
    from ragsys.ingestion.parsers.base import file_doc_id

    return parsed_dir / parser / f"{file_doc_id(path)}.json"


def _parse_one(path: Path, parser: Parser, fallback: Parser | None) -> tuple[Document, bool]:
    """Parse with the primary parser; on error or empty output use the fallback."""
    try:
        doc = parser.parse(path)
        if doc.text.strip() or fallback is None:
            return doc, False
    except Exception as exc:
        if fallback is None:
            raise
        log.warning("parse.primary_failed", path=str(path), error=str(exc))
    assert fallback is not None
    return fallback.parse(path), True


def load_sidecars(pdfs: Sequence[Path]) -> dict[Path, dict[str, str]]:
    """Per-PDF metadata from ``manifest.jsonl`` files next to the PDFs.

    Each JSON line needs ``file`` and may carry ``title``, ``date``, ``url``,
    ``sebi_id`` (the format written by ``ragsys fetch-sebi``).
    """
    meta: dict[Path, dict[str, str]] = {}
    for folder in {p.parent for p in pdfs}:
        manifest = folder / "manifest.jsonl"
        if not manifest.exists():
            continue
        for line in manifest.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("file"):
                keys = ("title", "date", "url", "sebi_id")
                meta[folder / entry["file"]] = {k: str(entry[k]) for k in keys if entry.get(k)}
    return meta


def enrich(doc: Document, sidecar: dict[str, str] | None) -> Document:
    """Apply authoritative sidecar metadata (official title, date, URL)."""
    if not sidecar:
        return doc
    title = sidecar.get("title") or doc.title
    return doc.model_copy(update={"title": title, "metadata": {**doc.metadata, **sidecar}})


def parse_corpus(
    pdfs: Sequence[Path], parsed_dir: Path, parser: Parser, fallback: Parser | None = None
) -> ParseReport:
    """Parse every PDF, reusing cached JSON from previous runs."""
    report = ParseReport()
    sidecars = load_sidecars(pdfs)
    for path in pdfs:
        cache = _cache_path(parsed_dir, parser.name, path)
        if cache.exists():
            cached = Document.model_validate_json(cache.read_text("utf-8"))
            report.documents.append(enrich(cached, sidecars.get(path)))
            report.from_cache += 1
            continue
        try:
            doc, used_fallback = _parse_one(path, parser, fallback)
        except Exception as exc:
            log.error("parse.failed", path=str(path), error=str(exc))
            report.failed[str(path)] = str(exc)
            continue
        if used_fallback:
            report.used_fallback.append(str(path))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(doc.model_dump_json(), "utf-8")  # cache stays parser-pure
        report.documents.append(enrich(doc, sidecars.get(path)))
        log.info("parse.done", path=str(path), parser=doc.parser, elements=len(doc.elements))
    return report


def chunk_corpus(docs: Sequence[Document], chunker: Chunker) -> list[Chunk]:
    """Chunk all documents (document order preserved)."""
    return [c for doc in docs for c in chunker.chunk(doc)]


def index_version(chunks: Sequence[Chunk], embedding_model: str, bm25: BM25Encoder) -> str:
    """Content hash of what would be indexed."""
    h = hashlib.sha256(f"{embedding_model}|{bm25.k1}|{bm25.b}".encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(hashlib.sha256(c.text.encode()).digest())
    return h.hexdigest()[:12]


def _batches(items: Sequence[Chunk], size: int) -> Iterator[Sequence[Chunk]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def index_chunks(
    chunks: Sequence[Chunk],
    *,
    alias: str,
    client: QdrantClient,
    embedder: Embedder,
    bm25: BM25Encoder,
    batch_size: int = 128,
    force: bool = False,
) -> IndexReport:
    """Build a versioned collection for ``chunks`` and point ``alias`` at it."""
    started = time.perf_counter()
    bm25.fit(c.text for c in chunks)
    version = index_version(chunks, embedder.model_name, bm25)
    target = QdrantStore(client, physical_name(alias, version))
    num_docs = len({c.doc_id for c in chunks})
    if target.exists() and not force:
        swap_alias(client, alias, target.collection)
        return IndexReport(alias, version, num_docs, len(chunks), True, 0.0)
    target.ensure_collection(embedder.dim, recreate=True)
    for batch in _batches(chunks, batch_size):
        dense = embedder.embed_documents([c.text for c in batch])
        sparse = [bm25.encode_document(c.text) for c in batch]
        target.upsert(batch, dense, sparse)
    target.set_metadata(
        {
            "index_version": version,
            "avgdl": bm25.avgdl,
            "k1": bm25.k1,
            "b": bm25.b,
            "embedding_model": embedder.model_name,
            "num_chunks": len(chunks),
            "num_documents": num_docs,
        }
    )
    swap_alias(client, alias, target.collection)
    elapsed = time.perf_counter() - started
    log.info("index.done", alias=alias, version=version, chunks=len(chunks), seconds=elapsed)
    return IndexReport(alias, version, num_docs, len(chunks), False, elapsed)
