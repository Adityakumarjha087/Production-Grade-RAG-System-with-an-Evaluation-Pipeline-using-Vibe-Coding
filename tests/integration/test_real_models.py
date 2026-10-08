"""Real models (downloads on first run): fastembed embedder + cross-encoder, Docling.

Run with:  uv run pytest -m slow
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from ragsys.config import Settings
from ragsys.embeddings.fastembed_embedder import FastEmbedEmbedder
from ragsys.factory import build_components
from ragsys.indexing.bm25 import BM25Encoder
from ragsys.ingestion.chunkers import get_chunker
from ragsys.ingestion.pipeline import chunk_corpus, index_chunks
from ragsys.models import Document, ElementType
from ragsys.retrieval.reranker import FastEmbedReranker
from tests.conftest import offline_config

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def embedder() -> FastEmbedEmbedder:
    e = FastEmbedEmbedder("BAAI/bge-base-en-v1.5", 768)
    e.warmup()
    return e


@pytest.fixture(scope="module", params=["Xenova/ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-base"])
def reranker(request: pytest.FixtureRequest) -> FastEmbedReranker:
    """Both rerankers in the eval matrix must work with the same thresholds."""
    r = FastEmbedReranker(request.param)
    r.warmup()
    return r


def test_embedding_dim_matches_config(embedder: FastEmbedEmbedder) -> None:
    assert len(embedder.embed_query("x")) == 768
    assert len(embedder.embed_documents(["a", "b"])) == 2


@pytest.fixture(scope="module")
def real_components(embedder, reranker, sample_docs: list[Document]):
    cfg = offline_config(
        embedding={"provider": "fastembed", "model": "BAAI/bge-base-en-v1.5", "dim": 768},
        retrieval={"rerank": {"provider": "fastembed", "model": reranker.model_name}},
        generation={"abstention": {"min_rerank_score": 0.15, "min_dense_score": 0.55}},
    )
    client = QdrantClient(location=":memory:")
    chunks = chunk_corpus(sample_docs, get_chunker(cfg.chunking))
    index_chunks(
        chunks, alias=cfg.collection_name(), client=client, embedder=embedder, bm25=BM25Encoder()
    )
    return build_components(
        cfg,
        Settings(_env_file=None),
        client=client,
        embedder=embedder,  # type: ignore[call-arg]
        reranker=reranker,
    )


@pytest.mark.parametrize(
    ("query", "title_word"),
    [
        ("When must funds be paid in under the shorter settlement cycle?", "Settlement"),
        ("For how many years should client identity documents be retained?", "KYC"),
        ("What is the cost cap for passive funds like ETFs?", "Expense"),
        ("How quickly must a company fix an investor's complaint?", "Grievance"),
    ],
)
def test_semantic_retrieval_with_reranker(real_components, query: str, title_word: str) -> None:
    result = real_components.service.retrieval.retrieve(query)
    assert title_word in result.chunks[0].chunk.doc_title
    assert set(result.signals) == {"rerank", "dense"}
    # Paraphrased questions must pass the gate even when the cross-encoder under-scores them
    assert real_components.service.policy.pre_check(result.signals, len(result.chunks)) is None


async def test_off_topic_question_abstains(real_components) -> None:
    answer = await real_components.service.answer("Who won the football world cup in 2018?")
    assert answer.abstained and answer.abstain_reason == "low_retrieval_confidence"


def test_docling_parser(sample_pdfs: list[Path]) -> None:
    pytest.importorskip("docling")
    from ragsys.ingestion.parsers.docling_parser import DoclingParser

    path = next(p for p in sample_pdfs if "expense" in p.name)
    doc = DoclingParser().parse(path)
    assert doc.title.startswith("Total Expense Ratio Limits")
    table = next(e for e in doc.elements if e.type == ElementType.TABLE)
    assert "2.25%" in table.text and "First 500" in table.text
    assert "SYNTHETIC SAMPLE DOCUMENT" not in doc.text  # page header furniture dropped
    assert any(e.text.startswith("3.1") for e in doc.elements)  # list markers kept
    for el in doc.elements:
        assert doc.text[el.char_start : el.char_end] == el.text
