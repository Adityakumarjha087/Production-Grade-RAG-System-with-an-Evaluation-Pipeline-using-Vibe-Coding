from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from ragsys.embeddings import HashEmbedder
from ragsys.indexing.bm25 import BM25Encoder, token_index, tokenize
from ragsys.indexing.qdrant_store import (
    QdrantStore,
    alias_target,
    make_client,
    physical_name,
    point_id,
    swap_alias,
)
from ragsys.models import Chunk


class TestTokenizer:
    def test_keeps_circular_numbers_whole_and_split(self) -> None:
        tokens = tokenize("See SEBI/HO/MRD/2023/12 now")
        assert "sebi/ho/mrd/2023/12" in tokens
        assert {"sebi", "ho", "mrd", "2023", "12"} <= set(tokens)

    def test_keeps_clause_references(self) -> None:
        assert "23(1)(b)" in tokenize("Regulation 23(1)(b) applies")

    def test_stems_and_drops_stopwords(self) -> None:
        tokens = tokenize("The intermediaries shall preserve the records")
        assert "the" not in tokens and "shall" not in tokens
        assert "intermediari" in tokens and "record" in tokens

    def test_token_index_stable_31_bit(self) -> None:
        assert token_index("kyc") == token_index("kyc")
        assert 0 <= token_index("kyc") < 2**31


class TestBM25Encoder:
    def test_requires_fit(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            BM25Encoder().encode_document("x")

    def test_term_frequency_saturates(self) -> None:
        enc = BM25Encoder(k1=1.2, b=0.0)
        enc.fit(["kyc"])
        one = enc.encode_document("kyc").values[0]
        many = enc.encode_document("kyc " * 50).values[0]
        assert one == pytest.approx(1.0)
        assert one < many < 1.2 + 1  # bounded by k1 + 1

    def test_longer_docs_penalised(self) -> None:
        enc = BM25Encoder(k1=1.2, b=0.75)
        enc.fit(["kyc records", "kyc " + "filler " * 30])
        short = enc.encode_document("kyc records")
        long = enc.encode_document("kyc " + "filler " * 30)
        idx = token_index("kyc")
        weight = lambda v: v.values[v.indices.index(idx)]
        assert weight(short) > weight(long)

    def test_query_vector_unique_terms_weight_one(self) -> None:
        q = BM25Encoder().encode_query("kyc kyc records")
        assert q.values == [1.0, 1.0] and q.indices == sorted(q.indices)


def _chunk(cid: str, text: str, doc: str = "d1") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc,
        doc_title="T",
        text=text,
        strategy="s",
        ordinal=0,
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )


CHUNKS = [
    _chunk("a" * 32, "KYC records must be preserved for eight years"),
    _chunk("b" * 32, "Settlement moves to T+1 under circular SEBI/HO/MRD/2023/12"),
    _chunk("c" * 32, "Mutual fund expense ratio limits by slab"),
]


@pytest.fixture
def store(qdrant: QdrantClient) -> QdrantStore:
    emb, enc = HashEmbedder(64), BM25Encoder()
    enc.fit(c.text for c in CHUNKS)
    s = QdrantStore(qdrant, "t")
    s.ensure_collection(64)
    s.upsert(
        CHUNKS,
        emb.embed_documents([c.text for c in CHUNKS]),
        [enc.encode_document(c.text) for c in CHUNKS],
    )
    return s


class TestQdrantStore:
    def test_count_and_metadata(self, store: QdrantStore) -> None:
        assert store.count() == 3
        store.set_metadata({"index_version": "v1"})
        assert store.metadata()["index_version"] == "v1"

    def test_sparse_search_exact_identifier(self, store: QdrantStore) -> None:
        hits = store.search_sparse(BM25Encoder().encode_query("SEBI/HO/MRD/2023/12"), 3)
        assert hits[0].chunk.chunk_id == "b" * 32 and hits[0].stage == "sparse"

    def test_dense_search_roundtrips_payload(self, store: QdrantStore) -> None:
        hits = store.search_dense(HashEmbedder(64).embed_query("expense ratio limits"), 1)
        assert hits[0].chunk == CHUNKS[2]

    def test_empty_sparse_query_returns_nothing(self, store: QdrantStore) -> None:
        assert store.search_sparse(BM25Encoder().encode_query("the of and"), 3) == []

    def test_upsert_is_idempotent(self, store: QdrantStore) -> None:
        enc = BM25Encoder()
        enc.fit(c.text for c in CHUNKS)
        store.upsert(
            CHUNKS,
            HashEmbedder(64).embed_documents([c.text for c in CHUNKS]),
            [enc.encode_document(c.text) for c in CHUNKS],
        )
        assert store.count() == 3

    def test_length_mismatch_rejected(self, store: QdrantStore) -> None:
        with pytest.raises(ValueError, match="same length"):
            store.upsert(CHUNKS, [], [])


def test_alias_swap_is_atomic_and_drops_old(qdrant: QdrantClient) -> None:
    for version in ("v1", "v2"):
        QdrantStore(qdrant, physical_name("idx", version)).ensure_collection(8)
    swap_alias(qdrant, "idx", "idx__v1")
    assert alias_target(qdrant, "idx") == "idx__v1"
    dropped = swap_alias(qdrant, "idx", "idx__v2")
    assert alias_target(qdrant, "idx") == "idx__v2"
    assert dropped == ["idx__v1"]
    assert QdrantStore(qdrant, "idx").exists()


def test_point_id_is_uuid() -> None:
    assert point_id("0" * 32) == "00000000-0000-0000-0000-000000000000"


def test_make_client_modes(tmp_path) -> None:
    assert make_client(":memory:") is not None
    local = make_client(str(tmp_path / "q"))
    local.close()
