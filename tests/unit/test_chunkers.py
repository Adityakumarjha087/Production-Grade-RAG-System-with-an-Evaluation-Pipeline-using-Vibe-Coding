from __future__ import annotations

import itertools

import pytest

from ragsys.config import ChunkingConfig
from ragsys.ingestion.chunkers import (
    FixedSizeChunker,
    RecursiveChunker,
    StructuralChunker,
    get_chunker,
)
from ragsys.ingestion.chunkers.base import overlapping_elements
from ragsys.ingestion.chunkers.fixed import window_spans
from ragsys.ingestion.chunkers.recursive import merge_pieces, split_recursive
from ragsys.models import Document
from tests.conftest import make_doc

SENTENCE = "The intermediary shall preserve records for eight years after closure. "


def long_doc() -> Document:
    table = "| Activity | Day |\n|---|---|\n" + "\n".join(
        f"| Row {i} item | T+{i} |" for i in range(40)
    )
    return make_doc(
        [
            ("h1", "Circular on Records"),
            ("h2", "1. Scope"),
            ("p", SENTENCE * 3),
            ("p", "Short paragraph."),
            ("h2", "2. Retention"),
            ("p", SENTENCE * 20),  # oversized paragraph
            ("h2", "3. Timelines"),
            ("table", table),  # oversized table
            ("h2", "4. Small"),
            ("p", "Tiny."),
        ],
        name="long",
    )


ALL = [
    FixedSizeChunker(300, 50),
    RecursiveChunker(300, 50),
    StructuralChunker(300, 50, 80),
]


@pytest.mark.parametrize("chunker", ALL, ids=lambda c: c.name)
class TestChunkerContract:
    def test_spans_are_valid_and_ordered(self, chunker) -> None:
        doc = long_doc()
        chunks = chunker.chunk(doc)
        assert chunks
        for c in chunks:
            assert 0 <= c.char_start < c.char_end <= len(doc.text)
            assert not doc.text[c.char_start].isspace()
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_every_word_is_covered(self, chunker) -> None:
        doc = long_doc()
        covered = [False] * len(doc.text)
        for c in chunker.chunk(doc):
            for i in range(c.char_start, c.char_end):
                covered[i] = True
        missing = [i for i, ch in enumerate(doc.text) if not ch.isspace() and not covered[i]]
        # structural drops nothing either; table row groups exclude only the repeated header
        allowed = chunker.name == "structural"
        assert not missing or (allowed and all(doc.text[i] in "|-:# ActivityDay" for i in missing))

    def test_deterministic_ids(self, chunker) -> None:
        a = [c.chunk_id for c in chunker.chunk(long_doc())]
        b = [c.chunk_id for c in chunker.chunk(long_doc())]
        assert a == b and len(set(a)) == len(a)

    def test_metadata_populated(self, chunker) -> None:
        for c in chunker.chunk(long_doc()):
            assert c.doc_id == "doc-long" and c.strategy == chunker.name
            assert 1 <= c.page_start <= c.page_end


@pytest.mark.parametrize("chunker", ALL[:2], ids=lambda c: c.name)
def test_size_bound_and_text_equals_span(chunker) -> None:
    doc = long_doc()
    for c in chunker.chunk(doc):
        assert len(c.text) <= 300
        assert c.text == doc.text[c.char_start : c.char_end]


def test_fixed_windows_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(200))
    spans = window_spans(text, 100, 30)
    for (s1, e1), (s2, _) in itertools.pairwise(spans):
        assert s2 < e1  # overlap
        assert s2 > s1  # progress


def test_fixed_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        window_spans("abc", 10, 10)


def test_recursive_prefers_paragraph_boundaries() -> None:
    text = ("a" * 80 + "\n\n") * 3
    pieces = split_recursive(text, 0, len(text), 100, ["\n\n", " "])
    assert all(text[s:e].strip() == "a" * 80 for s, e in pieces)


def test_merge_pieces_respects_size_and_overlap() -> None:
    pieces = [(i * 10, i * 10 + 10) for i in range(10)]  # contiguous 10-char pieces
    windows = merge_pieces(pieces, size=30, overlap=10)
    assert all(e - s <= 30 for s, e in windows)
    assert windows[0] == (0, 30) and windows[1][0] == 20  # 10 chars carried over


class TestStructural:
    def test_breadcrumb_prefix_and_section(self) -> None:
        chunks = StructuralChunker(300, 50, 80).chunk(long_doc())
        scope = chunks[0]
        assert scope.text.startswith("Circular on Records > 1. Scope\n\n")
        assert scope.section_path[-1] == "1. Scope"

    def test_does_not_mix_sections_once_min_size_reached(self) -> None:
        chunks = StructuralChunker(300, 50, 80).chunk(long_doc())
        for c in chunks:
            body = c.text.split("\n\n", 1)[1]
            assert not ("1. Scope" in body and "2. Retention" in body)

    def test_small_trailing_section_merged_not_dropped(self) -> None:
        chunks = StructuralChunker(300, 50, 80).chunk(long_doc())
        assert "Tiny." in chunks[-1].text

    def test_oversized_table_split_with_repeated_header(self) -> None:
        chunks = StructuralChunker(300, 50, 80).chunk(long_doc())
        table_chunks = [c for c in chunks if "| Row" in c.text]
        assert len(table_chunks) > 1
        for c in table_chunks:
            assert "| Activity | Day |\n|---|---|" in c.text
        rows = [
            line for c in table_chunks for line in c.text.splitlines() if line.startswith("| Row")
        ]
        assert len(rows) == 40  # every row exactly once

    def test_table_kept_whole_when_it_fits(self) -> None:
        doc = make_doc(
            [("h1", "T"), ("table", "| a | b |\n|---|---|\n| 1 | 2 |"), ("p", "after " * 5)]
        )
        chunks = StructuralChunker(1000, 50, 10).chunk(doc)
        assert any("| a | b |\n|---|---|\n| 1 | 2 |" in c.text for c in chunks)

    def test_oversized_paragraph_split_recursively(self) -> None:
        chunks = StructuralChunker(300, 50, 80).chunk(long_doc())
        retention = [c for c in chunks if c.section_path[-1:] == ["2. Retention"]]
        assert len(retention) > 2


def test_overlapping_elements() -> None:
    doc = long_doc()
    el = doc.elements[2]
    found = overlapping_elements(doc, el.char_start + 1, el.char_start + 5)
    assert found == [el]


@pytest.mark.parametrize("name", ["fixed", "recursive", "structural"])
def test_registry(name: str) -> None:
    assert get_chunker(ChunkingConfig(strategy=name)).name == name  # type: ignore[arg-type]


def test_recursive_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        RecursiveChunker(100, 100)
