from __future__ import annotations

from pathlib import Path

import pytest

from ragsys.ingestion.cleaning import is_page_number, normalize_text, repeating_lines
from ragsys.ingestion.parsers import RawElement, build_document, get_parser
from ragsys.ingestion.parsers.base import file_doc_id
from ragsys.ingestion.parsers.pymupdf_parser import merge_wrapped_headings, rows_to_markdown
from ragsys.models import ElementType


class TestNormalizeText:
    def test_rejoins_hyphenated_line_breaks(self) -> None:
        assert normalize_text("settle-\nment cycle") == "settlement cycle"

    def test_soft_line_breaks_become_spaces_but_paragraphs_survive(self) -> None:
        text = "first line\nwrapped here\n\nsecond paragraph"
        assert normalize_text(text) == "first line wrapped here\n\nsecond paragraph"

    def test_line_break_after_sentence_end_is_kept(self) -> None:
        assert normalize_text("Ends here.\nNext") == "Ends here.\nNext"

    def test_collapses_whitespace_and_unicode(self) -> None:
        assert normalize_text("  a  b\t c  ") == "a b c"
        assert normalize_text("ﬁle") == "file"  # NFKC ligature

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("effect from 7 th September 2026", "effect from 7th September 2026"),
            ("w.e.f. 1 st April 2026", "w.e.f. 1st April 2026"),
            ("the 2 nd and 3 rd tranche", "the 2nd and 3rd tranche"),
            ("1\nst April", "1st April"),
        ],
    )
    def test_reattaches_superscript_ordinal_suffixes(self, raw: str, expected: str) -> None:
        assert normalize_text(raw) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Regulation 23 the provisions apply",
            "Chapter 5 Standard operating procedure",
            "Rs. 10 thousand crore",
            "para 4 nder review",
        ],
    )
    def test_does_not_glue_words_that_merely_start_like_an_ordinal(self, text: str) -> None:
        assert normalize_text(text) == text


@pytest.mark.parametrize("line", ["3", "Page 3", "page 3 of 10", " 4/12 "])
def test_page_numbers_detected(line: str) -> None:
    assert is_page_number(line)


def test_page_number_rejects_content() -> None:
    assert not is_page_number("3. Applicability")


class TestRepeatingLines:
    def test_detects_running_header(self) -> None:
        pages = [["HEADER", "body a"], ["HEADER", "body b"], ["HEADER", "body c"]]
        assert repeating_lines(pages) == {"HEADER"}

    def test_single_page_never_strips(self) -> None:
        assert repeating_lines([["HEADER", "HEADER"]]) == set()

    def test_long_lines_are_never_treated_as_headers(self) -> None:
        long = "x" * 200
        assert repeating_lines([[long], [long]]) == set()


class TestBuildDocument:
    def _raw(self) -> list[RawElement]:
        return [
            RawElement(ElementType.HEADING, "Title", 1, level=1),
            RawElement(ElementType.HEADING, "1. Scope", 1, level=2),
            RawElement(ElementType.PARAGRAPH, "Applies to all.", 1),
            RawElement(ElementType.PARAGRAPH, "   ", 1),  # dropped
            RawElement(ElementType.HEADING, "2. Rules", 2, level=2),
            RawElement(ElementType.TABLE, "| a | b |\n|---|---|\n| 1 | 2 |", 2),
        ]

    def test_offsets_index_canonical_text(self) -> None:
        doc = build_document(Path("x.pdf"), self._raw(), "test", 2, doc_id="d")
        assert len(doc.elements) == 5
        for el in doc.elements:
            assert doc.text[el.char_start : el.char_end] == el.text

    def test_section_paths_follow_heading_levels(self) -> None:
        doc = build_document(Path("x.pdf"), self._raw(), "test", 2, doc_id="d")
        paths = [e.section_path for e in doc.elements]
        assert paths[2] == ["Title", "1. Scope"]
        assert paths[4] == ["Title", "2. Rules"]  # sibling replaced "1. Scope"

    def test_title_is_first_heading_and_table_newlines_preserved(self) -> None:
        doc = build_document(Path("x.pdf"), self._raw(), "test", 2, doc_id="d")
        assert doc.title == "Title"
        assert "\n|---|" in doc.elements[-1].text

    def test_doc_id_is_content_addressed(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.pdf", tmp_path / "renamed.pdf"
        a.write_bytes(b"same bytes")
        b.write_bytes(b"same bytes")
        assert file_doc_id(a) == file_doc_id(b)


def test_rows_to_markdown_cleans_cells() -> None:
    md = rows_to_markdown([["A", "B|C"], [None, "x\ny"], ["", ""]])
    assert md == "| A | B/C |\n|---|---|\n|  | x y |"


def test_rows_to_markdown_needs_header_and_body() -> None:
    assert rows_to_markdown([["only header"]]) == ""


def test_merge_wrapped_headings() -> None:
    els = [
        RawElement(ElementType.HEADING, "Long title", 1, 1),
        RawElement(ElementType.HEADING, "continued", 1, 1),
        RawElement(ElementType.HEADING, "Sub", 1, 2),
    ]
    merged = merge_wrapped_headings(els)
    assert [e.text for e in merged] == ["Long title continued", "Sub"]


def test_unknown_parser() -> None:
    with pytest.raises(ValueError, match="Unknown parser"):
        get_parser("nope")


class TestPyMuPDFOnSampleCorpus:
    def test_headers_footers_and_page_numbers_removed(self, sample_docs: list) -> None:
        for doc in sample_docs:
            assert "SYNTHETIC SAMPLE DOCUMENT" not in doc.text
            assert not any(e.text.startswith("Page ") for e in doc.elements)

    def test_tables_extracted_as_markdown(self, sample_docs: list) -> None:
        tables = [e for d in sample_docs for e in d.elements if e.type == ElementType.TABLE]
        assert len(tables) == 2
        assert tables[0].text.startswith("| Activity | Day | Deadline |")
        assert "| Pay-out of funds and securities | T+1 | 1:30 PM |" in tables[0].text

    def test_multi_page_and_headings(self, sample_docs: list) -> None:
        doc = next(d for d in sample_docs if "KYC" in d.title)
        assert doc.num_pages == 2
        sub = next(e for e in doc.elements if e.text.startswith("3.2.1"))
        assert sub.section_path[-1] == "3.2 Periodic updation"
        assert sub.page == 2


def test_text_layer_detection(sample_pdfs: list, tmp_path: Path) -> None:
    import pymupdf

    from ragsys.ingestion.parsers.docling_parser import has_text_layer

    scanned = tmp_path / "scan.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0, 0, 0))  # image-like, no text
        pdf.save(scanned)
    assert has_text_layer(sample_pdfs[0])
    assert not has_text_layer(scanned)
