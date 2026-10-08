"""SEBI downloader (offline, mocked HTTP) and real-corpus metadata handling."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from ragsys.devtools.sebi_fetch import (
    LISTING_URL,
    PAGE_URL,
    SebiFetcher,
    load_manifest,
    parse_listing,
    parse_pdf_url,
    total_pages,
)
from ragsys.ingestion.parsers import RawElement, build_document
from ragsys.ingestion.pipeline import enrich, load_sidecars, parse_corpus
from ragsys.models import ElementType
from tests.conftest import make_doc


def row(date: str, sid: str, title: str) -> str:
    url = f"https://www.sebi.gov.in/legal/circulars/sep-2026/c-{sid}_{sid}.html"
    return (
        f"<tr role='row' class='odd'><td>{date}</td><td><a href='{url}'  target=\"_blank\" "
        f"title=\"{title}\" class='points'>{title}</a></td></tr>"
    )


PAGE0 = f"<table>{row('Sep 07, 2026', '101', 'First &amp; one')}{row('Sep 06, 2026', '102', 'HTML only')}</table>"
PAGE1 = f"<input type='hidden' name='totalpage' value=2 />{row('Aug 01, 2026', '103', 'Third')}"
VIEWER = "<iframe src='../../../web/?file=https://www.sebi.gov.in/sebi_data/attachdocs/sep-2026/{n}.PDF' width='100%'>"


def test_parse_listing_and_pdf_url() -> None:
    refs = parse_listing(PAGE0)
    assert [(r.date, r.sebi_id, r.title) for r in refs] == [
        ("2026-09-07", "101", "First & one"),
        ("2026-09-06", "102", "HTML only"),
    ]
    assert (
        parse_pdf_url(VIEWER.format(n=7))
        == "https://www.sebi.gov.in/sebi_data/attachdocs/sep-2026/7.PDF"
    )
    assert parse_pdf_url("<p>no viewer</p>") is None
    assert total_pages(PAGE1) == 2 and total_pages(PAGE0) is None


class FakeSebi:
    """Mock transport for the listing, circular pages and PDFs."""

    def __init__(self, fail_pdf: str | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.fail_pdf = fail_pdf

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url == LISTING_URL:
            return httpx.Response(200, text=PAGE0, headers={"set-cookie": "JSESSIONID=abc; Path=/"})
        if url == PAGE_URL:
            assert request.headers["x-requested-with"] == "XMLHttpRequest"
            assert "JSESSIONID=abc" in request.headers.get("cookie", "")
            return httpx.Response(200, text=PAGE1)
        if url.endswith(".html"):
            sid = url.rsplit("_", 1)[1].removesuffix(".html")
            return httpx.Response(
                200, text="<p>no pdf</p>" if sid == "102" else VIEWER.format(n=sid)
            )
        if url.endswith(".PDF"):
            if self.fail_pdf and self.fail_pdf in url:
                return httpx.Response(503)
            return httpx.Response(200, content=b"%PDF-1.7 fake")
        return httpx.Response(404)


def fetcher(tmp_path: Path, transport: FakeSebi) -> SebiFetcher:
    client = httpx.Client(transport=httpx.MockTransport(transport))
    return SebiFetcher(tmp_path, delay=0, max_retries=1, client=client, sleep=lambda _: None)


def test_download_paginates_skips_html_only_and_writes_manifest(tmp_path: Path) -> None:
    counts = fetcher(tmp_path, FakeSebi()).run(limit=2)
    assert counts == {"downloaded": 2, "no_pdf": 1, "failed": 0, "skipped": 0}
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == [
        "2026-08-01_103.pdf",
        "2026-09-07_101.pdf",
    ]
    manifest = load_manifest(tmp_path / "manifest.jsonl")
    assert manifest["102"].status == "no_pdf"
    assert manifest["101"].title == "First & one" and manifest["101"].file == "2026-09-07_101.pdf"


def test_resume_skips_done_and_retries_failed(tmp_path: Path) -> None:
    first = fetcher(tmp_path, FakeSebi(fail_pdf="/103.PDF")).run(limit=2)
    assert first["failed"] == 1 and first["downloaded"] == 1
    transport = FakeSebi()
    second = fetcher(tmp_path, transport).run(limit=2)
    assert second == {"downloaded": 1, "no_pdf": 0, "failed": 0, "skipped": 2}
    assert not any("101" in str(r.url) for r in transport.requests)  # never re-downloaded


def test_non_pdf_payload_rejected(tmp_path: Path) -> None:
    class HtmlInsteadOfPdf(FakeSebi):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith(".PDF"):
                return httpx.Response(200, text="<html>blocked</html>")
            return super().__call__(request)

    counts = fetcher(tmp_path, HtmlInsteadOfPdf()).run(limit=1)
    assert counts["failed"] >= 1 and not list(tmp_path.glob("*.pdf"))


class TestTitles:
    def _doc(self, *texts: tuple[ElementType, str]):
        raw = [RawElement(t, x, 1, 1 if t == ElementType.HEADING else None) for t, x in texts]
        return build_document(Path("x.pdf"), raw, "test", 1, doc_id="d")

    def test_subject_line_beats_generic_heading(self) -> None:
        doc = self._doc(
            (ElementType.HEADING, "CIRCULAR"),
            (ElementType.PARAGRAPH, "Sub: Review of position limits"),
        )
        assert doc.title == "Review of position limits"

    def test_skips_generic_heading(self) -> None:
        doc = self._doc(
            (ElementType.HEADING, "MASTER CIRCULAR"), (ElementType.HEADING, "Real Title")
        )
        assert doc.title == "Real Title"

    def test_falls_back_to_filename(self) -> None:
        assert self._doc((ElementType.HEADING, "Circular")).title == "x"


def test_sidecar_metadata_flows_into_chunks_and_citations(
    tmp_path: Path, sample_pdfs: list[Path]
) -> None:
    pdf = tmp_path / "2026-09-07_101.pdf"
    pdf.write_bytes(sample_pdfs[0].read_bytes())
    entry = {
        "sebi_id": "101",
        "date": "2026-09-07",
        "title": "Official Title",
        "url": "https://x/101.html",
        "status": "downloaded",
        "file": pdf.name,
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(entry) + "\n")
    assert load_sidecars([pdf])[pdf]["title"] == "Official Title"

    from ragsys.ingestion.parsers import PyMuPDFParser

    first = parse_corpus([pdf], tmp_path / "parsed", PyMuPDFParser()).documents[0]
    cached = parse_corpus([pdf], tmp_path / "parsed", PyMuPDFParser()).documents[0]
    for doc in (first, cached):  # enrichment applies to cached parses too
        assert doc.title == "Official Title" and doc.metadata["url"] == "https://x/101.html"

    from ragsys.generation.citations import to_citation
    from ragsys.generation.prompts import passage_header
    from ragsys.ingestion.chunkers import StructuralChunker
    from ragsys.models import ScoredChunk

    chunk = StructuralChunker(600, 50, 100).chunk(first)[0]
    assert chunk.doc_date == "2026-09-07" and chunk.source_url == "https://x/101.html"
    hit = ScoredChunk(chunk=chunk, score=1.0, stage="x")
    assert to_citation(1, hit).source_url == "https://x/101.html"
    assert passage_header(1, hit).startswith("[1] Official Title (2026-09-07) — p. 1")


def test_enrich_without_sidecar_is_identity() -> None:
    doc = make_doc([("h1", "T"), ("p", "body")])
    assert enrich(doc, None) is doc
