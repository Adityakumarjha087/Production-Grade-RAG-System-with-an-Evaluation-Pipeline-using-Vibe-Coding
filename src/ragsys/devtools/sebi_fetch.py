"""Polite, resumable downloader for public SEBI circulars (sebi.gov.in → Legal → Circulars).

* One request at a time with a fixed delay (default 1.5 s) and exponential back-off
  on errors; robots.txt allows these paths.
* Resumable: ``manifest.jsonl`` records every circular processed (downloaded,
  no PDF, or failed); re-runs skip completed entries.
* Files are named ``<YYYY-MM-DD>_<sebi id>.pdf`` and verified to be PDFs.

The listing is an HTML table loaded page by page through the site's AJAX endpoint,
which requires the session cookie and referer of the listing page.
"""

from __future__ import annotations

import html
import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
import structlog

log = structlog.get_logger(__name__)

BASE = "https://www.sebi.gov.in"
LISTING_URL = f"{BASE}/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0"
PAGE_URL = f"{BASE}/sebiweb/ajax/home/getnewslistinfo.jsp"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140 Safari/537.36 ragsys-corpus-builder"
)

_ROW = re.compile(
    r"<td>\s*([A-Z][a-z]{2} \d{2}, \d{4})\s*</td>\s*<td>\s*<a href=['\"]"
    r"(https://www\.sebi\.gov\.in/legal/circulars/[^'\"]+?_(\d+)\.html)['\"][^>]*?"
    r"title=\"([^\"]*)\"",
    re.DOTALL,
)
_PDF = re.compile(r"file=(https?://[^'\"&\s]+?\.pdf)", re.IGNORECASE)
_TOTAL_PAGES = re.compile(r"name='totalpage' value=(\d+)")

Status = Literal["downloaded", "no_pdf", "failed"]


@dataclass(frozen=True)
class CircularRef:
    """One row of the circulars listing."""

    date: str  # ISO date
    title: str
    url: str
    sebi_id: str


@dataclass(frozen=True)
class ManifestEntry:
    """Outcome for one circular."""

    sebi_id: str
    date: str
    title: str
    url: str
    status: Status
    pdf_url: str = ""
    file: str = ""
    error: str = ""


def parse_listing(page_html: str) -> list[CircularRef]:
    """Circular rows from a listing page (static page or AJAX fragment)."""
    refs = []
    for date, url, sebi_id, title in _ROW.findall(page_html):
        iso = datetime.strptime(date, "%b %d, %Y").date().isoformat()
        clean_title = " ".join(html.unescape(title).split())
        refs.append(CircularRef(iso, clean_title, url, sebi_id))
    return refs


def parse_pdf_url(circular_html: str) -> str | None:
    """PDF URL embedded in a circular page's viewer iframe (None for HTML-only circulars)."""
    match = _PDF.search(circular_html)
    return match.group(1) if match else None


def total_pages(page_html: str) -> int | None:
    """Total listing pages, when the fragment reports it."""
    match = _TOTAL_PAGES.search(page_html)
    return int(match.group(1)) if match else None


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    """Entries keyed by SEBI id (latest wins)."""
    if not path.exists():
        return {}
    lines = path.read_text("utf-8").splitlines()
    entries = (ManifestEntry(**json.loads(line)) for line in lines if line.strip())
    return {e.sebi_id: e for e in entries}


class SebiFetcher:
    """See module docstring. ``sleep`` is injectable so tests don't wait."""

    def __init__(
        self,
        out_dir: Path,
        *,
        delay: float = 1.5,
        max_retries: int = 3,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.out_dir = out_dir
        self.delay = delay
        self.max_retries = max_retries
        self.client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True
        )
        self._sleep = sleep
        self.manifest_path = out_dir / "manifest.jsonl"

    def _request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        """HTTP with polite delay and exponential back-off."""
        for attempt in range(self.max_retries + 1):
            self._sleep(self.delay)
            try:
                response = self.client.request(method, url, **kwargs)  # type: ignore[arg-type]
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                if attempt == self.max_retries:
                    raise
                log.warning("sebi.retry", url=url, attempt=attempt + 1, error=str(exc))
                self._sleep(self.delay * 2 ** (attempt + 1))
        raise AssertionError("unreachable")

    def listing(self, page: int) -> str:
        """HTML of listing page ``page`` (0-based)."""
        if page == 0:
            return self._request("GET", LISTING_URL).text
        data = {
            "nextValue": str(page),
            "next": "n",
            "search": "",
            "fromDate": "",
            "toDate": "",
            "fromYear": "",
            "toYear": "",
            "deptId": "-1",
            "sid": "1",
            "ssid": "7",
            "smid": "0",
            "ssidhidden": "7",
            "intmid": "-1",
            "sText": "Legal",
            "ssText": "Circulars",
            "smText": "",
            "doDirect": str(page),
        }
        headers = {"Referer": LISTING_URL, "X-Requested-With": "XMLHttpRequest"}
        return self._request("POST", PAGE_URL, data=data, headers=headers).text

    def refs(self, limit: int) -> Iterator[CircularRef]:
        """Newest-first circular references, up to ``limit``.

        Page 0 is the static listing page, which also sets the session cookie the
        AJAX endpoint for later pages requires.
        """
        seen, page, pages = 0, 0, None
        fragment = self.listing(0)
        while True:
            pages = pages or total_pages(fragment)
            rows = parse_listing(fragment)
            for ref in rows[: limit - seen]:
                seen += 1
                yield ref
            page += 1
            if not rows or seen >= limit or (pages is not None and page >= pages):
                return
            fragment = self.listing(page)

    def _download(self, ref: CircularRef) -> ManifestEntry:
        pdf_url = parse_pdf_url(self._request("GET", ref.url).text)
        if pdf_url is None:
            return ManifestEntry(**asdict(ref), status="no_pdf")
        content = self._request("GET", pdf_url).content
        if not content.startswith(b"%PDF"):
            return ManifestEntry(**asdict(ref), status="failed", pdf_url=pdf_url, error="not a PDF")
        name = f"{ref.date}_{ref.sebi_id}.pdf"
        (self.out_dir / name).write_bytes(content)
        return ManifestEntry(**asdict(ref), status="downloaded", pdf_url=pdf_url, file=name)

    def run(self, limit: int) -> dict[str, int]:
        """Download up to ``limit`` circulars that have PDFs; returns counts by status."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        done = load_manifest(self.manifest_path)
        counts = {"downloaded": 0, "no_pdf": 0, "failed": 0, "skipped": 0}
        have = sum(1 for e in done.values() if e.status == "downloaded")
        with self.manifest_path.open("a", encoding="utf-8") as manifest:
            for ref in self.refs(limit=10 * limit):
                if have >= limit:
                    break
                previous = done.get(ref.sebi_id)
                if previous and previous.status != "failed":
                    counts["skipped"] += 1
                    continue
                try:
                    entry = self._download(ref)
                except httpx.HTTPError as exc:
                    entry = ManifestEntry(**asdict(ref), status="failed", error=str(exc))
                manifest.write(json.dumps(asdict(entry)) + "\n")
                manifest.flush()
                counts[entry.status] += 1
                have += entry.status == "downloaded"
                log.info("sebi.circular", id=ref.sebi_id, status=entry.status, total=have)
        return counts
