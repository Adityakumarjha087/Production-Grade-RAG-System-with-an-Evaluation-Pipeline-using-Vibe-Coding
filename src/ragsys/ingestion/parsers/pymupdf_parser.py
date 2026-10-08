"""Fast, dependency-light parser built on PyMuPDF.

Headings are inferred from font size relative to the body text; tables come from
PyMuPDF's ``find_tables`` and are rendered as markdown. Text blocks that fall inside
a table's bounding box are skipped so table content is not duplicated.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from ragsys.ingestion.cleaning import is_page_number, repeating_lines
from ragsys.ingestion.parsers.base import RawElement, build_document
from ragsys.models import Document, ElementType

HEADING_SIZE_RATIO = 1.15
MAX_HEADING_CHARS = 150
MAX_HEADING_LINES = 3


@dataclass(frozen=True)
class _Block:
    page: int
    text: str
    size: float
    bold: bool
    bbox: tuple[float, float, float, float]


def _block_from_dict(page_no: int, block: dict[str, Any]) -> _Block | None:
    spans = [s for line in block.get("lines", []) for s in line.get("spans", [])]
    lines = ["".join(s["text"] for s in line.get("spans", [])) for line in block.get("lines", [])]
    text = "\n".join(lines).strip()
    if not spans or not text:
        return None
    size = max(s["size"] for s in spans)
    bold = all(s["flags"] & 16 for s in spans if s["text"].strip())
    return _Block(page=page_no, text=text, size=size, bold=bold, bbox=tuple(block["bbox"]))


def _inside(bbox: tuple[float, ...], area: tuple[float, ...], tol: float = 2.0) -> bool:
    x0, y0, x1, y1 = bbox
    return (
        x0 >= area[0] - tol and y0 >= area[1] - tol and x1 <= area[2] + tol and y1 <= area[3] + tol
    )


def _heading_levels(blocks: list[_Block], body_size: float) -> dict[float, int]:
    """Map heading font sizes to levels (largest -> 1)."""
    sizes = sorted({round(b.size, 1) for b in blocks if b.size >= body_size * HEADING_SIZE_RATIO})
    return {size: level for level, size in enumerate(reversed(sizes), start=1)}


def _cell(value: object) -> str:
    return " ".join(str(value).split()).replace("|", "/") if value is not None else ""


def rows_to_markdown(rows: list[list[object]]) -> str:
    """Render extracted table rows as a markdown table (first row = header).

    We build the markdown ourselves rather than use ``Table.to_markdown`` because
    the latter guesses "external" headers from nearby text and HTML-escapes cells.
    """
    cleaned = [[_cell(c) for c in row] for row in rows if any(_cell(c) for c in row)]
    if len(cleaned) < 2:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    lines = ["| " + " | ".join(cleaned[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in cleaned[1:]]
    return "\n".join(lines)


def merge_wrapped_headings(elements: list[RawElement]) -> list[RawElement]:
    """Join consecutive same-level headings (a long title wrapped onto two lines)."""
    merged: list[RawElement] = []
    for el in elements:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and el.type == prev.type == ElementType.HEADING
            and el.level == prev.level
            and el.page == prev.page
        ):
            merged[-1] = RawElement(
                ElementType.HEADING, f"{prev.text} {el.text}", el.page, el.level
            )
        else:
            merged.append(el)
    return merged


def _classify(block: _Block, body_size: float, levels: dict[float, int]) -> RawElement:
    short = (
        len(block.text) <= MAX_HEADING_CHARS and block.text.strip().count("\n") < MAX_HEADING_LINES
    )
    level = levels.get(round(block.size, 1))
    if level is not None and short:
        return RawElement(ElementType.HEADING, block.text, block.page, level)
    if block.bold and short and block.size >= body_size and not block.text.endswith("."):
        return RawElement(ElementType.HEADING, block.text, block.page, len(levels) + 1)
    stripped = block.text.lstrip()
    if stripped[:1] in {"•", "-", "–", "▪"}:
        return RawElement(ElementType.LIST_ITEM, stripped.lstrip("•-–▪ "), block.page)
    return RawElement(ElementType.PARAGRAPH, block.text, block.page)


class PyMuPDFParser:
    """See module docstring."""

    name = "pymupdf"

    def parse(self, path: Path) -> Document:
        """Parse ``path`` into a Document."""
        with pymupdf.open(path) as pdf:
            per_page = [self._page_items(page, i + 1) for i, page in enumerate(pdf)]
            num_pages = pdf.page_count
        blocks = [b for page_blocks, _ in per_page for b in page_blocks]
        noise = repeating_lines([[b.text for b in page_blocks] for page_blocks, _ in per_page])
        body_size = statistics.median([b.size for b in blocks]) if blocks else 10.0
        levels = _heading_levels(blocks, body_size)
        raw: list[RawElement] = []
        for page_blocks, tables in per_page:
            raw.extend(self._merge_page(page_blocks, tables, noise, body_size, levels))
        return build_document(path, raw, self.name, num_pages)

    def _page_items(
        self, page: pymupdf.Page, page_no: int
    ) -> tuple[list[_Block], list[tuple[tuple[float, ...], RawElement]]]:
        tables = []
        for table in page.find_tables().tables:
            markdown = rows_to_markdown(table.extract())
            if markdown:
                tables.append((tuple(table.bbox), RawElement(ElementType.TABLE, markdown, page_no)))
        blocks = []
        for raw_block in page.get_text("dict")["blocks"]:
            if raw_block.get("type") != 0:
                continue
            block = _block_from_dict(page_no, raw_block)
            if block and not any(_inside(block.bbox, bbox) for bbox, _ in tables):
                blocks.append(block)
        return blocks, tables

    def _merge_page(
        self,
        blocks: list[_Block],
        tables: list[tuple[tuple[float, ...], RawElement]],
        noise: set[str],
        body_size: float,
        levels: dict[float, int],
    ) -> list[RawElement]:
        """Interleave text blocks and tables in reading (top-to-bottom) order."""
        items: list[tuple[float, RawElement]] = [(bbox[1], el) for bbox, el in tables]
        for block in blocks:
            if block.text in noise or is_page_number(block.text):
                continue
            items.append((block.bbox[1], _classify(block, body_size, levels)))
        items.sort(key=lambda pair: pair[0])
        return merge_wrapped_headings([el for _, el in items])
