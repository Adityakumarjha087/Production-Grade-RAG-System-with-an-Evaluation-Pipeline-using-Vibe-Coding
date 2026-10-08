"""Layout-aware parser built on IBM Docling.

Docling runs a layout model and a table-structure model, so it recovers heading
hierarchy and real table cells (including merged headers) far better than text
extraction. It is slow on CPU and pulls in torch, which is why it is an optional
extra and its output is cached by :mod:`ragsys.ingestion.pipeline`.

OCR is ~6x slower (measured 1.74 vs 0.28 s/page on SEBI circulars) and useless for
born-digital PDFs, so it is used only for PDFs without a text layer (scans).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pymupdf

from ragsys.ingestion.parsers.base import RawElement, build_document
from ragsys.models import Document, ElementType

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

_SKIP_LABELS = {"page_header", "page_footer", "picture", "caption", "footnote"}
MIN_CHARS_PER_PAGE = 100


def has_text_layer(path: Path, min_chars_per_page: int = MIN_CHARS_PER_PAGE) -> bool:
    """True if the PDF has extractable text (born-digital) rather than page images."""
    with pymupdf.open(path) as pdf:
        pages: int = max(int(pdf.page_count), 1)
        chars: int = sum(len(page.get_text("text").strip()) for page in pdf)
    return chars / pages >= min_chars_per_page


def _page_of(item: Any) -> int:
    prov = getattr(item, "prov", None) or []
    return int(prov[0].page_no) if prov else 1


def _to_raw(item: Any, doc: Any) -> RawElement | None:
    """Convert one Docling item to a RawElement (None = skip)."""
    from docling_core.types.doc.items.table.table import TableItem
    from docling_core.types.doc.items.text import ListItem, SectionHeaderItem, TextItem, TitleItem

    page = _page_of(item)
    if isinstance(item, TableItem):
        return RawElement(ElementType.TABLE, item.export_to_markdown(doc=doc), page)
    if isinstance(item, TitleItem):
        return RawElement(ElementType.HEADING, item.text, page, level=1)
    if isinstance(item, SectionHeaderItem):
        return RawElement(ElementType.HEADING, item.text, page, level=int(item.level) + 1)
    if isinstance(item, ListItem):
        marker = str(getattr(item, "marker", "") or "").strip()
        text = f"{marker} {item.text}" if marker and not item.text.startswith(marker) else item.text
        return RawElement(ElementType.LIST_ITEM, text, page)
    if isinstance(item, TextItem) and str(item.label) not in _SKIP_LABELS:
        return RawElement(ElementType.PARAGRAPH, item.text, page)
    return None


class DoclingParser:
    """See module docstring. Converters (and their models) are loaded lazily once."""

    name = "docling"

    def __init__(self) -> None:
        self._converters: dict[bool, DocumentConverter] = {}
        self.ocr_used: list[str] = []

    def _get_converter(self, ocr: bool) -> DocumentConverter:
        if ocr not in self._converters:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions(do_ocr=ocr)
            self._converters[ocr] = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._converters[ocr]

    def parse(self, path: Path) -> Document:
        """Parse ``path`` into a Document (OCR only when there is no text layer)."""
        ocr = not has_text_layer(path)
        if ocr:
            self.ocr_used.append(str(path))
        doc = self._get_converter(ocr).convert(path).document
        raw = [el for item, _ in doc.iterate_items() if (el := _to_raw(item, doc)) is not None]
        return build_document(path, raw, self.name, num_pages=len(doc.pages))
