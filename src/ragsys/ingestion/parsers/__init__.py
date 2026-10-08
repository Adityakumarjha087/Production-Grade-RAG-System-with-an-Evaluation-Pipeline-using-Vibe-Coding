"""PDF parsers behind a common :class:`Parser` protocol."""

from __future__ import annotations

from ragsys.ingestion.parsers.base import Parser, RawElement, build_document
from ragsys.ingestion.parsers.pymupdf_parser import PyMuPDFParser


def get_parser(name: str) -> Parser:
    """Instantiate a parser by name (``docling`` imports lazily)."""
    if name == "pymupdf":
        return PyMuPDFParser()
    if name == "docling":
        from ragsys.ingestion.parsers.docling_parser import DoclingParser

        return DoclingParser()
    raise ValueError(f"Unknown parser: {name}")


__all__ = ["Parser", "PyMuPDFParser", "RawElement", "build_document", "get_parser"]
