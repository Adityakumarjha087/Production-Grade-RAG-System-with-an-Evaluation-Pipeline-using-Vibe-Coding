"""Parser and chunker lookups accept the configured names and reject anything else by name."""

from __future__ import annotations

import pytest

from ragsys.config import ChunkingConfig
from ragsys.ingestion.chunkers import get_chunker
from ragsys.ingestion.parsers import PyMuPDFParser, get_parser


def test_pymupdf_parser_is_built_without_optional_dependencies() -> None:
    assert isinstance(get_parser("pymupdf"), PyMuPDFParser)


def test_unknown_parser_is_named_in_the_error() -> None:
    with pytest.raises(ValueError, match="tesseract"):
        get_parser("tesseract")


def test_unknown_chunking_strategy_is_named_in_the_error() -> None:
    with pytest.raises(ValueError, match="semantic"):
        get_chunker(ChunkingConfig.model_construct(strategy="semantic"))
