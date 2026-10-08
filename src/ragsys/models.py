"""Domain types shared by every stage of the pipeline.

This module is deliberately dependency-free (only pydantic) so that every other
module can import it without creating cycles.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ElementType(StrEnum):
    """Kind of a parsed document element."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"


class Element(BaseModel):
    """A contiguous, typed piece of a parsed document.

    ``char_start``/``char_end`` locate the element inside ``Document.text`` so that
    any chunk built from elements can be mapped back to an exact source span.
    """

    type: ElementType
    text: str
    page: int
    section_path: list[str] = Field(default_factory=list)
    level: int | None = None
    char_start: int = 0
    char_end: int = 0


class Document(BaseModel):
    """A parsed PDF: canonical text plus the elements it is made of."""

    doc_id: str
    source_path: str
    title: str
    text: str
    elements: list[Element]
    num_pages: int
    parser: str
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit of text.

    ``text`` is what gets embedded and shown to the LLM (it may carry a section
    breadcrumb); ``char_start``/``char_end`` is the source span in the document,
    which is what evaluation uses to judge relevance independent of chunking.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    strategy: str
    ordinal: int
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    section_path: list[str] = Field(default_factory=list)
    source_path: str = ""
    source_url: str = ""
    doc_date: str = ""


class ScoredChunk(BaseModel):
    """A chunk with the score assigned by the stage that produced it."""

    chunk: Chunk
    score: float
    stage: str


class Citation(BaseModel):
    """A validated reference from the answer to one context chunk."""

    marker: int
    chunk_id: str
    doc_id: str
    doc_title: str
    page_start: int
    page_end: int
    section: str
    source_path: str
    snippet: str
    source_url: str = ""
    doc_date: str = ""


class Usage(BaseModel):
    """LLM token usage for one request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class Answer(BaseModel):
    """Final result of a RAG request."""

    query: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[Citation] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: str | None = None
    confidence: float | None = None
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: dict[str, float] = Field(default_factory=dict)
    cached: bool = False
    request_id: str = ""
    trace_id: str | None = None
