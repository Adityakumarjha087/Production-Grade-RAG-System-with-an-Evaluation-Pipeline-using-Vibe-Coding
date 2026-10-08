"""Parser interface and the shared Document builder.

Parsers only need to emit a flat list of :class:`RawElement`; the builder assigns
character offsets, heading breadcrumbs and the canonical document text. Keeping
that logic in one place guarantees every parser produces documents with the same
invariants (``doc.text[e.char_start:e.char_end] == e.text``).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ragsys.ingestion.cleaning import normalize_text
from ragsys.models import Document, Element, ElementType

ELEMENT_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class RawElement:
    """Parser output before offsets/sections are assigned."""

    type: ElementType
    text: str
    page: int
    level: int | None = None


class Parser(Protocol):
    """Turns a PDF into a :class:`Document`."""

    name: str

    def parse(self, path: Path) -> Document:
        """Parse one file. Raises on unrecoverable errors."""
        ...


def file_doc_id(path: Path) -> str:
    """Content-addressed id: renaming a file does not change its id."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:16]


def _update_section_stack(stack: list[tuple[int, str]], level: int, title: str) -> None:
    """Pop headings at the same or deeper level, then push the new heading."""
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


GENERIC_TITLES = frozenset({"circular", "master circular", "notification", "press release"})
_SUBJECT = re.compile(r"^\s*(sub(ject)?\s*[:.\-])\s*", re.IGNORECASE)


def pick_title(elements: list[Element], fallback: str) -> str:
    """Document title: the ``Sub:`` line, else the first non-generic heading.

    Regulatory circulars all start with the heading "CIRCULAR"; using it as the title
    would make every citation read "CIRCULAR". The subject line is the real title.
    """
    for el in elements[:40]:
        if _SUBJECT.match(el.text):
            return _SUBJECT.sub("", el.text).strip()[:200]
    for el in elements:
        if el.type == ElementType.HEADING and el.text.strip().lower() not in GENERIC_TITLES:
            return el.text[:200]
    return fallback


def build_document(
    path: Path, raw: list[RawElement], parser: str, num_pages: int, doc_id: str | None = None
) -> Document:
    """Assemble a :class:`Document` from raw elements.

    Empty elements are dropped; text is normalised (tables keep their newlines).
    ``doc_id`` defaults to the content hash of ``path``.
    """
    elements: list[Element] = []
    parts: list[str] = []
    cursor = 0
    stack: list[tuple[int, str]] = []
    for item in raw:
        text = item.text.strip() if item.type == ElementType.TABLE else normalize_text(item.text)
        if not text:
            continue
        if item.type == ElementType.HEADING:
            _update_section_stack(stack, item.level or 1, text)
        if parts:
            cursor += len(ELEMENT_SEPARATOR)
        elements.append(
            Element(
                type=item.type,
                text=text,
                page=item.page,
                level=item.level,
                section_path=[title for _, title in stack],
                char_start=cursor,
                char_end=cursor + len(text),
            )
        )
        parts.append(text)
        cursor += len(text)
    title = pick_title(elements, path.stem)
    return Document(
        doc_id=doc_id or file_doc_id(path),
        source_path=str(path),
        title=title,
        text=ELEMENT_SEPARATOR.join(parts),
        elements=elements,
        num_pages=num_pages,
        parser=parser,
        metadata={"filename": path.name},
    )
