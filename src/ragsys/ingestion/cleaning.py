"""Text normalisation applied to every parser's output."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_BREAK = re.compile(r"(?<![\n.:;])\n(?!\n)")
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalise unicode, re-join hyphenated words and collapse whitespace.

    Single line breaks inside a paragraph (PDF line wrapping) become spaces;
    blank lines (paragraph breaks) are preserved.
    """
    text = unicodedata.normalize("NFKC", text).replace("\r", "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SOFT_BREAK.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def is_page_number(line: str) -> bool:
    """True for lines like ``3``, ``Page 3``, ``3 of 10``."""
    return bool(_PAGE_NUMBER.match(line))


def repeating_lines(
    pages: Sequence[Sequence[str]], min_fraction: float = 0.5, max_chars: int = 150
) -> set[str]:
    """Short lines that occur on at least ``min_fraction`` of pages (running headers/footers).

    Needs 2+ pages and only considers short lines, so real paragraphs are never
    removed from one-page circulars.
    """
    if len(pages) < 2:
        return set()
    counts = Counter(
        line for page in pages for line in set(page) if line.strip() and len(line) <= max_chars
    )
    threshold = max(2, round(len(pages) * min_fraction))
    return {line for line, n in counts.items() if n >= threshold}
