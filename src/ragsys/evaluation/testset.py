"""Synthetic test-set generation with chunking-independent gold labels.

Gold evidence is stored as a character span of ``Document.text`` (not a chunk id),
because chunk ids differ per chunking strategy. A retrieved chunk is relevant if
its source span overlaps the gold span enough (see ``retrieval_metrics``), so the
*same* test set fairly scores every chunker.

The LLM must return a *verbatim* evidence quote; items whose quote cannot be found
in the source passage are discarded (cheap guard against hallucinated Q/A pairs).
Every item starts as ``pending`` and only ``approved`` items are evaluated.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import anyio
import structlog
from pydantic import BaseModel, Field

from ragsys.generation.llm import LLMClient
from ragsys.generation.prompts import ABSTAIN_TEXT
from ragsys.models import Document, ElementType

log = structlog.get_logger(__name__)


class QType(StrEnum):
    """Question categories (each stresses a different part of the system)."""

    FACTUAL = "factual"
    TABLE = "table"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"


class ReviewStatus(StrEnum):
    """Human review state."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class EvidenceSpan(BaseModel):
    """Gold evidence as a span of ``Document.text``."""

    doc_id: str
    char_start: int
    char_end: int
    text: str


class TestItem(BaseModel):
    """One question with its reference answer and gold evidence."""

    __test__ = False  # not a pytest test class

    id: str
    question: str
    reference_answer: str
    qtype: QType
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    status: ReviewStatus = ReviewStatus.PENDING
    generator: str = ""
    notes: str = ""

    @property
    def answerable(self) -> bool:
        """Unanswerable items expect the system to abstain."""
        return self.qtype != QType.UNANSWERABLE


def load_testset(path: Path) -> list[TestItem]:
    """Read a JSONL test set."""
    with path.open(encoding="utf-8") as fh:
        return [TestItem.model_validate_json(line) for line in fh if line.strip()]


def save_testset(path: Path, items: Sequence[TestItem]) -> None:
    """Write JSONL atomically (never leaves a half-written file during review)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
    tmp.replace(path)


def hash_testset(items: Sequence[TestItem]) -> str:
    """Content hash recorded with every evaluation run."""
    blob = "".join(i.model_dump_json() for i in items).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# ---------------------------------------------------------------- passages


@dataclass(frozen=True)
class Passage:
    """A contiguous span of a document used as generation context."""

    doc: Document
    start: int
    end: int
    kind: str  # "text" | "table"

    @property
    def text(self) -> str:
        return self.doc.text[self.start : self.end]


def doc_passages(doc: Document, min_chars: int = 300, max_chars: int = 1500) -> list[Passage]:
    """Element-aligned windows of ``min_chars..max_chars`` (tables are their own passage)."""
    passages: list[Passage] = []
    window: list[tuple[int, int]] = []
    for el in doc.elements:
        if el.type == ElementType.TABLE:
            if len(el.text) >= 100:
                passages.append(
                    Passage(
                        doc, el.char_start, min(el.char_end, el.char_start + max_chars), "table"
                    )
                )
            continue
        if el.type == ElementType.HEADING:
            window = []
            continue
        if window and el.char_end - window[0][0] > max_chars:
            window = []
        window.append((el.char_start, el.char_end))
        if window[-1][1] - window[0][0] >= min_chars:
            passages.append(Passage(doc, window[0][0], window[-1][1], "text"))
            window = []
    return passages


def sample_passages(
    docs: Sequence[Document], n: int, rng: random.Random, kind: str | None = None
) -> list[Passage]:
    """Round-robin over shuffled documents so questions cover the corpus evenly."""
    pools = [[p for p in doc_passages(d) if kind is None or p.kind == kind] for d in docs]
    pools = [p for p in pools if p]
    for pool in pools:
        rng.shuffle(pool)
    rng.shuffle(pools)
    picked: list[Passage] = []
    while len(picked) < n and any(pools):
        for pool in pools:
            if pool and len(picked) < n:
                picked.append(pool.pop())
    return picked


# ---------------------------------------------------------------- quote location

_WORDS = re.compile(r"\S+")


def locate_quote(
    text: str, quote: str, start: int = 0, end: int | None = None
) -> tuple[int, int] | None:
    """Find ``quote`` in ``text[start:end]``: exact first, then whitespace-insensitive."""
    end = len(text) if end is None else end
    quote = quote.strip().strip('"').strip()
    if len(quote) < 5:
        return None
    idx = text.find(quote, start, end)
    if idx != -1:
        return idx, idx + len(quote)
    words = _WORDS.findall(quote)
    pattern = re.compile(r"\s+".join(re.escape(w) for w in words), re.IGNORECASE)
    match = pattern.search(text, start, end)
    return (match.start(), match.end()) if match else None


def reanchor_evidence(
    items: Sequence[TestItem], docs: Sequence[Document]
) -> tuple[list[TestItem], list[str]]:
    """Re-locate evidence spans by their text in the current parse.

    Spans are offsets into a specific parser's canonical text; if the corpus was
    re-parsed (other parser/version) offsets can shift. Items whose evidence text
    can no longer be found are dropped (their ids are returned) rather than scored
    against wrong spans.
    """
    by_id = {d.doc_id: d for d in docs}
    kept: list[TestItem] = []
    dropped: list[str] = []
    for item in items:
        new_spans = []
        for ev in item.evidence:
            doc = by_id.get(ev.doc_id)
            if doc is None:
                break
            if doc.text[ev.char_start : ev.char_end] == ev.text:
                new_spans.append(ev)
                continue
            span = locate_quote(doc.text, ev.text)
            if span is None:
                break
            new_spans.append(ev.model_copy(update={"char_start": span[0], "char_end": span[1]}))
        if len(new_spans) == len(item.evidence):
            kept.append(item.model_copy(update={"evidence": new_spans}))
        else:
            dropped.append(item.id)
    if dropped:
        log.warning("testset.evidence_not_found", dropped=len(dropped), ids=dropped[:10])
    return kept, dropped


# ---------------------------------------------------------------- hand-written specs


class TestSpec(BaseModel):
    """A hand-written item: evidence is given as verbatim quotes, located at build time."""

    __test__ = False  # not a pytest test class

    qtype: QType
    question: str
    answer: str = ""
    quotes: list[str] = Field(default_factory=list)
    doc: str | None = None  # substring of the source filename/doc id (narrows the search)


def _candidate_docs(docs: Sequence[Document], hint: str | None) -> list[Document]:
    if not hint:
        return list(docs)
    return [d for d in docs if hint in Path(d.source_path).name or hint == d.doc_id]


def _locate_spec_quote(quote: str, docs: Sequence[Document]) -> EvidenceSpan | None:
    for d in docs:
        span = locate_quote(d.text, quote)
        if span is not None:
            s, e = span
            return EvidenceSpan(doc_id=d.doc_id, char_start=s, char_end=e, text=d.text[s:e])
    return None


def build_from_specs(
    specs: Sequence[TestSpec],
    docs: Sequence[Document],
    *,
    status: ReviewStatus = ReviewStatus.PENDING,
    generator: str = "hand-written",
) -> tuple[list[TestItem], list[str]]:
    """Items whose quotes were all found, plus error messages for the rest."""
    items: list[TestItem] = []
    errors: list[str] = []
    for spec in specs:
        pool = _candidate_docs(docs, spec.doc)
        evidence = [_locate_spec_quote(q, pool) for q in spec.quotes]
        missing = [q for q, ev in zip(spec.quotes, evidence, strict=True) if ev is None]
        if spec.doc and not pool:
            errors.append(f"document {spec.doc!r} not in corpus: {spec.question}")
            continue
        if missing:
            errors.append(f"quote not found {missing!r}: {spec.question}")
            continue
        unanswerable = spec.qtype == QType.UNANSWERABLE
        items.append(
            TestItem(
                id=item_id(spec.question, []),
                question=spec.question,
                reference_answer=ABSTAIN_TEXT if unanswerable else spec.answer,
                qtype=spec.qtype,
                evidence=[ev for ev in evidence if ev is not None],
                status=status,
                generator=generator,
            )
        )
    return items, errors


def load_specs(path: Path) -> list[TestSpec]:
    """Read ``items:`` from a YAML spec file."""
    import yaml

    data = yaml.safe_load(path.read_text("utf-8")) or {}
    return [TestSpec.model_validate(raw) for raw in data.get("items", [])]


# ---------------------------------------------------------------- generation

SYSTEM = (
    "You create evaluation data for a question-answering system over Indian securities-"
    "market regulations (SEBI circulars). Questions must sound like a compliance officer "
    "or market participant asking a realistic question, must be answerable without seeing "
    "the passage (no 'according to the passage'), and must be specific."
)

_QA_PROPS = {
    "question": {"type": "string"},
    "answer": {"type": "string"},
    "evidence_quotes": {"type": "array", "items": {"type": "string"}},
}
QA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": _QA_PROPS,
    "required": ["question", "answer", "evidence_quotes"],
    "additionalProperties": False,
}

_INSTRUCTIONS = {
    QType.FACTUAL: "Write ONE question whose answer is a specific fact stated in the passage.",
    QType.TABLE: "Write ONE question that can only be answered by reading a value from the table.",
    QType.MULTI_HOP: (
        "Write ONE question that requires combining information from BOTH passages. "
        "Provide at least one evidence quote from each passage."
    ),
    QType.UNANSWERABLE: (
        "Write ONE realistic question on the same topic that the passage does NOT answer "
        "(e.g. asks for a detail, number or case it does not mention). Set answer to "
        f'"{ABSTAIN_TEXT}" and evidence_quotes to [].'
    ),
}


def build_generation_prompt(qtype: QType, passages: Sequence[Passage]) -> str:
    """User prompt for one test item."""
    blocks = "\n\n".join(
        f'<passage id="{i}" title="{p.doc.title}">\n{p.text}\n</passage>'
        for i, p in enumerate(passages, start=1)
    )
    return (
        f"{blocks}\n\n{_INSTRUCTIONS[qtype]}\n"
        "The answer must be complete but concise. Each evidence quote must be copied "
        "VERBATIM from a passage (a full sentence or table row, 10-300 characters)."
    )


def item_id(question: str, passages: Sequence[Passage]) -> str:
    """Stable id from the question and its source spans."""
    key = question + "|" + "|".join(f"{p.doc.doc_id}:{p.start}" for p in passages)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _locate_in_passages(quote: str, passages: Sequence[Passage]) -> EvidenceSpan | None:
    """Evidence span for ``quote`` inside the first passage that contains it."""
    for p in passages:
        span = locate_quote(p.doc.text, quote, p.start, p.end)
        if span is not None:
            start, end = span
            return EvidenceSpan(
                doc_id=p.doc.doc_id, char_start=start, char_end=end, text=p.doc.text[start:end]
            )
    return None


def parse_item(
    raw: dict[str, Any], qtype: QType, passages: Sequence[Passage], generator: str
) -> TestItem | None:
    """Validate an LLM response; None if any evidence quote can't be located."""
    question = str(raw.get("question", "")).strip()
    answer = str(raw.get("answer", "")).strip()
    if len(question) < 10 or not answer:
        return None
    evidence: list[EvidenceSpan] = []
    if qtype != QType.UNANSWERABLE:
        for quote in raw.get("evidence_quotes", []):
            found = _locate_in_passages(str(quote), passages)
            if found is None:
                return None
            evidence.append(found)
        if not evidence:
            return None
    else:
        answer = ABSTAIN_TEXT
    return TestItem(
        id=item_id(question, passages),
        question=question,
        reference_answer=answer,
        qtype=qtype,
        evidence=evidence,
        generator=generator,
    )


class TestsetGenerator:
    """Generates pending test items with an LLM (structured JSON output)."""

    __test__ = False  # not a pytest test class

    def __init__(self, llm: LLMClient, seed: int = 13, concurrency: int = 4) -> None:
        self.llm = llm
        self.rng = random.Random(seed)
        self.concurrency = concurrency

    async def generate_one(self, qtype: QType, passages: Sequence[Passage]) -> TestItem | None:
        """One item, or None if the model output fails validation."""
        completion = await self.llm.complete(
            system=SYSTEM,
            prompt=build_generation_prompt(qtype, passages),
            max_tokens=2048,
            json_schema=QA_SCHEMA,
        )
        try:
            raw = json.loads(completion.text)
        except json.JSONDecodeError:
            return None
        return parse_item(raw, qtype, passages, completion.model)

    def plan(
        self, docs: Sequence[Document], n: int, mix: dict[QType, float]
    ) -> list[tuple[QType, list[Passage]]]:
        """Decide which passages feed which question type."""
        jobs: list[tuple[QType, list[Passage]]] = []
        for qtype, share in mix.items():
            count = round(n * share)
            if qtype == QType.TABLE:
                jobs += [(qtype, [p]) for p in sample_passages(docs, count, self.rng, "table")]
            elif qtype == QType.MULTI_HOP:
                pool = sample_passages(docs, count * 2, self.rng, "text")
                jobs += [(qtype, pool[i : i + 2]) for i in range(0, len(pool) - 1, 2)]
            else:
                jobs += [(qtype, [p]) for p in sample_passages(docs, count, self.rng, "text")]
        return jobs

    async def generate(
        self, docs: Sequence[Document], n: int, mix: dict[QType, float]
    ) -> list[TestItem]:
        """Generate up to ``n`` items following ``mix`` (fractions per type)."""
        jobs = self.plan(docs, n, mix)
        results: list[TestItem | None] = [None] * len(jobs)
        limiter = anyio.Semaphore(self.concurrency)

        async def run(i: int, qtype: QType, passages: list[Passage]) -> None:
            async with limiter:
                try:
                    results[i] = await self.generate_one(qtype, passages)
                except Exception as exc:
                    log.warning("testset.item_failed", qtype=qtype, error=str(exc))

        async with anyio.create_task_group() as tg:
            for i, (qtype, passages) in enumerate(jobs):
                tg.start_soon(run, i, qtype, passages)
        items = [r for r in results if r is not None]
        log.info("testset.generated", requested=n, planned=len(jobs), valid=len(items))
        return items
