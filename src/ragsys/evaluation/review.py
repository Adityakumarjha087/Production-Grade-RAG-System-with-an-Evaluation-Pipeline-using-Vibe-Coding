"""Human review of generated test items.

The decision logic is pure (and unit-tested); the interactive loop is a thin
terminal UI over it. Progress is saved after every decision, so review can be
interrupted and resumed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from ragsys.evaluation.testset import ReviewStatus, TestItem, load_testset, save_testset

DECISIONS = {"a": "approve", "r": "reject", "e": "edit", "s": "skip", "q": "quit"}


def apply_decision(
    item: TestItem,
    decision: str,
    *,
    question: str | None = None,
    answer: str | None = None,
    note: str | None = None,
) -> TestItem:
    """Return the item updated for ``decision`` (approve | reject | edit)."""
    if decision == "approve":
        return item.model_copy(update={"status": ReviewStatus.APPROVED})
    if decision == "reject":
        return item.model_copy(
            update={"status": ReviewStatus.REJECTED, "notes": note or item.notes}
        )
    if decision == "edit":
        return item.model_copy(
            update={
                "question": (question or item.question).strip(),
                "reference_answer": (answer or item.reference_answer).strip(),
                "status": ReviewStatus.APPROVED,
                "notes": note or "edited",
            }
        )
    raise ValueError(f"unknown decision: {decision}")


def review_stats(items: Sequence[TestItem]) -> dict[str, int]:
    """Counts by status and by question type."""
    stats = Counter(f"status:{i.status.value}" for i in items)
    stats.update(f"type:{i.qtype.value}" for i in items)
    stats["total"] = len(items)
    return dict(sorted(stats.items()))


def render_item(item: TestItem, position: str) -> str:
    """Text shown to the reviewer."""
    evidence = "\n".join(f"  > {e.text}" for e in item.evidence) or "  (none: unanswerable)"
    return (
        f"\n[{position}] {item.qtype.value}  id={item.id}\n"
        f"Q: {item.question}\nA: {item.reference_answer}\nEvidence:\n{evidence}\n"
    )


def run_review(
    path: Path,
    ask: Callable[[str], str],
    show: Callable[[str], None],
    only_pending: bool = True,
) -> dict[str, int]:
    """Interactive loop. ``ask``/``show`` are injected (typer.prompt/echo in the CLI)."""
    items = load_testset(path)
    queue = [
        i for i, it in enumerate(items) if not only_pending or it.status == ReviewStatus.PENDING
    ]
    for n, idx in enumerate(queue, start=1):
        show(render_item(items[idx], f"{n}/{len(queue)}"))
        choice = DECISIONS.get(ask("[a]pprove [r]eject [e]dit [s]kip [q]uit").strip().lower()[:1])
        if choice == "quit":
            break
        if choice in (None, "skip"):
            continue
        if choice == "edit":
            q = ask("New question (enter = keep)") or None
            a = ask("New answer (enter = keep)") or None
            items[idx] = apply_decision(items[idx], "edit", question=q, answer=a)
        elif choice == "reject":
            items[idx] = apply_decision(items[idx], "reject", note=ask("Reason (optional)") or None)
        else:
            items[idx] = apply_decision(items[idx], "approve")
        save_testset(path, items)
    return review_stats(items)
