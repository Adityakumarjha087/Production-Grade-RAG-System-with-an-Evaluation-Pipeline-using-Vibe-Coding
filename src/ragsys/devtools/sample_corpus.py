"""Synthetic sample corpus + hand-written test set for tests, CI and demos.

All documents are FICTIONAL and clearly labelled as synthetic sample data; they
imitate the *structure* of regulatory circulars (numbered paragraphs, headings,
tables, running headers/footers) so every pipeline stage can be exercised
without the real corpus. Requires ``reportlab`` (dev dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragsys.evaluation.testset import QType, ReviewStatus, TestItem, TestSpec, build_from_specs
from ragsys.generation.prompts import ABSTAIN_TEXT
from ragsys.models import Document

DISCLAIMER = "SYNTHETIC SAMPLE DOCUMENT - fictional content for software testing only"

Block = tuple[str, object]  # ("h1"|"h2"|"p"|"table", content)


@dataclass(frozen=True)
class SampleDoc:
    filename: str
    number: str
    title: str
    blocks: list[Block]


DOCS: list[SampleDoc] = [
    SampleDoc(
        "sample_settlement_cycle.pdf",
        "SAMPLE/HO/MRD/2024/001",
        "Settlement Cycle for the Equity Cash Segment",
        [
            ("h1", "1. Background"),
            (
                "p",
                "1.1 Market infrastructure institutions have requested a shorter settlement "
                "cycle to reduce counterparty risk and to release investor funds sooner.",
            ),
            (
                "p",
                "1.2 After public consultation, it has been decided to move all scrips in the "
                "equity cash segment to a T+1 rolling settlement cycle in a phased manner.",
            ),
            ("h1", "2. Phased implementation"),
            (
                "p",
                "2.1 The bottom 100 scrips by market capitalisation shall move to T+1 settlement "
                "from 25 February 2025. Thereafter, the next 500 scrips shall move on the last "
                "Friday of every month.",
            ),
            (
                "p",
                "2.2 Stock exchanges shall publish the list of scrips moving to T+1 settlement "
                "at least one month before the effective date of each phase.",
            ),
            ("h1", "3. Pay-in and pay-out timelines"),
            (
                "p",
                "3.1 The timelines applicable to scrips under T+1 settlement are given below. "
                "All times are Indian Standard Time.",
            ),
            (
                "table",
                [
                    ["Activity", "Day", "Deadline"],
                    ["Funds pay-in by clearing members", "T+1", "10:30 AM"],
                    ["Securities pay-in by clearing members", "T+1", "10:30 AM"],
                    ["Pay-out of funds and securities", "T+1", "1:30 PM"],
                    ["Auction session for shortages", "T+1", "2:30 PM to 3:00 PM"],
                    ["Close-out of auction shortfalls", "T+2", "11:00 AM"],
                ],
            ),
            ("h1", "4. Applicability"),
            (
                "p",
                "4.1 This circular shall come into force with immediate effect and applies to "
                "all recognised stock exchanges, clearing corporations and depositories.",
            ),
        ],
    ),
    SampleDoc(
        "sample_kyc_records.pdf",
        "SAMPLE/HO/MIRSD/2024/014",
        "Maintenance of KYC Records by Intermediaries",
        [
            ("h1", "1. Introduction"),
            (
                "p",
                "1.1 Registered intermediaries collect Know Your Client (KYC) documents at the "
                "time of account opening. This circular specifies how long such records must be "
                "preserved and in what form.",
            ),
            ("h1", "2. Retention period"),
            (
                "p",
                "2.1 Intermediaries shall preserve KYC records and transaction records for a "
                "period of eight years from the date of cessation of the business relationship "
                "with the client.",
            ),
            (
                "p",
                "2.2 Where a client account is the subject of an ongoing investigation, records "
                "shall be preserved until the investigation is closed, even if the eight year "
                "period has elapsed.",
            ),
            ("h1", "3. Form of records"),
            (
                "p",
                "3.1 Records may be maintained in electronic form provided they are stored in a "
                "tamper-evident system and a backup copy is kept at a geographically separate "
                "location at least 100 kilometres away.",
            ),
            ("h2", "3.2 Periodic updation"),
            (
                "p",
                "3.2.1 KYC details of high-risk clients shall be updated every two years, of "
                "medium-risk clients every eight years, and of low-risk clients every ten years.",
            ),
            ("h1", "4. Reporting"),
            (
                "p",
                "4.1 Any loss or destruction of KYC records shall be reported to the regulator "
                "within 7 working days of the intermediary becoming aware of the loss.",
            ),
        ],
    ),
    SampleDoc(
        "sample_mf_expense_ratio.pdf",
        "SAMPLE/HO/IMD/2024/031",
        "Total Expense Ratio Limits for Mutual Fund Schemes",
        [
            ("h1", "1. Objective"),
            (
                "p",
                "1.1 To ensure that costs charged to investors remain reasonable, the maximum "
                "Total Expense Ratio (TER) that a scheme may charge is linked to the daily net "
                "assets of the scheme.",
            ),
            ("h1", "2. Slab-wise limits for equity-oriented schemes"),
            (
                "table",
                [
                    ["Daily net assets (INR crore)", "Maximum TER"],
                    ["First 500", "2.25%"],
                    ["Next 250", "2.00%"],
                    ["Next 1,250", "1.75%"],
                    ["Next 3,000", "1.60%"],
                    ["Next 5,000", "1.50%"],
                    ["Balance above 10,000", "1.05%"],
                ],
            ),
            ("h1", "3. Index funds and exchange traded funds"),
            (
                "p",
                "3.1 For index funds and exchange traded funds, the total expense ratio shall "
                "not exceed 1.00 per cent of the daily net assets.",
            ),
            ("h1", "4. Disclosure"),
            (
                "p",
                "4.1 Asset management companies shall disclose the TER of all schemes on their "
                "website on a daily basis and shall notify investors at least three working "
                "days before any increase in the TER.",
            ),
        ],
    ),
    SampleDoc(
        "sample_grievance_redressal.pdf",
        "SAMPLE/HO/OIAE/2024/007",
        "Investor Grievance Redressal Timelines",
        [
            ("h1", "1. Scope"),
            (
                "p",
                "1.1 This circular applies to complaints lodged by investors against listed "
                "companies and registered intermediaries through the online complaint portal.",
            ),
            ("h1", "2. Timelines for resolution"),
            (
                "p",
                "2.1 The entity against which a complaint is lodged shall resolve the complaint "
                "within 21 calendar days of receipt.",
            ),
            (
                "p",
                "2.2 If the investor is not satisfied with the resolution, the investor may "
                "request a first level review by the designated body within 15 days of the "
                "closure of the complaint.",
            ),
            (
                "p",
                "2.3 The designated body shall complete the first level review within 10 "
                "calendar days, after which the investor may approach the online dispute "
                "resolution mechanism.",
            ),
            ("h1", "3. Monitoring"),
            (
                "p",
                "3.1 Entities shall submit a quarterly report of complaints received, resolved "
                "and pending to the stock exchange within 21 days from the end of each quarter.",
            ),
        ],
    ),
]

# (qtype, question, reference answer, verbatim evidence quotes)
SAMPLE_QA: list[tuple[QType, str, str, list[str]]] = [
    (
        QType.FACTUAL,
        "From what date do the bottom 100 scrips move to T+1 settlement?",
        "The bottom 100 scrips by market capitalisation move to T+1 settlement from 25 February 2025.",
        [
            "The bottom 100 scrips by market capitalisation shall move to T+1 settlement from 25 February 2025."
        ],
    ),
    (
        QType.TABLE,
        "What is the deadline for pay-out of funds and securities under T+1?",
        "Pay-out of funds and securities happens on T+1 by 1:30 PM.",
        ["Pay-out of funds and securities"],
    ),
    (
        QType.FACTUAL,
        "How long must intermediaries keep KYC records after the client relationship ends?",
        "Eight years from the date of cessation of the business relationship with the client.",
        ["preserve KYC records and transaction records for a period of eight years"],
    ),
    (
        QType.FACTUAL,
        "How far away must the backup copy of electronic KYC records be stored?",
        "At a geographically separate location at least 100 kilometres away.",
        [
            "a backup copy is kept at a geographically separate location at least 100 kilometres away"
        ],
    ),
    (
        QType.FACTUAL,
        "Within how many days must loss of KYC records be reported to the regulator?",
        "Within 7 working days of the intermediary becoming aware of the loss.",
        ["within 7 working days of the intermediary becoming aware of the loss"],
    ),
    (
        QType.TABLE,
        "What is the maximum TER on the first 500 crore of daily net assets of an equity scheme?",
        "2.25%.",
        ["First 500"],
    ),
    (
        QType.FACTUAL,
        "What is the expense ratio cap for index funds and ETFs?",
        "It shall not exceed 1.00 per cent of daily net assets.",
        ["shall not exceed 1.00 per cent of the daily net assets"],
    ),
    (
        QType.FACTUAL,
        "Within how many days must a listed company resolve an investor complaint?",
        "Within 21 calendar days of receipt.",
        ["shall resolve the complaint within 21 calendar days of receipt"],
    ),
    (
        QType.MULTI_HOP,
        "After a complaint is closed, how long does the investor have to seek review and how long does the review take?",
        "The investor may request a first level review within 15 days of closure, and the review must be completed within 10 calendar days.",
        [
            "request a first level review by the designated body within 15 days",
            "shall complete the first level review within 10 calendar days",
        ],
    ),
    (
        QType.UNANSWERABLE,
        "What penalty applies to a broker that misses the T+1 funds pay-in deadline?",
        ABSTAIN_TEXT,
        [],
    ),
    (
        QType.UNANSWERABLE,
        "What is the maximum TER for debt-oriented mutual fund schemes?",
        ABSTAIN_TEXT,
        [],
    ),
]


def _build_pdf(doc: SampleDoc, path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()

    def on_page(canvas: object, pdf: object) -> None:
        canvas.saveState()  # type: ignore[attr-defined]
        canvas.setFont("Helvetica", 8)  # type: ignore[attr-defined]
        canvas.drawString(40, 815, DISCLAIMER)  # type: ignore[attr-defined]
        canvas.drawString(40, 25, f"Page {pdf.page}")  # type: ignore[attr-defined]
        canvas.restoreState()  # type: ignore[attr-defined]

    story: list[object] = [
        Paragraph(doc.title, styles["Title"]),
        Paragraph(f"Circular No. {doc.number}", styles["Normal"]),
        Spacer(1, 12),
    ]
    broke = False
    for i, (kind, content) in enumerate(doc.blocks):
        if kind == "h1":
            if not broke and i >= len(doc.blocks) // 2:
                story.append(PageBreak())  # multi-page docs exercise header/footer removal
                broke = True
            story.append(Paragraph(str(content), styles["Heading2"]))
        elif kind == "h2":
            story.append(Paragraph(str(content), styles["Heading3"]))
        elif kind == "p":
            story.append(Paragraph(str(content), styles["BodyText"]))
        elif kind == "table":
            table = Table(content, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ]
                )
            )
            story.extend([table, Spacer(1, 8)])
    SimpleDocTemplate(str(path), pagesize=A4, title=doc.title).build(
        story, onFirstPage=on_page, onLaterPages=on_page
    )


def build_sample_corpus(out_dir: Path) -> list[Path]:
    """Write the sample PDFs into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for doc in DOCS:
        path = out_dir / doc.filename
        _build_pdf(doc, path)
        paths.append(path)
    return paths


def build_sample_testset(docs: list[Document]) -> list[TestItem]:
    """Approved test items with evidence spans located in the parsed ``docs``."""
    specs = [TestSpec(qtype=q, question=qu, answer=a, quotes=qs) for q, qu, a, qs in SAMPLE_QA]
    items, errors = build_from_specs(specs, docs, status=ReviewStatus.APPROVED)
    if errors:
        raise LookupError(f"sample test set does not match the sample corpus: {errors}")
    return items
