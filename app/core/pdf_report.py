"""Render a summary (and optional transcript) to a PDF."""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .branding import APP_NAME
from .subtitles import Segment, format_timestamp
from .summarizer import ChronologicalSummary, SummaryResult

# Matches the app's theme in ui/main_window.py: mosaic-myna gold on paper.
ACCENT = colors.HexColor("#8C6416")
MUTED = colors.HexColor("#8A7F73")
RULE = colors.HexColor("#E8E0D5")
RULE_FAINT = colors.HexColor("#F1EAE0")
TOPIC_INK = colors.HexColor("#2B2622")


@dataclass
class ReportMeta:
    source_name: str
    duration_seconds: float
    language: str
    whisper_model: str
    device: str
    summary_model: str


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleX", parent=base["Title"], fontSize=20, leading=25,
            textColor=ACCENT, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleX", parent=base["Normal"], fontSize=9, leading=13,
            textColor=MUTED, spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "HeadingX", parent=base["Heading2"], fontSize=13, leading=17,
            textColor=ACCENT, spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyX", parent=base["BodyText"], fontSize=10, leading=15,
            alignment=TA_JUSTIFY, spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "BulletX", parent=base["BodyText"], fontSize=10, leading=14, spaceAfter=3,
        ),
        "mono": ParagraphStyle(
            "MonoX", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5,
            leading=12.5, spaceAfter=2,
        ),
        "topic": ParagraphStyle(
            "TopicX", parent=base["Heading3"], fontSize=11.5, leading=15,
            textColor=TOPIC_INK, spaceBefore=0, spaceAfter=1,
        ),
        "stamp": ParagraphStyle(
            "StampX", parent=base["BodyText"], fontName="Courier", fontSize=9,
            leading=13, textColor=ACCENT, spaceAfter=0,
        ),
        "who": ParagraphStyle(
            "WhoX", parent=base["BodyText"], fontSize=8.5, leading=12,
            textColor=MUTED, spaceAfter=3,
        ),
    }


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


def _page_furniture(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"Page {doc.page}")
    canvas.drawString(18 * mm, 12 * mm, f"Generated locally by {APP_NAME}")
    canvas.setStrokeColor(RULE)
    canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
    canvas.restoreState()


def _bullet_list(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(_esc(item), style), leftIndent=12) for item in items],
        bulletType="bullet",
        bulletFontSize=7,
        start="•",
        leftIndent=14,
    )


def build_pdf(
    output_path: Path,
    summary: SummaryResult,
    meta: ReportMeta,
    segments: list[Segment] | None = None,
    include_transcript: bool = False,
) -> Path:
    styles = _styles()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=f"Summary — {meta.source_name}",
        author=APP_NAME,
    )

    story: list = _header_story(styles, meta, "Summary")

    if summary.overview:
        story.append(Paragraph("Overview", styles["heading"]))
        for paragraph in [p for p in summary.overview.split("\n\n") if p.strip()]:
            story.append(Paragraph(_esc(paragraph.strip()), styles["body"]))

    if summary.key_points:
        story.append(Paragraph("Key points", styles["heading"]))
        story.append(_bullet_list(summary.key_points, styles["bullet"]))

    if summary.chapters:
        story.append(Paragraph("Chapters", styles["heading"]))
        story.append(
            _bullet_list(
                [f"{stamp} — {title}" for stamp, title in summary.chapters],
                styles["bullet"],
            )
        )

    if summary.action_items:
        story.append(Paragraph("Action items", styles["heading"]))
        story.append(_bullet_list(summary.action_items, styles["bullet"]))

    if include_transcript and segments:
        story.append(PageBreak())
        story.append(Paragraph("Full transcript", styles["heading"]))
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            stamp = format_timestamp(segment.start, ".")[:8]
            who = f"<b>{_esc(segment.speaker)}:</b> " if segment.speaker else ""
            story.append(
                Paragraph(
                    f'<font color="#A99C8C">[{stamp}]</font> {who}{_esc(text)}',
                    styles["mono"],
                )
            )

    doc.build(story, onFirstPage=_page_furniture, onLaterPages=_page_furniture)
    return output_path


def _header_story(styles: dict, meta: ReportMeta, heading: str) -> list:
    minutes = meta.duration_seconds / 60.0
    facts = " · ".join(
        part
        for part in [
            f"{minutes:.1f} min",
            f"language: {meta.language}" if meta.language else "",
            f"Whisper {meta.whisper_model} on {meta.device.upper()}",
            f"summary: {meta.summary_model}" if meta.summary_model else "",
            datetime.now().strftime("%d %b %Y, %H:%M"),
        ]
        if part
    )
    return [
        Paragraph(_esc(meta.source_name), styles["title"]),
        Paragraph(_esc(heading), styles["subtitle"]),
        Paragraph(_esc(facts), styles["subtitle"]),
        HRFlowable(width="100%", thickness=1, color=RULE),
        Spacer(1, 6),
    ]


def build_chronological_pdf(
    output_path: Path,
    summary: ChronologicalSummary,
    meta: ReportMeta,
    segments: list[Segment] | None = None,
    include_transcript: bool = False,
) -> Path:
    """A walk through the recording in order: each topic, when it ran, what was said."""
    styles = _styles()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=f"Chronological summary — {meta.source_name}",
        author=APP_NAME,
    )

    story: list = _header_story(
        styles, meta, "Chronological summary — sections in the order they occur"
    )

    if not summary.blocks:
        story.append(
            Paragraph(
                "No distinct sections could be identified in this recording.",
                styles["body"],
            )
        )
        doc.build(story, onFirstPage=_page_furniture, onLaterPages=_page_furniture)
        return output_path

    # A contents strip up front, so the reader can jump straight to a section.
    story.append(Paragraph("Contents", styles["heading"]))
    contents = [
        [
            Paragraph(f'<font face="Courier">{_esc(block.time_range)}</font>',
                      styles["bullet"]),
            Paragraph(_esc(block.title), styles["bullet"]),
        ]
        for block in summary.blocks
    ]
    table = Table(contents, colWidths=[32 * mm, None], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (0, -1), 0),
            ]
        )
    )
    story.append(table)

    story.append(Paragraph("Timeline", styles["heading"]))
    for index, block in enumerate(summary.blocks):
        if index:
            story.append(Spacer(1, 4))
            story.append(
                HRFlowable(width="100%", thickness=0.5, color=RULE_FAINT)
            )
            story.append(Spacer(1, 6))
        story.append(
            Paragraph(f"{_esc(block.title)}  [{_esc(block.time_range)}]", styles["topic"])
        )
        if block.speakers:
            story.append(
                Paragraph(
                    "Speaking: " + _esc(", ".join(block.speakers)), styles["who"]
                )
            )
        if block.summary:
            story.append(Paragraph(_esc(block.summary), styles["body"]))

    if include_transcript and segments:
        story.append(PageBreak())
        story.append(Paragraph("Full transcript", styles["heading"]))
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            stamp = format_timestamp(segment.start, ".")[:8]
            who = f"<b>{_esc(segment.speaker)}:</b> " if segment.speaker else ""
            story.append(
                Paragraph(
                    f'<font color="#A99C8C">[{stamp}]</font> {who}{_esc(text)}',
                    styles["mono"],
                )
            )

    doc.build(story, onFirstPage=_page_furniture, onLaterPages=_page_furniture)
    return output_path
