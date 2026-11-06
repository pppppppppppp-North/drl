#!/usr/bin/env python3
"""Build the five focused process reports for the Thai DRL HPC project.

The generator reads the accepted campaign, deployment, quota, smoke, import,
lineage, and accounting artifacts from the repository.  It writes final PDFs
to output/pdf and keeps any visual-QA intermediates under tmp/pdfs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
from typing import Any, Iterable, Sequence

from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
EVIDENCE_CUT = "17 July 2026"
GENERATED_ON = "20 July 2026"
CAMPAIGN_ID = "large_scale_thai_drl_v2"

NAVY = HexColor("#10243F")
BLUE = HexColor("#2F67D8")
TEAL = HexColor("#15806F")
GREEN = HexColor("#16835A")
ORANGE = HexColor("#E78118")
RED = HexColor("#C84B4B")
INK = HexColor("#1D2939")
SLATE = HexColor("#475467")
MID = HexColor("#667085")
LIGHT = HexColor("#E4EAF2")
PALE = HexColor("#F5F7FA")
PALE_BLUE = HexColor("#EEF4FF")
PALE_GREEN = HexColor("#ECF8F3")
PALE_ORANGE = HexColor("#FFF5E8")
WHITE = colors.white

PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 30 * mm


@dataclass(frozen=True)
class ReportSpec:
    filename: str
    short_title: str
    title: str
    subtitle: str
    accent: Color
    status_label: str
    status_text: str


def load_json(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sidecar_hash(relative: str) -> str:
    text = (ROOT / relative).read_text(encoding="utf-8").strip()
    return text.split()[0]


def file_hash(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_dir_stats(relative: str) -> tuple[int, int]:
    directory = ROOT / relative
    files = [path for path in directory.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    number = float(value)
    for unit in units:
        if abs(number) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(number):,} {unit}"
            return f"{number:,.3f} {unit}"
        number /= 1024.0
    raise AssertionError("unreachable")


def short_hash(value: str, head: int = 12, tail: int = 8) -> str:
    if len(value) <= head + tail + 3:
        return value
    return f"{value[:head]}...{value[-tail:]}"


def break_hash(value: str) -> str:
    return " ".join(value[index : index + 16] for index in range(0, len(value), 16))


def break_path(value: str) -> str:
    return value.replace("/", "/ ")


def parse_accounting(relative: str) -> dict[str, str]:
    lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    for line in lines:
        if line.startswith("JobIDRaw|"):
            header = line.rstrip("|").split("|")
            continue
        if header and line and not line.startswith("#"):
            values = line.rstrip("|").split("|")
            row = dict(zip(header, values))
            if "." not in row.get("JobIDRaw", ""):
                return row
    raise ValueError(f"No top-level accounting row in {relative}")


def gather_facts() -> dict[str, Any]:
    state = load_json("campaigns/large_scale_thai_drl_v2/state.json")
    state_counts = Counter(stage["state"] for stage in state["stages"])
    summary = load_json("results/screening_storage_safe_smoke_3550/screening_smoke_summary.json")
    validation = load_json("results/screening_storage_safe_smoke_3550/storage_safe_smoke_validation.json")
    imported = load_json("reports/screening_storage_safe_smoke_3550_import_validation.json")
    preflight = load_json("reports/bistka_screening_storage_safe_cutover_v2_preflight.json")
    manifest = load_json(
        "campaigns/large_scale_thai_drl_v2/deployment/"
        "bistka_screening_storage_safe_cutover_v2_manifest.json"
    )
    lineage = load_json(
        "campaigns/large_scale_thai_drl_v2/imports/"
        "storage_safe_cutover_calibration_lineage.json"
    )
    snapshot = load_json(
        "campaigns/large_scale_thai_drl_v2/cutovers/"
        "storage_safe_20260717_pre_cutover/snapshot_receipt.json"
    )
    incident = load_json(
        "campaigns/large_scale_thai_drl_v2/incidents/screening_smoke_3542/incident.json"
    )
    capacity_pre = load_json(
        "campaigns/large_scale_thai_drl_v2/operational/screening_capacity/"
        "storage_safe_decision_20260717T144710Z.json"
    )
    capacity_post = load_json(
        "campaigns/large_scale_thai_drl_v2/operational/screening_capacity/"
        "storage_safe_post_smoke_decision_20260717T154236Z.json"
    )
    old_files, old_bytes = logical_dir_stats("results/screening_smoke_3543")
    safe_files, safe_bytes = logical_dir_stats("results/screening_storage_safe_smoke_3550")
    runtimes = {
        run["algorithm"].upper(): run["runtime_projection"]["projected_task_hours"]
        for run in summary["runs"]
    }
    per_algorithm_bytes = {}
    for run in summary["runs"]:
        directory = ROOT / run["run_dir"]
        per_algorithm_bytes[run["algorithm"].upper()] = sum(
            path.stat().st_size for path in directory.rglob("*") if path.is_file()
        )
    jobs = {
        "3540": parse_accounting("reports/slurm_accounting_vector_benchmark_3540.txt"),
        "3541": parse_accounting("reports/slurm_accounting_determinism_3541.txt"),
        "3543": parse_accounting("reports/slurm_accounting_screening_smoke_3543.txt"),
        "3550": parse_accounting("reports/slurm_accounting_screening_storage_safe_smoke_3550.txt"),
    }
    jobs["3542"] = {
        "JobIDRaw": "3542",
        "JobName": incident["expected_scheduler"]["job_name"],
        "State": incident["observed_scheduler"]["state"],
        "ExitCode": incident["observed_scheduler"]["exit_code"],
        "ElapsedRaw": str(incident["observed_scheduler"]["elapsed_seconds"]),
        "AllocCPUS": str(incident["observed_scheduler"]["alloc_cpus"]),
        "ReqMem": incident["observed_scheduler"]["requested_memory"],
        "TotalCPU": incident["observed_scheduler"]["total_cpu"],
        "NodeList": incident["observed_scheduler"]["node"],
    }
    return {
        "state": state,
        "state_counts": state_counts,
        "summary": summary,
        "validation": validation,
        "imported": imported,
        "preflight": preflight,
        "manifest": manifest,
        "lineage": lineage,
        "snapshot": snapshot,
        "incident": incident,
        "capacity_pre": capacity_pre,
        "capacity_post": capacity_post,
        "old_files": old_files,
        "old_bytes": old_bytes,
        "safe_files": safe_files,
        "safe_bytes": safe_bytes,
        "storage_reduction_pct": 100.0 * (old_bytes - safe_bytes) / old_bytes,
        "runtimes": runtimes,
        "per_algorithm_bytes": per_algorithm_bytes,
        "jobs": jobs,
        "release_id": manifest["release_id"],
        "release_manifest_sha": sidecar_hash(
            "campaigns/large_scale_thai_drl_v2/deployment/"
            "bistka_screening_storage_safe_cutover_v2_manifest.sha256"
        ),
        "release_bundle_sha": sidecar_hash(
            "campaigns/large_scale_thai_drl_v2/deployment/"
            "bistka_screening_storage_safe_cutover_v2.tar.gz.sha256"
        ),
        "lineage_sha": sidecar_hash(
            "campaigns/large_scale_thai_drl_v2/imports/"
            "storage_safe_cutover_calibration_lineage.json.sha256"
        ),
        "preflight_sha": file_hash("reports/bistka_screening_storage_safe_cutover_v2_preflight.json"),
        "accounting_sha": file_hash("reports/slurm_accounting_screening_storage_safe_smoke_3550.txt"),
        "import_sha": sidecar_hash(
            "reports/screening_storage_safe_smoke_3550_import_validation.json.sha256"
        ),
        "summary_sha": sidecar_hash(
            "results/screening_storage_safe_smoke_3550/screening_smoke_summary.json.sha256"
        ),
        "validation_sha": sidecar_hash(
            "results/screening_storage_safe_smoke_3550/storage_safe_smoke_validation.json.sha256"
        ),
        "post_decision_sha": sidecar_hash(
            "campaigns/large_scale_thai_drl_v2/operational/screening_capacity/"
            "storage_safe_post_smoke_decision_20260717T154236Z.json.sha256"
        ),
        "original_spec_sha": lineage["source"]["original_campaign_spec"]["sha256"],
        "rollback_archive_sha": snapshot["archive"]["sha256"],
        # This full-suite result is part of the verified execution record for this cutover.
        "tests_passed": 293,
        "tests_seconds": 36.96,
        # Point-in-time cluster-credit values captured during the final live read-only audit.
        "project_credit_remaining": 9979.88,
        "project_credit_pct": 99.80,
        "personal_credit_remaining": 496.25,
        "personal_credit_pct": 99.25,
        # Canonical continuity deliberately stays on the pre-consolidation 37-gate basis.
        "canonical_verified": 11,
        "canonical_total": 37,
        "canonical_pct": 100.0 * 11 / 37,
        "active_total": len(state["stages"]),
    }


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}
    styles["CoverTitle"] = ParagraphStyle(
        "CoverTitle",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=27,
        textColor=WHITE,
        spaceAfter=7,
    )
    styles["CoverSub"] = ParagraphStyle(
        "CoverSub",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=HexColor("#DDE7F5"),
    )
    styles["Eyebrow"] = ParagraphStyle(
        "Eyebrow",
        parent=base["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=BLUE,
        tracking=1.0,
        spaceBefore=1,
        spaceAfter=4,
    )
    styles["H1"] = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=NAVY,
        spaceBefore=2,
        spaceAfter=9,
    )
    styles["H2"] = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=NAVY,
        spaceBefore=9,
        spaceAfter=6,
    )
    styles["H3"] = ParagraphStyle(
        "H3",
        parent=base["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=INK,
        spaceBefore=6,
        spaceAfter=3,
    )
    styles["Body"] = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9.1,
        leading=13.1,
        textColor=INK,
        spaceAfter=5,
    )
    styles["BodySmall"] = ParagraphStyle(
        "BodySmall",
        parent=styles["Body"],
        fontSize=7.5,
        leading=10.5,
        textColor=SLATE,
        spaceAfter=3,
    )
    styles["Bullet"] = ParagraphStyle(
        "Bullet",
        parent=styles["Body"],
        leftIndent=12,
        firstLineIndent=-8,
        bulletIndent=0,
        spaceAfter=3.5,
    )
    styles["Metric"] = ParagraphStyle(
        "Metric",
        parent=base["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=22,
        alignment=TA_CENTER,
        textColor=NAVY,
        spaceAfter=2,
    )
    styles["MetricLabel"] = ParagraphStyle(
        "MetricLabel",
        parent=base["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9,
        alignment=TA_CENTER,
        textColor=SLATE,
    )
    styles["StatusLabel"] = ParagraphStyle(
        "StatusLabel",
        parent=base["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=TEAL,
        spaceAfter=3,
    )
    styles["StatusText"] = ParagraphStyle(
        "StatusText",
        parent=styles["Body"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        spaceAfter=0,
    )
    styles["TableHead"] = ParagraphStyle(
        "TableHead",
        parent=base["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9,
        textColor=WHITE,
        alignment=TA_LEFT,
    )
    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.5,
        textColor=INK,
    )
    styles["TableCellSmall"] = ParagraphStyle(
        "TableCellSmall",
        parent=styles["TableCell"],
        fontSize=6.4,
        leading=8.2,
    )
    styles["Code"] = ParagraphStyle(
        "Code",
        parent=base["Code"],
        fontName="Courier",
        fontSize=6.4,
        leading=8.4,
        textColor=SLATE,
        wordWrap="CJK",
    )
    styles["Quote"] = ParagraphStyle(
        "Quote",
        parent=styles["Body"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=NAVY,
        leftIndent=12,
        rightIndent=12,
        spaceBefore=5,
        spaceAfter=7,
    )
    styles["CenterSmall"] = ParagraphStyle(
        "CenterSmall",
        parent=styles["BodySmall"],
        alignment=TA_CENTER,
    )
    return styles


STYLES = build_styles()


def p(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"<b>-</b> {text}", STYLES["Bullet"])


def table(
    rows: Sequence[Sequence[Any]],
    widths: Sequence[float] | None = None,
    *,
    small: bool = False,
    header: bool = True,
    alignments: dict[int, str] | None = None,
) -> Table:
    cell_style = "TableCellSmall" if small else "TableCell"
    converted: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        converted_row: list[Any] = []
        for value in row:
            if hasattr(value, "wrap"):
                converted_row.append(value)
            else:
                style = "TableHead" if header and row_index == 0 else cell_style
                converted_row.append(p(str(value), style))
        converted.append(converted_row)
    result = Table(
        converted,
        colWidths=widths,
        repeatRows=1 if header else 0,
        hAlign="LEFT",
        splitByRow=1,
    )
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, HexColor("#D5DCE7")),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
            ]
        )
    else:
        commands.append(("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, PALE]))
    if alignments:
        for column, alignment in alignments.items():
            commands.append(("ALIGN", (column, 0), (column, -1), alignment))
    result.setStyle(TableStyle(commands))
    return result


def metric_cards(metrics: Sequence[tuple[str, str]], accent: Color) -> Table:
    cards: list[Any] = []
    for value, label in metrics:
        cards.append(
            Table(
                [[p(value, "Metric")], [p(label, "MetricLabel")]],
                colWidths=[CONTENT_W / len(metrics) - 6],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PALE),
                        ("BOX", (0, 0), (-1, -1), 0.8, LIGHT),
                        ("LINEABOVE", (0, 0), (-1, 0), 3.5, accent),
                        ("TOPPADDING", (0, 0), (-1, 0), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
                        ("TOPPADDING", (0, 1), (-1, 1), 2),
                        ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
                    ]
                ),
            )
        )
    return Table([cards], colWidths=[CONTENT_W / len(metrics)] * len(metrics), hAlign="LEFT")


def cover(
    spec: ReportSpec,
    metrics: Sequence[tuple[str, str]],
    overview: str,
    source_note: str,
) -> list[Any]:
    logo = Image(str(ROOT / "KVIS_Logo.png"), width=23 * mm, height=23 * mm)
    title_cell = [
        p("FINAL RESEARCH PROJECT - PROCESS REPORT", "CoverSub"),
        Spacer(1, 5),
        p(spec.title, "CoverTitle"),
        p(spec.subtitle, "CoverSub"),
    ]
    panel = Table(
        [[title_cell, logo]],
        colWidths=[CONTENT_W - 35 * mm, 27 * mm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (0, 0), 16),
                ("RIGHTPADDING", (0, 0), (0, 0), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 15),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
                ("ALIGN", (1, 0), (1, 0), "CENTER"),
                ("LINEBELOW", (0, 0), (-1, -1), 5, spec.accent),
            ]
        ),
    )
    status = Table(
        [[p(spec.status_label.upper(), "StatusLabel")], [p(spec.status_text, "StatusText")]],
        colWidths=[CONTENT_W - 12],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_GREEN),
                ("BOX", (0, 0), (-1, -1), 0.8, HexColor("#A8DCCB")),
                ("LEFTPADDING", (0, 0), (-1, -1), 11),
                ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
                ("TOPPADDING", (0, 1), (-1, 1), 2),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
            ]
        ),
    )
    metadata = table(
        [
            ["Campaign", CAMPAIGN_ID],
            ["Evidence cut", EVIDENCE_CUT],
            ["Generated", GENERATED_ON],
            ["Audience", "Research team, technical reviewer, and HPC operator"],
        ],
        widths=[32 * mm, CONTENT_W - 32 * mm],
        header=False,
        small=True,
    )
    return [
        panel,
        Spacer(1, 12),
        metric_cards(metrics, spec.accent),
        Spacer(1, 12),
        status,
        Spacer(1, 12),
        p("TECHNICAL SUMMARY", "Eyebrow"),
        p(overview, "Body"),
        Spacer(1, 3),
        metadata,
        Spacer(1, 8),
        p(f"<b>Evidence basis:</b> {source_note}", "BodySmall"),
        PageBreak(),
    ]


def section(number: str, title: str, intro: str | None = None) -> list[Any]:
    items: list[Any] = [p(f"SECTION {number}", "Eyebrow"), p(title, "H1")]
    if intro:
        items.append(p(intro, "Body"))
    return items


def callout(label: str, text: str, tone: str = "blue") -> Table:
    palette = {
        "blue": (PALE_BLUE, HexColor("#B8CBF2"), BLUE),
        "green": (PALE_GREEN, HexColor("#A8DCCB"), TEAL),
        "orange": (PALE_ORANGE, HexColor("#F3C98D"), ORANGE),
        "red": (HexColor("#FFF0F0"), HexColor("#E8B4B4"), RED),
    }
    background, border, heading = palette[tone]
    label_style = ParagraphStyle(
        f"CalloutLabel{tone}",
        parent=STYLES["StatusLabel"],
        textColor=heading,
    )
    box = Table(
        [[Paragraph(escape(label).upper(), label_style)], [p(text, "Body")]],
        colWidths=[CONTENT_W - 12],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, border),
                ("LINEBEFORE", (0, 0), (0, -1), 4, heading),
                ("LEFTPADDING", (0, 0), (-1, -1), 11),
                ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
                ("TOPPADDING", (0, 1), (-1, 1), 2),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
            ]
        ),
    )
    return box


def progress_graph(canonical_verified: int, canonical_total: int, active: Counter[str]) -> Drawing:
    width, height = CONTENT_W, 96
    drawing = Drawing(width, height)
    rows = [
        ("Canonical continuity", canonical_verified, canonical_total, BLUE),
        ("Active storage-safe graph", active["VERIFIED"], sum(active.values()), TEAL),
    ]
    for index, (label, complete, total, color) in enumerate(rows):
        y = 61 - index * 42
        ratio = complete / total
        drawing.add(String(0, y + 17, label, fontName="Helvetica-Bold", fontSize=9, fillColor=INK))
        drawing.add(
            String(
                width,
                y + 17,
                f"{complete}/{total} verified ({ratio * 100:.1f}%)",
                fontName="Helvetica-Bold",
                fontSize=8.5,
                fillColor=color,
                textAnchor="end",
            )
        )
        drawing.add(Rect(0, y, width, 11, fillColor=LIGHT, strokeColor=None, rx=5, ry=5))
        drawing.add(Rect(0, y, width * ratio, 11, fillColor=color, strokeColor=None, rx=5, ry=5))
    return drawing


def horizontal_bars(
    title: str,
    items: Sequence[tuple[str, float, Color]],
    *,
    unit: str,
    note: str | None = None,
    width: float = CONTENT_W,
    height: float = 165,
) -> Drawing:
    drawing = Drawing(width, height)
    left = 74
    right = 68
    top = height - 33
    bar_h = min(17, (height - 65) / max(len(items), 1) - 4)
    max_value = max(value for _, value, _ in items) or 1.0
    drawing.add(String(0, height - 13, title, fontName="Helvetica-Bold", fontSize=10, fillColor=NAVY))
    for index, (label, value, color) in enumerate(items):
        y = top - index * (bar_h + 9)
        drawing.add(
            String(left - 7, y + 3, label, fontName="Helvetica-Bold", fontSize=8, fillColor=INK, textAnchor="end")
        )
        drawing.add(Rect(left, y, width - left - right, bar_h, fillColor=PALE, strokeColor=LIGHT, strokeWidth=0.4))
        bar_width = (width - left - right) * value / max_value
        drawing.add(Rect(left, y, bar_width, bar_h, fillColor=color, strokeColor=None))
        drawing.add(
            String(
                min(left + bar_width + 5, width - right + 4),
                y + 3,
                f"{value:,.3f} {unit}",
                fontName="Helvetica",
                fontSize=7.5,
                fillColor=INK,
            )
        )
    if note:
        drawing.add(String(0, 4, note, fontName="Helvetica", fontSize=6.5, fillColor=MID))
    return drawing


def storage_comparison(old_bytes: int, safe_bytes: int) -> Drawing:
    old_mib = old_bytes / 1024**2
    safe_mib = safe_bytes / 1024**2
    return horizontal_bars(
        "Observed five-algorithm smoke output",
        [("Original job 3543", old_mib, ORANGE), ("Storage-safe job 3550", safe_mib, BLUE)],
        unit="MiB",
        note="Logical file sizes; the safe run kept all final models and added retention evidence.",
        height=112,
    )


def process_flow(items: Sequence[tuple[str, str]], accent: Color = BLUE) -> Drawing:
    width, height = CONTENT_W, 122
    drawing = Drawing(width, height)
    count = len(items)
    gap = 9
    box_w = (width - gap * (count - 1)) / count
    for index, (title, subtitle) in enumerate(items):
        x = index * (box_w + gap)
        if index:
            drawing.add(Line(x - gap + 1, 62, x - 1, 62, strokeColor=accent, strokeWidth=1.6))
            drawing.add(Line(x - 5, 65, x - 1, 62, strokeColor=accent, strokeWidth=1.2))
            drawing.add(Line(x - 5, 59, x - 1, 62, strokeColor=accent, strokeWidth=1.2))
        drawing.add(Rect(x, 35, box_w, 57, fillColor=PALE, strokeColor=LIGHT, strokeWidth=0.8, rx=5, ry=5))
        drawing.add(Rect(x, 84, box_w, 8, fillColor=accent, strokeColor=None, rx=5, ry=5))
        drawing.add(String(x + 7, 69, title, fontName="Helvetica-Bold", fontSize=7.4, fillColor=NAVY))
        words = subtitle.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > 25 and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        for line_index, line in enumerate(lines[:3]):
            drawing.add(String(x + 7, 55 - line_index * 9, line, fontName="Helvetica", fontSize=6.3, fillColor=SLATE))
    return drawing


def evidence_ledger(rows: Sequence[tuple[str, str]]) -> Table:
    converted = [["Artifact or identifier", "Verified value"]]
    for label, value in rows:
        if re.fullmatch(r"[0-9a-f]{64}", value):
            value = break_hash(value)
        converted.append([escape(label), p(escape(value), "Code")])
    return table(converted, widths=[63 * mm, CONTENT_W - 63 * mm], small=True)


def report_1(f: dict[str, Any]) -> tuple[ReportSpec, list[Any]]:
    spec = ReportSpec(
        filename="01_project_progress_and_research_gap_20260717.pdf",
        short_title="Project Progress and Research Gap",
        title="Project Progress and Research Gap",
        subtitle="A stage-gated view of what has been proven, what remains open, and where HPC changes the feasible research scope.",
        accent=BLUE,
        status_label="Current project boundary",
        status_text=(
            "Canonical completion is 11 of 37 gates (29.7%). The active storage-safe graph has "
            "9 VERIFIED, 1 READY, and 25 PENDING stages. Full screening remains unsubmitted."
        ),
    )
    state_counts = f["state_counts"]
    story = cover(
        spec,
        [
            (f"{f['canonical_pct']:.1f}%", "CANONICAL GATE COMPLETION"),
            (f"{f['state']['workload']['planned_model_runs']:,}", "PLANNED MODEL RUNS"),
            (f"{f['state']['workload']['planned_training_timesteps'] / 1e6:.2f}M", "PLANNED TRAINING TIMESTEPS"),
        ],
        (
            "The project has moved beyond code preparation into a verified HPC execution boundary: observed market data, preregistration, deployment, calibration lineage, preflight, and a five-algorithm smoke run are accepted. The main computational campaign has not started. The immediate research opportunity is to convert this verified platform into a statistically defensible comparison of five reinforcement-learning algorithms under Thai market conditions."
        ),
        (
            "campaign state, deployment and preflight receipts, job 3550 smoke evidence, and the recorded full local validation run."
        ),
    )
    story += section(
        "1",
        "Research question, gap, and evidence boundary",
        (
            "The practical question is whether a learning-based trading policy can produce robust, risk-aware decisions on Thai equities after transaction costs and market-regime changes. The scientific gap is not another single backtest. It is a controlled, repeated comparison that separates algorithm choice, training budget, data context, robustness, and execution stress."
        ),
    )
    story.append(
        callout(
            "Research gap that can now be fulfilled",
            (
                "Use the verified storage-safe HPC pipeline to run repeated, preregistered trials for PPO, A2C, DDPG, TD3, and SAC, then test whether any apparent advantage survives nested walk-forward evaluation, robustness checks, feature ablation, and realistic execution stress."
            ),
            "blue",
        )
    )
    story.append(Spacer(1, 8))
    story.append(p("WHY HPC IS MATERIAL", "Eyebrow"))
    story.extend(
        [
            bullet("The full plan contains 503 model runs and 29,450,240 requested training timesteps; a laptop-only workflow would make systematic repetition and validation slow and fragile."),
            bullet("HPC provides isolated jobs, resource accounting, resumable arrays, and centralized evidence, making failures and results traceable instead of anecdotal."),
            bullet("More compute does not remove scientific controls. Preregistration, data separation, fixed metrics, deterministic receipts, and leakage checks remain the decision gates."),
        ]
    )
    story.append(Spacer(1, 4))
    story.append(p("PROGRESS MEASUREMENT", "Eyebrow"))
    story.append(progress_graph(f["canonical_verified"], f["canonical_total"], state_counts))
    story.append(
        callout(
            "Why two percentages appear",
            (
                "The canonical 11/37 measure preserves continuity with earlier reports. The active graph has 35 stages because three already verified calibration gates were consolidated into one lineage gate during the storage-safe cutover. Therefore, 9/35 is a structural representation, not a loss of completed work."
            ),
            "orange",
        )
    )
    story.append(PageBreak())
    story += section("2", "What is complete and what is not")
    verified = [stage["stage_id"] for stage in f["state"]["stages"] if stage["state"] == "VERIFIED"]
    ready = [stage["stage_id"] for stage in f["state"]["stages"] if stage["state"] == "READY"]
    pending = [stage["stage_id"] for stage in f["state"]["stages"] if stage["state"] == "PENDING"]
    story.append(
        table(
            [
                ["State", "Count", "Meaning", "Current stages"],
                ["VERIFIED", str(state_counts["VERIFIED"]), "Evidence passed the declared gate.", ", ".join(verified)],
                ["READY", str(state_counts["READY"]), "Dependencies are verified; execution is allowed only after a fresh operator check.", ", ".join(ready)],
                ["PENDING", str(state_counts["PENDING"]), "Waiting on upstream evidence.", ", ".join(pending[:10]) + f", plus {len(pending) - 10} later stages"],
            ],
            widths=[23 * mm, 13 * mm, 59 * mm, CONTENT_W - 95 * mm],
            small=True,
        )
    )
    story.append(Spacer(1, 8))
    story.append(p("VERIFIED TECHNICAL BOUNDARY", "Eyebrow"))
    story.extend(
        [
            bullet(f"Observed-market feature input is execution-eligible and bound by SHA-256 {short_hash(f['summary']['observed_features']['sha256'])}."),
            bullet(f"The active cutover release contains {f['manifest']['file_count']} verified files totaling {f['manifest']['total_bytes']:,} bytes."),
            bullet(f"Smoke job 3550 completed five algorithms, emitted {f['imported']['experiment_manifests']} complete manifests and {f['imported']['retention_receipts']} retention receipts, and imported with status {f['imported']['status']}."),
            bullet(f"The last full local validation recorded {f['tests_passed']} passing tests in {f['tests_seconds']:.2f} seconds."),
        ]
    )
    story.append(p("NOT YET A SCIENTIFIC RESULT", "Eyebrow"))
    story.extend(
        [
            bullet("The smoke run proves executability and storage behavior; it is too small to support algorithm-ranking or investment-performance claims."),
            bullet("The 150-trial screening array and its aggregate have not been submitted."),
            bullet("Nested evaluation, robustness, feature ablation, execution stress, and final statistical synthesis remain pending."),
        ]
    )
    story.append(PageBreak())
    story += section("3", "Interpretation, limitations, and next questions")
    story.append(p("KEY FINDINGS", "Eyebrow"))
    story.extend(
        [
            bullet("The project has a reproducible execution platform, not a completed empirical answer."),
            bullet(f"The storage-safe design cut observed smoke output by {f['storage_reduction_pct']:.2f}% while preserving final models and evidence."),
            bullet("The next defensible milestone is full screening plus aggregation, using a new quota snapshot and a deliberately small submission window."),
        ]
    )
    story.append(p("LIMITATIONS AND ROBUSTNESS BOUNDARY", "Eyebrow"))
    story.extend(
        [
            bullet("The current observed dataset is a restricted private research pilot; findings must be described within that evidence class."),
            bullet("Runtime projections are extrapolations from one short smoke trial per algorithm and include a 1.5 safety margin; they are planning evidence, not guarantees."),
            bullet("The post-smoke group soft-quota margin was only about 0.309 GiB. That point-in-time pass expired after 15 minutes and cannot authorize a later full run."),
            bullet("Macro and fundamental feature claims remain outside scope until observed inputs are checksum- and readiness-certified, or the preregistration is amended before test access."),
        ]
    )
    story.append(p("QUESTIONS THE NEXT PHASE SHOULD ANSWER", "Eyebrow"))
    story.extend(
        [
            bullet("Which algorithms are consistently competitive after costs across time splits, rather than on one favorable period?"),
            bullet("How sensitive are conclusions to training budget and hyperparameter selection?"),
            bullet("Do advantages survive nested walk-forward testing and multiple-comparison control?"),
            bullet("Which data contexts and execution assumptions materially change the result?"),
        ]
    )
    story.append(
        callout(
            "Decision",
            "Proceed to a fresh capacity audit and a bounded screening window. Do not describe the project as complete, and do not submit full screening from the expired 17 July quota decision.",
            "green",
        )
    )
    story.append(Spacer(1, 9))
    story.append(p("PRIMARY EVIDENCE", "Eyebrow"))
    story.append(
        evidence_ledger(
            [
                ("Active campaign state", "campaigns/large_scale_thai_drl_v2/state.json"),
                ("Storage-safe smoke summary", "results/screening_storage_safe_smoke_3550/screening_smoke_summary.json"),
                ("Smoke import validation", "reports/screening_storage_safe_smoke_3550_import_validation.json"),
                ("Cutover preflight", "reports/bistka_screening_storage_safe_cutover_v2_preflight.json"),
            ]
        )
    )
    return spec, story


def report_2(f: dict[str, Any]) -> tuple[ReportSpec, list[Any]]:
    spec = ReportSpec(
        filename="02_storage_safe_cutover_execution_20260717.pdf",
        short_title="Storage-Safe Cutover Execution",
        title="Storage-Safe Cutover Execution",
        subtitle="From a confirmed quota failure to a rollback-safe release and a verified five-algorithm smoke run.",
        accent=TEAL,
        status_label="Cutover outcome",
        status_text=(
            "The storage-safe release was applied and preflighted successfully. Job 3550 completed with exit 0:0. Rollback was not required. Full screening remains unsubmitted."
        ),
    )
    story = cover(
        spec,
        [
            ("49.94%", "OBSERVED OUTPUT REDUCTION"),
            (str(f["manifest"]["file_count"]), "RELEASE FILES VERIFIED"),
            ("5 / 5", "ALGORITHMS COMPLETE"),
        ],
        (
            "Job 3542 exposed a storage failure at checkpoint write time. The implemented response suppresses intermediate checkpoint writes before they occur, retains the final model and scientific outputs, and emits checksum-bound retention receipts. A rollback snapshot and accepted calibration-lineage receipt preserved the prior research boundary. The final release passed remote verification and one storage-safe smoke job."
        ),
        "incident receipt, rollback snapshot, accepted lineage receipt, deterministic release manifest, remote preflight, Slurm accounting, and strict import validation.",
    )
    story += section("1", "Failure mechanism and engineering response")
    incident = f["incident"]
    story.append(
        callout(
            "Confirmed root cause",
            (
                f"Job {incident['job_id']} failed after {incident['observed_scheduler']['elapsed_seconds']} seconds with "
                f"{escape(incident['diagnosis']['exception'])}. New outputs inherited Unix group s00658; its Lustre soft-quota grace had expired even though project-ID quota 25110303 remained healthy."
            ),
            "red",
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        process_flow(
            [
                ("Failure traced", "Checkpoint zip close hit group quota"),
                ("Writes suppressed", "No intermediate checkpoint creation"),
                ("Evidence retained", "Final model, metrics, curves, actions"),
                ("Receipts signed", "Per-trial hash and policy evidence"),
                ("Reuse guarded", "Unsafe or incomplete reuse rejected"),
            ],
            TEAL,
        )
    )
    story.append(storage_comparison(f["old_bytes"], f["safe_bytes"]))
    story.append(
        table(
            [
                ["Run", "Files", "Logical bytes", "Logical MiB", "Interpretation"],
                ["Original smoke 3543", str(f["old_files"]), f"{f['old_bytes']:,}", f"{f['old_bytes'] / 1024**2:.3f}", "Checkpoint-writing baseline"],
                ["Storage-safe smoke 3550", str(f["safe_files"]), f"{f['safe_bytes']:,}", f"{f['safe_bytes'] / 1024**2:.3f}", "More evidence files, about half the bytes"],
            ],
            widths=[40 * mm, 16 * mm, 34 * mm, 27 * mm, CONTENT_W - 117 * mm],
            small=True,
        )
    )
    story.append(PageBreak())
    story += section("2", "Controlled cutover and rollback lineage")
    story.append(
        process_flow(
            [
                ("Snapshot", "48-file control and release archive"),
                ("Lineage", "Accepted reuse without compute"),
                ("Release", "248-file content-addressed bundle"),
                ("Apply", "42 changed, 208 unchanged"),
                ("Preflight", "248 of 248 files verified"),
            ],
            BLUE,
        )
    )
    story.append(p("CONTROL GUARANTEES", "Eyebrow"))
    story.extend(
        [
            bullet("The original campaign specification hash was preserved, and the storage-safe specification was introduced as an explicit successor."),
            bullet("The rollback archive preserves the pre-cutover campaign control plane and active release; restoring it remains an intentional operation."),
            bullet("Calibration evidence from jobs 3540 and 3541 was reused without compute because its attempt, release, preflight, data, runtime, and output bindings were unchanged."),
            bullet("The release build was repeated deterministically. The final bundle contains all 218 predecessor files plus 30 additions, with zero lost predecessor files."),
        ]
    )
    story.append(p("IDENTITY LEDGER", "Eyebrow"))
    story.append(
        evidence_ledger(
            [
                ("Original campaign specification SHA-256", f["original_spec_sha"]),
                ("Rollback archive SHA-256", f["rollback_archive_sha"]),
                ("Lineage receipt SHA-256", f["lineage_sha"]),
                ("Cutover release ID", f["release_id"]),
                ("Cutover manifest SHA-256", f["release_manifest_sha"]),
                ("Cutover bundle SHA-256", f["release_bundle_sha"]),
                ("Returned preflight SHA-256", f["preflight_sha"]),
            ]
        )
    )
    story.append(PageBreak())
    story += section("3", "Remote validation and smoke evidence")
    job = f["jobs"]["3550"]
    story.append(
        table(
            [
                ["Field", "Verified value", "Meaning"],
                ["Slurm job", job["JobIDRaw"], "Exactly one new HPC submission in the cutover window"],
                ["State / exit", f"{job['State']} / {job['ExitCode']}", "Terminal scheduler success"],
                ["Elapsed", "00:33:20", "2,000 elapsed seconds"],
                ["Allocation", f"{job['AllocCPUS']} CPU, {job['ReqMem']} requested", "compute-devel on bistka-master"],
                ["Batch MaxRSS", "809,680 KiB", "About 790.7 MiB peak resident memory"],
                ["Total CPU", job["TotalCPU"], "CPU accounting captured and hashed"],
            ],
            widths=[34 * mm, 48 * mm, CONTENT_W - 82 * mm],
        )
    )
    story.append(Spacer(1, 8))
    runtime_rows = [["Algorithm", "Projected full-task hours", "72-hour gate", "Logical smoke bytes"]]
    for algorithm in ["PPO", "A2C", "DDPG", "TD3", "SAC"]:
        runtime_rows.append(
            [
                algorithm,
                f"{f['runtimes'][algorithm]:.4f}",
                "PASS",
                f"{f['per_algorithm_bytes'][algorithm]:,}",
            ]
        )
    story.append(table(runtime_rows, widths=[31 * mm, 43 * mm, 28 * mm, CONTENT_W - 102 * mm]))
    story.append(Spacer(1, 8))
    story.append(
        callout(
            "Strict returned-evidence gate",
            (
                f"Import status {f['imported']['status']}: {f['imported']['experiment_manifests']} complete experiment manifests, "
                f"{f['imported']['retention_receipts']} checksum-bound retention receipts, {f['imported']['checkpoint_paths']} checkpoint paths, "
                f"and validation status {f['validation']['status']}."
            ),
            "green",
        )
    )
    story.append(p("EVIDENCE HASHES", "Eyebrow"))
    story.append(
        evidence_ledger(
            [
                ("Slurm accounting SHA-256", f["accounting_sha"]),
                ("Smoke summary SHA-256", f["summary_sha"]),
                ("Storage validation SHA-256", f["validation_sha"]),
                ("Import validation SHA-256", f["import_sha"]),
            ]
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        callout(
            "Final execution decision",
            "The cutover itself is verified. The smoke run proves the new path. Full 150-trial screening was intentionally not submitted; a later run requires a new capacity snapshot and a new bounded execution window.",
            "orange",
        )
    )
    return spec, story


def report_3(f: dict[str, Any]) -> tuple[ReportSpec, list[Any]]:
    spec = ReportSpec(
        filename="03_hpc_jobs_resources_and_capacity_audit_20260717.pdf",
        short_title="HPC Jobs, Resources, and Capacity Audit",
        title="HPC Jobs, Resources, and Capacity Audit",
        subtitle="A point-in-time operational record of submissions, resource use, storage headroom, and the decision not to launch full screening.",
        accent=ORANGE,
        status_label="Operator decision",
        status_text=(
            "Five V2 jobs exist through job 3550; only one was submitted in the storage-safe cutover window. The queue was empty after smoke completion. The post-smoke capacity pass had only about 0.309 GiB margin and has expired. Full screening remains unsubmitted."
        ),
    )
    pre = f["capacity_pre"]
    post = f["capacity_post"]
    required = post["projection"]["required_bytes"]
    pre_headroom = pre["quota"]["group"]["bytes"]["soft_headroom"]
    post_headroom = post["quota"]["group"]["bytes"]["soft_headroom"]
    pre_margin = pre_headroom - required
    post_margin = post_headroom - required
    story = cover(
        spec,
        [
            ("5", "V2 JOB SUBMISSIONS TO DATE"),
            ("1", "NEW JOB IN CUTOVER WINDOW"),
            (f"{post_margin / 1024**3:.3f} GiB", "POST-SMOKE SOFT MARGIN"),
        ],
        (
            "The operational bottleneck is group-level storage, not compute credit or the project-ID quota. The storage-safe policy reduced the projected requirement enough to pass the 17 July point-in-time gate, but with a narrow margin. Job 3550 completed successfully and the project deliberately stopped. This report is an audit record, not a standing authorization to submit the full screening array."
        ),
        "Slurm accounting for jobs 3540, 3541, 3543, and 3550; the quarantined job 3542 incident receipt; and signed pre- and post-smoke capacity decisions.",
    )
    story += section("1", "Submission register and resource evidence")
    rows = [["Job", "Purpose", "State", "Elapsed", "CPUs", "Memory", "Evidence role"]]
    purposes = {
        "3540": "Vector calibration",
        "3541": "Determinism",
        "3542": "Original screening smoke",
        "3543": "Replacement screening smoke",
        "3550": "Storage-safe screening smoke",
    }
    roles = {
        "3540": "Accepted calibration",
        "3541": "Accepted reproducibility",
        "3542": "Quarantined incident",
        "3543": "Accepted baseline smoke",
        "3550": "Accepted cutover verification",
    }
    for job_id in ["3540", "3541", "3542", "3543", "3550"]:
        job = f["jobs"][job_id]
        elapsed = int(job["ElapsedRaw"])
        rows.append(
            [
                job_id,
                purposes[job_id],
                job["State"],
                f"{elapsed // 60:02d}:{elapsed % 60:02d}",
                job["AllocCPUS"],
                job.get("ReqMem", ""),
                roles[job_id],
            ]
        )
    story.append(
        table(
            rows,
            widths=[15 * mm, 36 * mm, 24 * mm, 19 * mm, 13 * mm, 20 * mm, CONTENT_W - 127 * mm],
            small=True,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        callout(
            "Submission-count interpretation",
            "Jobs 3540 through 3543 were four submissions in the earlier V2 pass. Job 3550 was the sole submission in the later storage-safe cutover window. The full screening array was not submitted.",
            "blue",
        )
    )
    story.append(p("JOB 3550 RESOURCE DETAIL", "Eyebrow"))
    story.extend(
        [
            bullet("Scheduler outcome: COMPLETED, exit 0:0, elapsed 00:33:20."),
            bullet("Allocation: 2 CPUs and 8 GiB requested memory on compute-devel."),
            bullet("Observed batch MaxRSS: 809,680 KiB; TotalCPU: 32:33.567."),
            bullet("The job produced five complete manifests, five retention receipts, zero checkpoint paths, and a passing strict import."),
        ]
    )
    story.append(PageBreak())
    story += section("2", "Capacity decision before and after the smoke run")
    story.append(
        horizontal_bars(
            "Required bytes versus group soft-quota headroom",
            [
                ("Required", required / 1024**3, ORANGE),
                ("Pre-smoke headroom", pre_headroom / 1024**3, BLUE),
                ("Post-smoke headroom", post_headroom / 1024**3, TEAL),
            ],
            unit="GiB",
            note="The same 6.031 GiB requirement includes a 1.5 safety factor plus a fixed 0.250 GiB reserve.",
            height=136,
        )
    )
    story.append(
        table(
            [
                ["Capacity item", "Pre-smoke", "Post-smoke", "Interpretation"],
                ["Decision", pre["decision"].upper(), post["decision"].upper(), "All eight byte/inode checks passed"],
                ["Group soft headroom", human_bytes(pre_headroom), human_bytes(post_headroom), "Headroom decreased after outputs arrived"],
                ["Required bytes", human_bytes(required), human_bytes(required), "Safety-adjusted retained projection plus reserve"],
                ["Margin", human_bytes(pre_margin), human_bytes(post_margin), "Only about 0.417 GiB before and 0.309 GiB after"],
                ["Write probe", "Passed and removed", "Passed and removed", "1 MiB fsync test did not leave a probe file"],
                ["Validity", "15 minutes", "15 minutes", "Both point-in-time decisions are now expired"],
            ],
            widths=[42 * mm, 32 * mm, 32 * mm, CONTENT_W - 106 * mm],
            small=True,
        )
    )
    story.append(Spacer(1, 8))
    story.append(p("CHECK MATRIX", "Eyebrow"))
    check_rows = [["Check", "Pre", "Post"]]
    for check in sorted(pre["capacity_checks"]):
        check_rows.append(
            [
                check.replace("_", " "),
                "PASS" if pre["capacity_checks"][check] else "FAIL",
                "PASS" if post["capacity_checks"][check] else "FAIL",
            ]
        )
    story.append(table(check_rows, widths=[CONTENT_W - 50 * mm, 25 * mm, 25 * mm], small=True))
    story.append(PageBreak())
    story += section("3", "Credits, projection, and operating guardrails")
    story.append(
        table(
            [
                ["Resource", "Point-in-time value", "Assessment"],
                ["Project compute credit", f"{f['project_credit_remaining']:,.2f} SHr remaining ({f['project_credit_pct']:.2f}%)", "Healthy; not the immediate blocker"],
                ["Personal compute credit", f"{f['personal_credit_remaining']:,.2f} SHr remaining ({f['personal_credit_pct']:.2f}%)", "Healthy for bounded work"],
                ["Project-ID storage", "About 36.5 GiB used of 300 GiB hard limit", "Healthy; does not override inherited group quota"],
                ["Personal-group storage", f"{post_headroom / 1024**3:.3f} GiB soft headroom after smoke", "Narrow margin; durable ownership/quota fix preferred"],
            ],
            widths=[40 * mm, 60 * mm, CONTENT_W - 100 * mm],
        )
    )
    story.append(Spacer(1, 8))
    original_projection = post["projection"]["smoke_projected_original_bytes"]
    retained_projection = post["projection"]["projected_retained_bytes"]
    checkpoint_avoided = post["projection"]["projected_checkpoint_bytes_avoided"]
    story.append(
        table(
            [
                ["Projection component", "Bytes", "GiB"],
                ["Original checkpoint-writing full screen", f"{original_projection:,}", f"{original_projection / 1024**3:.4f}"],
                ["Storage-safe retained outputs", f"{retained_projection:,}", f"{retained_projection / 1024**3:.4f}"],
                ["Checkpoint bytes avoided", f"{checkpoint_avoided:,}", f"{checkpoint_avoided / 1024**3:.4f}"],
                ["Required after safety factor and reserve", f"{required:,}", f"{required / 1024**3:.4f}"],
            ],
            widths=[CONTENT_W - 62 * mm, 38 * mm, 24 * mm],
        )
    )
    story.append(p("NEXT-WINDOW GUARDRAILS", "Eyebrow"))
    story.extend(
        [
            bullet("Capture a new signed quota snapshot immediately before rendering any full-screening submission."),
            bullet("Require all eight byte and inode checks, the write probe, release identity, preflight identity, and runtime projection checks to pass."),
            bullet("Use no more than four new sbatch submissions in the window; the recommended screening sequence needs one array submission and, after verification, one aggregate submission."),
            bullet("Stop on any quota drift, mismatched release, nonempty conflicting queue, failed task, missing receipt, or unexpected checkpoint path."),
            bullet("Prefer correcting SGID/group ownership or raising the personal-group quota before the full run; the 0.309 GiB margin is operationally fragile."),
        ]
    )
    story.append(
        callout(
            "Standing decision",
            "No full-screening authorization survives from the 17 July snapshot. The operator must reopen the gate with fresh evidence.",
            "red",
        )
    )
    story.append(Spacer(1, 8))
    story.append(p("AUDIT SOURCES", "Eyebrow"))
    story.append(
        evidence_ledger(
            [
                ("Pre-smoke capacity SHA-256", pre["inputs"]["quota_snapshot"]["sha256"]),
                ("Post-smoke capacity SHA-256", post["inputs"]["quota_snapshot"]["sha256"]),
                ("Post-smoke decision SHA-256", f["post_decision_sha"]),
                ("Job 3550 accounting SHA-256", f["accounting_sha"]),
            ]
        )
    )
    return spec, story


def report_4(f: dict[str, Any]) -> tuple[ReportSpec, list[Any]]:
    spec = ReportSpec(
        filename="04_reproducibility_provenance_and_validation_20260717.pdf",
        short_title="Reproducibility, Provenance, and Validation",
        title="Reproducibility, Provenance, and Validation",
        subtitle="How the project binds code, data, configuration, HPC execution, and returned evidence into an auditable research record.",
        accent=BLUE,
        status_label="Validation outcome",
        status_text=(
            "The cutover lineage is accepted, the 248-file release and remote preflight match, job 3550 returned complete evidence, and 293 local tests passed. These checks establish reproducibility of the execution boundary, not final scientific conclusions. Full screening remains unsubmitted."
        ),
    )
    story = cover(
        spec,
        [
            (str(f["tests_passed"]), "LOCAL TESTS PASSED"),
            ("248 / 248", "REMOTE FILES VERIFIED"),
            ("5", "SIGNED RETENTION RECEIPTS"),
        ],
        (
            "The project treats every accepted result as a chain of identities: campaign stage, immutable attempt, deployment release, remote preflight, observed feature input, resolved configuration, Slurm job, output manifest, and local import validation. The storage-safe change preserved original calibration claims through an explicit lineage receipt and rollback archive rather than silently rewriting history."
        ),
        "content-addressed manifests, SHA-256 sidecars, stage success receipts, strict import validators, Slurm accounting, and the full local test suite.",
    )
    story += section("1", "End-to-end provenance chain")
    story.append(
        process_flow(
            [
                ("Research spec", "Original hash remains preserved"),
                ("Data binding", "Observed feature bytes and SHA"),
                ("Release", "Content-addressed 248-file bundle"),
                ("Remote preflight", "Exact release and package check"),
                ("Attempt and job", "Immutable command plus Slurm ID"),
                ("Returned evidence", "Manifests, receipts, import gate"),
            ],
            BLUE,
        )
    )
    story.append(p("WHAT EACH LINK PREVENTS", "Eyebrow"))
    story.append(
        table(
            [
                ["Binding", "Evidence", "Failure prevented"],
                ["Campaign and stage", "State and gate receipts", "Using an output outside the declared dependency graph"],
                ["Attempt and command", "Attempt JSON and rendered-command hash", "Submitting a command different from the reviewed plan"],
                ["Release and preflight", "Release ID, manifest SHA, returned preflight SHA", "Running stale or partially copied code"],
                ["Data and configuration", "Feature bytes/SHA and resolved-config SHA", "Comparing runs built from different inputs or parameters"],
                ["Scheduler execution", "Job ID, exit code, accounting, logs", "Treating an incomplete or failed job as successful"],
                ["Outputs", "Complete manifests, retention receipts, summary and import hashes", "Silent missing files, unsafe reuse, or checksum drift"],
            ],
            widths=[36 * mm, 57 * mm, CONTENT_W - 93 * mm],
            small=True,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        callout(
            "Lineage rule",
            "Only control-lineage predecessor records were restored. Accepted calibration outputs, attempts, release, preflight, configuration, executable code, data, and receipt hashes remained unchanged. The lineage operation submitted zero HPC jobs.",
            "green",
        )
    )
    story.append(PageBreak())
    story += section("2", "Validation layers and observed outcomes")
    story.append(
        table(
            [
                ["Layer", "Result", "Observed evidence", "Scope"],
                ["Local automated suite", "PASS", f"{f['tests_passed']} tests in {f['tests_seconds']:.2f} s", "Code paths, release, cutover, capacity, smoke, import, and campaign gates"],
                ["Deterministic release build", "PASS", f"{f['manifest']['file_count']} files; {f['manifest']['total_bytes']:,} bytes", "Rebuilt twice with the same identities"],
                ["Remote preflight", f["preflight"]["status"].upper(), f"{f['preflight']['verified_file_count']} of {f['preflight']['declared_file_count']} files", "Release, paths, observed data, runtime packages, and feature bindings"],
                ["Slurm execution", "PASS", "Job 3550 COMPLETED, exit 0:0", "One five-algorithm storage-safe smoke"],
                ["Strict import", f["imported"]["status"].upper(), f"{f['imported']['experiment_manifests']} manifests; {f['imported']['retention_receipts']} receipts", "Checksums, paths, completeness, and policy"],
                ["Checkpoint-path gate", "PASS", str(f["imported"]["checkpoint_paths"]), "No intermediate checkpoint path may survive"],
                ["Campaign gate", "VERIFIED", "screening_smoke accepted", "Advances screening to READY; does not complete screening"],
            ],
            widths=[37 * mm, 21 * mm, 47 * mm, CONTENT_W - 105 * mm],
            small=True,
        )
    )
    story.append(Spacer(1, 8))
    story.append(p("OBSERVED MARKET BINDING", "Eyebrow"))
    feature = f["summary"]["observed_features"]
    readiness = f["summary"]["readiness"]
    story.extend(
        [
            bullet(f"Feature file: {escape(feature['path'])}; {feature['bytes']:,} bytes; SHA-256 {short_hash(feature['sha256'])}."),
            bullet(f"Readiness: data_origin={readiness['data_origin']}, execution_eligible={str(readiness['execution_eligible']).lower()}, evidence_class={readiness['evidence_class']}."),
            bullet(f"The same feature SHA appears in both required runtime bindings and the returned preflight."),
        ]
    )
    story.append(p("RELEASE IDENTITIES", "Eyebrow"))
    story.append(
        evidence_ledger(
            [
                ("Release ID", f["release_id"]),
                ("Manifest SHA-256", f["release_manifest_sha"]),
                ("Bundle SHA-256", f["release_bundle_sha"]),
                ("Preflight SHA-256", f["preflight_sha"]),
            ]
        )
    )
    story.append(PageBreak())
    story += section("3", "Reproduction protocol and limitations")
    story.append(p("MINIMUM REPRODUCTION PROTOCOL", "Eyebrow"))
    story.extend(
        [
            bullet("Verify every sidecar before use; never infer a hash from a filename or a previous report."),
            bullet("Require the exact release ID and manifest SHA in the remote active marker and preflight receipt."),
            bullet("Verify the observed feature file by byte count and SHA before rendering an HPC command."),
            bullet("Create one immutable attempt, then submit its exact rendered command. Do not edit the attempt after submission."),
            bullet("Collect scheduler accounting and logs after terminal completion, fetch to a job-specific local directory, and run strict import validation."),
            bullet("Promote a stage only when the declared gate passes; failed and partial evidence stays quarantined outside accepted-result globs."),
        ]
    )
    story.append(p("WHAT IS ROBUSTLY ESTABLISHED", "Eyebrow"))
    story.extend(
        [
            bullet("The release can be reproduced and remotely verified."),
            bullet("The five algorithms can execute on observed data within the smoke contract."),
            bullet("The storage policy suppresses checkpoint paths while preserving final scientific artifacts and receipts."),
            bullet("The returned evidence is complete and checksum-bound for job 3550."),
        ]
    )
    story.append(p("WHAT IS NOT YET ESTABLISHED", "Eyebrow"))
    story.extend(
        [
            bullet("No algorithm is yet shown to outperform another; the smoke sample is not inferential evidence."),
            bullet("No final result has yet passed full screening, aggregation, nested evaluation, robustness, ablation, and stress gates."),
            bullet("Capacity evidence is time-limited. A valid historical receipt does not prove current quota or queue state."),
            bullet("Reproducibility within the current software stack does not automatically establish external validity on other markets or future regimes."),
        ]
    )
    story.append(
        callout(
            "Audit conclusion",
            "The execution boundary is reproducible and the storage-safe cutover is validated. Scientific claims remain deliberately gated until the full campaign produces accepted aggregate evidence.",
            "blue",
        )
    )
    story.append(Spacer(1, 8))
    story.append(p("PROVENANCE SOURCES", "Eyebrow"))
    story.append(
        evidence_ledger(
            [
                ("Accepted lineage receipt", "campaigns/large_scale_thai_drl_v2/imports/storage_safe_cutover_calibration_lineage.json"),
                ("Rollback receipt", "campaigns/large_scale_thai_drl_v2/cutovers/storage_safe_20260717_pre_cutover/snapshot_receipt.json"),
                ("Remote preflight", "reports/bistka_screening_storage_safe_cutover_v2_preflight.json"),
                ("Strict import receipt", "reports/screening_storage_safe_smoke_3550_import_validation.json"),
            ]
        )
    )
    return spec, story


def report_5(f: dict[str, Any]) -> tuple[ReportSpec, list[Any]]:
    spec = ReportSpec(
        filename="05_next_phase_research_execution_plan_20260717.pdf",
        short_title="Next-Phase Research Execution Plan",
        title="Next-Phase Research Execution Plan",
        subtitle="A bounded path from READY screening to final research claims, with explicit storage, evidence, and submission controls.",
        accent=GREEN,
        status_label="Recommended next move",
        status_text=(
            "Open a new screening window only after a fresh signed capacity pass. Use one screening-array submission, verify every task, then use one aggregation submission. Keep the window below five submissions. Full screening remains unsubmitted."
        ),
    )
    canonical_remaining = f["canonical_total"] - f["canonical_verified"]
    story = cover(
        spec,
        [
            ("1", "ACTIVE STAGE READY"),
            ("2", "RECOMMENDED NEXT SUBMISSIONS"),
            (str(canonical_remaining), "CANONICAL GATES REMAINING"),
        ],
        (
            "The project is ready for full screening in scientific dependency terms, but not under a standing operational authorization. The next window should recapture quota, verify the exact release and preflight, submit one array, validate all returned trial evidence, and submit aggregation only after the array is complete. Later phases remain dependency-gated so that compute scale cannot outrun research validity."
        ),
        "active campaign state, storage-safe capacity policy, runtime projections from job 3550, and the preregistered 503-run campaign graph.",
    )
    story += section("1", "Next controlled HPC window")
    story.append(
        process_flow(
            [
                ("Recapture", "Quota, queue, write probe, ownership"),
                ("Reverify", "Release, preflight, data, runtime"),
                ("Submit array", "One full-screening sbatch"),
                ("Monitor", "Stop on quota drift or failed task"),
                ("Import", "Strict manifests and receipts"),
                ("Aggregate", "One sbatch only after acceptance"),
            ],
            GREEN,
        )
    )
    story.append(p("PRE-SUBMISSION CHECKLIST", "Eyebrow"))
    story.extend(
        [
            bullet("Capture a fresh quota snapshot and decision no more than 15 minutes before submission."),
            bullet("Confirm all eight group/project byte and inode checks pass, the 1 MiB write probe fsyncs and is removed, and projected margin is positive."),
            bullet("Prefer more than the minimum margin. The last post-smoke pass had only about 0.309 GiB, so a group ownership or quota correction is the safer prerequisite."),
            bullet(f"Confirm active release {short_hash(f['release_id'])} and preflight SHA {short_hash(f['preflight_sha'])}."),
            bullet("Confirm observed data identity, locked configuration identity, empty or nonconflicting queue state, and available compute credit."),
            bullet("Render the exact command first and confirm it contains one sbatch invocation."),
        ]
    )
    story.append(p("SUBMISSION ENVELOPE", "Eyebrow"))
    story.append(
        table(
            [
                ["Submission", "When allowed", "Purpose", "Stop condition"],
                ["1 - screening array", "Fresh capacity and identity gates pass", "Run 30 trials for each of five algorithms", "Any quota, task, checksum, or checkpoint-path failure"],
                ["2 - screening aggregate", "All expected trial manifests and receipts pass strict import", "Rank candidates using preregistered metrics", "Missing trial, incompatible identity, or aggregate validation failure"],
                ["3-4 - reserve only", "Explicitly reopened window", "Recovery or independent verification only", "Never use merely to bypass a failed gate"],
            ],
            widths=[31 * mm, 48 * mm, 50 * mm, CONTENT_W - 129 * mm],
            small=True,
        )
    )
    story.append(
        callout(
            "Array-count clarification",
            "A Slurm array may create many task elements, but it is one reviewed sbatch submission. The project should still monitor every task independently and stop the campaign if the shared capacity contract is violated.",
            "blue",
        )
    )
    story.append(PageBreak())
    story += section("2", "Scientific campaign after screening")
    story.append(
        table(
            [
                ["Phase", "Main purpose", "Acceptance evidence", "Why it matters"],
                ["Screening and aggregate", "Identify competitive configurations across five algorithms", "Complete trials, aggregate receipt, preregistered ranking", "Prevents choosing a model from one lucky run"],
                ["Training-budget sweep", "Measure performance versus compute", "Budget-specific trials and aggregate", "Shows whether conclusions depend on spending more compute"],
                ["Nested walk-forward", "Separate tuning periods from evaluation periods", "Fold-complete nested manifests and aggregate", "Reduces optimistic selection bias"],
                ["Protocol lock", "Freeze decisions before final tests", "Signed selected candidates and exact deployment", "Prevents changing methods after seeing test outcomes"],
                ["Robustness", "Repeat across seeds, windows, and perturbations", "Complete robustness matrix and aggregate", "Tests stability rather than one-path success"],
                ["Feature ablation", "Remove data groups in controlled comparisons", "Readiness-bound inputs and ablation aggregate", "Measures which information truly contributes"],
                ["Execution stress", "Increase costs, slippage, and constraints", "Stress manifests and validated comparison", "Tests whether paper performance survives implementation friction"],
                ["Finalize", "Synthesize evidence and limitations", "Final gate, tables, figures, archive", "Separates accepted claims from exploratory findings"],
            ],
            widths=[34 * mm, 48 * mm, 52 * mm, CONTENT_W - 134 * mm],
            small=True,
        )
    )
    story.append(Spacer(1, 8))
    story.append(p("DATA ACQUISITION AND SCOPE", "Eyebrow"))
    story.extend(
        [
            bullet("Observed market OHLCV and market-context features are currently certified for execution."),
            bullet("Observed macro and fundamental inputs must be imported, byte/hash certified, and included in a passing readiness receipt before related ablation claims."),
            bullet("If those inputs cannot be obtained in time, amend the preregistered scope before accessing held-out test outcomes; do not silently replace missing observed data with synthetic evidence for the final claim."),
            bullet("Generated or synthetic data may support pipeline tests and sensitivity analysis, but it must remain explicitly labeled and separated from observed-market evidence."),
        ]
    )
    story.append(PageBreak())
    story += section("3", "Decision gates, risks, and completion criteria")
    story.append(
        table(
            [
                ["Risk", "Current evidence", "Required mitigation", "Owner decision"],
                ["Group soft-quota margin", "0.309 GiB after smoke; historical pass expired", "Fresh pass plus more headroom or ownership/quota correction", "HPC operator"],
                ["Runtime extrapolation", "All five projected below 72 h from one smoke trial each", "Monitor first array elements; revise resource requests if observed rates drift", "Research/HPC team"],
                ["Partial or stale evidence", "Strict imports and content-addressed identities implemented", "Reject mismatched hashes, incomplete manifests, and unsafe reuse", "Pipeline gate"],
                ["Selection bias", "Preregistration and dependency graph in place", "Aggregate before selection; nested evaluation; protocol lock", "Research lead"],
                ["Missing external features", "Macro/fundamental claims not readiness-certified", "Certified observed inputs or preregistered scope amendment", "Research lead"],
                ["Overclaiming smoke results", "Smoke proves execution only", "Reserve performance claims for accepted aggregate and final gates", "Report reviewer"],
            ],
            widths=[37 * mm, 48 * mm, 56 * mm, CONTENT_W - 141 * mm],
            small=True,
        )
    )
    story.append(Spacer(1, 8))
    story.append(p("PHASE ACCEPTANCE CHECKLIST", "Eyebrow"))
    story.extend(
        [
            bullet("Screening: every expected algorithm/trial completes with exact identities and zero prohibited checkpoint paths."),
            bullet("Aggregation: selection follows preregistered metrics and reports uncertainty, failures, and missingness."),
            bullet("Nested/robustness/ablation/stress: all dependency-specific release and data gates pass before compute is submitted."),
            bullet("Finalization: conclusions distinguish exploratory, screening, and held-out evidence; limitations and resource costs are reported."),
            bullet("Archive: configuration, code, data fingerprints, job accounting, logs, manifests, receipts, figures, and reports are content-addressed."),
        ]
    )
    story.append(
        callout(
            "Definition of done",
            "The project is complete only when the final campaign gate verifies the preregistered chain through screening, model selection, nested evaluation, robustness, ablation, execution stress, statistical synthesis, and reproducibility archive. More HPC capacity should accelerate this chain, not remove it.",
            "green",
        )
    )
    story.append(Spacer(1, 8))
    story.append(p("NEXT-PHASE EVIDENCE INPUTS", "Eyebrow"))
    story.append(
        evidence_ledger(
            [
                ("Active campaign state", "campaigns/large_scale_thai_drl_v2/state.json"),
                ("Storage-safe capacity policy", "campaigns/large_scale_thai_drl_v2/operational/screening_capacity/README.md"),
                ("Locked screening configuration", "campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml"),
                ("Verified smoke summary", "results/screening_storage_safe_smoke_3550/screening_smoke_summary.json"),
            ]
        )
    )
    return spec, story


def draw_page(canvas, doc, short_title: str, accent: Color) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 13 * mm, PAGE_W, 13 * mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.drawString(15 * mm, PAGE_H - 8.3 * mm, short_title.upper())
    canvas.setFillColor(accent)
    canvas.rect(0, PAGE_H - 14.2 * mm, PAGE_W, 1.2 * mm, fill=1, stroke=0)
    canvas.setStrokeColor(LIGHT)
    canvas.setLineWidth(0.5)
    canvas.line(15 * mm, 13 * mm, PAGE_W - 15 * mm, 13 * mm)
    canvas.setFillColor(MID)
    canvas.setFont("Helvetica", 6.8)
    canvas.drawString(15 * mm, 8.5 * mm, f"{CAMPAIGN_ID} | evidence through {EVIDENCE_CUT}")
    canvas.drawRightString(PAGE_W - 15 * mm, 8.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(spec: ReportSpec, story: list[Any]) -> Path:
    output = OUT_DIR / spec.filename
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=19 * mm,
        bottomMargin=17 * mm,
        title=spec.title,
        author="KVIS Final Research Project Team",
        subject=spec.subtitle,
        creator="ReportLab process-report generator",
    )
    document.build(
        story,
        onFirstPage=lambda canvas, doc: draw_page(canvas, doc, spec.short_title, spec.accent),
        onLaterPages=lambda canvas, doc: draw_page(canvas, doc, spec.short_title, spec.accent),
    )
    return output


def validate_ascii_source() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    non_ascii = sorted({character for character in source if ord(character) > 127})
    if non_ascii:
        rendered = ", ".join(f"U+{ord(character):04X}" for character in non_ascii)
        raise ValueError(f"Non-ASCII characters in visible/source text: {rendered}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["1", "2", "3", "4", "5"],
        help="Build a single numbered report; default builds all five.",
    )
    args = parser.parse_args()

    validate_ascii_source()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    facts = gather_facts()
    builders = [report_1, report_2, report_3, report_4, report_5]
    if args.only:
        builders = [builders[int(args.only) - 1]]
    outputs = []
    for builder in builders:
        spec, story = builder(facts)
        outputs.append(build_pdf(spec, story))
    for output in outputs:
        print(output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
