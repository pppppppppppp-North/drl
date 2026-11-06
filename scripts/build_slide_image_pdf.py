#!/usr/bin/env python3
"""Build a presentation PDF from rendered slide PNG files.

This preserves complex-script typography exactly as rendered by the slide QA
pipeline. The resulting PDF is intended as a visually faithful display/export
copy; the editable, selectable-text source remains the accompanying PPTX.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a full-slide PDF from slide-*.png renderings."
    )
    parser.add_argument("slides_dir", type=Path)
    parser.add_argument("output_pdf", type=Path)
    parser.add_argument("--title", default="Presentation")
    parser.add_argument("--author", default="Thai Stock AI Research Project")
    parser.add_argument("--page-width", type=float, default=960.0)
    parser.add_argument("--page-height", type=float, default=540.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slides = sorted(args.slides_dir.glob("slide-*.png"), key=natural_key)
    if not slides:
        raise SystemExit(f"No slide-*.png files found in {args.slides_dir}")

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    page_size = (args.page_width, args.page_height)
    pdf = canvas.Canvas(str(args.output_pdf), pagesize=page_size, pageCompression=1)
    pdf.setTitle(args.title)
    pdf.setAuthor(args.author)
    pdf.setSubject("Self-guided HPC Showcase presentation about HPC in the project")

    for slide in slides:
        pdf.drawImage(
            ImageReader(str(slide)),
            0,
            0,
            width=page_size[0],
            height=page_size[1],
            preserveAspectRatio=True,
            anchor="c",
            mask="auto",
        )
        pdf.showPage()

    pdf.save()


if __name__ == "__main__":
    main()
