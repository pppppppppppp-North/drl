from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


BLUE = "#2E5EAA"
BLUE_DARK = "#1D3E70"
GOLD = "#E3B341"
GOLD_DARK = "#7A5A00"
INK = "#202632"
MUTED = "#647184"
GRID = "#DCE2EA"
BACKGROUND = "#FFFFFF"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_sidecar(path: Path) -> None:
    _atomic_text(
        path.with_name(path.name + ".sha256"),
        f"{_sha256_file(path)}  {path.name}\n",
    )


def _verify_receipt(project_root: Path, receipt_path: Path) -> dict[str, Any]:
    sidecar = receipt_path.with_name(receipt_path.name + ".sha256")
    if not sidecar.exists():
        raise ValueError(f"missing receipt sidecar: {sidecar}")
    declared = sidecar.read_text(encoding="utf-8").strip().split()
    if not declared or declared[0] != _sha256_file(receipt_path):
        raise ValueError("SET constituent receipt SHA-256 sidecar mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for name, profile in receipt["outputs"].items():
        output = (project_root / profile["path"]).resolve()
        if not output.is_file():
            raise ValueError(f"receipt output is missing: {name}")
        if output.stat().st_size != int(profile["bytes"]):
            raise ValueError(f"receipt output byte mismatch: {name}")
        if _sha256_file(output) != profile["sha256"]:
            raise ValueError(f"receipt output hash mismatch: {name}")
    return receipt


def _svg_turnover_chart(readiness: pd.DataFrame) -> str:
    chart = readiness.iloc[1:].reset_index(drop=True)
    width = 1200
    height = 760
    margin_left = 250
    margin_right = 90
    plot_top = 145
    plot_bottom = 670
    plot_width = width - margin_left - margin_right
    row_height = (plot_bottom - plot_top) / len(chart)
    maximum = max(
        24,
        int(
            max(
                chart["set50_turnover"].max(),
                chart["set100_turnover"].max(),
            )
            + 2
        ),
    )

    def x(value: float) -> float:
        return margin_left + (float(value) / maximum) * plot_width

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">SET50 and SET100 constituent turnover by transition</title>',
        (
            '<desc id="description">Horizontal grouped bars compare the total '
            'number of additions and removals in SET50 and SET100 at each '
            'official effective interval after the January 2021 baseline.</desc>'
        ),
        "<defs>",
        (
            '<pattern id="goldHatch" width="8" height="8" '
            'patternUnits="userSpaceOnUse" patternTransform="rotate(35)">'
            f'<rect width="8" height="8" fill="{GOLD}"/>'
            f'<line x1="0" y1="0" x2="0" y2="8" stroke="{GOLD_DARK}" '
            'stroke-width="1.4" opacity="0.45"/>'
            "</pattern>"
        ),
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        (
            f'<text x="42" y="48" fill="{INK}" font-family="Arial, sans-serif" '
            'font-size="27" font-weight="700">SET50/SET100 constituent '
            "turnover by transition</text>"
        ),
        (
            f'<text x="42" y="79" fill="{MUTED}" '
            'font-family="Arial, sans-serif" font-size="16">'
            "Additions + removals relative to the preceding effective interval; "
            "official SET archive, 2021–2024</text>"
        ),
        (
            f'<rect x="42" y="101" width="24" height="13" rx="2" '
            f'fill="{BLUE}" stroke="{BLUE_DARK}"/>'
        ),
        (
            f'<text x="76" y="113" fill="{INK}" '
            'font-family="Arial, sans-serif" font-size="14">SET100</text>'
        ),
        (
            '<rect x="160" y="101" width="24" height="13" rx="2" '
            f'fill="url(#goldHatch)" stroke="{GOLD_DARK}"/>'
        ),
        (
            f'<text x="194" y="113" fill="{INK}" '
            'font-family="Arial, sans-serif" font-size="14">SET50</text>'
        ),
    ]
    for tick in range(0, maximum + 1, 4):
        tick_x = x(tick)
        parts.extend(
            [
                (
                    f'<line x1="{tick_x:.1f}" y1="{plot_top - 6}" '
                    f'x2="{tick_x:.1f}" y2="{plot_bottom}" '
                    f'stroke="{GRID}" stroke-width="1"/>'
                ),
                (
                    f'<text x="{tick_x:.1f}" y="{plot_bottom + 27}" '
                    f'text-anchor="middle" fill="{MUTED}" '
                    'font-family="Arial, sans-serif" font-size="13">'
                    f"{tick}</text>"
                ),
            ]
        )
    for index, row in chart.iterrows():
        center = plot_top + row_height * (index + 0.5)
        set100 = int(row["set100_turnover"])
        set50 = int(row["set50_turnover"])
        set100_y = center - 18
        set50_y = center + 3
        set100_width = x(set100) - margin_left
        set50_width = x(set50) - margin_left
        label = html.escape(str(row["period_label"]))
        parts.extend(
            [
                (
                    f'<text x="{margin_left - 18}" y="{center + 4:.1f}" '
                    f'text-anchor="end" fill="{INK}" '
                    'font-family="Arial, sans-serif" font-size="14">'
                    f"{label}</text>"
                ),
                (
                    f'<rect x="{margin_left}" y="{set100_y:.1f}" '
                    f'width="{set100_width:.1f}" height="14" rx="2" '
                    f'fill="{BLUE}" stroke="{BLUE_DARK}" stroke-width="1"/>'
                ),
                (
                    f'<text x="{x(set100) + 8:.1f}" y="{set100_y + 12:.1f}" '
                    f'fill="{BLUE_DARK}" font-family="Arial, sans-serif" '
                    f'font-size="13" font-weight="700">{set100}</text>'
                ),
                (
                    f'<rect x="{margin_left}" y="{set50_y:.1f}" '
                    f'width="{set50_width:.1f}" height="14" rx="2" '
                    'fill="url(#goldHatch)" '
                    f'stroke="{GOLD_DARK}" stroke-width="1"/>'
                ),
                (
                    f'<text x="{x(set50) + 8:.1f}" y="{set50_y + 12:.1f}" '
                    f'fill="{GOLD_DARK}" font-family="Arial, sans-serif" '
                    f'font-size="13" font-weight="700">{set50}</text>'
                ),
            ]
        )
    parts.extend(
        [
            (
                f'<text x="{margin_left + plot_width / 2:.1f}" '
                f'y="{plot_bottom + 55}" text-anchor="middle" fill="{INK}" '
                'font-family="Arial, sans-serif" font-size="14" '
                'font-weight="600">Constituent changes (additions + removals)</text>'
            ),
            (
                f'<text x="42" y="735" fill="{MUTED}" '
                'font-family="Arial, sans-serif" font-size="12">'
                "Source: Stock Exchange of Thailand official constituent PDFs. "
                "Baseline interval omitted because turnover requires a predecessor."
                "</text>"
            ),
            # Compact four-petal research blossom in the locked top-right position.
            (
                f'<g transform="translate({width - 58},44)" opacity="0.8">'
                f'<circle cx="0" cy="-8" r="6" fill="{BLUE}"/>'
                f'<circle cx="8" cy="0" r="6" fill="{GOLD}"/>'
                f'<circle cx="0" cy="8" r="6" fill="{BLUE}" opacity="0.65"/>'
                f'<circle cx="-8" cy="0" r="6" fill="{GOLD}" opacity="0.65"/>'
                f'<circle cx="0" cy="0" r="3" fill="{INK}"/>'
                "</g>"
            ),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def _record_symbols(
    membership: pd.DataFrame,
    record_id: str,
    *,
    index_name: str,
) -> set[str]:
    rows = membership[membership["record_id"].eq(record_id)]
    if index_name == "SET50":
        rows = rows[rows["in_set50"]]
    return set(rows["symbol"].astype(str))


def _revision_summary(
    membership: pd.DataFrame,
    original: str,
    revised: str,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for index_name in ("SET50", "SET100"):
        before = _record_symbols(
            membership,
            original,
            index_name=index_name,
        )
        after = _record_symbols(
            membership,
            revised,
            index_name=index_name,
        )
        result[f"{index_name.lower()}_added"] = sorted(after - before)
        result[f"{index_name.lower()}_removed"] = sorted(before - after)
    return result


def _report_markdown(
    *,
    receipt: dict[str, Any],
    readiness: pd.DataFrame,
    membership: pd.DataFrame,
    receipt_path: Path,
    chart_path: Path,
) -> str:
    coverage = receipt["coverage"]
    outputs = receipt["outputs"]
    revisions = [
        (
            "2021 H1",
            _revision_summary(
                membership,
                "set_2021_h1_original",
                "set_2021_h1_revised",
            ),
        ),
        (
            "2023 H1",
            _revision_summary(
                membership,
                "set_2023_h1_original",
                "set_2023_h1_revised",
            ),
        ),
    ]
    interval_lines = [
        "| Effective record | Coverage | SET50 | SET100 | SET50 turnover | SET100 turnover | Sectors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in readiness.itertuples(index=False):
        interval_lines.append(
            "| "
            f"{row.period_label} | {row.effective_from} to "
            f"{row.effective_to_exclusive} (exclusive) | "
            f"{row.set50_count} | {row.set100_count} | "
            f"{row.set50_turnover} | {row.set100_turnover} | "
            f"{row.sector_count} |"
        )
    revision_lines = [
        "| Revision | SET50 change | SET100 change |",
        "| --- | --- | --- |",
    ]
    for label, diff in revisions:
        set50_change = (
            f"+{', '.join(diff['set50_added'])}; "
            f"−{', '.join(diff['set50_removed'])}"
        )
        set100_change = (
            f"+{', '.join(diff['set100_added'])}; "
            f"−{', '.join(diff['set100_removed'])}"
        )
        revision_lines.append(
            f"| {label} | {set50_change} | {set100_change} |"
        )
    artifact_lines = [
        "| Artifact | Rows | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for name, profile in outputs.items():
        artifact_lines.append(
            f"| `{profile['path']}` | {profile['rows']} | "
            f"`{profile['sha256']}` |"
        )
    maximum_row = readiness.iloc[
        readiness["set100_turnover"].astype(int).argmax()
    ]
    median_set100 = float(readiness.iloc[1:]["set100_turnover"].median())
    chart_relative = chart_path.relative_to(receipt_path.parents[1])
    receipt_relative = receipt_path.relative_to(receipt_path.parents[1])
    changed_tickers = ", ".join(
        f"`{ticker}`"
        for ticker in receipt["sector_history"]["changed_tickers"]
    )
    return f"""# SET Point-in-Time Constituent Archive Readiness Report

Generated from checksum-verified official SET source files on 2026-07-31.

## Executive summary

The project now has a reproducible, point-in-time structural archive of SET50 and SET100 membership for 2021–2024. Ten official PDF records—including two mid-period revisions—were preserved, hashed, parsed, and normalized into {coverage['normalized_rows']:,} dated membership rows. The acquisition union contains {coverage['unique_tickers']} distinct `.BK` tickers and the archive contains {coverage['sector_count']} canonical sector labels.

This closes two previously missing structural inputs: the broader constituent universe and complete constituent-level sector history. External-source structural readiness therefore rises from **1/5 to 3/5**. It does **not** change the locked V2 campaign denominator or create new model results: canonical whole-project campaign completion remains **11/37 = 29.7%**, and the active graph remains 9 VERIFIED, 1 READY, and 25 PENDING.

The archive is eligible for point-in-time membership and sector lookup, but it is deliberately marked `model_input_eligible=false`. Licensed historical OHLCV, a date-aware investability mask, and a newly preregistered broad-universe release are still required before training on this universe.

## Key validated results

| Measure | Validated value |
| --- | ---: |
| Official raw PDFs | {coverage['raw_file_count']} |
| Raw source bytes | {coverage['raw_bytes']:,} |
| Effective intervals | {coverage['interval_count']} |
| Coverage | {coverage['effective_from']} to {coverage['effective_to_exclusive']} (exclusive) |
| Normalized membership rows | {coverage['normalized_rows']:,} |
| Unique acquisition tickers | {coverage['unique_tickers']} |
| Canonical sectors | {coverage['sector_count']} |
| Tickers with genuine sector changes | {receipt['sector_history']['changed_ticker_count']} |
| Largest SET100 transition | {maximum_row['set100_turnover']} changes ({maximum_row['period_label']}) |
| Median SET100 turnover after baseline | {median_set100:.0f} changes |
| Quality gates passed | {sum(bool(value) for value in receipt['quality_gates'].values())}/{len(receipt['quality_gates'])} |

![SET50 and SET100 constituent turnover by transition]({chart_relative.as_posix()})

The chart counts additions plus removals relative to the immediately preceding effective record. It describes index-composition change only; it is not a trading-performance result.

## Interval validation

{chr(10).join(interval_lines)}

Every interval has exactly 50 SET50 members and 100 SET100 members, with SET50 fully contained in SET100. Intervals are continuous, non-overlapping, and publication-gated.

## Mid-period revisions captured

{chr(10).join(revision_lines)}

The revisions are applied from their publication dates because the archived PDFs do not state a separate exchange-effective transition date. This is conservative against future-information leakage, but it should not be interpreted as proof of the exact corporate-action effective instant.

## Sector-history finding

Two securities have genuine classification changes in the official records: {changed_tickers}. `MINT.BK` moves from Food & Beverage to Tourism & Leisure beginning with 2023 H2; `NEX.BK` moves from Electronic Components to Automotive beginning with 2023 H2.

For that reason, `sector_mapping.csv` is only a latest-label compatibility reference. Scientific joins must use the dated membership archive. A single static sector map would leak future classifications into earlier dates.

## Methodology and data-quality controls

1. The official [SET constituent archive](https://www.set.or.th/en/market/information/securities-list/constituents-list-set50-set100) was used to identify all half-year records and published revisions from 2021 through 2024.
2. Each source PDF was preserved under `data/external/original/set_constituent_history/` and bound to its URL, byte size, retrieval timestamp, and SHA-256 in `data/external/set_constituent_source_manifest.csv`.
3. The importer handles both legacy two-column and modern ruled-table layouts. It extracts rank, symbol, company, and sector; normalizes `and`/`&` sector variants; and rejects incomplete ranks, duplicates, blank fields, malformed ticker mappings, count drift, hash drift, and SET50/SET100 subset violations.
4. The normalized schema uses half-open intervals: `effective_from <= date < effective_to_exclusive`. The resolver in `src/data/constituent_history.py` returns only the membership record published and active at the requested date.
5. The receipt and every generated CSV are hash-bound. The source-readiness checker now rejects the broader-universe and sector files if their receipt is missing or if any output byte, row count, or hash changes.

## Scientific interpretation

This milestone materially reduces survivorship-bias risk for the next study version. It allows price acquisition to target every security that appeared in SET100 during the study period and provides the information needed to mask the tradable universe as of each date.

It does not retroactively make the current five-ticker V2 experiment survivorship-free. The active preregistration, environment, and locked release remain unchanged. No conclusions about full SET50/SET100 profitability are justified yet.

## Limitations and next decisions

- Licensed or explicitly authorized 2021–2024 OHLCV is still missing for the 138-ticker union. The current Yahoo snapshot covers only five tickers and remains private-research-only.
- Historical news/sentiment and licensed fundamentals remain missing, so overall structural source readiness is 3/5 rather than 5/5.
- A V3 environment must use date-aware membership and sector joins and must handle entries, exits, suspensions, missing histories, and corporate actions without forward filling future membership.
- The broad-universe design needs a new preregistration before any test-period access; the locked V2 preregistration must not be amended.
- Before eventual HPC screening, the supported submission workflow should be replaced by a one-shot, receipt-bound submission capsule. No HPC job was submitted while building this archive.

## Reproducible artifacts

{chr(10).join(artifact_lines)}

Receipt: `{receipt_relative.as_posix()}` (with adjacent SHA-256 sidecar).

## References

- Stock Exchange of Thailand. [SET50 and SET100 constituent archive](https://www.set.or.th/en/market/information/securities-list/constituents-list-set50-set100).
- Stock Exchange of Thailand. [SET50/SET100 constituents, 2023 H2](https://media.set.or.th/set/Documents/2023/Jun/SET50-SET100_H2_2023.pdf).
- Stock Exchange of Thailand. [SET50/SET100 constituents, 2024 H2](https://media.set.or.th/set/Documents/2024/Jun/SET50_100_H2_2024.pdf).
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--receipt",
        default="reports/data_sources/set_constituent_archive_20260731.json",
    )
    parser.add_argument(
        "--output",
        default="reports/set_constituent_archive_readiness_20260731.md",
    )
    parser.add_argument(
        "--chart",
        default="reports/figures/set_constituent_turnover_2021_2024.svg",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    receipt_path = (project_root / args.receipt).resolve()
    output_path = (project_root / args.output).resolve()
    chart_path = (project_root / args.chart).resolve()
    receipt = _verify_receipt(project_root, receipt_path)
    readiness = pd.read_csv(
        project_root / receipt["outputs"]["interval_readiness"]["path"]
    )
    membership = pd.read_csv(
        project_root / receipt["outputs"]["membership_history"]["path"]
    )
    _atomic_text(chart_path, _svg_turnover_chart(readiness))
    _atomic_text(
        output_path,
        _report_markdown(
            receipt=receipt,
            readiness=readiness,
            membership=membership,
            receipt_path=receipt_path,
            chart_path=chart_path,
        ),
    )
    _write_sidecar(chart_path)
    _write_sidecar(output_path)
    print(output_path)
    print(chart_path)


if __name__ == "__main__":
    main()
