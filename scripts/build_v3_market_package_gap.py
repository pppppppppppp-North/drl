"""Build the deterministic V3 licensed-market package gap assessment."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.v3.market_data_readiness import (
    assess_v3_market_package,
    checks_for_csv,
    sha256_file,
    verify_market_package_bindings,
)


DEFAULT_BARS = "data/raw/prices_real_ohlcv.csv"
DEFAULT_MEMBERSHIP = (
    "data/processed/v3_development/"
    "set100_membership_session_mask_2021_2024.csv.gz"
)
DEFAULT_UNIVERSE = "data/reference/set50_or_set100_universe.csv"
DEFAULT_MEMBERSHIP_RECEIPT = (
    "reports/data_quality_v3/"
    "v3_membership_mask_readiness_20260731.json"
)
DEFAULT_CHECKS = "reports/data_quality_v3/v3_market_package_checks.csv"
DEFAULT_COVERAGE = "reports/data_quality_v3/v3_market_package_coverage.csv"
DEFAULT_RECEIPT = (
    "reports/data_quality_v3/v3_market_package_gap_20260731.json"
)
DEFAULT_CHART = (
    "reports/figures/v3_market_package_coverage_20260731.png"
)
DEFAULT_REPORT = "reports/v3_market_package_gap_20260731.md"

V2_IMMUTABILITY_BINDINGS = {
    "locked_preregistration": {
        "path": "config/research_preregistration.yaml",
        "sha256": "69a02d38f64ce65f3d593cf28dbe4135e0965102380aa18d6f000a9514b5e045",
    },
    "research_campaign": {
        "path": "config/research_campaign.yaml",
        "sha256": "9d0d02ad1890379e72ce8d7513cf93dc83a54e8afc1c6f527c5275e76476a0b4",
    },
    "active_storage_safe_release_manifest": {
        "path": (
            "campaigns/large_scale_thai_drl_v2/deployment/"
            "bistka_screening_storage_safe_cutover_v2_manifest.json"
        ),
        "sha256": "439b590bc4a85f03a6d98723ee3d28e40acdc00d95bdfbf48d310f07d867e25c",
    },
}

IMPLEMENTATION_PATHS = {
    "market_package_validator": "src/v3/market_data_readiness.py",
    "dated_sector_context": "src/v3/sector_context.py",
    "mask_aware_environment": "src/v3/trading_env.py",
    "self_financing_execution_ledger": "src/v3/execution_ledger.py",
    "investability_policy": "src/v3/investability_policy.py",
    "investability_policy_draft": "config/v3/investability_policy_draft.yaml",
    "shared_environment_runtime": "src/envs/trading_env.py",
    "manifest_template": "config/v3/market_package_manifest_template.yaml",
    "companion_notebook": "notebooks/v3_market_package_readiness.ipynb",
    "artifact_builder": "scripts/build_v3_market_package_gap.py",
}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_sidecar(path: Path) -> Path:
    sidecar = Path(f"{path}.sha256")
    _atomic_text(sidecar, f"{sha256_file(path)}  {path.name}\n")
    return sidecar


def _descriptor(project_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _require_project_file(project_root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(project_root) or not resolved.is_file():
        raise FileNotFoundError(
            f"required artifact is missing or outside the project: {resolved}"
        )
    return resolved


def _verify_sidecar(path: Path) -> Path:
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    tokens = sidecar.read_text(encoding="utf-8").split()
    if not tokens or tokens[0].lower() != sha256_file(path):
        raise ValueError(f"SHA-256 sidecar does not match: {path}")
    return sidecar


def _verify_membership_receipt_binding(
    project_root: Path,
    *,
    membership_path: Path,
    membership_sidecar: Path,
    receipt_path: Path,
) -> None:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read membership readiness receipt: {exc}") from exc
    if not isinstance(receipt, dict):
        raise ValueError("membership readiness receipt must be a JSON object")
    if (
        receipt.get("status") != "pass"
        or receipt.get("artifact_class") != "v3_membership_development_only"
        or receipt.get("membership_point_in_time_eligible") is not True
        or receipt.get("training_enabled") is not False
        or receipt.get("hpc_jobs_submitted") != 0
    ):
        raise ValueError("membership readiness receipt is not an eligible V3 input")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("membership readiness receipt outputs must be a mapping")
    expected = {
        "membership_panel": _descriptor(project_root, membership_path),
        "membership_panel_sidecar": _descriptor(
            project_root,
            membership_sidecar,
        ),
    }
    for label, descriptor in expected.items():
        if outputs.get(label) != descriptor:
            raise ValueError(
                f"membership receipt does not bind the exact {label}"
            )


def _plot_coverage(path: Path, coverage: pd.DataFrame) -> None:
    selected_metrics = [
        "acquisition_ticker_coverage",
        "active_member_bar_coverage",
        "active_member_status_coverage",
        "active_bar_sector_peer_coverage",
        "membership_session_calendar_coverage",
    ]
    labels = {
        "acquisition_ticker_coverage": "Acquisition tickers",
        "active_member_bar_coverage": "Active member bars",
        "active_member_status_coverage": "Active member status",
        "active_bar_sector_peer_coverage": "Covered active rows with a sector peer",
        "membership_session_calendar_coverage": "Authoritative session calendar",
    }
    selected = (
        coverage.set_index("metric")
        .reindex(selected_metrics)
        .reset_index()
    )
    if selected[["numerator", "denominator", "coverage_percent"]].isna().any().any():
        raise ValueError("coverage table lacks required chart metrics")

    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), "#FAFBFC")
    draw = ImageDraw.Draw(image)
    regular_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    title_font = ImageFont.truetype(bold_path, 46)
    subtitle_font = ImageFont.truetype(regular_path, 25)
    label_font = ImageFont.truetype(regular_path, 25)
    value_font = ImageFont.truetype(bold_path, 23)
    tick_font = ImageFont.truetype(regular_path, 20)
    note_font = ImageFont.truetype(regular_path, 18)

    draw.text(
        (70, 55),
        "V3 market-package coverage by readiness dimension",
        fill="#1B2430",
        font=title_font,
    )
    draw.text(
        (70, 120),
        "Current five-ticker snapshot assessed against point-in-time V3 requirements",
        fill="#596579",
        font=subtitle_font,
    )

    chart_left = 610
    chart_right = 1690
    chart_top = 235
    chart_bottom = 875
    chart_width = chart_right - chart_left
    for tick in (0, 25, 50, 75, 100):
        x = chart_left + int(chart_width * tick / 100)
        draw.line(
            (x, chart_top - 18, x, chart_bottom),
            fill="#DDE2E8",
            width=2,
        )
        tick_label = str(tick)
        tick_box = draw.textbbox((0, 0), tick_label, font=tick_font)
        draw.text(
            (x - (tick_box[2] - tick_box[0]) / 2, chart_bottom + 18),
            tick_label,
            fill="#4E5968",
            font=tick_font,
        )

    row_gap = 122
    bar_height = 58
    for index, row in enumerate(selected.itertuples(index=False)):
        center_y = chart_top + 38 + index * row_gap
        metric_label = labels[row.metric]
        label_box = draw.textbbox((0, 0), metric_label, font=label_font)
        draw.text(
            (
                chart_left - 28 - (label_box[2] - label_box[0]),
                center_y - (label_box[3] - label_box[1]) / 2 - 3,
            ),
            metric_label,
            fill="#27313F",
            font=label_font,
        )
        bar_width = max(
            0,
            int(chart_width * float(row.coverage_percent) / 100.0),
        )
        if bar_width:
            draw.rounded_rectangle(
                (
                    chart_left,
                    center_y - bar_height // 2,
                    chart_left + bar_width,
                    center_y + bar_height // 2,
                ),
                radius=8,
                fill="#3568A8",
                outline="#234B7D",
                width=2,
            )
        else:
            draw.line(
                (
                    chart_left,
                    center_y - bar_height // 2,
                    chart_left,
                    center_y + bar_height // 2,
                ),
                fill="#234B7D",
                width=3,
            )
        value_label = (
            f"{int(row.numerator):,}/{int(row.denominator):,} "
            f"({row.coverage_percent:.1f}%)"
        )
        draw.text(
            (max(chart_left + bar_width + 18, chart_left + 18), center_y - 15),
            value_label,
            fill="#1B2430",
            font=value_font,
        )

    axis_label = "Coverage (%) — each dimension uses its own stated denominator"
    axis_box = draw.textbbox((0, 0), axis_label, font=subtitle_font)
    draw.text(
        (
            chart_left + (chart_width - (axis_box[2] - axis_box[0])) / 2,
            945,
        ),
        axis_label,
        fill="#27313F",
        font=subtitle_font,
    )
    draw.text(
        (70, 1022),
        (
            "Source: current project market snapshot, checksum-bound 138-ticker "
            "acquisition union, and V3 membership panel. No performance result."
        ),
        fill="#667386",
        font=note_font,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(
        temporary,
        format="PNG",
        optimize=True,
        compress_level=9,
    )
    os.replace(temporary, path)


def _percentage(coverage: pd.DataFrame, metric: str) -> str:
    row = coverage.set_index("metric").loc[metric]
    return (
        f"{int(row['numerator']):,}/{int(row['denominator']):,} "
        f"({float(row['coverage_percent']):.2f}%)"
    )


def _report_markdown(
    receipt: dict[str, Any],
    checks: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    chart_relative_path: str,
) -> str:
    metrics = receipt["metrics"]
    failed = checks[checks["status"].ne("pass")]
    failed_by_severity = (
        failed.groupby("severity").size().to_dict()
        if not failed.empty
        else {}
    )
    priority_checks = [
        "bar_required_columns",
        "bar_acquisition_ticker_contract",
        "active_member_status_coverage",
        "authoritative_session_calendar_match",
        "corporate_action_required_columns",
        "symbol_history_required_columns",
        "membership_exit_path_coverage",
        "license_allows_research_and_derived_publication",
        "manifest_bindings_verified",
    ]
    check_index = checks.set_index("check")
    audit_rows = []
    for check_id in priority_checks:
        row = check_index.loc[check_id]
        audit_rows.append(
            f"| `{check_id}` | {str(row['status']).upper()} | "
            f"{row['severity']} | {row['requirement']} |"
        )
    audit_table = "\n".join(audit_rows)
    return f"""# V3 Licensed-Market Package Readiness Gap

## Technical summary

The V3 intake gate is now implemented, but the current market snapshot **fails** it. The snapshot covers **{_percentage(coverage, 'acquisition_ticker_coverage')}** of the checksum-bound acquisition universe and **{_percentage(coverage, 'active_member_bar_coverage')}** of the point-in-time active SET100 session keys. It contains no authoritative trading-status package, no licensed session-calendar binding, no corporate-action table, no stable-instrument history, and no candidate license/checksum manifest.

This is a useful fail-closed result: the old five-ticker private-pilot readiness receipt can no longer be mistaken for broad-universe V3 evidence. The new gate does **not** enable model input, training, test access, or a publication claim. No HPC job was submitted.

## Five percent bar coverage is not a trainable market package

![V3 market-package coverage by readiness dimension]({chart_relative_path})

The bars show separate coverage questions with separate denominators. The 5.00% active-key result means that all five pilot tickers have a bar on each of the **965 provisional membership sessions**, but the other active members are absent. The zero status and calendar values mean the corresponding authoritative package components do not exist locally—not that no suspensions or calendar differences occurred.

| Readiness measure | Evidence |
| --- | ---: |
| Acquisition tickers with any bar | {_percentage(coverage, 'acquisition_ticker_coverage')} |
| Active SET100 session keys with a bar | {_percentage(coverage, 'active_member_bar_coverage')} |
| Active keys with explicit trading status | {_percentage(coverage, 'active_member_status_coverage')} |
| Covered active keys with at least one covered dated-sector peer | {_percentage(coverage, 'active_bar_sector_peer_coverage')} |
| Sessions backed by a licensed authoritative calendar | {_percentage(coverage, 'membership_session_calendar_coverage')} |
| Membership exits with a certified liquidation/settlement path | {_percentage(coverage, 'membership_exit_path_coverage')} |
| Current raw bar rows | {metrics['current_bar_rows']:,} |
| Current raw bar dates | {metrics['current_bar_dates']:,} |
| Bar dates outside the provisional membership grid | {metrics['bar_dates_outside_development_membership_grid']} |
| Gate checks passed | {metrics['checks_passed']}/{metrics['checks_total']} |
| Gate checks failed | {metrics['checks_failed']} |
| HPC jobs submitted | 0 |

The three extra stock dates are 2 April 2021, 25 December 2023, and 25 December 2024. Their presence demonstrates why the existing observed Yahoo SET-index dates cannot be treated as an authoritative exchange calendar. A licensed session package must be supplied and the membership grid rebuilt if the calendars differ.

## The gate now checks the evidence that V3 was missing

| Priority check | Result | Severity | Acceptance rule |
| --- | --- | --- | --- |
{audit_table}

The complete audit table is stored separately in `reports/data_quality_v3/v3_market_package_checks.csv`, and `notebooks/v3_market_package_readiness.ipynb` is the inspectable companion analysis. This assessment has **{failed_by_severity.get('critical', 0)} critical**, **{failed_by_severity.get('high', 0)} high**, and **{failed_by_severity.get('medium', 0)} medium** failed checks.

## Scope, data, and metric definitions

- **Acquisition ticker coverage** is the number of the 138 checksum-bound union tickers with at least one bar. The union is only a provider acquisition list; it is never treated as static membership.
- **Active member bar coverage** uses the {metrics['active_membership_keys']:,} date/ticker keys marked as SET100 members by the point-in-time V3 panel.
- **Status coverage** requires an explicit daily state. A missing bar is never reinterpreted as a suspension. A suspended, halted, or no-trade active key may lack a bar only when that non-trading state is present.
- **Sector-peer coverage** requires at least one other covered active member in the same dated sector. It is a context sufficiency diagnostic, not a performance metric.
- **Session-calendar coverage** requires a provider-authorized SET session table with lineage. The current 965-date grid remains a development timing scaffold.
- **Membership-exit path coverage** requires continuous status after each active-to-inactive transition until a valid liquidation bar or explicit terminal settlement. Index removal alone is not execution evidence.

The source period is 2021-01-04 through 2024-12-30. This period has already been examined and cannot be relabeled as a new untouched V3 holdout.

## Methodology and failure behavior

The validator evaluates five separately bound CSV components: adjusted OHLCV bars, trading status, authoritative SET sessions, corporate actions, and stable-instrument symbol history. A candidate YAML manifest must bind each file by project-relative path, SHA-256, byte count, and row count; the manifest itself also needs a checksum sidecar. The license must allow research use and publication of derived findings, although raw redistribution may remain forbidden.

Market completeness is evaluated at point-in-time active membership keys rather than against an artificial dense 138-ticker price panel. Every active key must have either a valid trading bar or an explicit suspended, halted, or no-trade status. Daily bars and statuses must be available on their own session, every trading key must have a valid bar, OHLC envelopes must be coherent, prices must be finite and positive, and volume must be nonnegative. Corporate actions and symbol intervals are checked for identifiers, timing, factors, successor symbols, overlaps, and source lineage.

The gate also evaluates all {metrics['membership_exit_transitions']} observed membership exits. A suspended or halted security remains a holding with a pending exit; it cannot be converted to cash until a contiguous status path reaches a valid trading bar or an explicit provider-authorized terminal settlement.

## Dated sector features and environment evidence were strengthened

The isolated V3 sector module now uses `sector_as_of` from the validated membership panel. Only active members with an explicit observed return contribute; inactive acquisition rows cannot influence a sector mean. It exposes contributor counts, coverage, inclusive active-sector returns, separately named leave-one-out peer returns, and availability flags. It preserves the real MINT and NEX 2023 sector changes and rejects duplicates, missing active status rows, fabricated unobserved zeroes, and future availability.

Four adversarial V3 environment defects were also corrected. Every feature and close-price matrix is explicitly reindexed to the caller's fixed asset-slot order, and step evidence records the decision mask separately from the next mask. Security turnover now excludes the balancing cash leg, so transaction costs are charged once on actual traded notional. Post-return weights drift with realized asset returns and reconcile exactly with cash and costs before the next decision. The minimal no-Gymnasium fallback also preserves a callable `close()` lifecycle method, restoring deterministic-worker compatibility without changing the locked V2 release artifacts.

## Limitations, uncertainty, and robustness boundary

- A package-level PASS would certify intake only. The new lifecycle policy covers entry, exit, suspension, delisting, stale valuation, corporate actions, symbol changes, and self-financing accounting, but remains an unapproved draft with four explicit author decisions.
- Date-level after-close availability supports use in a later decision; it does not prove that a same-close trade could execute using the just-finished bar.
- The current five tickers occupy five different dated sectors throughout the covered active rows, so none has a covered same-sector peer. The new sector code is structurally tested, but the current data cannot support a meaningful broad-sector feature result.
- The 2021–2024 market and membership evidence is already known. A genuinely untouched post-2024 period and extended constituent archive are still required.
- Historical news/sentiment and licensed fundamentals remain separate multi-source blockers; this market-package gate does not certify them.
- The canonical V2 campaign remains **11/37 verified stages (29.7%)** because no scientific screening stage was completed here.

## Recommended next steps

1. Send the manifest contract and 138-ticker acquisition union to the licensed provider.
2. Receive the five market-package components without overwriting any locked V2 input.
3. Verify the manifest checksum bindings and rebuild membership on the authoritative session calendar.
4. Resolve every critical/high gate failure, then issue a passing licensed-package receipt.
5. Lock the execution policy and bind the passing package receipt into an author-approved V3 protocol.
6. Extend constituent history and embargo an untouched post-2024 holdout.
7. Only after those gates pass should a small train/validation smoke run be considered; no HPC submission is needed for the current intake work.

## Further questions

- Which provider field is authoritative for suspended, halted, and no-trade states?
- Does the license permit derived tables and figures in the final paper while forbidding raw redistribution?
- Are adjusted prices point-in-time safe, or must raw prices and event adjustments be reconstructed?
- What execution price and valuation rule should apply when an active member stops trading?
"""


def build_gap_report(
    project_root: Path,
    *,
    bars_path: Path,
    membership_path: Path,
    universe_path: Path,
    membership_receipt_path: Path,
    candidate_manifest_path: Path | None,
    checks_path: Path,
    coverage_path: Path,
    receipt_path: Path,
    chart_path: Path,
    report_path: Path,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    bars_path = _require_project_file(root, bars_path)
    membership_path = _require_project_file(root, membership_path)
    universe_path = _require_project_file(root, universe_path)
    membership_receipt_path = _require_project_file(
        root,
        membership_receipt_path,
    )
    membership_sidecar = _verify_sidecar(membership_path)
    membership_receipt_sidecar = _verify_sidecar(membership_receipt_path)
    _verify_membership_receipt_binding(
        root,
        membership_path=membership_path,
        membership_sidecar=membership_sidecar,
        receipt_path=membership_receipt_path,
    )

    v2_descriptors: dict[str, Any] = {}
    for label, binding in V2_IMMUTABILITY_BINDINGS.items():
        path = _require_project_file(root, root / binding["path"])
        actual_hash = sha256_file(path)
        if actual_hash != binding["sha256"]:
            raise ValueError(
                f"locked V2 artifact changed: {binding['path']} "
                f"expected={binding['sha256']} actual={actual_hash}"
            )
        v2_descriptors[label] = _descriptor(root, path)
    implementation: dict[str, Any] = {}
    for label, relative in IMPLEMENTATION_PATHS.items():
        implementation[label] = _descriptor(
            root,
            _require_project_file(root, root / relative),
        )

    bars = pd.read_csv(bars_path)
    membership = pd.read_csv(membership_path)
    universe = pd.read_csv(universe_path)
    trading_status = None
    sessions = None
    corporate_actions = None
    symbol_history = None
    manifest = None
    binding_results = None
    candidate_inputs: dict[str, Any] = {}
    if candidate_manifest_path is not None:
        candidate_manifest_path = _require_project_file(
            root,
            candidate_manifest_path,
        )
        (
            manifest,
            frames,
            binding_results,
            referenced,
        ) = verify_market_package_bindings(
            candidate_manifest_path,
            project_root=root,
        )
        bars = frames.get("bars", bars)
        trading_status = frames.get("trading_status")
        sessions = frames.get("sessions")
        corporate_actions = frames.get("corporate_actions")
        symbol_history = frames.get("symbol_history")
        candidate_inputs = {
            path.relative_to(root).as_posix(): _descriptor(root, path)
            for path in referenced
        }

    generated = generated_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    receipt, checks, coverage = assess_v3_market_package(
        bars,
        membership,
        universe,
        trading_status=trading_status,
        sessions=sessions,
        corporate_actions=corporate_actions,
        symbol_history=symbol_history,
        manifest=manifest,
        binding_results=binding_results,
        generated_at_utc=generated,
    )
    receipt["metrics"]["current_bar_rows"] = int(len(bars))
    receipt["metrics"]["current_bar_dates"] = (
        int(pd.to_datetime(bars["date"], errors="coerce").nunique())
        if "date" in bars.columns
        else 0
    )

    _atomic_csv(checks_path, checks_for_csv(checks))
    checks_sidecar = _write_sidecar(checks_path)
    _atomic_csv(coverage_path, coverage)
    coverage_sidecar = _write_sidecar(coverage_path)
    _plot_coverage(chart_path, coverage)
    chart_sidecar = _write_sidecar(chart_path)

    receipt["inputs"] = {
        "bars_assessed": _descriptor(root, bars_path),
        "membership_panel": _descriptor(root, membership_path),
        "membership_panel_sidecar": _descriptor(root, membership_sidecar),
        "acquisition_universe": _descriptor(root, universe_path),
        "membership_readiness_receipt": _descriptor(
            root,
            membership_receipt_path,
        ),
        "membership_readiness_receipt_sidecar": _descriptor(
            root,
            membership_receipt_sidecar,
        ),
        "candidate_package": candidate_inputs,
        "implementation": implementation,
        "v2_immutability_baseline": v2_descriptors,
    }
    receipt["outputs"] = {
        "checks": _descriptor(root, checks_path),
        "checks_sidecar": _descriptor(root, checks_sidecar),
        "coverage": _descriptor(root, coverage_path),
        "coverage_sidecar": _descriptor(root, coverage_sidecar),
        "coverage_chart": _descriptor(root, chart_path),
        "coverage_chart_sidecar": _descriptor(root, chart_sidecar),
    }
    _atomic_text(
        receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    _write_sidecar(receipt_path)
    chart_relative = os.path.relpath(chart_path, start=report_path.parent)
    _atomic_text(
        report_path,
        _report_markdown(
            receipt,
            checks,
            coverage,
            chart_relative_path=Path(chart_relative).as_posix(),
        ),
    )
    _write_sidecar(report_path)
    return receipt


def _project_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the deterministic V3 licensed-market gap report"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--bars", default=DEFAULT_BARS)
    parser.add_argument("--membership", default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument(
        "--membership-receipt",
        default=DEFAULT_MEMBERSHIP_RECEIPT,
    )
    parser.add_argument("--candidate-manifest", default=None)
    parser.add_argument("--checks-output", default=DEFAULT_CHECKS)
    parser.add_argument("--coverage-output", default=DEFAULT_COVERAGE)
    parser.add_argument("--receipt-output", default=DEFAULT_RECEIPT)
    parser.add_argument("--chart-output", default=DEFAULT_CHART)
    parser.add_argument("--report-output", default=DEFAULT_REPORT)
    parser.add_argument(
        "--generated-at-utc",
        default=None,
        help="Optional fixed ISO timestamp for deterministic evidence rebuilds.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero when the candidate package is not ready.",
    )
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    receipt = build_gap_report(
        root,
        bars_path=_project_path(root, args.bars),
        membership_path=_project_path(root, args.membership),
        universe_path=_project_path(root, args.universe),
        membership_receipt_path=_project_path(
            root,
            args.membership_receipt,
        ),
        candidate_manifest_path=(
            _project_path(root, args.candidate_manifest)
            if args.candidate_manifest
            else None
        ),
        checks_path=_project_path(root, args.checks_output),
        coverage_path=_project_path(root, args.coverage_output),
        receipt_path=_project_path(root, args.receipt_output),
        chart_path=_project_path(root, args.chart_output),
        report_path=_project_path(root, args.report_output),
        generated_at_utc=args.generated_at_utc,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "metrics": receipt["metrics"],
                "licensed_market_package_eligible": receipt[
                    "licensed_market_package_eligible"
                ],
                "hpc_jobs_submitted": receipt["hpc_jobs_submitted"],
            },
            indent=2,
        )
    )
    if args.require_pass and not receipt["licensed_market_package_eligible"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
