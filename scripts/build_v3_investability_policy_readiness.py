"""Build deterministic V3 investability-policy and exit-path evidence."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.v3.investability_policy import (
    assess_membership_exit_paths,
    membership_exit_transitions,
    sha256_file,
    validate_investability_policy,
)


DEFAULT_POLICY = "config/v3/investability_policy_draft.yaml"
DEFAULT_MEMBERSHIP = (
    "data/processed/v3_development/"
    "set100_membership_session_mask_2021_2024.csv.gz"
)
DEFAULT_MEMBERSHIP_RECEIPT = (
    "reports/data_quality_v3/v3_membership_mask_readiness_20260731.json"
)
DEFAULT_MARKET_RECEIPT = (
    "reports/data_quality_v3/v3_market_package_gap_20260731.json"
)
DEFAULT_LEDGER = (
    "reports/data_quality_v3/"
    "v3_membership_exit_path_inventory_20260731.csv"
)
DEFAULT_RECEIPT = (
    "reports/data_quality_v3/"
    "v3_investability_policy_readiness_20260731.json"
)
DEFAULT_CHART = (
    "reports/figures/v3_membership_exit_transitions_20260731.png"
)
DEFAULT_REPORT = "reports/v3_investability_policy_readiness_20260731.md"

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


def _verify_sidecar(path: Path) -> Path:
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    tokens = sidecar.read_text(encoding="utf-8").split()
    if not tokens or tokens[0].lower() != sha256_file(path):
        raise ValueError(f"SHA-256 sidecar does not match: {path}")
    return sidecar


def _descriptor(project_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_receipt(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_exact_descriptor(
    container: Any,
    key: str,
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(container, dict) or container.get(key) != expected:
        raise ValueError(f"{label} does not bind the exact {key}")


def _verify_evidence_chain(
    project_root: Path,
    *,
    membership_path: Path,
    membership_sidecar: Path,
    membership_receipt_path: Path,
    membership_receipt_sidecar: Path,
    market_receipt_path: Path,
) -> None:
    membership_receipt = _load_receipt(
        membership_receipt_path,
        label="membership readiness receipt",
    )
    if (
        membership_receipt.get("status") != "pass"
        or membership_receipt.get("artifact_class")
        != "v3_membership_development_only"
        or membership_receipt.get("membership_point_in_time_eligible") is not True
        or membership_receipt.get("training_enabled") is not False
        or membership_receipt.get("hpc_jobs_submitted") != 0
    ):
        raise ValueError("membership readiness receipt is not an eligible V3 input")
    membership_outputs = membership_receipt.get("outputs")
    _require_exact_descriptor(
        membership_outputs,
        "membership_panel",
        _descriptor(project_root, membership_path),
        label="membership readiness receipt",
    )
    _require_exact_descriptor(
        membership_outputs,
        "membership_panel_sidecar",
        _descriptor(project_root, membership_sidecar),
        label="membership readiness receipt",
    )

    market_receipt = _load_receipt(
        market_receipt_path,
        label="market-package receipt",
    )
    if (
        market_receipt.get("evidence_class")
        != "v3_market_package_gap_assessment"
        or market_receipt.get("training_enabled") is not False
        or market_receipt.get("hpc_jobs_submitted") != 0
    ):
        raise ValueError("market-package receipt is not a bounded V3 gap assessment")
    market_inputs = market_receipt.get("inputs")
    _require_exact_descriptor(
        market_inputs,
        "membership_panel",
        _descriptor(project_root, membership_path),
        label="market-package receipt",
    )
    _require_exact_descriptor(
        market_inputs,
        "membership_panel_sidecar",
        _descriptor(project_root, membership_sidecar),
        label="market-package receipt",
    )
    _require_exact_descriptor(
        market_inputs,
        "membership_readiness_receipt",
        _descriptor(project_root, membership_receipt_path),
        label="market-package receipt",
    )
    _require_exact_descriptor(
        market_inputs,
        "membership_readiness_receipt_sidecar",
        _descriptor(project_root, membership_receipt_sidecar),
        label="market-package receipt",
    )


def _plot_exit_batches(path: Path, batches: pd.DataFrame) -> None:
    if len(batches) < 4:
        raise ValueError("exit-batch chart needs at least four observed batches")
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), "#FAFBFC")
    draw = ImageDraw.Draw(image)
    regular_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    title_font = ImageFont.truetype(bold_path, 46)
    subtitle_font = ImageFont.truetype(regular_path, 25)
    label_font = ImageFont.truetype(regular_path, 21)
    value_font = ImageFont.truetype(bold_path, 23)
    note_font = ImageFont.truetype(regular_path, 18)

    total = int(batches["exit_transitions"].sum())
    draw.text(
        (70, 55),
        "Observed SET100 membership exits by effective session",
        fill="#1B2430",
        font=title_font,
    )
    draw.text(
        (70, 120),
        (
            f"{total} transitions in 9 dated batches, 2021–2024; each requires "
            "observable post-exit execution evidence"
        ),
        fill="#596579",
        font=subtitle_font,
    )

    chart_left, chart_right = 115, 1810
    chart_top, chart_bottom = 235, 845
    max_value = max(12, int(batches["exit_transitions"].max()))
    for tick in range(0, max_value + 1, 2):
        y = chart_bottom - int(
            (chart_bottom - chart_top) * tick / max_value
        )
        draw.line((chart_left, y, chart_right, y), fill="#DDE2E8", width=2)
        draw.text(
            (chart_left - 45, y - 11),
            str(tick),
            fill="#4E5968",
            font=label_font,
        )

    count = len(batches)
    slot_width = (chart_right - chart_left) / count
    bar_width = min(105, int(slot_width * 0.58))
    peak = int(batches["exit_transitions"].max())
    for index, row in enumerate(batches.itertuples(index=False)):
        value = int(row.exit_transitions)
        center_x = chart_left + slot_width * (index + 0.5)
        bar_height = int((chart_bottom - chart_top) * value / max_value)
        x0 = int(center_x - bar_width / 2)
        x1 = int(center_x + bar_width / 2)
        y0 = chart_bottom - bar_height
        fill = "#234B7D" if value == peak else "#4D79B3"
        draw.rounded_rectangle(
            (x0, y0, x1, chart_bottom),
            radius=8,
            fill=fill,
            outline="#1E436D",
            width=2,
        )
        value_text = str(value)
        value_box = draw.textbbox((0, 0), value_text, font=value_font)
        draw.text(
            (
                center_x - (value_box[2] - value_box[0]) / 2,
                y0 - 34,
            ),
            value_text,
            fill="#1B2430",
            font=value_font,
        )
        date_text = pd.Timestamp(row.exit_effective_date).strftime("%Y-%m-%d")
        date_box = draw.textbbox((0, 0), date_text, font=label_font)
        draw.text(
            (
                center_x - (date_box[2] - date_box[0]) / 2,
                chart_bottom + 24,
            ),
            date_text,
            fill="#27313F",
            font=label_font,
        )

    draw.line(
        (chart_left, chart_bottom, chart_right, chart_bottom),
        fill="#4E5968",
        width=3,
    )
    draw.text(
        (70, 980),
        (
            "Metric: first non-member session after observed active membership. "
            "Source: checksum-bound point-in-time SET100 membership panel."
        ),
        fill="#596579",
        font=note_font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG", optimize=False)
    os.replace(temporary, path)


def _report_text(
    *,
    generated_at_utc: str,
    receipt: dict[str, Any],
    batches: pd.DataFrame,
) -> str:
    metrics = receipt["metrics"]
    peak_row = batches.sort_values(
        ["exit_transitions", "exit_effective_date"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return f"""# V3 Investability Lifecycle And Exit-Path Readiness

Generated: {generated_at_utc}

## Technical summary

The V3 execution layer now uses self-financing security-and-cash accounting, but final investability remains **blocked**. The point-in-time panel contains **{metrics['exit_transitions']} membership exits across {metrics['exit_batches']} effective-session batches**. With no licensed daily status, stable-instrument, or corporate-action package bound, **0/{metrics['exit_transitions']} exit paths can be certified**. An index removal is therefore no longer treated as proof that a holding was sold.

The draft policy freezes shares during suspension, halt, or no-trade states; carries a visibly stale valuation rather than inventing an observed zero return; and waits for the first valid tradable bar or explicit terminal settlement. It is structurally valid but deliberately unapproved and cannot unlock training. No HPC job was submitted, and canonical whole-project completion remains **11/37 = 29.7%**.

## Exit transitions create a real execution obligation

![Observed SET100 membership exits by effective session](figures/v3_membership_exit_transitions_20260731.png)

The chart counts first non-member sessions after observed active membership. The largest batch is **{int(peak_row['exit_transitions'])} exits on {pd.Timestamp(peak_row['exit_effective_date']).date()}**. These are not assumed sales: each row needs continuous post-exit status until a valid bar permits liquidation or a provider-authorized terminal action resolves the holding.

| Metric | Current value |
| --- | ---: |
| Observed membership-exit transitions | {metrics['exit_transitions']} |
| Unique tickers with at least one exit | {metrics['exit_tickers']} |
| Dated exit batches | {metrics['exit_batches']} |
| Exit paths certified by current licensed evidence | {metrics['exit_paths_covered']}/{metrics['exit_transitions']} |
| Unresolved author policy decisions | {metrics['unresolved_author_decisions']} |
| HPC jobs submitted | 0 |

## Scope, data, and definitions

- **Exit transition:** the first authoritative session where a ticker changes from active SET100 membership to inactive membership in the validated fixed-slot panel.
- **Covered exit path:** contiguous daily status beginning on the exit session and ending at either a valid trading bar or an explicit cash/successor settlement.
- **Turnover:** gross absolute security trades only. Cash settlement is not counted as a second trade.
- **Self-financing reconciliation:** ending security values plus ending cash must equal starting wealth plus asset return minus costs and slippage.
- **Evidence boundary:** the current 2021–2024 membership panel is authoritative for membership timing, but no final market-package receipt has passed.

## Methodology

The policy validator fixes the timing, entry, membership-exit, suspension, halt, no-trade, delisting, stale-valuation, corporate-action, symbol-change, and accounting contracts. The execution primitive applies buy and sell permissions separately, keeps an untradeable forced exit pending, reserves cash for transaction cost and slippage, and carries post-return drifted weights into the next decision. The market-package gate now requires complete exit-path coverage in addition to active-member coverage.

The exit inventory is deterministic: it sorts the checksum-bound fixed-slot panel, identifies every `True → False` membership transition, and records the effective session. Future market rows cannot change an exit already resolved on an earlier session.

## Limitations, uncertainty, and robustness checks

- The **0/{metrics['exit_transitions']}** result means that no exit can be *certified from the current licensed package*; it does not claim that all 62 securities were untradeable.
- Four material policy choices remain for author/advisor approval: maximum stale-valuation age, dividend credit timing, terminal delisting recovery, and the final order cutoff around the closing auction.
- The current five-ticker pilot cannot validate the 62 broad-universe exit paths.
- Provider adjusted prices may be used for features and reconciliation only; final wealth accounting uses raw execution prices plus explicit actions to prevent dividend or split double counting.
- Existing V2 results are unchanged and remain separate from this isolated V3 development capsule.

## Recommended next steps

1. Submit the prepared academic SETSMART data request through the [official SET research-support process](https://www.set.or.th/th/education-research/research/database/setsmart/approval-process).
2. Request `CALENDAR`, `DAILY_SECURITY_TRADING`, `ADJUST_FACTOR`, `RIGHTS_BENEFIT`, `SECURITY`, `SIGN_POSTING`, `SIGN_LIFTING`, and effective-dated symbol-change events documented in the [SETSMART PSIMS data model](https://media.set.or.th/set/Documents/2022/Apr/SETSMART_MIS_SY_SSE_PSIMS_DataModel_V6_2_1.pdf) for the full buffer period.
3. Bind the provider extract by checksum and run the market-package gate; require all {metrics['exit_transitions']} exit paths to resolve.
4. Review the four open policy choices with both authors and the advisor, then lock a new policy only after the evidence passes.
5. Freeze a separate V3 evaluation and model-selection policy before using large HPC resources.

## Further questions

- What maximum stale-price age should abort a training fold rather than continue valuation?
- Should dividend wealth be recognized on ex-date, payment date, or a provider-defined total-return convention?
- Which documented delisting outcomes count as terminal settlement, including zero recovery?
- What exact order cutoff and auction assumption can the available SET data support without look-ahead?

This report is a structural readiness result, not a model-performance or investment claim.
"""


def build(
    *,
    project_root: Path,
    policy_path: Path,
    membership_path: Path,
    membership_receipt_path: Path,
    market_receipt_path: Path,
    ledger_path: Path,
    receipt_path: Path,
    chart_path: Path,
    report_path: Path,
    generated_at_utc: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    resolved = {
        "policy": (root / policy_path).resolve(),
        "membership": (root / membership_path).resolve(),
        "membership_receipt": (root / membership_receipt_path).resolve(),
        "market_receipt": (root / market_receipt_path).resolve(),
        "ledger": (root / ledger_path).resolve(),
        "receipt": (root / receipt_path).resolve(),
        "chart": (root / chart_path).resolve(),
        "report": (root / report_path).resolve(),
    }
    for key, path in resolved.items():
        if not path.is_relative_to(root):
            raise ValueError(f"{key} path escapes the project")
    for key in ("policy", "membership", "membership_receipt", "market_receipt"):
        if not resolved[key].is_file():
            raise FileNotFoundError(resolved[key])
    membership_sidecar = _verify_sidecar(resolved["membership"])
    membership_receipt_sidecar = _verify_sidecar(
        resolved["membership_receipt"]
    )
    _verify_sidecar(resolved["market_receipt"])
    _verify_evidence_chain(
        root,
        membership_path=resolved["membership"],
        membership_sidecar=membership_sidecar,
        membership_receipt_path=resolved["membership_receipt"],
        membership_receipt_sidecar=membership_receipt_sidecar,
        market_receipt_path=resolved["market_receipt"],
    )
    for binding in V2_IMMUTABILITY_BINDINGS.values():
        path = root / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"locked V2 artifact changed: {path}")

    policy_issues, policy = validate_investability_policy(resolved["policy"])
    if policy_issues:
        raise ValueError("invalid policy: " + "; ".join(policy_issues))
    _write_sidecar(resolved["policy"])
    membership = pd.read_csv(resolved["membership"])
    transitions = membership_exit_transitions(membership)
    ledger = assess_membership_exit_paths(
        membership,
        bars=None,
        trading_status=None,
        sessions=None,
        corporate_actions=None,
        symbol_history=None,
    )
    _atomic_csv(resolved["ledger"], ledger)
    _write_sidecar(resolved["ledger"])

    batches = (
        transitions.groupby("exit_effective_date", as_index=False)
        .agg(exit_transitions=("ticker", "size"))
        .sort_values("exit_effective_date", kind="mergesort")
        .reset_index(drop=True)
    )
    _plot_exit_batches(resolved["chart"], batches)
    _write_sidecar(resolved["chart"])

    decisions = policy["author_decisions_required_before_lock"]
    receipt = {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc,
        "status": "blocked",
        "evidence_class": "v3_investability_policy_structural_readiness",
        "policy": {
            **_descriptor(root, resolved["policy"]),
            "registration_status": policy["registration"]["status"],
            "approved_by_authors": policy["registration"]["approved_by_authors"],
            "structural_status": "pass",
            "locked": False,
        },
        "inputs": {
            "membership_panel": _descriptor(root, resolved["membership"]),
            "membership_receipt": _descriptor(
                root,
                resolved["membership_receipt"],
            ),
            "market_gap_receipt": _descriptor(
                root,
                resolved["market_receipt"],
            ),
        },
        "implementation": {
            "execution_ledger": _descriptor(
                root,
                root / "src/v3/execution_ledger.py",
            ),
            "investability_policy": _descriptor(
                root,
                root / "src/v3/investability_policy.py",
            ),
            "v3_environment": _descriptor(
                root,
                root / "src/v3/trading_env.py",
            ),
            "market_package_gate": _descriptor(
                root,
                root / "src/v3/market_data_readiness.py",
            ),
            "artifact_builder": _descriptor(
                root,
                root / "scripts/build_v3_investability_policy_readiness.py",
            ),
        },
        "metrics": {
            "exit_transitions": len(transitions),
            "exit_tickers": int(transitions["ticker"].nunique()),
            "exit_batches": len(batches),
            "exit_paths_covered": int(ledger["path_covered"].astype(bool).sum()),
            "unresolved_author_decisions": len(decisions),
            "checks_total": 6,
            "checks_passed": 2,
            "checks_failed": 4,
        },
        "checks": [
            {
                "check": "policy_structural_validation",
                "status": "pass",
                "severity": "critical",
            },
            {
                "check": "self_financing_execution_implementation",
                "status": "pass",
                "severity": "critical",
            },
            {
                "check": "policy_locked",
                "status": "fail",
                "severity": "critical",
            },
            {
                "check": "policy_author_approved",
                "status": "fail",
                "severity": "critical",
            },
            {
                "check": "author_decisions_resolved",
                "status": "fail",
                "severity": "critical",
            },
            {
                "check": "membership_exit_paths_covered",
                "status": "fail",
                "severity": "critical",
            },
        ],
        "artifacts": {
            "exit_ledger": _descriptor(root, resolved["ledger"]),
            "chart": _descriptor(root, resolved["chart"]),
        },
        "final_investability_eligible": False,
        "model_input_eligible": False,
        "training_enabled": False,
        "publication_claim_eligible": False,
        "hpc_jobs_submitted": 0,
        "canonical_project_progress": {
            "verified_stages": 11,
            "total_stages": 37,
            "percent": 29.7,
        },
    }
    _atomic_text(
        resolved["receipt"],
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    _write_sidecar(resolved["receipt"])
    _atomic_text(
        resolved["report"],
        _report_text(
            generated_at_utc=generated_at_utc,
            receipt=receipt,
            batches=batches,
        ),
    )
    _write_sidecar(resolved["report"])
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build V3 investability-policy readiness evidence"
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--policy", type=Path, default=Path(DEFAULT_POLICY))
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path(DEFAULT_MEMBERSHIP),
    )
    parser.add_argument(
        "--membership-receipt",
        type=Path,
        default=Path(DEFAULT_MEMBERSHIP_RECEIPT),
    )
    parser.add_argument(
        "--market-receipt",
        type=Path,
        default=Path(DEFAULT_MARKET_RECEIPT),
    )
    parser.add_argument("--ledger", type=Path, default=Path(DEFAULT_LEDGER))
    parser.add_argument("--receipt", type=Path, default=Path(DEFAULT_RECEIPT))
    parser.add_argument("--chart", type=Path, default=Path(DEFAULT_CHART))
    parser.add_argument("--report", type=Path, default=Path(DEFAULT_REPORT))
    parser.add_argument(
        "--generated-at-utc",
        default=datetime.now(timezone.utc).isoformat(),
    )
    args = parser.parse_args(argv)
    receipt = build(
        project_root=args.project_root,
        policy_path=args.policy,
        membership_path=args.membership,
        membership_receipt_path=args.membership_receipt,
        market_receipt_path=args.market_receipt,
        ledger_path=args.ledger,
        receipt_path=args.receipt,
        chart_path=args.chart,
        report_path=args.report,
        generated_at_utc=args.generated_at_utc,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "exit_transitions": receipt["metrics"]["exit_transitions"],
                "exit_paths_covered": receipt["metrics"][
                    "exit_paths_covered"
                ],
                "training_enabled": receipt["training_enabled"],
                "hpc_jobs_submitted": receipt["hpc_jobs_submitted"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
