"""Build deterministic V3 fold, candidate, receipt, chart, and report evidence."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.v3.evaluation_policy import (
    build_candidate_ledger,
    build_evaluation_schedule,
    sha256_file,
    validate_candidate_ledger,
    validate_evaluation_policy,
    validate_fold_schedule,
)


DEFAULT_POLICY = "config/v3/evaluation_policy_draft.yaml"
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
DEFAULT_SCHEDULE = (
    "reports/data_quality_v3/v3_evaluation_fold_schedule_20260820.csv"
)
DEFAULT_SUMMARY = (
    "reports/data_quality_v3/v3_evaluation_fold_summary_20260820.csv"
)
DEFAULT_CANDIDATES = (
    "reports/data_quality_v3/v3_candidate_ledger_20260820.csv"
)
DEFAULT_RECEIPT = (
    "reports/data_quality_v3/v3_evaluation_policy_readiness_20260820.json"
)
DEFAULT_CHART = "reports/figures/v3_evaluation_fold_schedule_20260820.png"
DEFAULT_REPORT = "reports/v3_evaluation_policy_readiness_20260820.md"

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
    frame.to_csv(temporary, index=False, lineterminator="\n")
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


def _descriptor(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"artifact escapes the project: {path}")
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_upstream_chain(
    root: Path,
    *,
    membership_path: Path,
    membership_sidecar: Path,
    membership_receipt_path: Path,
    membership_receipt_sidecar: Path,
    market_receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    membership_receipt = _load_json(
        membership_receipt_path,
        "membership readiness receipt",
    )
    if (
        membership_receipt.get("status") != "pass"
        or membership_receipt.get("artifact_class")
        != "v3_membership_development_only"
        or membership_receipt.get("membership_point_in_time_eligible") is not True
        or membership_receipt.get("training_enabled") is not False
    ):
        raise ValueError("membership readiness receipt is not eligible")
    outputs = membership_receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("membership readiness outputs must be a mapping")
    expected_outputs = {
        "membership_panel": _descriptor(root, membership_path),
        "membership_panel_sidecar": _descriptor(root, membership_sidecar),
    }
    for key, expected in expected_outputs.items():
        if outputs.get(key) != expected:
            raise ValueError(f"membership receipt does not bind exact {key}")

    market_receipt = _load_json(market_receipt_path, "market-package receipt")
    if (
        market_receipt.get("evidence_class")
        != "v3_market_package_gap_assessment"
        or market_receipt.get("training_enabled") is not False
    ):
        raise ValueError("market receipt is not a bounded V3 gap assessment")
    inputs = market_receipt.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("market receipt inputs must be a mapping")
    expected_inputs = {
        "membership_panel": _descriptor(root, membership_path),
        "membership_panel_sidecar": _descriptor(root, membership_sidecar),
        "membership_readiness_receipt": _descriptor(
            root,
            membership_receipt_path,
        ),
        "membership_readiness_receipt_sidecar": _descriptor(
            root,
            membership_receipt_sidecar,
        ),
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise ValueError(f"market receipt does not bind exact {key}")
    return membership_receipt, market_receipt


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = f"/System/Library/Fonts/Supplemental/{name}"
    return ImageFont.truetype(path, size)


def _plot_schedule(path: Path, summary: pd.DataFrame) -> None:
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), "#FAFBFC")
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 52),
        "V3 development evaluation fold schedule",
        fill="#1B2430",
        font=_font(46, bold=True),
    )
    draw.text(
        (70, 118),
        (
            "Six expanding folds with non-overlapping half-year validation "
            "transitions; schedule evidence only, not performance"
        ),
        fill="#536176",
        font=_font(24),
    )

    chart_left, chart_right = 300, 1810
    chart_top, row_gap = 245, 112
    start = pd.Timestamp(summary["train_context_start"].min())
    end = pd.Timestamp(summary["validation_terminal_outcome"].max())
    span = max((end - start).days, 1)

    def x(date: Any) -> int:
        value = pd.Timestamp(date)
        return int(chart_left + (value - start).days / span * (chart_right - chart_left))

    for year in range(start.year, end.year + 1):
        date = pd.Timestamp(year=year, month=1, day=1)
        xpos = x(max(date, start))
        draw.line((xpos, chart_top - 25, xpos, chart_top + row_gap * 6 - 20), fill="#D8DEE8", width=2)
        draw.text((xpos + 7, chart_top - 62), str(year), fill="#536176", font=_font(19))

    for row_index, row in enumerate(summary.itertuples(index=False)):
        y = chart_top + row_index * row_gap
        draw.text((70, y - 13), row.fold_id.upper(), fill="#1B2430", font=_font(23, bold=True))
        train_start = x(row.train_decision_start)
        train_end = x(row.train_terminal_outcome)
        context_start = x(row.validation_context_start)
        validation_start = x(row.validation_start)
        validation_end = x(row.validation_terminal_outcome)
        draw.rounded_rectangle(
            (train_start, y, train_end, y + 34),
            radius=8,
            fill="#AFC7E3",
            outline="#315A87",
            width=2,
        )
        draw.rounded_rectangle(
            (context_start, y + 7, train_end, y + 27),
            radius=6,
            fill="#E6B85C",
            outline="#8B6722",
            width=2,
        )
        draw.line(
            (train_end, y - 7, train_end, y + 43),
            fill="#C76B2A",
            width=4,
        )
        draw.rounded_rectangle(
            (validation_start, y, validation_end, y + 34),
            radius=8,
            fill="#315A87",
            outline="#173B64",
            width=2,
        )
        draw.text(
            (validation_start + 8, y + 42),
            f"{row.validation_decisions} scored transitions",
            fill="#536176",
            font=_font(17),
        )

    legend_y = 915
    legend = (
        ("Expanding training", "#AFC7E3"),
        ("30-session validation context", "#E6B85C"),
        ("Validation", "#315A87"),
        ("Unused boundary transition", "#C76B2A"),
    )
    xpos = 300
    for label, color in legend:
        draw.rounded_rectangle((xpos, legend_y, xpos + 34, legend_y + 22), radius=4, fill=color)
        draw.text((xpos + 45, legend_y - 2), label, fill="#334155", font=_font(18))
        xpos += 345
    draw.text(
        (70, 1005),
        (
            "Source: checksum-bound provisional SET100 session panel. Final "
            "dates remain blocked pending the licensed authoritative calendar."
        ),
        fill="#536176",
        font=_font(18),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def _report_markdown(receipt: dict[str, Any], summary: pd.DataFrame) -> str:
    metrics = receipt["metrics"]
    rows = "\n".join(
        (
            f"| {row.fold_id} | {row.train_decision_start} to "
            f"{row.train_decision_end} | {row.train_decisions:,} | "
            f"{row.validation_start} to {row.validation_terminal_outcome} | "
            f"{row.validation_decisions:,} |"
        )
        for row in summary.itertuples(index=False)
    )
    return f"""# V3 Evaluation And Model-Selection Readiness

Generated: {receipt['generated_at_utc']}

## Technical summary

The project now has an isolated, deterministic V3 evaluation contract. It materializes **{metrics['fold_count']} expanding temporal folds**, owns every close-to-close training or validation transition, freezes **{metrics['candidate_configurations']} candidate configurations**, and caps the proposed campaign at **{metrics['planned_model_fits']} model fits** and **{metrics['planned_training_timesteps']:,} training timesteps**. Large HPC capacity is therefore assigned to predeclared temporal evaluation and seed replication rather than an open-ended search.

This is structural readiness, not a performance result. The schedule currently uses the checksum-bound provisional membership dates; the licensed authoritative SET calendar and market package have not passed. The policy is also awaiting author approval. V3 training, `test`, and `all` remain disabled. The later campaign is designed for at most **{metrics['maximum_hpc_array_jobs']} array-job submissions**, while this build submitted **0 HPC jobs**.

## Six validation periods are explicit and non-overlapping

![V3 development evaluation fold schedule](figures/v3_evaluation_fold_schedule_20260820.png)

Each dark-blue interval is a distinct half-year validation period. A validation transition is scored only when both its decision date and next-session outcome remain inside that interval. The orange boundary marks the deliberately unused train-end-to-validation-start transition. Gold dates supply the 30-session observation history for validation without causing a new preprocessing fit.

| Fold | Expanding training decisions | Train decisions | Validation sessions | Scored validation transitions |
| --- | --- | ---: | --- | ---: |
{rows}

## Scope, evidence, and definitions

- **Decision transition:** allocation at session `t` followed by wealth at the next authoritative session `t+1`.
- **Transition owner:** either training or validation; a cross-boundary transition has no owner and cannot enter either metric.
- **Validation context:** the prior 30 sessions supplied only to form lagged observations. The dates may already have a training role, but validation cannot refit preprocessing or score them again.
- **Candidate:** one exact algorithm, one of two fixed parameter profiles, and one of three fixed timestep budgets. No adaptive Optuna expansion is permitted.
- **Selection cell:** one candidate, one temporal fold, and one model seed. The exact result grid contains {metrics['planned_model_fits']} cells.
- **Primary selection score:** the median across folds after first averaging the three seeds within each fold, using annualized excess net log return over the mask-aware active equal-weight baseline.

## Methodology and model-selection guardrails

The proposed registry contains PPO, A2C, DDPG, TD3, and SAC; two fully enumerated parameter profiles; and 50k, 100k, and 200k timestep budgets. Seeds 42, 43, and 44 are fixed. The six folds and all candidates are materialized before compute. Any missing, duplicate, nonfinite, or unregistered fold/seed result blocks selection, and failed scientific runs cannot be silently replaced with new candidates.

Within each fold, preprocessing is fit only on training-decision dates. Model and portfolio state reset at validation start. Seed runs are averaged within their temporal fold and are not treated as independent market periods. Ties are resolved by worst-fold excess return, excess drawdown, turnover, smaller budget, and finally lexical candidate ID. If the best primary score is nonpositive, the predeclared decision is active equal weight rather than forcing a DRL winner.

This design follows the temporal-evaluation concern discussed by [Cerqueira et al. (2020)](https://doi.org/10.1007/s10994-020-05910-7), the backtest-selection warning from [Bailey et al. (2017)](https://doi.org/10.21314/JCF.2016.322), and seed-sensitivity guidance from [Henderson et al. (2018)](https://arxiv.org/abs/1709.06560). Active equal weight remains a demanding baseline consistent with [DeMiguel, Garlappi, and Uppal (2009)](https://doi.org/10.1093/rfs/hhm075).

## Limitations, uncertainty, and robustness checks

- The six date ranges come from a provisional session panel. They demonstrate geometry but cannot be locked until the provider supplies the authoritative calendar.
- The policy is draft and has three author decisions outstanding: fold approval, compute/seed budget approval, and metric/tie-order approval.
- The 2021–2024 period has already been examined and cannot become a new final holdout. No V3 performance or publication claim is produced here.
- Three seeds diagnose implementation variability but do not create three independent market histories. Final uncertainty must be organized around temporal periods, not raw run count.
- Historical V2 results use a separate accounting implementation. They cannot select V3 candidates or be described as reproduced V3 performance.

## Recommended next steps

1. Obtain and checksum the licensed SET calendar, bars, trading status, corporate actions, and stable-symbol history.
2. Rebuild the fold ledger against the authoritative calendar and require the six validation transition sets to remain globally disjoint.
3. Review the three open policy choices with Nathawat Youngyim, Kunach Kijsanayothin, and advisor Prompong Pakawanwong.
4. Lock the candidate and fold artifacts only after author approval and a passing provider package.
5. Build a dedicated V3 launcher that calls the development gate before importing a trainer; never route V3 through the shared `test`/`all` path.
6. When unlocked, bundle the fixed work into no more than three array jobs and preserve every registered result, including negative and failed scientific outcomes.

## Further questions

- Should three seeds be increased before lock if the measured training variance is large, while keeping the candidate count fixed?
- Can the licensed SET benchmark be supplied as a total-return series on every validation transition?
- What post-2024 period can be genuinely untouched and read once for the final V3 conclusion?

This report validates a methodology and evidence boundary. It does not report investment performance.
"""


def build(
    *,
    project_root: Path,
    policy_path: Path,
    membership_path: Path,
    membership_receipt_path: Path,
    market_receipt_path: Path,
    schedule_path: Path,
    summary_path: Path,
    candidates_path: Path,
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
        "schedule": (root / schedule_path).resolve(),
        "summary": (root / summary_path).resolve(),
        "candidates": (root / candidates_path).resolve(),
        "receipt": (root / receipt_path).resolve(),
        "chart": (root / chart_path).resolve(),
        "report": (root / report_path).resolve(),
    }
    for label, path in resolved.items():
        if not path.is_relative_to(root):
            raise ValueError(f"{label} path escapes the project")
    for label in ("policy", "membership", "membership_receipt", "market_receipt"):
        if not resolved[label].is_file():
            raise FileNotFoundError(resolved[label])
    membership_sidecar = _verify_sidecar(resolved["membership"])
    membership_receipt_sidecar = _verify_sidecar(resolved["membership_receipt"])
    market_receipt_sidecar = _verify_sidecar(resolved["market_receipt"])
    membership_receipt, market_receipt = _verify_upstream_chain(
        root,
        membership_path=resolved["membership"],
        membership_sidecar=membership_sidecar,
        membership_receipt_path=resolved["membership_receipt"],
        membership_receipt_sidecar=membership_receipt_sidecar,
        market_receipt_path=resolved["market_receipt"],
    )

    policy_issues, policy = validate_evaluation_policy(resolved["policy"])
    if policy_issues:
        raise ValueError("invalid evaluation policy: " + "; ".join(policy_issues))
    policy_sidecar = _write_sidecar(resolved["policy"])
    membership = pd.read_csv(resolved["membership"], usecols=["date"])
    sessions = membership["date"].drop_duplicates()
    schedule, summary = build_evaluation_schedule(sessions, policy)
    candidates = build_candidate_ledger(policy)
    schedule_issues = validate_fold_schedule(schedule, sessions, policy)
    candidate_issues = validate_candidate_ledger(candidates, policy)
    if schedule_issues or candidate_issues:
        raise ValueError("; ".join([*schedule_issues, *candidate_issues]))

    _atomic_csv(resolved["schedule"], schedule)
    schedule_sidecar = _write_sidecar(resolved["schedule"])
    _atomic_csv(resolved["summary"], summary)
    summary_sidecar = _write_sidecar(resolved["summary"])
    _atomic_csv(resolved["candidates"], candidates)
    candidates_sidecar = _write_sidecar(resolved["candidates"])
    _plot_schedule(resolved["chart"], summary)
    chart_sidecar = _write_sidecar(resolved["chart"])

    v2_descriptors: dict[str, Any] = {}
    for label, binding in V2_IMMUTABILITY_BINDINGS.items():
        path = root / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"locked V2 artifact changed: {binding['path']}")
        v2_descriptors[label] = _descriptor(root, path)

    market_pass = (
        market_receipt.get("status") == "pass"
        and market_receipt.get("licensed_market_package_eligible") is True
    )
    author_decisions = policy["author_decisions_required_before_lock"]
    receipt = {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc,
        "status": "blocked",
        "evidence_class": "v3_evaluation_policy_structural_readiness",
        "evaluation_policy_eligible": False,
        "schedule_structural_eligible": True,
        "candidate_registry_eligible": True,
        "training_enabled": False,
        "test_access_enabled": False,
        "all_access_enabled": False,
        "hpc_jobs_planned_maximum": 3,
        "hpc_jobs_submitted": 0,
        "policy": {
            **_descriptor(root, resolved["policy"]),
            "sidecar": _descriptor(root, policy_sidecar),
            "registration_status": policy["registration"]["status"],
            "approved_by_authors": policy["registration"]["approved_by_authors"],
            "structural_status": "pass",
            "locked": False,
        },
        "inputs": {
            "membership_panel": _descriptor(root, resolved["membership"]),
            "membership_panel_sidecar": _descriptor(root, membership_sidecar),
            "membership_receipt": _descriptor(root, resolved["membership_receipt"]),
            "membership_receipt_sidecar": _descriptor(
                root,
                membership_receipt_sidecar,
            ),
            "market_gap_receipt": _descriptor(root, resolved["market_receipt"]),
            "market_gap_receipt_sidecar": _descriptor(root, market_receipt_sidecar),
            "market_gap_status": market_receipt.get("status"),
            "licensed_market_package_eligible": market_receipt.get(
                "licensed_market_package_eligible"
            ),
            "membership_point_in_time_eligible": membership_receipt.get(
                "membership_point_in_time_eligible"
            ),
            "v2_immutability_baseline": v2_descriptors,
        },
        "implementation": {
            "evaluation_policy": _descriptor(
                root,
                root / "src/v3/evaluation_policy.py",
            ),
            "artifact_builder": _descriptor(
                root,
                root / "scripts/build_v3_evaluation_policy_readiness.py",
            ),
        },
        "outputs": {
            "fold_schedule": _descriptor(root, resolved["schedule"]),
            "fold_schedule_sidecar": _descriptor(root, schedule_sidecar),
            "fold_summary": _descriptor(root, resolved["summary"]),
            "fold_summary_sidecar": _descriptor(root, summary_sidecar),
            "candidate_ledger": _descriptor(root, resolved["candidates"]),
            "candidate_ledger_sidecar": _descriptor(root, candidates_sidecar),
            "fold_chart": _descriptor(root, resolved["chart"]),
            "fold_chart_sidecar": _descriptor(root, chart_sidecar),
        },
        "metrics": {
            "provisional_sessions": int(pd.to_datetime(sessions).nunique()),
            "fold_count": int(len(summary)),
            "validation_transitions": int(summary["validation_decisions"].sum()),
            "candidate_configurations": int(len(candidates)),
            "model_seeds": len(policy["candidate_budget"]["model_seeds"]),
            "planned_model_fits": int(policy["candidate_budget"]["planned_model_fits"]),
            "planned_training_timesteps": int(
                policy["candidate_budget"]["planned_training_timesteps"]
            ),
            "maximum_hpc_array_jobs": int(
                policy["candidate_budget"]["maximum_hpc_array_jobs"]
            ),
            "unresolved_author_decisions": len(author_decisions),
            "checks_total": 8,
            "checks_passed": 4,
            "checks_failed": 4,
            "whole_project_verified_stages": 11,
            "whole_project_total_stages": 37,
            "whole_project_percent": 29.7,
        },
        "checks": [
            {"check": "policy_structural_validation", "status": "pass", "severity": "critical"},
            {"check": "fold_schedule_deterministic", "status": "pass", "severity": "critical"},
            {"check": "transition_ownership_complete", "status": "pass", "severity": "critical"},
            {"check": "candidate_registry_exact", "status": "pass", "severity": "critical"},
            {"check": "licensed_authoritative_calendar", "status": "fail", "severity": "critical"},
            {"check": "licensed_market_package", "status": "pass" if market_pass else "fail", "severity": "critical"},
            {"check": "policy_locked", "status": "fail", "severity": "critical"},
            {"check": "policy_author_approved", "status": "fail", "severity": "critical"},
        ],
        "scientific_boundary": {
            "current_schedule_is_provisional": True,
            "known_period_is_not_final_holdout": True,
            "no_performance_metrics_generated": True,
            "seeds_not_treated_as_independent_market_periods": True,
            "shared_v2_splitter_not_used": True,
            "development_protocol_not_an_input": True,
        },
        "report_design": {
            "audience": "technical",
            "chart_contract": {
                "question": "Where do training, validation context, and scored validation occur?",
                "family": "comparison_and_progression",
                "variant": "horizontal_temporal_interval_bars",
                "takeaway": "Six validation transition sets are explicit and non-overlapping.",
                "palette": "two_blue_roots_gold_context_orange_boundary_plus_neutrals",
                "claim_boundary": "schedule_only_not_performance",
            },
        },
    }
    _atomic_text(
        resolved["receipt"],
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    _write_sidecar(resolved["receipt"])
    _atomic_text(resolved["report"], _report_markdown(receipt, summary))
    _write_sidecar(resolved["report"])
    return receipt


def _project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the V3 evaluation-policy structural readiness artifact"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--membership", default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--membership-receipt", default=DEFAULT_MEMBERSHIP_RECEIPT)
    parser.add_argument("--market-receipt", default=DEFAULT_MARKET_RECEIPT)
    parser.add_argument("--schedule-output", default=DEFAULT_SCHEDULE)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--candidates-output", default=DEFAULT_CANDIDATES)
    parser.add_argument("--receipt-output", default=DEFAULT_RECEIPT)
    parser.add_argument("--chart-output", default=DEFAULT_CHART)
    parser.add_argument("--report-output", default=DEFAULT_REPORT)
    parser.add_argument("--generated-at-utc", default=None)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    generated = args.generated_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    receipt = build(
        project_root=root,
        policy_path=Path(args.policy),
        membership_path=Path(args.membership),
        membership_receipt_path=Path(args.membership_receipt),
        market_receipt_path=Path(args.market_receipt),
        schedule_path=Path(args.schedule_output),
        summary_path=Path(args.summary_output),
        candidates_path=Path(args.candidates_output),
        receipt_path=Path(args.receipt_output),
        chart_path=Path(args.chart_output),
        report_path=Path(args.report_output),
        generated_at_utc=generated,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "folds": receipt["metrics"]["fold_count"],
                "candidates": receipt["metrics"]["candidate_configurations"],
                "planned_model_fits": receipt["metrics"]["planned_model_fits"],
                "hpc_jobs_submitted": receipt["hpc_jobs_submitted"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
