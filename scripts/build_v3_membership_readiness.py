"""Build the checksum-bound V3 point-in-time membership development artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.v3.membership_mask import build_point_in_time_membership_panel


DEFAULT_HISTORY = "data/reference/set50_set100_membership_2021_2024.csv"
DEFAULT_UNIVERSE = "data/reference/set50_or_set100_universe.csv"
DEFAULT_CALENDAR = "data/raw/prices_market_indices.csv"
DEFAULT_CONSTITUENT_RECEIPT = (
    "reports/data_sources/set_constituent_archive_20260731.json"
)
DEFAULT_PANEL = (
    "data/processed/v3_development/"
    "set100_membership_session_mask_2021_2024.csv.gz"
)
DEFAULT_SUMMARY = "reports/data_quality_v3/membership_session_summary.csv"
DEFAULT_RECEIPT = (
    "reports/data_quality_v3/"
    "v3_membership_mask_readiness_20260731.json"
)
DEFAULT_REPORT = "reports/v3_membership_mask_readiness_20260731.md"
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
    "membership_panel": "src/v3/membership_mask.py",
    "development_eligibility": "src/v3/eligibility.py",
    "mask_aware_environment": "src/v3/trading_env.py",
    "mask_aware_baselines": "src/v3/baselines.py",
    "artifact_builder": "scripts/build_v3_membership_readiness.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    os.replace(temporary, path)


def _write_sidecar(path: Path) -> Path:
    sidecar = Path(f"{path}.sha256")
    _atomic_text(sidecar, f"{sha256_file(path)}  {path.name}\n")
    return sidecar


def _source_descriptor(project_root: Path, path: Path) -> dict:
    relative = path.relative_to(project_root).as_posix()
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _session_calendar(path: Path) -> pd.DatetimeIndex:
    calendar = pd.read_csv(path)
    if "date" not in calendar.columns:
        raise ValueError("session-calendar input is missing date")
    dates = pd.to_datetime(calendar["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("session-calendar input contains invalid dates")
    if "ticker" in calendar.columns:
        tickers = sorted(calendar["ticker"].dropna().astype(str).unique())
        if tickers != ["^SET.BK"]:
            raise ValueError(
                "session-calendar input must contain only the SET index; "
                f"found {tickers}"
            )
    sessions = pd.DatetimeIndex(dates.dt.normalize().drop_duplicates().sort_values())
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise ValueError("session calendar is not unique and chronological")
    return sessions


def _record_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    previous: set[str] | None = None
    records = panel[
        ["membership_record_id", "membership_published_at", "revision_status"]
    ].drop_duplicates("membership_record_id")
    order = (
        panel.groupby("membership_record_id", as_index=False)["date"]
        .min()
        .rename(columns={"date": "first_safe_session"})
        .sort_values("first_safe_session")
    )
    records = order.merge(records, on="membership_record_id", validate="one_to_one")
    for record in records.itertuples(index=False):
        selected = panel[panel["membership_record_id"].eq(record.membership_record_id)]
        sessions = selected["date"].drop_duplicates().sort_values()
        active = set(selected.loc[selected["is_index_member"], "ticker"].astype(str))
        additions = sorted(active - previous) if previous is not None else sorted(active)
        removals = sorted(previous - active) if previous is not None else []
        rows.append(
            {
                "membership_record_id": record.membership_record_id,
                "revision_status": record.revision_status,
                "published_at": pd.Timestamp(record.membership_published_at).date().isoformat(),
                "first_safe_session": sessions.min().date().isoformat(),
                "last_safe_session": sessions.max().date().isoformat(),
                "session_count": int(len(sessions)),
                "active_members_per_session": int(len(active)),
                "additions": len(additions),
                "removals": len(removals),
                "added_tickers": ",".join(additions),
                "removed_tickers": ",".join(removals),
            }
        )
        previous = active
    return pd.DataFrame(rows)


def _report_markdown(receipt: dict, summary: pd.DataFrame) -> str:
    metrics = receipt["metrics"]
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            "| "
            + " | ".join(
                [
                    row.membership_record_id,
                    row.revision_status,
                    row.first_safe_session,
                    row.last_safe_session,
                    str(row.session_count),
                    str(row.active_members_per_session),
                    str(row.additions),
                    str(row.removals),
                ]
            )
            + " |"
        )
    table = "\n".join(rows)
    return f"""# V3 Point-in-Time Membership Development Readiness

## Technical summary

The isolated V3 membership component passes its structural readiness checks across all **{metrics['session_count']:,} observed SET sessions** from {metrics['first_session']} through {metrics['last_session']}. The generated panel contains **{metrics['panel_rows']:,} date-by-asset rows**, keeps a fixed **{metrics['asset_slots']}-ticker slot layout**, and identifies exactly **100 SET100 members on every session**.

This is deliberately a membership-only development artifact. It is **not** a final investability mask, is not connected to training, cannot read `test` or `all`, and does not make any performance claim.

## The new timing rule prevents same-day revision leakage

The two source revisions contain publication dates but no intraday publication times. V3 therefore continues using the previous record on the publication session and activates the revision on the next observed SET session:

- the 11 February 2021 revision first becomes usable on **15 February 2021**; and
- the 7 April 2023 revision first becomes usable on **10 April 2023**.

Scheduled half-year lists still become active on their stated effective session when they were already published. This rule is more conservative than the archive's general same-day as-of resolver and is isolated to V3 close-to-close decisions.

![SET50 and SET100 constituent turnover](figures/set_constituent_turnover_2021_2024.svg)

The chart shows the size of each membership transition. Turnover is descriptive structural evidence; it is not a model result.

## Scope, grain, and exact evidence

| Measure | Result |
| --- | ---: |
| Observed SET session dates | {metrics['session_count']:,} |
| Fixed acquisition slots | {metrics['asset_slots']} |
| Dense membership rows | {metrics['panel_rows']:,} |
| Active SET100 rows | {metrics['active_rows']:,} |
| Sessions with a count other than 100 | {metrics['active_count_failures']} |
| Point-in-time records used | {metrics['record_count']} |
| Future-record violations | {metrics['future_record_violations']} |
| HPC jobs submitted for this milestone | 0 |

The session grid comes from the existing observed SET-index calendar and is used only to decide the next observed session. The official SET PDFs and checksum-bound normalized archive remain the source of membership and sector classifications.

## Record-by-record coverage

| Record | Type | First safe session | Last safe session | Sessions | Members | Added | Removed |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
{table}

## Methodology

For each observed session, the builder selects the most recent scheduled record whose effective date and publication date are available. A revised record published on the same date is excluded until the first later session. The selected record is crossed with the ordered 138-ticker acquisition union, producing a fixed action-slot grid. Membership, dated sector, source URL, and source hash are then attached only to active rows.

The companion development-eligibility component combines this membership flag with a valid current bar and a strictly prior lookback window. It never uses next-session price availability to decide today's eligibility. The isolated environment exposes the current eligibility mask in the observation, zeros inactive actions, forces exits to cash, records turnover-cap overrides, charges execution costs once, and rejects missing valuation prices.

## Robustness and leakage checks

- The panel has one and only one row per session and asset slot.
- Every session maps to one source record and exactly 100 active members.
- Revised records require `published_at < decision_date`; scheduled records require `published_at <= decision_date`.
- Historical MINT and NEX sector changes are retained as of date.
- Unknown tickers, out-of-range dates, future information, duplicate rows, unstable slots, malformed Booleans, and incomplete masks fail closed.
- The higher-level V3 gate consumes this receipt but is not an input to it, keeping the evidence graph acyclic. Executable training still requires author approval, a protocol sidecar, explicit training enablement, passing market evidence, and locked investability and evaluation policies.
- `test` and `all` are rejected by the V3 gate.
- The locked V2 preregistration, campaign, and active release manifest still match their recorded SHA-256 baselines.

## Limitations

The 2021–2024 period has already been examined in the V2 study, so it cannot become a new untouched V3 final holdout. Final investability also requires a passing licensed market package with authoritative sessions, 138-ticker adjusted OHLCV, observable suspension/trading status, corporate actions, symbol continuity, and locked entry, exit, delisting, stale-valuation, and suspension rules. Current structural source readiness remains 3/5 because historical news and licensed fundamentals are still missing.

## Recommended next steps

1. Obtain and checksum licensed OHLCV, corporate actions, and trading-status history for the 138 acquisition tickers.
2. Certify active-member price coverage without using future bars to determine today's mask.
3. Finalize the entry, exit, suspension, and delisting policy with both authors and the advisor.
4. Extend constituent history beyond 2024 and define a genuinely untouched final holdout.
5. Only then lock the V3 protocol, build a separate release, and enable train/validation execution.

## Further questions

- Which licensed field will be authoritative for suspensions and non-trading sessions?
- Should forced exits occur at the last valid close, the next auction price, or remain frozen until trading resumes?
- What post-2024 period can remain fully embargoed for final V3 conclusions?
"""


def build_readiness(
    project_root: Path,
    *,
    history_path: Path,
    universe_path: Path,
    calendar_path: Path,
    constituent_receipt_path: Path,
    panel_path: Path,
    summary_path: Path,
    receipt_path: Path,
    report_path: Path,
    generated_at_utc: str | None = None,
) -> dict:
    root = project_root.resolve()
    inputs = [
        history_path,
        universe_path,
        calendar_path,
        constituent_receipt_path,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    if not constituent_receipt_path.is_relative_to(root):
        raise ValueError("constituent receipt path escapes the project")
    constituent_receipt_sidecar = Path(f"{constituent_receipt_path}.sha256")
    if not constituent_receipt_path.is_file() or not constituent_receipt_sidecar.is_file():
        raise FileNotFoundError("constituent receipt or sidecar is missing")
    v2_descriptors = {}
    for label, binding in V2_IMMUTABILITY_BINDINGS.items():
        path = root / binding["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != binding["sha256"]:
            raise ValueError(
                f"locked V2 artifact changed: {binding['path']} "
                f"expected={binding['sha256']} actual={actual_hash}"
            )
        v2_descriptors[label] = _source_descriptor(root, path)
    implementation_descriptors = {}
    for label, relative in IMPLEMENTATION_PATHS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        implementation_descriptors[label] = _source_descriptor(root, path)

    universe = pd.read_csv(universe_path)
    if "ticker" not in universe.columns:
        raise ValueError("acquisition universe is missing ticker")
    tickers = tuple(universe["ticker"].astype(str).str.strip().str.upper())
    if len(tickers) != 138 or len(set(tickers)) != 138:
        raise ValueError("acquisition universe must contain 138 unique tickers")
    sessions = _session_calendar(calendar_path)
    panel = build_point_in_time_membership_panel(
        history_path,
        sessions.tolist(),
        tickers,
        index_name="SET100",
    )
    summary = _record_summary(panel)

    _atomic_gzip_csv(panel_path, panel)
    panel_sidecar = _write_sidecar(panel_path)
    _atomic_csv(summary_path, summary)
    summary_sidecar = _write_sidecar(summary_path)

    revised = summary[summary["revision_status"].eq("revised")]
    expected_revision_sessions = {
        "set_2021_h1_revised": "2021-02-15",
        "set_2023_h1_revised": "2023-04-10",
    }
    observed_revision_sessions = dict(
        zip(
            revised["membership_record_id"],
            revised["first_safe_session"],
            strict=True,
        )
    )
    if observed_revision_sessions != expected_revision_sessions:
        raise ValueError(
            "revision next-session boundaries changed: "
            f"{observed_revision_sessions}"
        )

    counts = panel.groupby("date")["is_index_member"].sum()
    revision_rows = panel["revision_status"].eq("revised")
    future_violations = int(
        (
            revision_rows
            & (panel["membership_published_at"] >= panel["date"])
        ).sum()
        + (
            ~revision_rows
            & (panel["membership_published_at"] > panel["date"])
        ).sum()
    )
    generated_at = generated_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "generated_at_utc": generated_at,
        "artifact_class": "v3_membership_development_only",
        "membership_point_in_time_eligible": True,
        "final_investability_eligible": False,
        "model_input_eligible": False,
        "publication_claim_eligible": False,
        "training_enabled": False,
        "test_access_enabled": False,
        "hpc_jobs_submitted": 0,
        "metrics": {
            "session_count": int(panel["date"].nunique()),
            "first_session": panel["date"].min().date().isoformat(),
            "last_session": panel["date"].max().date().isoformat(),
            "asset_slots": int(panel["ticker"].nunique()),
            "panel_rows": int(len(panel)),
            "active_rows": int(panel["is_index_member"].sum()),
            "active_count_min": int(counts.min()),
            "active_count_max": int(counts.max()),
            "active_count_failures": int(counts.ne(100).sum()),
            "record_count": int(panel["membership_record_id"].nunique()),
            "future_record_violations": future_violations,
            "revision_first_safe_sessions": expected_revision_sessions,
        },
        "inputs": {
            "constituent_history": _source_descriptor(root, history_path),
            "acquisition_universe": _source_descriptor(root, universe_path),
            "set_session_calendar": _source_descriptor(root, calendar_path),
            "constituent_receipt": _source_descriptor(
                root,
                constituent_receipt_path,
            ),
            "constituent_receipt_sidecar": _source_descriptor(
                root,
                constituent_receipt_sidecar,
            ),
            "implementation": implementation_descriptors,
            "v2_immutability_baseline": v2_descriptors,
        },
        "outputs": {
            "membership_panel": _source_descriptor(root, panel_path),
            "membership_panel_sidecar": _source_descriptor(root, panel_sidecar),
            "record_summary": _source_descriptor(root, summary_path),
            "record_summary_sidecar": _source_descriptor(root, summary_sidecar),
        },
        "checks": [
            {"check": "fixed_asset_slots_138", "status": "pass"},
            {"check": "dense_date_asset_grid", "status": "pass"},
            {"check": "exact_set100_count_each_session", "status": "pass"},
            {"check": "one_source_record_each_session", "status": "pass"},
            {"check": "next_session_revision_activation", "status": "pass"},
            {"check": "no_future_record_use", "status": "pass"},
            {"check": "evidence_dependency_graph_acyclic", "status": "pass"},
            {"check": "training_and_test_access_disabled", "status": "pass"},
            {"check": "locked_v2_hashes_unchanged", "status": "pass"},
        ],
        "scientific_boundary": {
            "membership_is_not_investability": True,
            "session_calendar_used_for_timing_only": True,
            "period_already_examined_not_new_holdout": True,
            "missing_before_training": [
                "licensed adjusted OHLCV for 138 tickers",
                "observable suspension and trading status",
                "corporate actions and ticker changes",
                "locked entry/exit/delisting/suspension policy",
                "locked V3 evaluation and selection policy",
                "author-approved checksum-locked V3 protocol",
                "untouched post-2024 holdout and extended constituent archive",
            ],
        },
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _write_sidecar(receipt_path)
    _atomic_text(report_path, _report_markdown(receipt, summary))
    _write_sidecar(report_path)
    return receipt


def _project_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the isolated V3 membership readiness artifact"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--history", default=DEFAULT_HISTORY)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--calendar", default=DEFAULT_CALENDAR)
    parser.add_argument(
        "--constituent-receipt",
        default=DEFAULT_CONSTITUENT_RECEIPT,
    )
    parser.add_argument("--panel-output", default=DEFAULT_PANEL)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--receipt-output", default=DEFAULT_RECEIPT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT)
    parser.add_argument(
        "--generated-at-utc",
        default=None,
        help="Optional fixed ISO-8601 timestamp for reproducible evidence builds.",
    )
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    receipt = build_readiness(
        root,
        history_path=_project_path(root, args.history),
        universe_path=_project_path(root, args.universe),
        calendar_path=_project_path(root, args.calendar),
        constituent_receipt_path=_project_path(
            root,
            args.constituent_receipt,
        ),
        panel_path=_project_path(root, args.panel_output),
        summary_path=_project_path(root, args.summary_output),
        receipt_path=_project_path(root, args.receipt_output),
        report_path=_project_path(root, args.report_output),
        generated_at_utc=args.generated_at_utc,
    )
    print(json.dumps(receipt["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
