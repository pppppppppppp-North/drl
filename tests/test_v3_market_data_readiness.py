from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.v3.market_data_readiness import (
    CORPORATE_ACTION_COLUMNS,
    REQUIRED_BINDINGS,
    assess_v3_market_package,
    sha256_file,
    verify_market_package_bindings,
)
from src.v3.membership_mask import (
    MASK_KIND,
    REVISION_AVAILABILITY_RULE,
    validate_membership_panel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _membership() -> pd.DataFrame:
    rows = []
    tickers = ("AAA.BK", "BBB.BK", "CCC.BK")
    sectors = ("Energy", "Energy", "Banking")
    for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
        for slot, (ticker, sector) in enumerate(zip(tickers, sectors, strict=True)):
            rows.append(
                {
                    "date": date,
                    "asset_slot": slot,
                    "ticker": ticker,
                    "index_name": "SET100",
                    "is_index_member": True,
                    "membership_record_id": "synthetic_market_record",
                    "membership_published_at": pd.Timestamp("2023-12-01"),
                    "revision_status": "original",
                    "sector_as_of": sector,
                    "symbol": ticker.removesuffix(".BK"),
                    "company_name": ticker,
                    "source_url": "https://example.invalid/market-test",
                    "source_file_sha256": "a" * 64,
                    "revision_availability_rule": REVISION_AVAILABILITY_RULE,
                    "mask_kind": MASK_KIND,
                }
            )
    return validate_membership_panel(pd.DataFrame(rows))


def _complete_package() -> dict[str, object]:
    membership = _membership()
    universe = pd.DataFrame({"ticker": ["AAA.BK", "BBB.BK", "CCC.BK"]})
    bars = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "adjusted_close": 100.0,
                "volume": 0.0 if ticker == "CCC.BK" else 1_000.0,
                "bar_available_at": f"{date.date().isoformat()}T17:00:00+07:00",
                "source_record_id": f"bar-{date.date()}-{ticker}",
            }
            for date in membership["date"].drop_duplicates()
            for ticker in universe["ticker"]
        ]
    )
    status = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "trading_status": "trading",
                "status_available_at": f"{date.date().isoformat()}T17:00:00+07:00",
                "source_record_id": f"status-{date.date()}-{ticker}",
            }
            for date in membership["date"].drop_duplicates()
            for ticker in universe["ticker"]
        ]
    )
    sessions = pd.DataFrame(
        [
            {
                "date": date,
                "session_available_at": "2023-12-01T00:00:00+07:00",
                "source_record_id": f"session-{date.date()}",
            }
            for date in membership["date"].drop_duplicates()
        ]
    )
    corporate_actions = pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)
    symbol_history = pd.DataFrame(
        [
            {
                "instrument_id": f"instrument-{ticker}",
                "ticker": ticker,
                "effective_from": "2024-01-01",
                "effective_to_exclusive": "2025-01-01",
                "source_record_id": f"symbol-{ticker}",
            }
            for ticker in universe["ticker"]
        ]
    )
    manifest = {
        "schema_version": 1,
        "package_id": "synthetic-complete-package",
        "provider": "synthetic structural fixture",
        "retrieved_at_utc": "2026-07-31T00:00:00+00:00",
        "license": {
            "reference": "synthetic-test-only",
            "research_use_allowed": True,
            "derived_publication_allowed": True,
            "raw_redistribution_allowed": False,
        },
        "attestations": {
            "authoritative_session_calendar_complete": True,
            "trading_status_complete": True,
            "membership_exit_path_status_complete": True,
            "corporate_actions_complete": True,
            "symbol_history_complete": True,
            "adjustment_method_documented": True,
            "point_in_time_feature_safe": True,
        },
    }
    binding_results = pd.DataFrame(
        [
            {"check": "manifest_sidecar", "status": "pass"},
            *[
                {"check": f"binding_{label}", "status": "pass"}
                for label in REQUIRED_BINDINGS
            ],
        ]
    )
    return {
        "bars": bars,
        "membership_panel": membership,
        "acquisition_universe": universe,
        "trading_status": status,
        "sessions": sessions,
        "corporate_actions": corporate_actions,
        "symbol_history": symbol_history,
        "manifest": manifest,
        "binding_results": binding_results,
    }


def test_complete_structural_package_can_pass_intake_but_not_unlock_training() -> None:
    receipt, checks, coverage = assess_v3_market_package(
        **_complete_package(),
        generated_at_utc="2026-07-31T00:00:00+00:00",
    )

    assert checks["status"].eq("pass").all()
    assert receipt["licensed_market_package_eligible"] is True
    assert receipt["final_investability_eligible"] is False
    assert receipt["model_input_eligible"] is False
    assert receipt["training_enabled"] is False
    assert receipt["publication_claim_eligible"] is False
    assert receipt["metrics"]["membership_exit_transitions"] == 0
    assert (
        coverage.set_index("metric")
        .loc["active_member_status_coverage", "coverage_ratio"]
        == 1.0
    )


def test_suspension_can_explain_a_missing_bar_but_missing_status_cannot() -> None:
    package = _complete_package()
    bars = package["bars"]
    status = package["trading_status"]
    missing_key = (
        bars["date"].eq(pd.Timestamp("2024-01-03"))
        & bars["ticker"].eq("AAA.BK")
    )
    package["bars"] = bars.loc[~missing_key].reset_index(drop=True)
    status.loc[
        status["date"].eq(pd.Timestamp("2024-01-03"))
        & status["ticker"].eq("AAA.BK"),
        "trading_status",
    ] = "suspended"

    receipt, checks, _ = assess_v3_market_package(**package)
    by_check = checks.set_index("check")

    assert by_check.loc[
        "active_member_market_observation_coverage",
        "status",
    ] == "pass"
    assert by_check.loc["trading_status_requires_bar", "status"] == "pass"
    assert receipt["licensed_market_package_eligible"] is True

    package["trading_status"] = status[
        ~(
            status["date"].eq(pd.Timestamp("2024-01-03"))
            & status["ticker"].eq("AAA.BK")
        )
    ]
    receipt, checks, _ = assess_v3_market_package(**package)
    assert (
        checks.set_index("check")
        .loc["active_member_status_coverage", "status"]
        == "fail"
    )
    assert receipt["licensed_market_package_eligible"] is False


def test_trading_status_without_a_bar_fails_closed() -> None:
    package = _complete_package()
    bars = package["bars"]
    package["bars"] = bars[
        ~(
            bars["date"].eq(pd.Timestamp("2024-01-03"))
            & bars["ticker"].eq("AAA.BK")
        )
    ]

    receipt, checks, _ = assess_v3_market_package(**package)

    assert (
        checks.set_index("check")
        .loc["trading_status_requires_bar", "status"]
        == "fail"
    )
    assert receipt["licensed_market_package_eligible"] is False


def test_inactive_membership_exit_requires_status_until_liquidation() -> None:
    package = _complete_package()
    membership = package["membership_panel"].copy()
    exit_date = pd.Timestamp("2024-01-03")
    exiting = membership["date"].eq(exit_date) & membership["ticker"].eq("AAA.BK")
    membership.loc[exiting, "is_index_member"] = False
    membership.loc[exiting, "sector_as_of"] = ""
    package["membership_panel"] = validate_membership_panel(membership)
    status = package["trading_status"]
    package["trading_status"] = status[
        ~(
            status["date"].eq(exit_date)
            & status["ticker"].eq("AAA.BK")
        )
    ]

    receipt, checks, coverage = assess_v3_market_package(**package)
    by_check = checks.set_index("check")
    by_coverage = coverage.set_index("metric")

    assert by_check.loc["active_member_status_coverage", "status"] == "pass"
    assert by_check.loc["membership_exit_path_coverage", "status"] == "fail"
    assert receipt["metrics"]["membership_exit_transitions"] == 1
    assert receipt["metrics"]["membership_exit_paths_covered"] == 0
    assert (
        by_coverage.loc["membership_exit_path_coverage", "coverage_ratio"]
        == 0.0
    )
    assert receipt["licensed_market_package_eligible"] is False


def test_authoritative_buffer_sessions_are_allowed_outside_membership_window() -> None:
    package = _complete_package()
    buffer_date = pd.Timestamp("2024-01-01")
    buffer_bars = package["bars"].iloc[:3].copy()
    buffer_bars["date"] = buffer_date
    buffer_bars["bar_available_at"] = "2024-01-01T17:00:00+07:00"
    buffer_bars["source_record_id"] = [
        f"buffer-bar-{ticker}" for ticker in buffer_bars["ticker"]
    ]
    package["bars"] = pd.concat(
        [buffer_bars, package["bars"]],
        ignore_index=True,
    )
    buffer_status = package["trading_status"].iloc[:3].copy()
    buffer_status["date"] = buffer_date
    buffer_status["status_available_at"] = "2024-01-01T17:00:00+07:00"
    buffer_status["source_record_id"] = [
        f"buffer-status-{ticker}" for ticker in buffer_status["ticker"]
    ]
    package["trading_status"] = pd.concat(
        [buffer_status, package["trading_status"]],
        ignore_index=True,
    )
    package["sessions"] = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "date": buffer_date,
                        "session_available_at": "2023-12-01T00:00:00+07:00",
                        "source_record_id": "session-buffer",
                    }
                ]
            ),
            package["sessions"],
        ],
        ignore_index=True,
    )

    receipt, checks, _ = assess_v3_market_package(**package)

    assert checks["status"].eq("pass").all()
    assert (
        receipt["metrics"]["bar_dates_outside_development_membership_grid"]
        == 1
    )
    assert receipt["licensed_market_package_eligible"] is True


def test_current_five_ticker_snapshot_is_measured_as_a_v3_gap() -> None:
    bars = pd.read_csv(PROJECT_ROOT / "data/raw/prices_real_ohlcv.csv")
    membership = pd.read_csv(
        PROJECT_ROOT
        / "data/processed/v3_development/"
        "set100_membership_session_mask_2021_2024.csv.gz"
    )
    universe = pd.read_csv(
        PROJECT_ROOT / "data/reference/set50_or_set100_universe.csv"
    )

    receipt, checks, coverage = assess_v3_market_package(
        bars,
        membership,
        universe,
        generated_at_utc="2026-07-31T00:00:00+00:00",
    )
    coverage_by_metric = coverage.set_index("metric")

    assert receipt["status"] == "fail"
    assert receipt["metrics"]["bar_tickers_observed"] == 5
    assert receipt["metrics"]["active_membership_keys"] == 96_500
    assert receipt["metrics"]["active_bar_keys"] == 4_825
    assert receipt["metrics"]["active_status_keys"] == 0
    assert receipt["metrics"]["membership_exit_transitions"] == 62
    assert receipt["metrics"]["membership_exit_paths_covered"] == 0
    assert (
        receipt["metrics"]["bar_dates_outside_development_membership_grid"]
        == 3
    )
    assert coverage_by_metric.loc[
        "acquisition_ticker_coverage",
        "coverage_ratio",
    ] == pytest.approx(5 / 138)
    assert coverage_by_metric.loc[
        "active_member_bar_coverage",
        "coverage_ratio",
    ] == pytest.approx(0.05)
    assert (
        checks.set_index("check")
        .loc["license_allows_research_and_derived_publication", "status"]
        == "fail"
    )


def test_manifest_binding_verifier_detects_hash_drift(tmp_path) -> None:
    package = _complete_package()
    frames = {
        "bars": package["bars"],
        "trading_status": package["trading_status"],
        "sessions": package["sessions"],
        "corporate_actions": package["corporate_actions"],
        "symbol_history": package["symbol_history"],
    }
    bindings = {}
    for label, frame in frames.items():
        path = tmp_path / "data" / f"{label}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        bindings[label] = {
            "path": str(path.relative_to(tmp_path)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": len(frame),
        }
    manifest = dict(package["manifest"])
    manifest["bindings"] = bindings
    manifest_path = tmp_path / "config" / "market_package.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    Path(f"{manifest_path}.sha256").write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n",
        encoding="utf-8",
    )

    _, loaded, results, _ = verify_market_package_bindings(
        manifest_path,
        project_root=tmp_path,
    )
    assert results["status"].eq("pass").all()
    assert set(loaded) == set(REQUIRED_BINDINGS)

    bars_path = tmp_path / bindings["bars"]["path"]
    bars_path.write_text(
        bars_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    _, _, results, _ = verify_market_package_bindings(
        manifest_path,
        project_root=tmp_path,
    )
    assert (
        results.set_index("check").loc["binding_bars", "status"]
        == "fail"
    )
