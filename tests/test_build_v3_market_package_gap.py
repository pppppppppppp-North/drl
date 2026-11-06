from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.build_v3_market_package_gap import (
    IMPLEMENTATION_PATHS,
    V2_IMMUTABILITY_BINDINGS,
    build_gap_report,
)
from src.v3.market_data_readiness import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy(root: Path, relative: str) -> Path:
    source = PROJECT_ROOT / relative
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def test_v3_market_gap_build_is_deterministic_and_fail_closed(tmp_path) -> None:
    bars = _copy(tmp_path, "data/raw/prices_real_ohlcv.csv")
    membership = _copy(
        tmp_path,
        "data/processed/v3_development/"
        "set100_membership_session_mask_2021_2024.csv.gz",
    )
    _copy(
        tmp_path,
        "data/processed/v3_development/"
        "set100_membership_session_mask_2021_2024.csv.gz.sha256",
    )
    universe = _copy(
        tmp_path,
        "data/reference/set50_or_set100_universe.csv",
    )
    membership_receipt = _copy(
        tmp_path,
        "reports/data_quality_v3/"
        "v3_membership_mask_readiness_20260731.json",
    )
    _copy(
        tmp_path,
        "reports/data_quality_v3/"
        "v3_membership_mask_readiness_20260731.json.sha256",
    )
    for binding in V2_IMMUTABILITY_BINDINGS.values():
        _copy(tmp_path, binding["path"])
    for relative in IMPLEMENTATION_PATHS.values():
        _copy(tmp_path, relative)

    checks = tmp_path / "reports/data_quality_v3/v3_market_package_checks.csv"
    coverage = tmp_path / "reports/data_quality_v3/v3_market_package_coverage.csv"
    receipt = (
        tmp_path
        / "reports/data_quality_v3/v3_market_package_gap_20260731.json"
    )
    chart = (
        tmp_path
        / "reports/figures/v3_market_package_coverage_20260731.png"
    )
    report = tmp_path / "reports/v3_market_package_gap_20260731.md"
    arguments = {
        "project_root": tmp_path,
        "bars_path": bars,
        "membership_path": membership,
        "universe_path": universe,
        "membership_receipt_path": membership_receipt,
        "candidate_manifest_path": None,
        "checks_path": checks,
        "coverage_path": coverage,
        "receipt_path": receipt,
        "chart_path": chart,
        "report_path": report,
        "generated_at_utc": "2026-07-31T00:00:00+00:00",
    }

    first = build_gap_report(**arguments)
    output_paths = [
        checks,
        coverage,
        receipt,
        chart,
        report,
        *[Path(f"{path}.sha256") for path in (checks, coverage, receipt, chart, report)],
    ]
    first_hashes = {path: sha256_file(path) for path in output_paths}
    second = build_gap_report(**arguments)
    second_hashes = {path: sha256_file(path) for path in output_paths}

    assert first == second
    assert first_hashes == second_hashes
    assert first["status"] == "fail"
    assert first["metrics"]["bar_tickers_observed"] == 5
    assert first["metrics"]["active_membership_keys"] == 96_500
    assert first["metrics"]["active_bar_keys"] == 4_825
    assert first["metrics"]["active_status_keys"] == 0
    assert first["licensed_market_package_eligible"] is False
    assert first["training_enabled"] is False
    assert first["hpc_jobs_submitted"] == 0
    assert chart.stat().st_size > 50_000
    assert "11/37 verified stages (29.7%)" in report.read_text(
        encoding="utf-8"
    )

    membership_receipt_value = json.loads(
        membership_receipt.read_text(encoding="utf-8")
    )
    membership_receipt_value["outputs"]["membership_panel"]["sha256"] = "0" * 64
    membership_receipt.write_text(
        json.dumps(membership_receipt_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(f"{membership_receipt}.sha256").write_text(
        f"{sha256_file(membership_receipt)}  {membership_receipt.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="membership receipt does not bind the exact membership_panel",
    ):
        build_gap_report(**arguments)
