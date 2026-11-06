from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from scripts.build_v3_membership_readiness import (
    IMPLEMENTATION_PATHS,
    V2_IMMUTABILITY_BINDINGS,
    build_readiness,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy(root: Path, relative: str) -> Path:
    source = PROJECT_ROOT / relative
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def test_v3_readiness_build_is_deterministic_and_receipt_bound(tmp_path) -> None:
    history = _copy(
        tmp_path,
        "data/reference/set50_set100_membership_2021_2024.csv",
    )
    universe = _copy(
        tmp_path,
        "data/reference/set50_or_set100_universe.csv",
    )
    calendar = _copy(tmp_path, "data/raw/prices_market_indices.csv")
    receipt_source = _copy(
        tmp_path,
        "reports/data_sources/set_constituent_archive_20260731.json",
    )
    _copy(
        tmp_path,
        "reports/data_sources/set_constituent_archive_20260731.json.sha256",
    )
    for binding in V2_IMMUTABILITY_BINDINGS.values():
        _copy(tmp_path, binding["path"])
    for relative in IMPLEMENTATION_PATHS.values():
        _copy(tmp_path, relative)

    panel = (
        tmp_path
        / "data/processed/v3_development/"
        "set100_membership_session_mask_2021_2024.csv.gz"
    )
    summary = tmp_path / "reports/data_quality_v3/membership_session_summary.csv"
    receipt = (
        tmp_path
        / "reports/data_quality_v3/"
        "v3_membership_mask_readiness_20260731.json"
    )
    report = tmp_path / "reports/v3_membership_mask_readiness_20260731.md"
    arguments = {
        "project_root": tmp_path,
        "history_path": history,
        "universe_path": universe,
        "calendar_path": calendar,
        "constituent_receipt_path": receipt_source,
        "panel_path": panel,
        "summary_path": summary,
        "receipt_path": receipt,
        "report_path": report,
        "generated_at_utc": "2026-07-31T00:00:00+00:00",
    }

    first = build_readiness(**arguments)
    first_hashes = {
        path: sha256_file(path)
        for path in (panel, summary, receipt, report)
    }
    second = build_readiness(**arguments)
    second_hashes = {
        path: sha256_file(path)
        for path in (panel, summary, receipt, report)
    }

    assert first_hashes == second_hashes
    assert first == second
    assert first["metrics"]["session_count"] == 965
    assert first["metrics"]["panel_rows"] == 133_170
    assert first["metrics"]["active_count_failures"] == 0
    assert first["model_input_eligible"] is False
    assert first["hpc_jobs_submitted"] == 0
    assert first["inputs"]["constituent_receipt"]["sha256"] == sha256_file(
        receipt_source
    )
    assert set(first["inputs"]["implementation"]) == set(IMPLEMENTATION_PATHS)
    materialized = pd.read_csv(panel, compression="gzip")
    assert len(materialized) == 133_170
    assert materialized.groupby("date")["is_index_member"].sum().eq(100).all()
