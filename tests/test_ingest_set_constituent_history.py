from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.ingest_set_constituent_history import (
    _csv_bytes,
    _validate_source_file,
    build_constituent_archive,
    load_source_manifest,
    validate_manifest_intervals,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    PROJECT_ROOT / "data/external/set_constituent_source_manifest.csv"
)


def test_builds_complete_checksum_bound_official_archive() -> None:
    (
        membership,
        acquisition_universe,
        sector_mapping,
        readiness,
        receipt,
    ) = build_constituent_archive(PROJECT_ROOT, MANIFEST)

    assert len(membership) == 1000
    assert membership["record_id"].nunique() == 10
    assert membership["ticker"].nunique() == 138
    assert membership["sector"].nunique() == 23
    assert len(acquisition_universe) == 138
    assert len(sector_mapping) == 138
    assert len(readiness) == 10
    assert readiness["set50_count"].eq(50).all()
    assert readiness["set100_count"].eq(100).all()
    assert readiness["set50_subset_of_set100"].eq(True).all()
    assert receipt["coverage"]["raw_file_count"] == 10
    assert receipt["coverage"]["raw_bytes"] == 2_723_035
    assert receipt["eligibility"]["point_in_time_membership_eligible"] is True
    assert receipt["eligibility"]["model_input_eligible"] is False
    assert receipt["sector_history"]["changed_tickers"] == [
        "MINT.BK",
        "NEX.BK",
    ]


def test_archive_normalization_is_deterministic() -> None:
    first = build_constituent_archive(PROJECT_ROOT, MANIFEST)
    second = build_constituent_archive(PROJECT_ROOT, MANIFEST)

    for first_frame, second_frame in zip(first[:4], second[:4]):
        assert _csv_bytes(first_frame) == _csv_bytes(second_frame)
    assert first[4] == second[4]


def test_source_file_hash_drift_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4\nfixture\n")
    row = SimpleNamespace(
        record_id="fixture",
        source_file_path="source.pdf",
        source_bytes=source.stat().st_size,
        source_file_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="source hash mismatch"):
        _validate_source_file(tmp_path, row)


def test_manifest_interval_gap_fails_closed() -> None:
    manifest = load_source_manifest(MANIFEST)
    broken = manifest.copy()
    broken.loc[1, "effective_from"] = "2021-02-12"

    with pytest.raises(ValueError, match="interval gap"):
        validate_manifest_intervals(broken)


def test_manifest_rejects_future_publication() -> None:
    manifest = load_source_manifest(MANIFEST)
    broken = manifest.copy()
    broken.loc[0, "published_at"] = "2021-01-02"

    with pytest.raises(ValueError, match="before it was published"):
        validate_manifest_intervals(broken)


def test_manifest_has_expected_source_contract() -> None:
    manifest = pd.read_csv(MANIFEST)

    assert len(manifest) == 10
    assert manifest["source_file_sha256"].str.fullmatch(
        r"[0-9a-f]{64}"
    ).all()
    assert manifest["source_bytes"].sum() == 2_723_035
    assert set(manifest["revision_status"]) == {"original", "revised"}
