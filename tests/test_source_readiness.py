from __future__ import annotations

import csv

from src.data.source_readiness import SourceRequirement, check_source_readiness


def _write_csv(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_source_readiness_reports_missing_file(tmp_path) -> None:
    requirement = SourceRequirement(
        name="missing",
        path=tmp_path / "missing.csv",
        required_columns=("ticker",),
    )

    report = check_source_readiness((requirement,))

    assert bool(report[0]["ready"]) is False
    assert report[0]["reason"] == "missing file"


def test_source_readiness_accepts_valid_csv(tmp_path) -> None:
    path = tmp_path / "universe.csv"
    _write_csv(path, [{"ticker": "PTT.BK", "name": "PTT"}])
    requirement = SourceRequirement(
        name="universe",
        path=path,
        required_columns=("ticker",),
    )

    report = check_source_readiness((requirement,))

    assert bool(report[0]["ready"]) is True
    assert report[0]["rows"] == 1


def test_source_readiness_reports_missing_columns(tmp_path) -> None:
    path = tmp_path / "universe.csv"
    _write_csv(path, [{"symbol": "PTT"}])
    requirement = SourceRequirement(
        name="universe",
        path=path,
        required_columns=("ticker",),
    )

    report = check_source_readiness((requirement,))

    assert bool(report[0]["ready"]) is False
    assert report[0]["reason"] == "missing required columns"
    assert report[0]["missing_columns"] == ["ticker"]
