from __future__ import annotations

import csv
from datetime import date

from src.data.intake_validation import IntakePaths, validate_external_intake


def _write_csv(path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _valid_paths(tmp_path) -> IntakePaths:
    paths = IntakePaths(
        universe=tmp_path / "set50_or_set100_universe.csv",
        sector_mapping=tmp_path / "sector_mapping.csv",
        official_macro=tmp_path / "bot_historical_macro.csv",
        historical_news=tmp_path / "historical_news.csv",
        fundamentals=tmp_path / "fundamentals.csv",
        manifest=tmp_path / "external_data_manifest.csv",
    )
    _write_csv(paths.universe, [{"ticker": "PTT.BK"}])
    _write_csv(paths.sector_mapping, [{"ticker": "PTT.BK", "sector": "Energy"}])
    _write_csv(paths.official_macro, [{"period": "2021-01-31", "metric": "lei", "value": 1.0}])
    _write_csv(paths.historical_news, [{"date": "2021-02-01", "ticker": "PTT.BK"}])
    _write_csv(paths.fundamentals, [{"ticker": "PTT.BK", "metric": "roe", "period_end": "2021-03-31", "value": 0.1}])
    _write_csv(
        paths.manifest,
        [
            {"source": "universe", "raw_file_path": str(paths.universe)},
            {"source": "sector", "raw_file_path": str(paths.sector_mapping)},
            {"source": "macro", "raw_file_path": str(paths.official_macro)},
            {"source": "news", "raw_file_path": str(paths.historical_news)},
            {"source": "fundamentals", "raw_file_path": str(paths.fundamentals)},
        ],
    )
    return paths


def test_validate_external_intake_accepts_consistent_package(tmp_path) -> None:
    report = validate_external_intake(_valid_paths(tmp_path))

    assert all(bool(row["passed"]) for row in report)


def test_validate_external_intake_reports_missing_sector_coverage(tmp_path) -> None:
    paths = _valid_paths(tmp_path)
    _write_csv(paths.universe, [{"ticker": "PTT.BK"}, {"ticker": "AOT.BK"}])

    report = validate_external_intake(paths)
    sector_check = next(row for row in report if row["check"] == "sector_covers_universe")

    assert bool(sector_check["passed"]) is False
    assert "AOT.BK" in str(sector_check["detail"])


def test_validate_external_intake_reports_missing_manifest_file(tmp_path) -> None:
    paths = _valid_paths(tmp_path)
    paths.manifest.unlink()

    report = validate_external_intake(paths)
    manifest_check = next(row for row in report if row["check"] == "external_manifest_file")

    assert bool(manifest_check["passed"]) is False


def test_validate_external_intake_honors_custom_window(tmp_path) -> None:
    paths = _valid_paths(tmp_path)

    report = validate_external_intake(paths, start=date(2022, 1, 1), end=date(2022, 12, 31))
    macro_check = next(row for row in report if row["check"] == "official_macro_overlaps_window")

    assert bool(macro_check["passed"]) is False
    assert macro_check["detail"] == "2021-01-31 to 2021-01-31"
