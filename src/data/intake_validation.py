from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


DEFAULT_START = date(2021, 1, 1)
DEFAULT_END = date(2024, 12, 31)


@dataclass(frozen=True)
class IntakePaths:
    universe: Path = Path("data/reference/set50_or_set100_universe.csv")
    sector_mapping: Path = Path("data/reference/sector_mapping.csv")
    official_macro: Path = Path("data/external/bot_historical_macro.csv")
    historical_news: Path = Path("data/external/historical_news.csv")
    fundamentals: Path = Path("data/external/fundamentals.csv")
    manifest: Path = Path("data/external/external_data_manifest.csv")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _unique_values(rows: list[dict[str, str]], column: str) -> set[str]:
    return {str(row.get(column, "")).strip() for row in rows if str(row.get(column, "")).strip()}


def _parse_date(value: str) -> date | None:
    value = str(value or "").strip()
    if not value:
        return None
    for fmt, length in (
        ("%Y-%m-%d", 10),
        ("%Y/%m/%d", 10),
        ("%Y-%m", 7),
        ("%Y/%m", 7),
        ("%Y", 4),
    ):
        try:
            parsed = datetime.strptime(value[:length], fmt)
        except ValueError:
            continue
        return parsed.date()
    return None


def _date_values(rows: list[dict[str, str]], columns: tuple[str, ...]) -> list[date]:
    values: list[date] = []
    for row in rows:
        for column in columns:
            parsed = _parse_date(row.get(column, ""))
            if parsed is not None:
                values.append(parsed)
                break
    return values


def _overlaps_window(values: list[date], start: date, end: date) -> bool:
    return any(start <= value <= end for value in values)


def _result(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": passed, "detail": detail}


def validate_external_intake(
    paths: IntakePaths = IntakePaths(),
    *,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    required_paths = {
        "universe_file": paths.universe,
        "sector_mapping_file": paths.sector_mapping,
        "official_macro_file": paths.official_macro,
        "historical_news_file": paths.historical_news,
        "fundamentals_file": paths.fundamentals,
        "external_manifest_file": paths.manifest,
    }
    for check, path in required_paths.items():
        results.append(_result(check, path.exists(), str(path)))

    if not all(path.exists() for path in required_paths.values()):
        return results

    universe = _read_rows(paths.universe)
    sectors = _read_rows(paths.sector_mapping)
    official_macro = _read_rows(paths.official_macro)
    news = _read_rows(paths.historical_news)
    fundamentals = _read_rows(paths.fundamentals)
    manifest = _read_rows(paths.manifest)

    universe_tickers = _unique_values(universe, "ticker")
    sector_tickers = _unique_values(sectors, "ticker")
    news_tickers = _unique_values(news, "ticker")
    fundamental_tickers = _unique_values(fundamentals, "ticker")
    manifest_paths = _unique_values(manifest, "raw_file_path")

    missing_sector = sorted(universe_tickers - sector_tickers)
    results.append(
        _result(
            "sector_covers_universe",
            not missing_sector,
            "missing=" + ";".join(missing_sector[:20]) if missing_sector else f"tickers={len(universe_tickers)}",
        )
    )

    extra_sector = sorted(sector_tickers - universe_tickers)
    results.append(
        _result(
            "sector_has_no_extra_tickers",
            not extra_sector,
            "extra=" + ";".join(extra_sector[:20]) if extra_sector else f"tickers={len(sector_tickers)}",
        )
    )

    unknown_news = sorted(news_tickers - universe_tickers)
    results.append(
        _result(
            "news_tickers_in_universe",
            not unknown_news,
            "unknown=" + ";".join(unknown_news[:20]) if unknown_news else f"tickers={len(news_tickers)}",
        )
    )

    unknown_fundamentals = sorted(fundamental_tickers - universe_tickers)
    results.append(
        _result(
            "fundamental_tickers_in_universe",
            not unknown_fundamentals,
            "unknown=" + ";".join(unknown_fundamentals[:20])
            if unknown_fundamentals
            else f"tickers={len(fundamental_tickers)}",
        )
    )

    for check, path in {
        "manifest_lists_universe": paths.universe,
        "manifest_lists_sector_mapping": paths.sector_mapping,
        "manifest_lists_official_macro": paths.official_macro,
        "manifest_lists_historical_news": paths.historical_news,
        "manifest_lists_fundamentals": paths.fundamentals,
    }.items():
        results.append(_result(check, str(path) in manifest_paths, str(path)))

    macro_dates = _date_values(official_macro, ("release_date", "period", "date"))
    news_dates = _date_values(news, ("published_at", "date"))
    fundamental_dates = _date_values(fundamentals, ("effective_date", "report_date", "period_end"))

    for check, values in {
        "official_macro_overlaps_window": macro_dates,
        "historical_news_overlaps_window": news_dates,
        "fundamentals_overlap_window": fundamental_dates,
    }.items():
        detail = "no parseable dates"
        if values:
            detail = f"{min(values).isoformat()} to {max(values).isoformat()}"
        results.append(_result(check, _overlaps_window(values, start, end), detail))

    return results


def _write_report_csv(report: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(report)


def _format_report(report: list[dict[str, object]]) -> str:
    lines = ["check,passed,detail"]
    lines.extend(f"{row['check']},{row['passed']},{row['detail']}" for row in report)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Optional CSV path for the intake validation report.")
    args = parser.parse_args()

    report = validate_external_intake()
    if args.output:
        _write_report_csv(report, Path(args.output))
    print(_format_report(report))
    if not all(bool(row["passed"]) for row in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
