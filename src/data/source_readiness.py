from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceRequirement:
    name: str
    path: Path
    required_columns: tuple[str, ...]
    allow_empty: bool = False


DEFAULT_REQUIREMENTS = (
    SourceRequirement(
        name="broader_ticker_universe",
        path=Path("data/reference/set50_or_set100_universe.csv"),
        required_columns=("ticker",),
    ),
    SourceRequirement(
        name="sector_mapping",
        path=Path("data/reference/sector_mapping.csv"),
        required_columns=("ticker", "sector"),
    ),
    SourceRequirement(
        name="official_macro_history",
        path=Path("data/external/bot_historical_macro.csv"),
        required_columns=("period", "metric", "value"),
    ),
    SourceRequirement(
        name="historical_news_sentiment",
        path=Path("data/external/historical_news.csv"),
        required_columns=("date", "ticker"),
    ),
    SourceRequirement(
        name="licensed_fundamentals",
        path=Path("data/external/fundamentals.csv"),
        required_columns=("ticker", "metric", "period_end", "value"),
    ),
)


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), [])


def _count_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def check_source_requirement(requirement: SourceRequirement) -> dict[str, object]:
    if not requirement.path.exists():
        return {
            "name": requirement.name,
            "path": str(requirement.path),
            "ready": False,
            "reason": "missing file",
            "missing_columns": list(requirement.required_columns),
            "rows": 0,
        }

    header = _read_header(requirement.path)
    missing_columns = [column for column in requirement.required_columns if column not in header]
    if missing_columns:
        return {
            "name": requirement.name,
            "path": str(requirement.path),
            "ready": False,
            "reason": "missing required columns",
            "missing_columns": missing_columns,
            "rows": 0,
        }

    row_count = _count_data_rows(requirement.path)
    ready = requirement.allow_empty or row_count > 0
    return {
        "name": requirement.name,
        "path": str(requirement.path),
        "ready": ready,
        "reason": "ready" if ready else "empty file",
        "missing_columns": [],
        "rows": max(row_count, 0),
    }


def check_source_readiness(requirements: tuple[SourceRequirement, ...] = DEFAULT_REQUIREMENTS) -> list[dict[str, object]]:
    return [check_source_requirement(requirement) for requirement in requirements]


def _write_report_csv(report: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "path", "ready", "reason", "missing_columns", "rows"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report:
            serialized = dict(row)
            serialized["missing_columns"] = ";".join(serialized["missing_columns"])
            writer.writerow(serialized)


def _format_report(report: list[dict[str, object]]) -> str:
    lines = ["name,path,ready,reason,missing_columns,rows"]
    for row in report:
        lines.append(
            ",".join(
                [
                    str(row["name"]),
                    str(row["path"]),
                    str(row["ready"]),
                    str(row["reason"]),
                    ";".join(row["missing_columns"]),
                    str(row["rows"]),
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Optional CSV path for the readiness report.")
    args = parser.parse_args()

    report = check_source_readiness()
    if args.output:
        _write_report_csv(report, Path(args.output))
    print(_format_report(report))
    if not all(bool(row["ready"]) for row in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
