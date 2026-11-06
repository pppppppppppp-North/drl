from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceRequirement:
    name: str
    path: Path
    required_columns: tuple[str, ...]
    allow_empty: bool = False
    minimum_rows: int = 1
    receipt_path: Path | None = None
    receipt_output_key: str | None = None
    receipt_project_root: Path = Path(".")


DEFAULT_REQUIREMENTS = (
    SourceRequirement(
        name="broader_ticker_universe",
        path=Path("data/reference/set50_or_set100_universe.csv"),
        required_columns=("ticker",),
        minimum_rows=100,
        receipt_path=Path(
            "reports/data_sources/set_constituent_archive_20260731.json"
        ),
        receipt_output_key="acquisition_universe",
    ),
    SourceRequirement(
        name="sector_mapping",
        path=Path("data/reference/sector_mapping.csv"),
        required_columns=("ticker", "sector"),
        minimum_rows=100,
        receipt_path=Path(
            "reports/data_sources/set_constituent_archive_20260731.json"
        ),
        receipt_output_key="sector_mapping",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_receipt_binding(
    requirement: SourceRequirement,
    *,
    row_count: int,
) -> tuple[bool, str]:
    if requirement.receipt_path is None:
        return True, ""
    if not requirement.receipt_output_key:
        return False, "receipt output key is not configured"
    project_root = requirement.receipt_project_root.resolve()
    receipt_path = (
        requirement.receipt_path
        if requirement.receipt_path.is_absolute()
        else project_root / requirement.receipt_path
    )
    if not receipt_path.exists():
        return False, "missing checksum-bound receipt"
    sidecar = receipt_path.with_name(receipt_path.name + ".sha256")
    if not sidecar.exists():
        return False, "missing receipt SHA-256 sidecar"
    sidecar_parts = sidecar.read_text(encoding="utf-8").strip().split()
    if not sidecar_parts:
        return False, "empty receipt SHA-256 sidecar"
    if _sha256_file(receipt_path) != sidecar_parts[0]:
        return False, "receipt SHA-256 sidecar mismatch"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, "unreadable receipt"
    if not bool(
        receipt.get("eligibility", {}).get("structural_source_ready")
    ):
        return False, "receipt does not mark the source structurally ready"
    profile = receipt.get("outputs", {}).get(requirement.receipt_output_key)
    if not isinstance(profile, dict):
        return False, "receipt is missing the required output binding"
    declared_path = Path(str(profile.get("path", "")))
    if declared_path.is_absolute():
        return False, "receipt output path must be project-relative"
    expected_path = (project_root / declared_path).resolve()
    actual_path = (
        requirement.path
        if requirement.path.is_absolute()
        else project_root / requirement.path
    ).resolve()
    if expected_path != actual_path:
        return False, "receipt output path does not match the source file"
    if int(profile.get("rows", -1)) != row_count:
        return False, "receipt row count does not match the source file"
    if int(profile.get("bytes", -1)) != actual_path.stat().st_size:
        return False, "receipt byte count does not match the source file"
    if str(profile.get("sha256", "")) != _sha256_file(actual_path):
        return False, "receipt SHA-256 does not match the source file"
    return True, ""


def check_source_requirement(requirement: SourceRequirement) -> dict[str, object]:
    if not requirement.path.exists():
        return {
            "name": requirement.name,
            "path": str(requirement.path),
            "ready": False,
            "reason": "missing file",
            "missing_columns": list(requirement.required_columns),
            "rows": 0,
            "evidence_path": (
                str(requirement.receipt_path)
                if requirement.receipt_path is not None
                else ""
            ),
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
            "evidence_path": (
                str(requirement.receipt_path)
                if requirement.receipt_path is not None
                else ""
            ),
        }

    row_count = _count_data_rows(requirement.path)
    ready = requirement.allow_empty or row_count >= requirement.minimum_rows
    reason = (
        "ready"
        if ready
        else f"too few rows: {row_count} < {requirement.minimum_rows}"
    )
    if ready:
        bound, binding_reason = _verify_receipt_binding(
            requirement,
            row_count=row_count,
        )
        ready = bound
        if not bound:
            reason = binding_reason
    return {
        "name": requirement.name,
        "path": str(requirement.path),
        "ready": ready,
        "reason": reason,
        "missing_columns": [],
        "rows": max(row_count, 0),
        "evidence_path": (
            str(requirement.receipt_path)
            if requirement.receipt_path is not None
            else ""
        ),
    }


def check_source_readiness(requirements: tuple[SourceRequirement, ...] = DEFAULT_REQUIREMENTS) -> list[dict[str, object]]:
    return [check_source_requirement(requirement) for requirement in requirements]


def _write_report_csv(report: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "path",
        "ready",
        "reason",
        "missing_columns",
        "rows",
        "evidence_path",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report:
            serialized = dict(row)
            serialized["missing_columns"] = ";".join(serialized["missing_columns"])
            writer.writerow(serialized)


def _format_report(report: list[dict[str, object]]) -> str:
    lines = [
        "name,path,ready,reason,missing_columns,rows,evidence_path"
    ]
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
                    str(row["evidence_path"]),
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
