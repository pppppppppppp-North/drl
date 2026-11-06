from __future__ import annotations

import csv
import hashlib
import json

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


def _write_bound_receipt(tmp_path, output, *, key: str = "universe"):
    receipt = {
        "eligibility": {"structural_source_ready": True},
        "outputs": {
            key: {
                "path": output.name,
                "rows": 1,
                "bytes": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        },
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    receipt_path.with_name("receipt.json.sha256").write_text(
        f"{digest}  receipt.json\n",
        encoding="utf-8",
    )
    return receipt_path


def test_source_readiness_requires_declared_receipt(tmp_path) -> None:
    path = tmp_path / "universe.csv"
    _write_csv(path, [{"ticker": "PTT.BK"}])
    requirement = SourceRequirement(
        name="universe",
        path=path,
        required_columns=("ticker",),
        receipt_path=tmp_path / "missing.json",
        receipt_output_key="universe",
        receipt_project_root=tmp_path,
    )

    report = check_source_readiness((requirement,))

    assert bool(report[0]["ready"]) is False
    assert report[0]["reason"] == "missing checksum-bound receipt"


def test_source_readiness_accepts_checksum_bound_output(tmp_path) -> None:
    path = tmp_path / "universe.csv"
    _write_csv(path, [{"ticker": "PTT.BK"}])
    receipt = _write_bound_receipt(tmp_path, path)
    requirement = SourceRequirement(
        name="universe",
        path=path,
        required_columns=("ticker",),
        receipt_path=receipt,
        receipt_output_key="universe",
        receipt_project_root=tmp_path,
    )

    report = check_source_readiness((requirement,))

    assert bool(report[0]["ready"]) is True
    assert report[0]["reason"] == "ready"


def test_source_readiness_rejects_output_drift(tmp_path) -> None:
    path = tmp_path / "universe.csv"
    _write_csv(path, [{"ticker": "PTT.BK"}])
    receipt = _write_bound_receipt(tmp_path, path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("AOT.BK\n")
    requirement = SourceRequirement(
        name="universe",
        path=path,
        required_columns=("ticker",),
        receipt_path=receipt,
        receipt_output_key="universe",
        receipt_project_root=tmp_path,
    )

    report = check_source_readiness((requirement,))

    assert bool(report[0]["ready"]) is False
    assert "receipt" in str(report[0]["reason"])
