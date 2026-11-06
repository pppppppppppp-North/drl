from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.campaigns.calibration import sha256_file
from src.experiments import run_screening_smoke as base_smoke
from src.experiments.run_screening_storage_safe_smoke import (
    STORAGE_SAFE_POLICY,
    VALIDATION_RECEIPT,
    VALIDATION_SIDECAR,
    validate_storage_safe_smoke_summary,
)
from src.experiments.screening_storage import RETENTION_RECEIPT, RETENTION_SIDECAR


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_sidecar(path: Path, *, label: str) -> Path:
    sidecar = Path(str(path) + ".sha256")
    if path.is_symlink() or sidecar.is_symlink() or not path.is_file() or not sidecar.is_file():
        raise ValueError(f"{label} and its sidecar must be regular files")
    recorded = sidecar.read_text(encoding="utf-8")
    if recorded != f"{sha256_file(path)}\n":
        raise ValueError(f"{label} SHA-256 sidecar does not match")
    return sidecar


def validate_run(
    run_dir: str | Path,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    run = Path(run_dir)
    run = run.resolve() if run.is_absolute() else (root / run).resolve()
    if (
        not root.is_dir()
        or root.is_symlink()
        or not run.is_relative_to(root)
        or not run.is_dir()
        or run.is_symlink()
    ):
        raise ValueError("storage-safe smoke run directory is missing or unsafe")
    symlinks = sorted(path for path in run.rglob("*") if path.is_symlink())
    if symlinks:
        raise ValueError("storage-safe smoke run contains a symlink")
    checkpoint_paths = sorted(
        path for path in run.rglob("*") if "checkpoints" in path.relative_to(run).parts
    )
    if checkpoint_paths:
        raise ValueError("storage-safe smoke run contains a checkpoints path")

    summary_path = run / base_smoke.SUMMARY_NAME
    _verify_sidecar(summary_path, label="screening smoke summary")
    summary = _json_object(summary_path, label="screening smoke summary")
    issues = validate_storage_safe_smoke_summary(summary, project_root=root)
    if issues:
        raise ValueError(f"storage-safe smoke summary is invalid: {issues}")

    validation_path = run / VALIDATION_RECEIPT
    validation_sidecar = _verify_sidecar(
        validation_path,
        label="storage-safe smoke validation receipt",
    )
    if validation_sidecar.name != VALIDATION_SIDECAR:
        raise ValueError("storage-safe smoke validation sidecar name does not match")
    validation = _json_object(validation_path, label="storage-safe smoke validation receipt")
    retention = summary.get("checkpoint_retention")
    receipts = retention.get("receipts") if isinstance(retention, dict) else None
    expected_algorithms = list(base_smoke.SCREENING_ALGORITHMS)
    if (
        validation.get("schema_version") != 1
        or validation.get("status") != "pass"
        or validation.get("policy") != STORAGE_SAFE_POLICY
        or validation.get("run_count") != len(expected_algorithms)
        or validation.get("validation_errors") != []
        or validation.get("receipts") != receipts
    ):
        raise ValueError("storage-safe smoke validation receipt contract does not match")
    summary_record = validation.get("summary")
    if not isinstance(summary_record, dict) or summary_record != {
        "path": summary_path.name,
        "bytes": summary_path.stat().st_size,
        "sha256": sha256_file(summary_path),
    }:
        raise ValueError("storage-safe smoke validation summary binding does not match")

    manifest_paths = sorted(run.rglob("experiment_manifest.json"))
    if len(manifest_paths) != len(expected_algorithms):
        raise ValueError("storage-safe smoke must contain exactly five experiment manifests")
    for algorithm in expected_algorithms:
        trial = run / f"algorithm_{algorithm}"
        required = (
            trial / "experiment_manifest.json",
            trial / RETENTION_RECEIPT,
            trial / RETENTION_SIDECAR,
        )
        if any(path.is_symlink() or not path.is_file() for path in required):
            raise ValueError(f"storage-safe smoke {algorithm} evidence is incomplete")

    return {
        "schema_version": 1,
        "status": "pass",
        "run_dir": str(run.relative_to(root)),
        "summary_sha256": sha256_file(summary_path),
        "validation_sha256": sha256_file(validation_path),
        "algorithms": expected_algorithms,
        "experiment_manifests": len(manifest_paths),
        "retention_receipts": len(expected_algorithms),
        "checkpoint_paths": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an imported storage-safe five-algorithm screening smoke"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_run(args.run_dir, project_root=args.project_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
