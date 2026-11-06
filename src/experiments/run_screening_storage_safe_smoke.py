from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from src.campaigns.calibration import atomic_json, atomic_text, sha256_file
from src.experiments import run_screening_smoke as base_smoke
from src.experiments.run_screening_storage_safe import _storage_safe_training
from src.experiments.screening_storage import (
    RETENTION_RECEIPT,
    RETENTION_SIDECAR,
    validate_checkpoint_retention,
)


_SMOKE_PATCH_LOCK = threading.Lock()
_BASE_REQUIRED_ARTIFACTS = base_smoke._required_artifacts
STORAGE_SAFE_POLICY = "suppress_checkpoint_writes_retain_final_model"
VALIDATION_RECEIPT = "storage_safe_smoke_validation.json"
VALIDATION_SIDECAR = f"{VALIDATION_RECEIPT}.sha256"


def _has_checkpoint_component(path: str | Path) -> bool:
    return "checkpoints" in PurePosixPath(str(path).replace("\\", "/")).parts


def _storage_safe_required_artifacts(
    run_dir: Path,
    algorithm: str,
    timesteps: int,
) -> list[Path]:
    """Replace the legacy smoke checkpoint with signed no-write evidence."""

    original = _BASE_REQUIRED_ARTIFACTS(run_dir, algorithm, timesteps)
    checkpoint_artifacts = [
        path
        for path in original
        if _has_checkpoint_component(path.relative_to(run_dir))
    ]
    if len(checkpoint_artifacts) != 1:
        raise RuntimeError(
            "storage-safe smoke expected exactly one legacy checkpoint contract; "
            f"found {len(checkpoint_artifacts)}"
        )
    issues = validate_checkpoint_retention(run_dir, algorithm)
    if issues:
        raise ValueError(
            f"{algorithm} storage-safe smoke retention is invalid: {issues}"
        )
    retained = [path for path in original if path not in checkpoint_artifacts]
    retained.extend([run_dir / RETENTION_RECEIPT, run_dir / RETENTION_SIDECAR])
    if any(_has_checkpoint_component(path.relative_to(run_dir)) for path in retained):
        raise RuntimeError("storage-safe smoke artifact inventory retained a checkpoint path")
    return retained


@contextmanager
def _storage_safe_smoke_hooks() -> Iterator[None]:
    """Temporarily route the existing scientific smoke through no-write training."""

    original_train = base_smoke.train_with_stable_baselines
    original_required = base_smoke._required_artifacts

    def storage_safe_train(config: dict[str, Any], output_dir: Path) -> bool:
        return _storage_safe_training(original_train, config, output_dir)

    base_smoke.train_with_stable_baselines = storage_safe_train
    base_smoke._required_artifacts = _storage_safe_required_artifacts
    try:
        yield
    finally:
        base_smoke.train_with_stable_baselines = original_train
        base_smoke._required_artifacts = original_required


def validate_storage_safe_smoke_summary(
    summary: dict[str, Any],
    *,
    project_root: Path,
) -> list[str]:
    issues: list[str] = []
    policy = summary.get("checkpoint_retention")
    if not isinstance(policy, dict):
        return ["storage-safe smoke summary has no checkpoint-retention evidence"]
    if policy.get("schema_version") != 1 or policy.get("policy") != STORAGE_SAFE_POLICY:
        issues.append("storage-safe smoke checkpoint policy does not match")
    algorithms = list(base_smoke.SCREENING_ALGORITHMS)
    if policy.get("run_count") != len(algorithms):
        issues.append("storage-safe smoke checkpoint run count is incomplete")
    receipts = policy.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != len(algorithms):
        issues.append("storage-safe smoke receipt inventory is incomplete")
        receipts = []

    runs = summary.get("runs")
    if not isinstance(runs, list) or len(runs) != len(algorithms):
        issues.append("storage-safe smoke run inventory is incomplete")
        runs = []
    observed_algorithms: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            issues.append("storage-safe smoke contains a malformed run")
            continue
        algorithm = str(run.get("algorithm", "")).lower()
        observed_algorithms.append(algorithm)
        run_relative = Path(str(run.get("run_dir", "")))
        run_dir = (project_root / run_relative).resolve()
        if (
            algorithm not in algorithms
            or run_relative.is_absolute()
            or ".." in run_relative.parts
            or not run_dir.is_relative_to(project_root.resolve())
        ):
            issues.append(f"storage-safe smoke {algorithm!r} run path is unsafe")
            continue
        issues.extend(
            f"storage-safe smoke {algorithm}: {issue}"
            for issue in validate_checkpoint_retention(run_dir, algorithm)
        )
        artifact_paths = [
            str(record.get("path", ""))
            for record in run.get("artifacts", [])
            if isinstance(record, dict)
        ]
        if any(_has_checkpoint_component(path) for path in artifact_paths):
            issues.append(f"storage-safe smoke {algorithm} inventories a checkpoint")
        expected_receipt = str(run_relative / RETENTION_RECEIPT)
        expected_sidecar = str(run_relative / RETENTION_SIDECAR)
        if expected_receipt not in artifact_paths or expected_sidecar not in artifact_paths:
            issues.append(
                f"storage-safe smoke {algorithm} artifact inventory lacks retention evidence"
            )
    if observed_algorithms != algorithms:
        issues.append("storage-safe smoke algorithm order or coverage is invalid")

    receipt_algorithms: list[str] = []
    for record in receipts:
        if not isinstance(record, dict):
            issues.append("storage-safe smoke has a malformed receipt record")
            continue
        algorithm = str(record.get("algorithm", "")).lower()
        receipt_algorithms.append(algorithm)
        relative = Path(str(record.get("path", "")))
        receipt_path = (project_root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not receipt_path.is_relative_to(project_root.resolve())
            or not receipt_path.is_file()
            or record.get("bytes") != receipt_path.stat().st_size
            or record.get("sha256") != sha256_file(receipt_path)
        ):
            issues.append(f"storage-safe smoke {algorithm} receipt record is invalid")
    if receipt_algorithms != algorithms:
        issues.append("storage-safe smoke receipt order or coverage is invalid")
    return sorted(set(issues))


def _annotate_summary(
    summary: dict[str, Any],
    *,
    project_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    receipts: list[dict[str, Any]] = []
    for algorithm in base_smoke.SCREENING_ALGORITHMS:
        receipt_path = output_path / f"algorithm_{algorithm}" / RETENTION_RECEIPT
        sidecar_path = output_path / f"algorithm_{algorithm}" / RETENTION_SIDECAR
        receipts.append(
            {
                "algorithm": algorithm,
                "path": str(receipt_path.relative_to(project_root)),
                "bytes": receipt_path.stat().st_size,
                "sha256": sha256_file(receipt_path),
                "sidecar_path": str(sidecar_path.relative_to(project_root)),
            }
        )
    annotated = dict(summary)
    annotated["checkpoint_retention"] = {
        "schema_version": 1,
        "policy": STORAGE_SAFE_POLICY,
        "run_count": len(receipts),
        "receipts": receipts,
    }
    issues = validate_storage_safe_smoke_summary(annotated, project_root=project_root)
    if issues:
        raise ValueError(f"storage-safe smoke summary is invalid: {issues}")
    summary_path = output_path / base_smoke.SUMMARY_NAME
    atomic_json(summary_path, annotated)
    atomic_text(Path(str(summary_path) + ".sha256"), sha256_file(summary_path) + "\n")
    validation_path = output_path / VALIDATION_RECEIPT
    atomic_json(
        validation_path,
        {
            "schema_version": 1,
            "status": "pass",
            "policy": STORAGE_SAFE_POLICY,
            "summary": {
                "path": summary_path.name,
                "bytes": summary_path.stat().st_size,
                "sha256": sha256_file(summary_path),
            },
            "run_count": len(receipts),
            "receipts": receipts,
            "validation_errors": [],
        },
    )
    atomic_text(
        output_path / VALIDATION_SIDECAR,
        sha256_file(validation_path) + "\n",
    )
    return annotated


def run_screening_storage_safe_smoke(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    timesteps: int = base_smoke.DEFAULT_TIMESTEPS,
) -> dict[str, Any]:
    if not _SMOKE_PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError("storage-safe screening smoke supports one run per process")
    try:
        with _storage_safe_smoke_hooks():
            summary = base_smoke.run_screening_smoke(
                config_path=config_path,
                output_dir=output_dir,
                timesteps=timesteps,
            )
        project_root = Path.cwd().resolve()
        output_path = base_smoke._project_path(project_root, output_dir)
        try:
            return _annotate_summary(
                summary,
                project_root=project_root,
                output_path=output_path,
            )
        except BaseException:
            summary_path = output_path / base_smoke.SUMMARY_NAME
            summary_path.unlink(missing_ok=True)
            Path(str(summary_path) + ".sha256").unlink(missing_ok=True)
            (output_path / VALIDATION_RECEIPT).unlink(missing_ok=True)
            (output_path / VALIDATION_SIDECAR).unlink(missing_ok=True)
            raise
    finally:
        _SMOKE_PATCH_LOCK.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the five-algorithm smoke without checkpoint writes"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--timesteps", type=int, default=base_smoke.DEFAULT_TIMESTEPS
    )
    args = parser.parse_args()
    summary = run_screening_storage_safe_smoke(
        config_path=args.config,
        output_dir=args.output_dir,
        timesteps=args.timesteps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
