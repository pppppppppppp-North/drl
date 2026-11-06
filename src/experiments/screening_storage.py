from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_ALGORITHM_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
RETENTION_RECEIPT = "checkpoint_retention.json"
RETENTION_SIDECAR = f"{RETENTION_RECEIPT}.sha256"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_trial_paths(trial_dir: Path, algorithm: str) -> tuple[Path, Path, Path]:
    if _ALGORITHM_PATTERN.fullmatch(algorithm) is None:
        raise ValueError(f"unsafe screening algorithm name: {algorithm!r}")
    root = trial_dir.resolve()
    if not root.is_dir() or trial_dir.is_symlink():
        raise ValueError(f"screening trial directory is missing or unsafe: {trial_dir}")
    checkpoint_root = root / "checkpoints"
    algorithm_root = checkpoint_root / algorithm
    if checkpoint_root.is_symlink() or algorithm_root.is_symlink():
        raise ValueError("screening checkpoint directories must not be symlinks")
    return root, checkpoint_root, algorithm_root


def purge_checkpoint_artifacts(trial_dir: Path, algorithm: str) -> dict[str, Any]:
    """Remove only files below the exact per-trial checkpoint directory.

    The removal is deliberately narrow and rejects symlinks. Final models,
    metrics, manifests, actions, equity curves, and TensorBoard files are never
    selected by this function.
    """

    root, checkpoint_root, algorithm_root = _validated_trial_paths(trial_dir, algorithm)
    records: list[dict[str, Any]] = []
    if algorithm_root.exists():
        if not algorithm_root.is_dir():
            raise ValueError("screening algorithm checkpoint path is not a directory")
        paths = sorted(algorithm_root.rglob("*"), key=lambda path: (len(path.parts), str(path)))
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"screening checkpoint tree contains a symlink: {path}")
            if path.is_file():
                if not path.resolve().is_relative_to(algorithm_root.resolve()):
                    raise ValueError(f"screening checkpoint file escapes its trial: {path}")
                records.append(
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
        for path in (root / record["path"] for record in records):
            path.unlink()
        directories = sorted(
            (path for path in algorithm_root.rglob("*") if path.is_dir()),
            key=lambda path: (-len(path.parts), str(path)),
        )
        for directory in directories:
            directory.rmdir()
        algorithm_root.rmdir()
    if checkpoint_root.exists():
        if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
            raise ValueError("screening checkpoint root is unsafe after cleanup")
        if any(checkpoint_root.iterdir()):
            raise ValueError("screening checkpoint root contains unexpected retained entries")
        checkpoint_root.rmdir()
    return {
        "removed_file_count": len(records),
        "removed_bytes": sum(int(record["bytes"]) for record in records),
        "removed_files": records,
    }


def finalize_checkpoint_free_trial(
    trial_dir: Path,
    algorithm: str,
    *,
    callback_instances: int,
    callback_on_step_calls: int,
) -> Path:
    """Prove that training retained a final model without checkpoint writes.

    Any unexpected checkpoint is removed to recover quota, but makes the trial
    fail.  A successful receipt therefore always records zero removed files;
    post-hoc deletion is never accepted as storage-safe execution.
    """

    root, _, _ = _validated_trial_paths(trial_dir, algorithm)
    model_path = root / f"{algorithm}_model.zip"
    if model_path.is_symlink() or not model_path.is_file() or model_path.stat().st_size <= 0:
        raise ValueError("a nonempty final model is required before checkpoint cleanup")
    removal = purge_checkpoint_artifacts(root, algorithm)
    if int(removal["removed_file_count"]) != 0:
        raise RuntimeError(
            "storage-safe screening detected and removed an unexpected "
            f"checkpoint write ({removal['removed_file_count']} file(s), "
            f"{removal['removed_bytes']} bytes)"
        )
    if callback_instances != 1 or callback_on_step_calls <= 0:
        raise RuntimeError(
            "storage-safe screening requires exactly one active no-write "
            "checkpoint callback"
        )
    receipt = {
        "schema_version": 1,
        "policy": "suppress_checkpoint_writes_retain_final_model",
        "algorithm": algorithm,
        "completed_at": _utc_now(),
        "callback": {
            "implementation": "no_write_checkpoint_callback",
            "instances": callback_instances,
            "on_step_calls": callback_on_step_calls,
        },
        "final_model": {
            "path": model_path.name,
            "bytes": model_path.stat().st_size,
            "sha256": _sha256(model_path),
        },
        **removal,
    }
    receipt_path = root / RETENTION_RECEIPT
    _atomic_json(receipt_path, receipt)
    _atomic_text(root / RETENTION_SIDECAR, f"{_sha256(receipt_path)}\n")
    return receipt_path


def validate_checkpoint_retention(trial_dir: Path, algorithm: str) -> list[str]:
    issues: list[str] = []
    try:
        root, checkpoint_root, _ = _validated_trial_paths(trial_dir, algorithm)
    except ValueError as exc:
        return [str(exc)]
    if checkpoint_root.exists():
        issues.append("checkpoint directory remains after storage-safe screening")
    receipt_path = root / RETENTION_RECEIPT
    sidecar_path = root / RETENTION_SIDECAR
    if receipt_path.is_symlink() or sidecar_path.is_symlink():
        return [*issues, "checkpoint retention evidence must not use symlinks"]
    try:
        receipt_text = receipt_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [*issues, f"checkpoint retention receipt is unreadable: {exc}"]
    try:
        sidecar = sidecar_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [*issues, f"checkpoint retention sidecar is unreadable: {exc}"]
    if sidecar != f"{_sha256(receipt_path)}\n":
        issues.append("checkpoint retention sidecar does not match the receipt")
    try:
        receipt = json.loads(receipt_text)
    except json.JSONDecodeError as exc:
        return [*issues, f"checkpoint retention receipt is unreadable: {exc}"]
    if not isinstance(receipt, dict):
        return [*issues, "checkpoint retention receipt must be a JSON object"]
    if receipt.get("schema_version") != 1:
        issues.append("checkpoint retention receipt schema version is not one")
    if receipt.get("policy") != "suppress_checkpoint_writes_retain_final_model":
        issues.append("checkpoint retention policy is not the storage-safe policy")
    if receipt.get("algorithm") != algorithm:
        issues.append("checkpoint retention algorithm does not match the trial")
    final_model = receipt.get("final_model")
    model_path = root / f"{algorithm}_model.zip"
    if not isinstance(final_model, dict):
        issues.append("checkpoint retention receipt has no final-model record")
    elif not model_path.is_file() or model_path.is_symlink():
        issues.append("retained final model is missing or unsafe")
    else:
        if final_model.get("path") != model_path.name:
            issues.append("checkpoint retention final-model path does not match")
        if final_model.get("bytes") != model_path.stat().st_size:
            issues.append("checkpoint retention final-model byte count does not match")
        if final_model.get("sha256") != _sha256(model_path):
            issues.append("checkpoint retention final-model hash does not match")
    removed_files = receipt.get("removed_files")
    if not isinstance(removed_files, list):
        issues.append("checkpoint retention removed-files inventory is not a list")
    else:
        try:
            removed_bytes = sum(int(record["bytes"]) for record in removed_files)
        except (KeyError, TypeError, ValueError):
            issues.append("checkpoint retention removed-files inventory is malformed")
        else:
            if receipt.get("removed_file_count") != len(removed_files):
                issues.append("checkpoint retention removed-file count does not match")
            if receipt.get("removed_bytes") != removed_bytes:
                issues.append("checkpoint retention removed-byte total does not match")
            if removed_files or receipt.get("removed_file_count") != 0 or removed_bytes != 0:
                issues.append("checkpoint retention receipt records a checkpoint write")
    callback = receipt.get("callback")
    if not isinstance(callback, dict):
        issues.append("checkpoint retention receipt has no callback evidence")
    else:
        if callback.get("implementation") != "no_write_checkpoint_callback":
            issues.append("checkpoint retention callback implementation does not match")
        if callback.get("instances") != 1:
            issues.append("checkpoint retention callback instance count does not match")
        try:
            on_step_calls = int(callback.get("on_step_calls", 0))
        except (TypeError, ValueError):
            on_step_calls = 0
        if on_step_calls <= 0:
            issues.append("checkpoint retention callback was never active")
    return issues
