from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.campaigns.calibration import sha256_file
from src.experiments.aggregate_screening import aggregate_screening
from src.experiments.run_screening import screening_algorithms
from src.experiments.screening_storage import (
    RETENTION_RECEIPT,
    RETENTION_SIDECAR,
    validate_checkpoint_retention,
)
from src.utils.config import load_config


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_PATTERN = re.compile(r"^screening_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _project_path(
    root: Path,
    raw_path: str | Path,
    *,
    label: str,
    kind: str,
) -> Path:
    raw = Path(raw_path)
    if ".." in raw.parts:
        raise ValueError(f"{label} must not contain a parent traversal")
    candidate = raw if raw.is_absolute() else root / raw
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        relative = None
    if relative is not None:
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"{label} must not use symlinks")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} must stay inside the project root")
    if kind == "file":
        valid = resolved.is_file() and not resolved.is_symlink()
    elif kind == "directory":
        valid = resolved.is_dir() and not resolved.is_symlink()
    else:
        raise ValueError(f"unsupported path kind: {kind}")
    if not valid:
        raise ValueError(f"{label} is missing or unsafe")
    return resolved


def _require_nonempty_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")


def _require_hash(value: Any, *, label: str) -> str:
    text = str(value or "")
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return text


def _content_fingerprint(run: Path) -> tuple[int, int, str]:
    records: list[str] = []
    total_bytes = 0
    file_count = 0
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError("screening run contains a symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(run).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        records.append(f"{relative}\0{size}\0{digest}\n")
        total_bytes += size
        file_count += 1
    fingerprint = hashlib.sha256("".join(records).encode("utf-8")).hexdigest()
    return file_count, total_bytes, fingerprint


def _validate_attempt(
    attempt: dict[str, Any],
    *,
    attempt_path: Path,
    config_path: Path,
    root: Path,
) -> dict[str, str]:
    attempt_id = str(attempt.get("attempt_id", ""))
    if (
        attempt.get("schema_version") != 2
        or attempt.get("status") != "rendered"
        or attempt.get("campaign_id") != "large_scale_thai_drl_v2"
        or attempt.get("stage_id") != "screening"
        or _ATTEMPT_PATTERN.fullmatch(attempt_id) is None
        or attempt_path.name != f"{attempt_id}.json"
    ):
        raise ValueError("screening attempt identity contract does not match")

    attempt_sha256 = _require_hash(
        attempt.get("attempt_sha256"),
        label="campaign attempt hash",
    )
    attempt_payload = dict(attempt)
    attempt_payload.pop("attempt_sha256", None)
    computed_attempt_sha256 = hashlib.sha256(
        json.dumps(
            attempt_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if attempt_sha256 != computed_attempt_sha256:
        raise ValueError("campaign attempt hash does not match its payload")
    release_id = _require_hash(
        attempt.get("deployment_release_id"),
        label="deployment release ID",
    )
    preflight_sha256 = _require_hash(
        attempt.get("preflight_sha256"),
        label="BistKA preflight hash",
    )
    config_sha256 = _require_hash(
        attempt.get("runtime_config_sha256"),
        label="runtime configuration hash",
    )
    feature_sha256 = _require_hash(
        attempt.get("runtime_feature_sha256"),
        label="runtime feature hash",
    )
    if config_sha256 != sha256_file(config_path):
        raise ValueError("locked screening configuration does not match the attempt")
    expected_config_path = str(config_path.relative_to(root))
    if attempt.get("runtime_config_path") != expected_config_path:
        raise ValueError("attempt runtime configuration path does not match")

    inputs = attempt.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("screening attempt has no immutable input inventory")
    input_by_path = {
        str(record.get("path")): record
        for record in inputs
        if isinstance(record, dict) and record.get("path")
    }
    config_record = input_by_path.get(expected_config_path)
    if (
        not isinstance(config_record, dict)
        or config_record.get("sha256") != config_sha256
        or config_record.get("bytes") != config_path.stat().st_size
    ):
        raise ValueError("screening attempt input inventory does not bind the config")

    return {
        "campaign_id": "large_scale_thai_drl_v2",
        "stage_id": "screening",
        "attempt_id": attempt_id,
        "attempt_sha256": attempt_sha256,
        "release_id": release_id,
        "preflight_sha256": preflight_sha256,
        "config_sha256": config_sha256,
        "feature_sha256": feature_sha256,
    }


def _validate_task_summary(
    path: Path,
    *,
    task_id: int,
    algorithm: str,
    trials_per_algorithm: int,
    total_timesteps: int,
    objective_metric: str,
) -> None:
    _require_nonempty_file(path, label=f"screening task {task_id} summary")
    summary = _json_object(path, label=f"screening task {task_id} summary")
    reused = summary.get("reused_trials")
    if (
        summary.get("task_id") != task_id
        or summary.get("algorithm") != algorithm
        or summary.get("trials_per_algorithm") != trials_per_algorithm
        or summary.get("total_timesteps") != total_timesteps
        or summary.get("objective_metric") != objective_metric
        or summary.get("selection_split") != "validation"
        or summary.get("completed_trials") != trials_per_algorithm
        or summary.get("failed_trials") != 0
        or summary.get("failures") != []
        or not isinstance(reused, int)
        or isinstance(reused, bool)
        or not 0 <= reused <= trials_per_algorithm
    ):
        raise ValueError(f"screening task {task_id} summary is incomplete or inconsistent")


def _validate_trial(
    trial: Path,
    *,
    root: Path,
    algorithm: str,
    task_id: int,
    trial_id: int,
    trial_number: int,
    trials_per_algorithm: int,
    total_timesteps: int,
    objective_metric: str,
    base_seed: int,
    feature_path: str,
    lineage: dict[str, str],
    job_id: int,
) -> float:
    if trial.is_symlink() or not trial.is_dir():
        raise ValueError(f"screening trial {trial_id} directory is missing or unsafe")
    if any(path.is_symlink() for path in trial.rglob("*")):
        raise ValueError(f"screening trial {trial_id} contains a symlink")
    if any("checkpoints" in path.relative_to(trial).parts for path in trial.rglob("*")):
        raise ValueError(f"screening trial {trial_id} contains a checkpoints path")

    required = (
        trial / "experiment_manifest.json",
        trial / "resolved_config.json",
        trial / "trial_metadata.json",
        trial / "split_summary.csv",
        trial / f"{algorithm}_model.zip",
        trial / f"{algorithm}_metrics.csv",
        trial / f"{algorithm}_equity_curve.csv",
        trial / f"{algorithm}_actions.csv",
        trial / RETENTION_RECEIPT,
        trial / RETENTION_SIDECAR,
    )
    for path in required:
        _require_nonempty_file(
            path,
            label=f"screening trial {trial_id} artifact {path.name}",
        )
    retention_issues = validate_checkpoint_retention(trial, algorithm)
    if retention_issues:
        raise ValueError(
            f"screening trial {trial_id} violates checkpoint retention: "
            f"{retention_issues}"
        )

    manifest = _json_object(
        trial / "experiment_manifest.json",
        label=f"screening trial {trial_id} manifest",
    )
    identity = manifest.get("identity")
    runtime = manifest.get("runtime")
    data_files = manifest.get("data_files")
    feature_record = data_files.get("features") if isinstance(data_files, dict) else None
    expected_seed = base_seed + trial_id
    expected_identity = {
        "experiment_id": f"screen_{algorithm}_{trial_number}",
        "algorithm": algorithm,
        "window": None,
        "seed": expected_seed,
        "feature_group": None,
        "cost_scenario": "base",
        "selection_split": "validation",
        "trial_id": trial_id,
        "trial_number": trial_number,
    }
    if not isinstance(identity, dict):
        raise ValueError(f"screening trial {trial_id} manifest has no identity")
    for key, expected in expected_identity.items():
        if key == "feature_group":
            continue
        if identity.get(key) != expected:
            raise ValueError(
                f"screening trial {trial_id} manifest identity mismatch for {key}"
            )
    if (
        manifest.get("experiment_id") != expected_identity["experiment_id"]
        or manifest.get("status") != "complete"
        or manifest.get("error") is not None
        or manifest.get("campaign_id") != lineage["campaign_id"]
        or manifest.get("campaign_stage_id") != lineage["stage_id"]
        or manifest.get("campaign_attempt_id") != lineage["attempt_id"]
        or manifest.get("campaign_attempt_sha256") != lineage["attempt_sha256"]
        or manifest.get("deployment_release_id") != lineage["release_id"]
        or manifest.get("bistka_preflight_sha256") != lineage["preflight_sha256"]
    ):
        raise ValueError(f"screening trial {trial_id} manifest lineage does not match")
    if (
        not isinstance(runtime, dict)
        or str(runtime.get("slurm_array_job_id")) != str(job_id)
        or str(runtime.get("slurm_array_task_id")) != str(task_id)
        or re.fullmatch(
            r"[1-9][0-9]*(?:_[0-9]+)?",
            str(runtime.get("slurm_job_id") or ""),
        )
        is None
    ):
        raise ValueError(f"screening trial {trial_id} SLURM array lineage does not match")
    if (
        not isinstance(feature_record, dict)
        or feature_record.get("sha256") != lineage["feature_sha256"]
    ):
        raise ValueError(f"screening trial {trial_id} feature hash does not match")

    resolved = _json_object(
        trial / "resolved_config.json",
        label=f"screening trial {trial_id} resolved configuration",
    )
    training = resolved.get("training")
    screening = resolved.get("screening")
    paths = resolved.get("paths")
    if (
        not isinstance(training, dict)
        or training.get("algo") != algorithm
        or training.get("seed") != expected_seed
        or training.get("total_timesteps") != total_timesteps
        or not isinstance(screening, dict)
        or screening.get("trials_per_algorithm") != trials_per_algorithm
        or screening.get("objective_metric") != objective_metric
        or not isinstance(paths, dict)
        or paths.get("features") != feature_path
    ):
        raise ValueError(
            f"screening trial {trial_id} resolved configuration does not match"
        )

    metadata = _json_object(
        trial / "trial_metadata.json",
        label=f"screening trial {trial_id} metadata",
    )
    objective = metadata.get("objective_value")
    if (
        metadata.get("trial_id") != trial_id
        or metadata.get("trial_number") != trial_number
        or metadata.get("seed") != expected_seed
        or metadata.get("config_path") != "resolved_config.json"
        or metadata.get("algorithm") != algorithm
        or metadata.get("objective_metric") != objective_metric
        or isinstance(objective, bool)
        or not isinstance(objective, (int, float))
        or not math.isfinite(float(objective))
    ):
        raise ValueError(f"screening trial {trial_id} metadata does not match")
    return float(objective)


def validate_run(
    run_dir: str | Path,
    *,
    config_path: str | Path,
    attempt_path: str | Path,
    job_id: int,
    project_root: str | Path = ".",
    expected_algorithms: tuple[str, ...] = ("ppo", "a2c", "ddpg", "td3", "sac"),
    expected_trials_per_algorithm: int = 30,
) -> dict[str, Any]:
    root_raw = Path(project_root)
    if root_raw.is_symlink() or not root_raw.is_dir():
        raise ValueError("project root is missing or unsafe")
    root = root_raw.resolve()
    if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0:
        raise ValueError("job ID must be one positive integer")
    if expected_trials_per_algorithm <= 0:
        raise ValueError("expected trials per algorithm must be positive")

    expected_run = Path("results") / f"large_scale_screening_{job_id}"
    run = _project_path(root, run_dir, label="screening run directory", kind="directory")
    if run.relative_to(root) != expected_run:
        raise ValueError("screening run directory is not bound to the requested job ID")
    config = _project_path(
        root,
        config_path,
        label="locked screening configuration",
        kind="file",
    )
    attempt_file = _project_path(
        root,
        attempt_path,
        label="immutable screening attempt",
        kind="file",
    )
    lineage = _validate_attempt(
        _json_object(attempt_file, label="immutable screening attempt"),
        attempt_path=attempt_file,
        config_path=config,
        root=root,
    )
    config_value = load_config(config)
    algorithms = tuple(screening_algorithms(config_value))
    screening = config_value.get("screening", {})
    trials_per_algorithm = int(screening.get("trials_per_algorithm", 0))
    total_timesteps = int(screening.get("total_timesteps", 0))
    objective_metric = str(screening.get("objective_metric", ""))
    feature_path = str(config_value.get("paths", {}).get("features", ""))
    base_seed = int(config_value.get("project", {}).get("seed", 42))
    if algorithms != expected_algorithms:
        raise ValueError("screening algorithm order does not match the locked contract")
    if trials_per_algorithm != expected_trials_per_algorithm:
        raise ValueError("screening trial count does not match the locked contract")
    if total_timesteps <= 0 or not objective_metric or not feature_path:
        raise ValueError("locked screening configuration is incomplete")

    symlinks = sorted(path for path in run.rglob("*") if path.is_symlink())
    if symlinks:
        raise ValueError("screening run contains a symlink")
    checkpoint_paths = sorted(
        path for path in run.rglob("*") if "checkpoints" in path.relative_to(run).parts
    )
    if checkpoint_paths:
        raise ValueError("screening run contains a checkpoints path")

    expected_trial_ids = set(range(len(algorithms) * trials_per_algorithm))
    observed_trial_ids: set[int] = set()
    for path in run.glob("trial_*"):
        if not path.is_dir() or path.is_symlink():
            raise ValueError("screening run contains an unsafe trial entry")
        suffix = path.name.removeprefix("trial_")
        if not suffix.isdigit():
            raise ValueError(f"screening run contains an unexpected trial directory: {path.name}")
        observed_trial_ids.add(int(suffix))
    if observed_trial_ids != expected_trial_ids:
        missing = sorted(expected_trial_ids - observed_trial_ids)
        extra = sorted(observed_trial_ids - expected_trial_ids)
        raise ValueError(
            f"screening trial grid is incomplete; missing={missing[:10]} extra={extra[:10]}"
        )

    objective_values: list[float] = []
    for task_id, algorithm in enumerate(algorithms):
        _validate_task_summary(
            run / f"screening_task_{task_id}.json",
            task_id=task_id,
            algorithm=algorithm,
            trials_per_algorithm=trials_per_algorithm,
            total_timesteps=total_timesteps,
            objective_metric=objective_metric,
        )
        for trial_number in range(trials_per_algorithm):
            trial_id = task_id * trials_per_algorithm + trial_number
            objective_values.append(
                _validate_trial(
                    run / f"trial_{trial_id}",
                    root=root,
                    algorithm=algorithm,
                    task_id=task_id,
                    trial_id=trial_id,
                    trial_number=trial_number,
                    trials_per_algorithm=trials_per_algorithm,
                    total_timesteps=total_timesteps,
                    objective_metric=objective_metric,
                    base_seed=base_seed,
                    feature_path=feature_path,
                    lineage=lineage,
                    job_id=job_id,
                )
            )

    manifests = sorted(run.rglob("experiment_manifest.json"))
    receipts = sorted(run.rglob(RETENTION_RECEIPT))
    if len(manifests) != len(expected_trial_ids) or len(receipts) != len(expected_trial_ids):
        raise ValueError("screening run contains an unexpected manifest or receipt count")

    with tempfile.TemporaryDirectory(prefix="screening_import_dry_run_") as temporary:
        aggregate_dir = Path(temporary) / "aggregate"
        results, top = aggregate_screening(
            run,
            config=config_value,
            output_dir=aggregate_dir,
        )
        ranking = pd.read_csv(aggregate_dir / "screening_algorithm_ranking.csv")
        selected = json.loads(
            (aggregate_dir / "screening_selected_algorithms.json").read_text(
                encoding="utf-8"
            )
        )
    if len(results) != len(expected_trial_ids):
        raise ValueError("screening aggregate dry run did not retain every trial")
    if len(top) != len(algorithms) * int(screening.get("top_n_per_algorithm", 5)):
        raise ValueError("screening aggregate dry run produced an unexpected top-trial grid")

    file_count, total_bytes, content_fingerprint = _content_fingerprint(run)
    return {
        "schema_version": 1,
        "status": "pass",
        "validation_kind": "storage_safe_full_screening_import",
        "run_dir": str(run.relative_to(root)),
        "job_id": job_id,
        "campaign_id": lineage["campaign_id"],
        "campaign_stage_id": lineage["stage_id"],
        "campaign_attempt_id": lineage["attempt_id"],
        "campaign_attempt_sha256": lineage["attempt_sha256"],
        "deployment_release_id": lineage["release_id"],
        "bistka_preflight_sha256": lineage["preflight_sha256"],
        "config_path": str(config.relative_to(root)),
        "config_sha256": lineage["config_sha256"],
        "algorithms": list(algorithms),
        "array_tasks": len(algorithms),
        "trials_per_algorithm": trials_per_algorithm,
        "complete_trials": len(expected_trial_ids),
        "experiment_manifests": len(manifests),
        "retention_receipts": len(receipts),
        "checkpoint_paths": 0,
        "selection_split": "validation",
        "objective_metric": objective_metric,
        "finite_objectives": len(objective_values),
        "aggregate_dry_run": {
            "complete_trials": len(results),
            "top_trials": len(top),
            "ranked_algorithms": ranking["algorithm"].astype(str).tolist(),
            "selected_algorithms": selected["selected_algorithms"],
        },
        "content": {
            "files": file_count,
            "bytes": total_bytes,
            "sha256": content_fingerprint,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate an imported five-algorithm, storage-safe screening run"
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_run(
                args.run_dir,
                config_path=args.config,
                attempt_path=args.attempt,
                job_id=args.job_id,
                project_root=args.project_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
