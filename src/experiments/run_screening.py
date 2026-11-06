from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.experiment_manifest import tracked_experiment
from src.experiments.run_trial import (
    DEFAULT_ALGORITHMS,
    apply_trial_config,
    read_objective,
    suggest_hyperparameters,
    write_trial_metadata,
)
from src.train import ensure_feature_inputs, train_with_stable_baselines, write_split_summary
from src.utils.config import ensure_dirs, load_config


@dataclass(frozen=True)
class ScreeningTask:
    task_id: int
    algorithm: str


_MANIFEST_ENV_BINDINGS = {
    "campaign_id": "CAMPAIGN_ID",
    "campaign_attempt_id": "CAMPAIGN_ATTEMPT_ID",
    "campaign_attempt_sha256": "CAMPAIGN_ATTEMPT_SHA256",
    "campaign_stage_id": "CAMPAIGN_STAGE_ID",
    "deployment_release_id": "DEPLOYMENT_RELEASE_ID",
    "bistka_preflight_sha256": "BISTKA_PREFLIGHT_SHA256",
    "protocol_lock_hash": "PROTOCOL_LOCK_HASH",
}

_RUNTIME_ENV_BINDINGS = {
    "slurm_job_id": "SLURM_JOB_ID",
    "slurm_array_job_id": "SLURM_ARRAY_JOB_ID",
    "slurm_array_task_id": "SLURM_ARRAY_TASK_ID",
}


def screening_algorithms(config: dict[str, Any]) -> list[str]:
    raw = config.get("screening", {}).get("algorithms", list(DEFAULT_ALGORITHMS))
    values = raw.split(",") if isinstance(raw, str) else raw
    algorithms = [str(value).strip().lower() for value in values if str(value).strip()]
    invalid = sorted(set(algorithms) - set(DEFAULT_ALGORITHMS))
    if invalid:
        raise ValueError(f"unsupported screening algorithms: {', '.join(invalid)}")
    if not algorithms:
        raise ValueError("screening.algorithms must not be empty")
    return algorithms


def screening_task_grid(config: dict[str, Any]) -> list[ScreeningTask]:
    return [ScreeningTask(task_id=index, algorithm=algorithm) for index, algorithm in enumerate(screening_algorithms(config))]


def completed_screening_trial_is_reusable(
    trial_dir: Path,
    *,
    trial_config: dict[str, Any],
    identity: dict[str, Any],
    objective_metric: str,
) -> bool:
    """Return true only for a complete trial from this exact execution context."""

    manifest_path = trial_dir / "experiment_manifest.json"
    resolved_config_path = trial_dir / "resolved_config.json"
    metadata_path = trial_dir / "trial_metadata.json"
    model_path = trial_dir / f"{identity['algorithm']}_model.zip"
    metrics_path = trial_dir / f"{identity['algorithm']}_metrics.csv"
    equity_path = trial_dir / f"{identity['algorithm']}_equity_curve.csv"
    actions_path = trial_dir / f"{identity['algorithm']}_actions.csv"
    required_paths = (
        manifest_path,
        resolved_config_path,
        metadata_path,
        model_path,
        metrics_path,
        equity_path,
        actions_path,
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required_paths):
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if manifest.get("status") != "complete" or manifest.get("identity") != identity:
        return False
    if resolved_config != json.loads(json.dumps(trial_config, default=str)):
        return False

    for manifest_key, env_key in _MANIFEST_ENV_BINDINGS.items():
        expected = os.environ.get(env_key)
        if expected is not None and manifest.get(manifest_key) != expected:
            return False
    runtime = manifest.get("runtime", {})
    if not isinstance(runtime, dict):
        return False
    for manifest_key, env_key in _RUNTIME_ENV_BINDINGS.items():
        expected = os.environ.get(env_key)
        if expected is not None and runtime.get(manifest_key) != expected:
            return False

    configured_feature = trial_config.get("paths", {}).get("features")
    fallback_feature = trial_config.get("paths", {}).get("fallback_features_csv")
    if configured_feature:
        feature_path = Path(str(configured_feature))
        feature_path = feature_path if feature_path.is_absolute() else Path.cwd() / feature_path
        if not feature_path.is_file() and fallback_feature:
            feature_path = Path(str(fallback_feature))
            feature_path = feature_path if feature_path.is_absolute() else Path.cwd() / feature_path
        feature_record = manifest.get("data_files", {}).get("features", {})
        if not feature_path.is_file() or not isinstance(feature_record, dict):
            return False
        digest = hashlib.sha256()
        with feature_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if feature_record.get("sha256") != digest.hexdigest():
            return False

    if metadata.get("algorithm") != identity["algorithm"]:
        return False
    if metadata.get("trial_id") != identity["trial_id"]:
        return False
    if metadata.get("trial_number") != identity["trial_number"]:
        return False
    if metadata.get("seed") != identity["seed"]:
        return False
    if metadata.get("objective_metric") != objective_metric:
        return False

    try:
        read_objective(trial_dir, str(identity["algorithm"]), objective_metric)
    except (OSError, ValueError, KeyError):
        return False
    return True


def archive_previous_screening_trial_state(trial_dir: Path) -> None:
    """Preserve the prior manifest/config before retrying an incomplete trial."""

    manifest_path = trial_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        return
    history_dir = trial_dir / "retry_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    sequence = len(list(history_dir.glob("experiment_manifest_*.json"))) + 1
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = str(manifest.get("status", "unknown"))
    except (OSError, json.JSONDecodeError):
        status = "malformed"
    safe_status = "".join(char for char in status.lower() if char.isalnum() or char in "-_")
    safe_status = safe_status or "unknown"
    archived_manifest = history_dir / f"experiment_manifest_{sequence:03d}_{safe_status}.json"
    archived_manifest.write_bytes(manifest_path.read_bytes())
    resolved_config_path = trial_dir / "resolved_config.json"
    if resolved_config_path.is_file():
        archived_config = history_dir / f"resolved_config_{sequence:03d}_{safe_status}.json"
        archived_config.write_bytes(resolved_config_path.read_bytes())


def run_screening_task(config: dict[str, Any], *, task_id: int, output_dir: Path) -> ScreeningTask:
    tasks = screening_task_grid(config)
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"task_id must be between 0 and {len(tasks) - 1}; got {task_id}")
    task = tasks[task_id]
    screening = config.get("screening", {})
    trials_per_algorithm = int(screening.get("trials_per_algorithm", 30))
    total_timesteps = int(screening.get("total_timesteps", config.get("training", {}).get("total_timesteps", 20000)))
    objective_metric = str(screening.get("objective_metric", "sharpe"))
    base_seed = int(config.get("project", {}).get("seed", 42))
    if trials_per_algorithm <= 0:
        raise ValueError("screening.trials_per_algorithm must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    task_summary = {
        "task_id": task.task_id,
        "algorithm": task.algorithm,
        "trials_per_algorithm": trials_per_algorithm,
        "total_timesteps": total_timesteps,
        "objective_metric": objective_metric,
        "selection_split": "validation",
        "completed_trials": 0,
        "reused_trials": 0,
        "failed_trials": 0,
        "failures": [],
    }
    task_summary_path = output_dir / f"screening_task_{task.task_id}.json"
    task_summary_path.write_text(json.dumps(task_summary, indent=2), encoding="utf-8")

    for trial_number in range(trials_per_algorithm):
        trial_id = task.task_id * trials_per_algorithm + trial_number
        params = suggest_hyperparameters(
            task.algorithm,
            trial_number,
            base_seed,
            require_optuna=True,
        )
        trial_config = apply_trial_config(
            config,
            algo=task.algorithm,
            trial_id=trial_id,
            params=params,
            total_timesteps=total_timesteps,
        )
        trial_dir = output_dir / f"trial_{trial_id}"
        identity = {
            "experiment_id": f"screen_{task.algorithm}_{trial_number}",
            "algorithm": task.algorithm,
            "window": None,
            "seed": int(trial_config["training"]["seed"]),
            "feature_group": trial_config.get("project", {}).get("name"),
            "cost_scenario": "base",
            "selection_split": "validation",
            "trial_id": trial_id,
            "trial_number": trial_number,
        }
        if completed_screening_trial_is_reusable(
            trial_dir,
            trial_config=trial_config,
            identity=identity,
            objective_metric=objective_metric,
        ):
            task_summary["completed_trials"] += 1
            task_summary["reused_trials"] += 1
            task_summary_path.write_text(json.dumps(task_summary, indent=2), encoding="utf-8")
            continue
        archive_previous_screening_trial_state(trial_dir)
        try:
            with tracked_experiment(trial_dir, config=trial_config, identity=identity):
                ensure_dirs(trial_dir, trial_config["paths"]["models_dir"])
                ensure_feature_inputs(trial_config)
                write_split_summary(trial_config, trial_dir)
                if not train_with_stable_baselines(trial_config, trial_dir):
                    raise RuntimeError("stable-baselines3 is required for screening trials")
                objective_value, metrics = read_objective(trial_dir, task.algorithm, objective_metric)
                write_trial_metadata(
                    trial_dir,
                    trial_id=trial_id,
                    trial_number=trial_number,
                    seed=int(trial_config["training"]["seed"]),
                    config_path="resolved_config.json",
                    algo=task.algorithm,
                    params=params,
                    objective_metric=objective_metric,
                    objective_value=objective_value,
                    metrics=metrics,
                )
        except Exception as exc:
            task_summary["failed_trials"] += 1
            task_summary["failures"].append(
                {
                    "trial_id": trial_id,
                    "trial_number": trial_number,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            task_summary["completed_trials"] += 1
        task_summary_path.write_text(json.dumps(task_summary, indent=2), encoding="utf-8")
    if task_summary["failed_trials"]:
        failed_ids = ", ".join(str(item["trial_id"]) for item in task_summary["failures"])
        raise RuntimeError(
            f"{task_summary['failed_trials']} screening trial(s) failed for "
            f"{task.algorithm}: {failed_ids}"
        )
    return task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/large_scale_screening.yaml")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-task-grid", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.print_task_grid:
        print(json.dumps([task.__dict__ for task in screening_task_grid(config)], indent=2))
        return
    task = run_screening_task(config, task_id=args.task_id, output_dir=Path(args.output_dir))
    print(json.dumps(task.__dict__, indent=2))


if __name__ == "__main__":
    main()
