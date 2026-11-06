from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.campaigns.calibration import (
    atomic_json,
    atomic_text,
    load_observed_inputs,
    sha256_file,
)
from src.experiments.experiment_manifest import tracked_experiment
from src.experiments.run_trial import (
    DEFAULT_ALGORITHMS,
    apply_trial_config,
    read_objective,
    suggest_hyperparameters,
    write_trial_metadata,
)
from src.train import ensure_feature_inputs, train_with_stable_baselines, write_split_summary
from src.utils.config import ensure_dirs


SCREENING_ALGORITHMS = tuple(DEFAULT_ALGORITHMS)
DEFAULT_TIMESTEPS = 2_048
SUMMARY_NAME = "screening_smoke_summary.json"
RUNTIME_PROJECTION_LIMIT_HOURS = 72.0
RUNTIME_PROJECTION_MARGIN = 1.5
REPRESENTATIVE_TRIAL_NUMBERS = {
    "ppo": 1,
    "a2c": 7,
    "ddpg": 17,
    "td3": 1,
    "sac": 1,
}


def _project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"path escapes the project root: {raw_path}")
    return resolved


def _project_relative(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"path escapes the project root: {path}")
    return str(resolved.relative_to(project_root))


def _readiness_record(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    raw_path = config.get("data", {}).get(
        "readiness_receipt",
        "reports/data_quality_market/market_data_readiness.json",
    )
    receipt_path = _project_path(project_root, str(raw_path))
    if not receipt_path.is_file():
        raise FileNotFoundError(f"observed-data readiness receipt is missing: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"observed-data readiness receipt is invalid JSON: {exc}") from exc
    if not isinstance(receipt, dict):
        raise ValueError("observed-data readiness receipt must contain a JSON object")
    if receipt.get("status") != "pass" or receipt.get("execution_eligible") is not True:
        raise ValueError("observed-data readiness receipt is not execution eligible")
    if str(receipt.get("data_origin", "")).strip().lower() != "observed_market":
        raise ValueError("screening smoke requires an observed-market readiness receipt")
    return {
        "path": _project_relative(project_root, receipt_path),
        "bytes": receipt_path.stat().st_size,
        "sha256": sha256_file(receipt_path),
        "data_origin": receipt["data_origin"],
        "evidence_class": receipt.get("evidence_class"),
        "execution_eligible": True,
    }


def build_smoke_config(
    locked_config: dict[str, Any],
    *,
    algorithm: str,
    timesteps: int,
) -> dict[str, Any]:
    if algorithm not in SCREENING_ALGORITHMS:
        raise ValueError(f"unsupported screening smoke algorithm: {algorithm}")
    if timesteps <= 0:
        raise ValueError("screening smoke timesteps must be positive")
    trial = smoke_trial_spec(locked_config, algorithm)
    config = apply_trial_config(
        copy.deepcopy(locked_config),
        algo=algorithm,
        trial_id=int(trial["trial_id"]),
        params=dict(trial["sampled_params"]),
        total_timesteps=timesteps,
    )
    training = config.setdefault("training", {})
    training["checkpoint_freq"] = int(timesteps)
    training["checkpoint_keep_last"] = 1
    training["save_replay_buffer"] = False
    return config


def smoke_trial_spec(
    locked_config: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    if algorithm not in SCREENING_ALGORITHMS:
        raise ValueError(f"unsupported screening smoke algorithm: {algorithm}")
    trials_per_algorithm = int(
        locked_config.get("screening", {}).get("trials_per_algorithm", 30)
    )
    if trials_per_algorithm <= 0:
        raise ValueError("locked screening trials_per_algorithm must be positive")
    trial_number = int(REPRESENTATIVE_TRIAL_NUMBERS[algorithm])
    if trial_number >= trials_per_algorithm:
        raise ValueError(
            f"representative {algorithm} trial {trial_number} is outside the locked grid"
        )
    task_id = SCREENING_ALGORITHMS.index(algorithm)
    trial_id = task_id * trials_per_algorithm + trial_number
    base_seed = int(locked_config.get("project", {}).get("seed", 42))
    params = suggest_hyperparameters(
        algorithm,
        trial_number,
        base_seed,
        require_optuna=True,
    )
    return {
        "task_id": task_id,
        "trial_id": trial_id,
        "trial_number": trial_number,
        "seed": base_seed + trial_id,
        "sampled_params": params,
    }


def _required_artifacts(
    run_dir: Path,
    algorithm: str,
    timesteps: int,
) -> list[Path]:
    return [
        run_dir / "experiment_manifest.json",
        run_dir / "resolved_config.json",
        run_dir / "split_summary.csv",
        run_dir / f"{algorithm}_model.zip",
        run_dir / f"{algorithm}_metrics.csv",
        run_dir / f"{algorithm}_equity_curve.csv",
        run_dir / f"{algorithm}_actions.csv",
        run_dir / "trial_metadata.json",
        run_dir
        / "checkpoints"
        / algorithm
        / f"{algorithm}_checkpoint_{timesteps}_steps.zip",
        run_dir / "screening_smoke_result.json",
    ]


def _require_artifacts(
    *,
    project_root: Path,
    run_dir: Path,
    algorithm: str,
    timesteps: int,
) -> list[Path]:
    artifacts = _required_artifacts(run_dir, algorithm, timesteps)
    tensorboard_events = sorted(
        path
        for path in (run_dir / "tensorboard").rglob("events.out.tfevents.*")
        if path.is_file() and path.stat().st_size > 0
    )
    if not tensorboard_events:
        raise ValueError(f"{algorithm} screening smoke is missing TensorBoard events")
    artifacts.extend(tensorboard_events)
    missing = [
        _project_relative(project_root, path)
        for path in artifacts
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        raise ValueError(f"{algorithm} screening smoke is missing required artifacts: {missing}")
    return artifacts


def _validate_complete_run(
    *,
    project_root: Path,
    run_dir: Path,
    algorithm: str,
    timesteps: int,
    full_grid_work: int,
    trials_per_algorithm: int,
    trial_spec: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = run_dir / "experiment_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {algorithm} screening smoke manifest: {exc}") from exc
    identity = manifest.get("identity", {}) if isinstance(manifest, dict) else {}
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or not isinstance(identity, dict)
        or identity.get("algorithm") != algorithm
        or identity.get("smoke_test") is not True
        or identity.get("trial_id") != trial_spec["trial_id"]
        or identity.get("trial_number") != trial_spec["trial_number"]
        or identity.get("seed") != trial_spec["seed"]
    ):
        raise ValueError(f"{algorithm} screening smoke manifest is not complete and correctly identified")

    artifacts = _require_artifacts(
        project_root=project_root,
        run_dir=run_dir,
        algorithm=algorithm,
        timesteps=timesteps,
    )
    result_path = run_dir / "screening_smoke_result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {algorithm} screening smoke result: {exc}") from exc
    if (
        not isinstance(result, dict)
        or result.get("status") != "complete"
        or result.get("algorithm") != algorithm
        or result.get("trial_id") != trial_spec["trial_id"]
        or result.get("trial_number") != trial_spec["trial_number"]
        or result.get("sampled_params") != trial_spec["sampled_params"]
    ):
        raise ValueError(f"{algorithm} screening smoke result does not match its sampled trial")
    try:
        started_at = datetime.fromisoformat(str(manifest["started_at"]))
        completed_at = datetime.fromisoformat(str(manifest["completed_at"]))
        elapsed_seconds = (completed_at - started_at).total_seconds()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{algorithm} screening smoke has invalid runtime timestamps") from exc
    if elapsed_seconds <= 0:
        raise ValueError(f"{algorithm} screening smoke elapsed time must be positive")
    smoke_work = _training_work_units(algorithm, timesteps, trial_spec["sampled_params"])
    projection_scale = full_grid_work / smoke_work
    projected_task_seconds = (
        elapsed_seconds
        * projection_scale
        * RUNTIME_PROJECTION_MARGIN
    )
    projected_task_hours = projected_task_seconds / 3600.0
    return {
        "algorithm": algorithm,
        "trial_id": trial_spec["trial_id"],
        "trial_number": trial_spec["trial_number"],
        "seed": trial_spec["seed"],
        "sampled_params": trial_spec["sampled_params"],
        "run_dir": _project_relative(project_root, run_dir),
        "runtime_projection": {
            "elapsed_seconds": elapsed_seconds,
            "smoke_work_units": smoke_work,
            "full_grid_work_units": full_grid_work,
            "full_to_smoke_work_ratio": projection_scale,
            "trials_per_algorithm": trials_per_algorithm,
            "safety_margin": RUNTIME_PROJECTION_MARGIN,
            "projected_task_seconds": projected_task_seconds,
            "projected_task_hours": projected_task_hours,
            "limit_hours": RUNTIME_PROJECTION_LIMIT_HOURS,
            "within_limit": projected_task_hours <= RUNTIME_PROJECTION_LIMIT_HOURS,
        },
        "manifest": {
            "path": _project_relative(project_root, manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "status": manifest["status"],
        },
        "artifacts": [
            {
                "path": _project_relative(project_root, path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ],
    }


def _training_work_units(
    algorithm: str,
    timesteps: int,
    sampled_params: dict[str, Any],
) -> int:
    if algorithm not in {"ddpg", "td3", "sac"}:
        return int(timesteps)
    learning_starts = int(sampled_params.get("learning_starts", 0))
    train_freq = max(int(sampled_params.get("train_freq", 1)), 1)
    gradient_steps = max(int(sampled_params.get("gradient_steps", 1)), 1)
    batch_size = max(int(sampled_params.get("batch_size", 1)), 1)
    update_events = max(int(timesteps) - learning_starts, 0) // train_freq
    return int(timesteps) + update_events * gradient_steps * batch_size


def _full_grid_work_units(
    locked_config: dict[str, Any],
    algorithm: str,
    full_timesteps: int,
) -> int:
    base_seed = int(locked_config.get("project", {}).get("seed", 42))
    trials_per_algorithm = int(
        locked_config.get("screening", {}).get("trials_per_algorithm", 30)
    )
    total = 0
    for trial_number in range(trials_per_algorithm):
        params = suggest_hyperparameters(
            algorithm,
            trial_number,
            base_seed,
            require_optuna=True,
        )
        total += _training_work_units(algorithm, full_timesteps, params)
    return total


def run_screening_smoke(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    timesteps: int = DEFAULT_TIMESTEPS,
) -> dict[str, Any]:
    if timesteps <= 0:
        raise ValueError("screening smoke timesteps must be positive")
    inputs = load_observed_inputs(config_path)
    project_root = inputs.project_root.resolve()
    if Path.cwd().resolve() != project_root:
        raise ValueError(f"screening smoke must run from the project root: {project_root}")
    output_path = _project_path(project_root, output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"screening smoke output directory is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    readiness = _readiness_record(inputs.config, project_root)
    locked_num_envs = int(inputs.config.get("training", {}).get("num_envs", 0))
    if locked_num_envs <= 0:
        raise ValueError("locked screening configuration must declare a positive training.num_envs")
    objective_metric = str(
        inputs.config.get("screening", {}).get("objective_metric", "sharpe")
    )
    full_timesteps = int(
        inputs.config.get("screening", {}).get(
            "total_timesteps",
            inputs.config.get("training", {}).get("total_timesteps", 20_000),
        )
    )
    trials_per_algorithm = int(
        inputs.config.get("screening", {}).get("trials_per_algorithm", 30)
    )
    if full_timesteps <= timesteps:
        raise ValueError("locked full screening timesteps must exceed smoke timesteps")
    if trials_per_algorithm <= 0:
        raise ValueError("locked screening trials_per_algorithm must be positive")

    run_records: list[dict[str, Any]] = []
    for algorithm in SCREENING_ALGORITHMS:
        algorithm_dir = output_path / f"algorithm_{algorithm}"
        trial_spec = smoke_trial_spec(inputs.config, algorithm)
        full_grid_work = _full_grid_work_units(
            inputs.config,
            algorithm,
            full_timesteps,
        )
        smoke_config = build_smoke_config(
            inputs.config,
            algorithm=algorithm,
            timesteps=timesteps,
        )
        identity = {
            "experiment_id": f"screening_smoke_{algorithm}",
            "algorithm": algorithm,
            "window": None,
            "seed": int(smoke_config["training"]["seed"]),
            "feature_group": smoke_config.get("project", {}).get("name"),
            "cost_scenario": "base",
            "selection_split": "validation",
            "trial_id": trial_spec["trial_id"],
            "trial_number": trial_spec["trial_number"],
            "smoke_test": True,
            "smoke_timesteps": int(timesteps),
        }
        with tracked_experiment(algorithm_dir, config=smoke_config, identity=identity):
            ensure_dirs(algorithm_dir, smoke_config["paths"]["models_dir"])
            ensure_feature_inputs(smoke_config)
            write_split_summary(smoke_config, algorithm_dir)
            if not train_with_stable_baselines(smoke_config, algorithm_dir):
                raise RuntimeError(
                    f"stable-baselines3 is required for the {algorithm} screening smoke run"
                )
            objective_value, metrics = read_objective(
                algorithm_dir,
                algorithm,
                objective_metric,
            )
            write_trial_metadata(
                algorithm_dir,
                trial_id=trial_spec["trial_id"],
                trial_number=trial_spec["trial_number"],
                seed=int(smoke_config["training"]["seed"]),
                config_path="resolved_config.json",
                algo=algorithm,
                params=trial_spec["sampled_params"],
                objective_metric=objective_metric,
                objective_value=objective_value,
                metrics=metrics,
            )
            result_path = algorithm_dir / "screening_smoke_result.json"
            atomic_json(
                result_path,
                {
                    "schema_version": 1,
                    "status": "complete",
                    "algorithm": algorithm,
                    "seed": int(smoke_config["training"]["seed"]),
                    "trial_id": trial_spec["trial_id"],
                    "trial_number": trial_spec["trial_number"],
                    "sampled_params": trial_spec["sampled_params"],
                    "num_envs": int(smoke_config["training"]["num_envs"]),
                    "timesteps": int(timesteps),
                    "objective_metric": objective_metric,
                    "objective_value": objective_value,
                    "metrics": metrics,
                },
            )
            _require_artifacts(
                project_root=project_root,
                run_dir=algorithm_dir,
                algorithm=algorithm,
                timesteps=timesteps,
            )
        run_records.append(
            _validate_complete_run(
                project_root=project_root,
                run_dir=algorithm_dir,
                algorithm=algorithm,
                timesteps=timesteps,
                full_grid_work=full_grid_work,
                trials_per_algorithm=trials_per_algorithm,
                trial_spec=trial_spec,
            )
        )

    manifest_paths = sorted(output_path.rglob("experiment_manifest.json"))
    if len(manifest_paths) != len(SCREENING_ALGORITHMS):
        raise ValueError(
            "screening smoke must produce exactly "
            f"{len(SCREENING_ALGORITHMS)} experiment manifests; found {len(manifest_paths)}"
        )

    assert inputs.config_path is not None and inputs.feature_path is not None
    full_screening_eligible = all(
        bool(run["runtime_projection"]["within_limit"]) for run in run_records
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "smoke_kind": "observed_data_five_algorithm_screening",
        "algorithms": list(SCREENING_ALGORITHMS),
        "run_count": len(run_records),
        "complete_manifest_count": len(manifest_paths),
        "timesteps_per_algorithm": int(timesteps),
        "total_requested_timesteps": int(timesteps) * len(SCREENING_ALGORITHMS),
        "full_screening_timesteps_per_trial": full_timesteps,
        "full_screening_trials_per_algorithm": trials_per_algorithm,
        "runtime_projection_limit_hours": RUNTIME_PROJECTION_LIMIT_HOURS,
        "runtime_projection_margin": RUNTIME_PROJECTION_MARGIN,
        "full_screening_eligible": full_screening_eligible,
        "num_envs": locked_num_envs,
        "objective_metric": objective_metric,
        "source_config": {
            "path": _project_relative(project_root, inputs.config_path),
            "bytes": inputs.config_path.stat().st_size,
            "file_sha256": inputs.config_file_sha256,
            "resolved_sha256": inputs.config_sha256,
        },
        "observed_features": {
            "path": _project_relative(project_root, inputs.feature_path),
            "bytes": inputs.feature_path.stat().st_size,
            "sha256": inputs.data_sha256,
            "data_source": inputs.data_source,
        },
        "readiness": readiness,
        "runs": run_records,
    }
    summary_path = output_path / SUMMARY_NAME
    atomic_json(summary_path, summary)
    atomic_text(Path(str(summary_path) + ".sha256"), sha256_file(summary_path) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a sequential observed-data smoke test for all screening algorithms"
    )
    parser.add_argument(
        "--config",
        default="campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    args = parser.parse_args()
    summary = run_screening_smoke(
        config_path=args.config,
        output_dir=args.output_dir,
        timesteps=args.timesteps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
