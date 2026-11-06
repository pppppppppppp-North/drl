from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.validate_screening_storage_safe_results import validate_run
from src.campaigns.calibration import sha256_file
from src.experiments.screening_storage import finalize_checkpoint_free_trial


ALGORITHMS = ("ppo", "a2c", "ddpg", "td3", "sac")
JOB_ID = 12345
ATTEMPT_ID = "screening_20260725T120000Z_deadbeef"


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _materialize_valid_run(root: Path, *, trials_per_algorithm: int = 2) -> Path:
    feature_path = root / "data/processed/features.csv"
    feature_path.parent.mkdir(parents=True)
    feature_path.write_text("date,ticker,return_1d\n2024-01-01,A,0.0\n", encoding="utf-8")
    config_relative = Path(
        "campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml"
    )
    config_path = root / config_relative
    config_path.parent.mkdir(parents=True)
    config = {
        "project": {"name": "test_screening", "seed": 42},
        "paths": {
            "features": "data/processed/features.csv",
            "models_dir": "models",
        },
        "training": {"algo": "ppo", "seed": 42, "total_timesteps": 20},
        "screening": {
            "algorithms": list(ALGORITHMS),
            "trials_per_algorithm": trials_per_algorithm,
            "total_timesteps": 20,
            "objective_metric": "sharpe",
            "top_n_per_algorithm": 1,
            "selected_algorithm_count": 2,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    attempt_path = (
        root
        / "campaigns/large_scale_thai_drl_v2/attempts"
        / f"{ATTEMPT_ID}.json"
    )
    attempt = {
        "schema_version": 2,
        "status": "rendered",
        "campaign_id": "large_scale_thai_drl_v2",
        "stage_id": "screening",
        "attempt_id": ATTEMPT_ID,
        "rendered_at_utc": "2026-07-25T12:00:00+00:00",
        "inputs": [
            {
                "path": config_relative.as_posix(),
                "bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            }
        ],
        "deployment_manifest_path": (
            "campaigns/large_scale_thai_drl_v2/deployment/release.json"
        ),
        "deployment_manifest_sha256": "1" * 64,
        "deployment_release_id": "2" * 64,
        "preflight_receipt_path": "reports/preflight.json",
        "preflight_sha256": "3" * 64,
        "runtime_configs": [],
        "runtime_config_path": config_relative.as_posix(),
        "runtime_config_sha256": sha256_file(config_path),
        "runtime_feature_path": "data/processed/features.csv",
        "runtime_feature_sha256": sha256_file(feature_path),
        "predecessor_gates": [],
    }
    attempt["attempt_sha256"] = hashlib.sha256(
        json.dumps(attempt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _write_json(attempt_path, attempt)

    run = root / f"results/large_scale_screening_{JOB_ID}"
    run.mkdir(parents=True)
    for task_id, algorithm in enumerate(ALGORITHMS):
        _write_json(
            run / f"screening_task_{task_id}.json",
            {
                "task_id": task_id,
                "algorithm": algorithm,
                "trials_per_algorithm": trials_per_algorithm,
                "total_timesteps": 20,
                "objective_metric": "sharpe",
                "selection_split": "validation",
                "completed_trials": trials_per_algorithm,
                "reused_trials": 0,
                "failed_trials": 0,
                "failures": [],
            },
        )
        for trial_number in range(trials_per_algorithm):
            trial_id = task_id * trials_per_algorithm + trial_number
            seed = 42 + trial_id
            trial = run / f"trial_{trial_id}"
            trial.mkdir()
            identity = {
                "experiment_id": f"screen_{algorithm}_{trial_number}",
                "algorithm": algorithm,
                "window": None,
                "seed": seed,
                "feature_group": "test_screening",
                "cost_scenario": "base",
                "selection_split": "validation",
                "trial_id": trial_id,
                "trial_number": trial_number,
            }
            _write_json(
                trial / "experiment_manifest.json",
                {
                    "experiment_id": identity["experiment_id"],
                    "status": "complete",
                    "identity": identity,
                    "data_files": {
                        "features": {
                            "path": str(feature_path),
                            "bytes": feature_path.stat().st_size,
                            "sha256": sha256_file(feature_path),
                        }
                    },
                    "runtime": {
                        "slurm_job_id": f"{JOB_ID}_{task_id}",
                        "slurm_array_job_id": str(JOB_ID),
                        "slurm_array_task_id": str(task_id),
                    },
                    "campaign_id": "large_scale_thai_drl_v2",
                    "campaign_attempt_id": ATTEMPT_ID,
                    "campaign_attempt_sha256": attempt["attempt_sha256"],
                    "campaign_stage_id": "screening",
                    "deployment_release_id": "2" * 64,
                    "bistka_preflight_sha256": "3" * 64,
                    "error": None,
                },
            )
            resolved = json.loads(json.dumps(config))
            resolved["project"]["seed"] = seed
            resolved["training"]["algo"] = algorithm
            resolved["training"]["seed"] = seed
            _write_json(trial / "resolved_config.json", resolved)
            objective = float(task_id + 1) + trial_number / 10
            _write_json(
                trial / "trial_metadata.json",
                {
                    "trial_id": trial_id,
                    "trial_number": trial_number,
                    "seed": seed,
                    "config_path": "resolved_config.json",
                    "algorithm": algorithm,
                    "sampled_params": {"learning_rate": 0.001},
                    "objective_metric": "sharpe",
                    "objective_value": objective,
                    "metrics": {"sharpe": objective},
                },
            )
            (trial / "split_summary.csv").write_text(
                "split,rows\ntrain,1\nvalidation,1\n",
                encoding="utf-8",
            )
            (trial / f"{algorithm}_model.zip").write_bytes(b"model")
            (trial / f"{algorithm}_metrics.csv").write_text(
                f"sharpe,max_drawdown\n{objective},-0.1\n",
                encoding="utf-8",
            )
            (trial / f"{algorithm}_equity_curve.csv").write_text(
                "date,portfolio_value\n2024-01-01,1.0\n",
                encoding="utf-8",
            )
            (trial / f"{algorithm}_actions.csv").write_text(
                "date,ticker,weight\n2024-01-01,A,1.0\n",
                encoding="utf-8",
            )
            finalize_checkpoint_free_trial(
                trial,
                algorithm,
                callback_instances=1,
                callback_on_step_calls=1,
            )
    return run


def test_validator_accepts_complete_array_bound_screening(tmp_path: Path) -> None:
    run = _materialize_valid_run(tmp_path)

    result = validate_run(
        run,
        config_path=(
            "campaigns/large_scale_thai_drl_v2/derived/"
            "locked_large_scale_screening.yaml"
        ),
        attempt_path=(
            "campaigns/large_scale_thai_drl_v2/attempts/"
            f"{ATTEMPT_ID}.json"
        ),
        job_id=JOB_ID,
        project_root=tmp_path,
        expected_trials_per_algorithm=2,
    )

    assert result["status"] == "pass"
    assert result["complete_trials"] == 10
    assert result["array_tasks"] == 5
    assert result["retention_receipts"] == 10
    assert result["checkpoint_paths"] == 0
    assert result["aggregate_dry_run"]["complete_trials"] == 10
    assert len(result["aggregate_dry_run"]["selected_algorithms"]) == 2
    assert result["content"]["files"] > 0


def test_validator_rejects_one_missing_trial_before_aggregation(tmp_path: Path) -> None:
    run = _materialize_valid_run(tmp_path)
    missing = run / "trial_9"
    for path in sorted(missing.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    missing.rmdir()

    with pytest.raises(ValueError, match="trial grid is incomplete"):
        validate_run(
            run,
            config_path=(
                "campaigns/large_scale_thai_drl_v2/derived/"
                "locked_large_scale_screening.yaml"
            ),
            attempt_path=(
                "campaigns/large_scale_thai_drl_v2/attempts/"
                f"{ATTEMPT_ID}.json"
            ),
            job_id=JOB_ID,
            project_root=tmp_path,
            expected_trials_per_algorithm=2,
        )


def test_validator_rejects_mixed_attempt_lineage(tmp_path: Path) -> None:
    run = _materialize_valid_run(tmp_path)
    manifest_path = run / "trial_0/experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["campaign_attempt_id"] = "screening_20260725T120001Z_cafebabe"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="manifest lineage does not match"):
        validate_run(
            run,
            config_path=(
                "campaigns/large_scale_thai_drl_v2/derived/"
                "locked_large_scale_screening.yaml"
            ),
            attempt_path=(
                "campaigns/large_scale_thai_drl_v2/attempts/"
                f"{ATTEMPT_ID}.json"
            ),
            job_id=JOB_ID,
            project_root=tmp_path,
            expected_trials_per_algorithm=2,
        )


def test_validator_rejects_any_checkpoint_path(tmp_path: Path) -> None:
    run = _materialize_valid_run(tmp_path)
    checkpoint = run / "trial_0/checkpoints/ppo/unexpected.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="contains a checkpoints path"):
        validate_run(
            run,
            config_path=(
                "campaigns/large_scale_thai_drl_v2/derived/"
                "locked_large_scale_screening.yaml"
            ),
            attempt_path=(
                "campaigns/large_scale_thai_drl_v2/attempts/"
                f"{ATTEMPT_ID}.json"
            ),
            job_id=JOB_ID,
            project_root=tmp_path,
            expected_trials_per_algorithm=2,
        )
