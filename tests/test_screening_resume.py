from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.experiments.run_screening import (
    archive_previous_screening_trial_state,
    completed_screening_trial_is_reusable,
    run_screening_task,
)


@pytest.fixture(autouse=True)
def _deterministic_screening_sampler(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.experiments.run_screening.suggest_hyperparameters",
        lambda algorithm, trial_number, seed, require_optuna=False: {
            "learning_rate": 0.001,
            "n_steps": 64,
            **({"batch_size": 32} if algorithm == "ppo" else {}),
        },
    )


def _write_complete_trial(
    trial_dir: Path,
    *,
    config: dict,
    identity: dict,
    attempt_id: str = "attempt-1",
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    algorithm = identity["algorithm"]
    (trial_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": identity["experiment_id"],
                "status": "complete",
                "identity": identity,
                "campaign_id": "campaign",
                "campaign_attempt_id": attempt_id,
                "campaign_attempt_sha256": "a" * 64,
                "campaign_stage_id": "screening",
                "deployment_release_id": "b" * 64,
                "bistka_preflight_sha256": "c" * 64,
                "protocol_lock_hash": None,
                "runtime": {
                    "slurm_job_id": "123_0",
                    "slurm_array_job_id": "123",
                    "slurm_array_task_id": "0",
                },
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    (trial_dir / "trial_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": algorithm,
                "trial_id": identity["trial_id"],
                "trial_number": identity["trial_number"],
                "seed": identity["seed"],
                "objective_metric": "sharpe",
                "objective_value": 1.2,
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / f"{algorithm}_model.zip").write_bytes(b"model")
    pd.DataFrame(
        [
            {
                "policy": algorithm,
                "sharpe": 1.2,
                "max_drawdown": -0.1,
                "turnover": 0.2,
            }
        ]
    ).to_csv(trial_dir / f"{algorithm}_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2026-01-01", "ticker": "A", "weight": 0.4},
            {"date": "2026-01-01", "ticker": "B", "weight": 0.6},
        ]
    ).to_csv(trial_dir / f"{algorithm}_actions.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2026-01-01", "portfolio_value": 1.0},
            {"date": "2026-01-02", "portfolio_value": 1.1},
        ]
    ).to_csv(trial_dir / f"{algorithm}_equity_curve.csv", index=False)


def test_completed_screening_trial_is_reusable_for_exact_same_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = {"project": {"seed": 42}, "training": {"algo": "ppo", "seed": 42}}
    identity = {
        "experiment_id": "screen_ppo_0",
        "algorithm": "ppo",
        "window": None,
        "seed": 42,
        "feature_group": "screening",
        "cost_scenario": "base",
        "selection_split": "validation",
        "trial_id": 0,
        "trial_number": 0,
    }
    _write_complete_trial(tmp_path, config=config, identity=identity)
    environment = {
        "CAMPAIGN_ID": "campaign",
        "CAMPAIGN_ATTEMPT_ID": "attempt-1",
        "CAMPAIGN_ATTEMPT_SHA256": "a" * 64,
        "CAMPAIGN_STAGE_ID": "screening",
        "DEPLOYMENT_RELEASE_ID": "b" * 64,
        "BISTKA_PREFLIGHT_SHA256": "c" * 64,
        "SLURM_JOB_ID": "123_0",
        "SLURM_ARRAY_JOB_ID": "123",
        "SLURM_ARRAY_TASK_ID": "0",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    assert completed_screening_trial_is_reusable(
        tmp_path,
        trial_config=config,
        identity=identity,
        objective_metric="sharpe",
    )


def test_completed_screening_trial_rejects_different_attempt_or_invalid_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = {"project": {"seed": 42}, "training": {"algo": "ppo", "seed": 42}}
    identity = {
        "experiment_id": "screen_ppo_0",
        "algorithm": "ppo",
        "window": None,
        "seed": 42,
        "feature_group": "screening",
        "cost_scenario": "base",
        "selection_split": "validation",
        "trial_id": 0,
        "trial_number": 0,
    }
    _write_complete_trial(tmp_path, config=config, identity=identity)
    monkeypatch.setenv("CAMPAIGN_ATTEMPT_ID", "attempt-2")
    assert not completed_screening_trial_is_reusable(
        tmp_path,
        trial_config=config,
        identity=identity,
        objective_metric="sharpe",
    )
    monkeypatch.delenv("CAMPAIGN_ATTEMPT_ID")
    (tmp_path / "ppo_model.zip").write_bytes(b"")
    assert not completed_screening_trial_is_reusable(
        tmp_path,
        trial_config=config,
        identity=identity,
        objective_metric="sharpe",
    )


def _screening_config(tmp_path: Path) -> dict:
    return {
        "project": {"name": "resume_test", "seed": 42},
        "data": {"source": "synthetic"},
        "paths": {"models_dir": str(tmp_path / "models")},
        "training": {"seed": 42},
        "screening": {
            "algorithms": ["ppo"],
            "trials_per_algorithm": 2,
            "total_timesteps": 10,
            "objective_metric": "sharpe",
        },
    }


def _write_training_outputs(output_dir: Path, algorithm: str) -> None:
    (output_dir / f"{algorithm}_model.zip").write_bytes(b"model")
    pd.DataFrame(
        [{"policy": algorithm, "sharpe": 0.5, "max_drawdown": -0.1, "turnover": 0.2}]
    ).to_csv(output_dir / f"{algorithm}_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2026-01-01", "ticker": "A", "weight": 0.5},
            {"date": "2026-01-01", "ticker": "B", "weight": 0.5},
        ]
    ).to_csv(output_dir / f"{algorithm}_actions.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2026-01-01", "portfolio_value": 1.0},
            {"date": "2026-01-02", "portfolio_value": 1.1},
        ]
    ).to_csv(output_dir / f"{algorithm}_equity_curve.csv", index=False)


def test_screening_task_continues_after_failure_then_raises_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _screening_config(tmp_path)
    monkeypatch.setattr("src.experiments.run_screening.ensure_dirs", lambda *args: None)
    monkeypatch.setattr("src.experiments.run_screening.ensure_feature_inputs", lambda *args: None)
    monkeypatch.setattr("src.experiments.run_screening.write_split_summary", lambda *args: None)

    def fake_train(trial_config: dict, output_dir: Path) -> bool:
        if output_dir.name == "trial_0":
            raise RuntimeError("intentional trial failure")
        _write_training_outputs(output_dir, trial_config["training"]["algo"])
        return True

    monkeypatch.setattr("src.experiments.run_screening.train_with_stable_baselines", fake_train)

    with pytest.raises(RuntimeError, match="1 screening trial.*failed.*0"):
        run_screening_task(config, task_id=0, output_dir=tmp_path / "run")

    summary = json.loads((tmp_path / "run" / "screening_task_0.json").read_text())
    assert summary["completed_trials"] == 1
    assert summary["failed_trials"] == 1
    assert json.loads(
        (tmp_path / "run" / "trial_0" / "experiment_manifest.json").read_text()
    )["status"] == "failed"
    assert json.loads(
        (tmp_path / "run" / "trial_1" / "experiment_manifest.json").read_text()
    )["status"] == "complete"


def test_retry_archives_prior_manifest_and_resolved_config(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial_0"
    trial_dir.mkdir()
    (trial_dir / "experiment_manifest.json").write_text(
        json.dumps({"status": "failed", "error": "old failure"}),
        encoding="utf-8",
    )
    (trial_dir / "resolved_config.json").write_text(
        json.dumps({"training": {"seed": 42}}),
        encoding="utf-8",
    )

    archive_previous_screening_trial_state(trial_dir)

    assert json.loads(
        (trial_dir / "retry_history" / "experiment_manifest_001_failed.json").read_text()
    )["error"] == "old failure"
    assert json.loads(
        (trial_dir / "retry_history" / "resolved_config_001_failed.json").read_text()
    )["training"]["seed"] == 42


def test_screening_task_same_attempt_reuses_all_valid_completed_trials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _screening_config(tmp_path)
    monkeypatch.setattr("src.experiments.run_screening.ensure_dirs", lambda *args: None)
    monkeypatch.setattr("src.experiments.run_screening.ensure_feature_inputs", lambda *args: None)
    monkeypatch.setattr("src.experiments.run_screening.write_split_summary", lambda *args: None)

    def fake_train(trial_config: dict, output_dir: Path) -> bool:
        _write_training_outputs(output_dir, trial_config["training"]["algo"])
        return True

    monkeypatch.setattr("src.experiments.run_screening.train_with_stable_baselines", fake_train)
    run_dir = tmp_path / "run"
    run_screening_task(config, task_id=0, output_dir=run_dir)

    def unexpected_train(*args, **kwargs) -> bool:
        raise AssertionError("a valid completed trial must not be retrained")

    monkeypatch.setattr("src.experiments.run_screening.train_with_stable_baselines", unexpected_train)
    run_screening_task(config, task_id=0, output_dir=run_dir)

    summary = json.loads((run_dir / "screening_task_0.json").read_text())
    assert summary["completed_trials"] == 2
    assert summary["reused_trials"] == 2
    assert summary["failed_trials"] == 0
