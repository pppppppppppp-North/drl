from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.campaigns.calibration import sha256_file
from src.campaigns.control import load_definition
from src.campaigns.gates import _screening_smoke_summary_validator
from src.campaigns.slurm import validate_slurm_stage
from src.experiments import run_screening_smoke as smoke_module
from src.experiments.run_screening_smoke import (
    DEFAULT_TIMESTEPS,
    REPRESENTATIVE_TRIAL_NUMBERS,
    SCREENING_ALGORITHMS,
    build_smoke_config,
    run_screening_smoke,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _deterministic_screening_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sample(
        algorithm: str,
        trial_number: int,
        seed: int,
        *,
        require_optuna: bool = False,
    ) -> dict:
        del trial_number, seed, require_optuna
        if algorithm == "ppo":
            return {"n_steps": 64, "batch_size": 32, "learning_rate": 0.001}
        if algorithm == "a2c":
            return {"n_steps": 64, "learning_rate": 0.001}
        params = {
            "learning_starts": 100,
            "buffer_size": 1_000,
            "batch_size": 64,
            "train_freq": 1,
            "gradient_steps": 1,
            "learning_rate": 0.001,
        }
        if algorithm == "td3":
            params["policy_delay"] = 2
        if algorithm == "sac":
            params["ent_coef"] = "auto"
        return params

    monkeypatch.setattr(smoke_module, "suggest_hyperparameters", sample)


def _smoke_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    for directory in (
        project / "config",
        project / "campaigns" / "test" / "derived",
        project / "data",
        project / "reports" / "data_quality_market",
        project / "src",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    features = pd.DataFrame(
        [
            {
                "date": f"2024-01-{day:02d}",
                "ticker": ticker,
                "close": 100.0 + day,
                "return_1d": 0.001 * day,
            }
            for day in range(1, 8)
            for ticker in ("AAA.BK", "BBB.BK")
        ]
    )
    features.to_csv(project / "data" / "features.csv", index=False)
    readiness = {
        "schema_version": 1,
        "status": "pass",
        "data_origin": "observed_market",
        "evidence_class": "test_observed",
        "execution_eligible": True,
    }
    (project / "reports" / "data_quality_market" / "market_data_readiness.json").write_text(
        json.dumps(readiness),
        encoding="utf-8",
    )
    config = {
        "project": {"name": "screening_smoke_test", "seed": 42},
        "paths": {
            "features": "data/features.csv",
            "fallback_features_csv": "data/features.csv",
            "models_dir": "models/screening",
        },
        "data": {"source": "observed_test"},
        "features": {"lookback": 2, "columns": ["return_1d"]},
        "splits": {
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "train_split": "train",
            "eval_split": "validation",
        },
        "environment": {
            "initial_cash": 1_000.0,
            "transaction_cost": 0.0,
            "reward": {},
        },
        "training": {
            "algo": "ppo",
            "seed": 42,
            "total_timesteps": 20_000,
            "num_envs": 1,
        },
    }
    config_path = project / "campaigns" / "test" / "derived" / "locked_screening.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return project, config_path


def _fake_training(config: dict, output_dir: Path) -> bool:
    algorithm = str(config["training"]["algo"])
    timesteps = int(config["training"]["total_timesteps"])
    (output_dir / f"{algorithm}_model.zip").write_bytes(b"model")
    checkpoint_dir = output_dir / "checkpoints" / algorithm
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / f"{algorithm}_checkpoint_{timesteps}_steps.zip").write_bytes(
        b"checkpoint"
    )
    tensorboard_dir = output_dir / "tensorboard" / f"{algorithm}_1"
    tensorboard_dir.mkdir(parents=True)
    (tensorboard_dir / "events.out.tfevents.test").write_bytes(b"events")
    pd.DataFrame(
        [
            {
                "policy": algorithm,
                "sharpe": 1.0,
                "max_drawdown": -0.1,
                "turnover": 0.2,
            }
        ]
    ).to_csv(output_dir / f"{algorithm}_metrics.csv", index=False)
    pd.DataFrame(
        [{"date": "2024-01-01", "portfolio_value": 1_000.0}]
    ).to_csv(output_dir / f"{algorithm}_equity_curve.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "AAA.BK", "weight": 0.5},
            {"date": "2024-01-01", "ticker": "BBB.BK", "weight": 0.5},
        ]
    ).to_csv(output_dir / f"{algorithm}_actions.csv", index=False)
    return True


def _fake_split_summary(_: dict, output_dir: Path) -> None:
    pd.DataFrame([{"split": "train", "rows": 10}]).to_csv(
        output_dir / "split_summary.csv",
        index=False,
    )


def test_smoke_config_changes_only_algorithm_seed_and_budget() -> None:
    locked = {
        "project": {"name": "locked", "seed": 42},
        "paths": {"features": "data/features.csv"},
        "data": {"source": "observed"},
        "training": {"algo": "ppo", "seed": 7, "total_timesteps": 20_000, "num_envs": 1},
    }
    smoke = build_smoke_config(locked, algorithm="sac", timesteps=DEFAULT_TIMESTEPS)

    assert smoke["training"]["algo"] == "sac"
    assert smoke["training"]["seed"] == 42 + 4 * 30 + REPRESENTATIVE_TRIAL_NUMBERS["sac"]
    assert smoke["training"]["total_timesteps"] == DEFAULT_TIMESTEPS
    assert smoke["training"]["num_envs"] == 1
    assert smoke["training"]["checkpoint_freq"] == DEFAULT_TIMESTEPS
    assert smoke["training"]["save_replay_buffer"] is False
    assert "learning_starts" in smoke["training"]
    assert smoke["project"]["name"] == locked["project"]["name"]
    assert smoke["project"]["seed"] == smoke["training"]["seed"]
    assert locked["project"]["seed"] == 42
    assert locked["training"]["algo"] == "ppo"


def test_observed_screening_smoke_requires_all_five_complete_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, config_path = _smoke_project(tmp_path)
    monkeypatch.chdir(project)
    monkeypatch.setattr(smoke_module, "train_with_stable_baselines", _fake_training)
    monkeypatch.setattr(smoke_module, "write_split_summary", _fake_split_summary)

    summary = run_screening_smoke(
        config_path=config_path,
        output_dir="results/screening_smoke_123",
    )

    assert summary["status"] == "complete"
    assert summary["algorithms"] == list(SCREENING_ALGORITHMS)
    assert summary["complete_manifest_count"] == 5
    assert summary["total_requested_timesteps"] == 10_240
    assert summary["num_envs"] == 1
    assert summary["full_screening_eligible"] is True
    assert all(run["sampled_params"] for run in summary["runs"])
    assert all(run["runtime_projection"]["within_limit"] for run in summary["runs"])
    assert all(run["manifest"]["status"] == "complete" for run in summary["runs"])
    summary_path = project / "results" / "screening_smoke_123" / "screening_smoke_summary.json"
    assert Path(str(summary_path) + ".sha256").read_text(encoding="utf-8").strip() == sha256_file(
        summary_path
    )
    options = {
        "algorithms": list(SCREENING_ALGORITHMS),
        "timesteps_per_algorithm": DEFAULT_TIMESTEPS,
        "representative_trials": REPRESENTATIVE_TRIAL_NUMBERS,
        "runtime_limit_hours": 72,
        "required_runtime_packages": [],
    }
    assert _screening_smoke_summary_validator(summary_path, options) == []

    tampered = json.loads(summary_path.read_text(encoding="utf-8"))
    tampered["timesteps_per_algorithm"] = 1
    summary_path.write_text(json.dumps(tampered), encoding="utf-8")
    issues = _screening_smoke_summary_validator(summary_path, options)
    assert "screening smoke timesteps do not match the declared fixed budget" in issues
    assert any("sidecar does not match" in issue for issue in issues)


def test_screening_smoke_fails_if_an_algorithm_omits_a_required_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, config_path = _smoke_project(tmp_path)
    monkeypatch.chdir(project)

    def incomplete_training(config: dict, output_dir: Path) -> bool:
        result = _fake_training(config, output_dir)
        if config["training"]["algo"] == "td3":
            (output_dir / "td3_actions.csv").unlink()
        return result

    monkeypatch.setattr(smoke_module, "train_with_stable_baselines", incomplete_training)
    monkeypatch.setattr(smoke_module, "write_split_summary", _fake_split_summary)

    with pytest.raises((FileNotFoundError, ValueError), match="td3"):
        run_screening_smoke(
            config_path=config_path,
            output_dir="results/screening_smoke_456",
        )
    failed_manifest = json.loads(
        (
            project
            / "results"
            / "screening_smoke_456"
            / "algorithm_td3"
            / "experiment_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert failed_manifest["status"] == "failed"
    assert not (
        project / "results" / "screening_smoke_456" / "screening_smoke_summary.json"
    ).exists()


def test_campaign_requires_single_job_smoke_before_full_screening() -> None:
    definition, _ = load_definition(
        PROJECT_ROOT / "config" / "research_campaign.yaml",
        campaign_dir=PROJECT_ROOT / "campaigns" / "large_scale_thai_drl_v2",
    )
    smoke = definition.stage_map["screening_smoke"]
    screening = definition.stage_map["screening"]
    resources = validate_slurm_stage(definition, smoke)

    assert smoke.expected_manifests == 5
    assert smoke.planned_runs == 5
    assert smoke.planned_timesteps == 10_240
    assert smoke.array_tasks == 0
    assert "screening_smoke" in screening.depends_on
    assert resources == {
        "script": "slurm/screening_smoke.sbatch",
        "partition": "compute-devel",
        "account": "25sfcs03",
        "cpus_per_task": 2,
        "memory": "8GB",
        "walltime": "02:00:00",
        "array_tasks": 0,
        "array_concurrency": 1,
        "output": "logs/screening_smoke_%j.out",
        "error": "logs/screening_smoke_%j.err",
    }
