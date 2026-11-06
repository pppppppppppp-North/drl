from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.benchmark_vectorized_training import benchmark_vector_counts, select_num_envs, trial_command
from scripts.verify_determinism import run_determinism, trajectory_command
from src.campaigns.calibration import CalibrationRun, load_observed_inputs, receipt_path_for


def _observed_config(tmp_path: Path) -> Path:
    dates = pd.bdate_range("2024-01-02", periods=14)
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(("AAA.BK", "BBB.BK")):
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "close": 100.0 + date_index + ticker_index,
                    "return_1d": 0.001 * (date_index + 1),
                }
            )
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    config_dir.mkdir()
    pd.DataFrame(rows).to_csv(data_dir / "features.csv", index=False)
    config = {
        "project": {"name": "calibration_test", "seed": 7},
        "paths": {
            "features": "data/features.parquet",
            "fallback_features_csv": "data/features.csv",
        },
        "data": {"source": "observed_test"},
        "features": {"lookback": 2, "columns": ["return_1d"]},
        "environment": {"initial_cash": 1_000.0, "transaction_cost": 0.0, "reward": {}},
        "training": {"vectorization_backend": "dummy"},
    }
    path = config_dir / "observed.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_observed_inputs_load_only_a_declared_existing_feature_table(tmp_path: Path) -> None:
    config_path = _observed_config(tmp_path)
    loaded = load_observed_inputs(config_path)

    assert loaded.synthetic is False
    assert loaded.data_source == "observed_test"
    assert loaded.feature_path == (tmp_path / "data" / "features.csv").resolve()
    assert len(loaded.data_sha256) == len(loaded.config_sha256) == 64

    loaded.feature_path.unlink()
    with pytest.raises(FileNotFoundError, match="will not synthesize"):
        load_observed_inputs(config_path)


def test_observed_loader_rejects_a_synthetic_config(tmp_path: Path) -> None:
    config_path = _observed_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["source"] = "synthetic"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="explicit --synthetic"):
        load_observed_inputs(config_path)


def test_benchmark_builds_one_fresh_worker_command_per_repeat_and_selects_smallest_near_peak() -> None:
    calls = []

    def fake_trial(**kwargs):
        calls.append(kwargs)
        throughput = {1: 96.0, 2: 100.0, 4: 98.0}[kwargs["num_envs"]]
        return {
            "num_envs": kwargs["num_envs"],
            "repeat": kwargs["repeat"],
            "samples_per_second": throughput,
            "peak_rss_mib": float(kwargs["num_envs"] * 10),
        }

    results = benchmark_vector_counts(
        [1, 2, 4],
        timesteps=32,
        backend="dummy",
        seed=7,
        repeats=2,
        synthetic=True,
        trial_runner=fake_trial,
    )
    selected, summary = select_num_envs(results)

    assert len(calls) == 6
    assert {(item["num_envs"], item["repeat"]) for item in calls} == {
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (4, 1),
        (4, 2),
    }
    assert selected == 1
    assert summary["repeats"].tolist() == [2, 2, 2]

    command = trial_command(
        config_path=None,
        synthetic=True,
        num_envs=4,
        timesteps=32,
        backend="dummy",
        seed=7,
        repeat=2,
        output="trial.json",
    )
    assert "--trial-worker" in command
    assert "--synthetic" in command


def test_determinism_uses_two_independent_observed_data_subprocesses(tmp_path: Path) -> None:
    config_path = _observed_config(tmp_path)
    result = run_determinism(13, config_path=config_path)

    assert result["deterministic"] is True
    assert result["first_sha256"] == result["second_sha256"]
    assert result["independent_subprocess_runs"] == 2
    assert len(set(result["process_ids"])) == 2
    assert result["data_provenance"]["source"] == "observed_test"
    assert result["data_provenance"]["synthetic"] is False
    assert all(len(result["data_provenance"][key]) == 64 for key in ("data_sha256", "config_sha256", "code_sha256"))

    command = trajectory_command(
        config_path=config_path,
        synthetic=False,
        seed=13,
        output=tmp_path / "trajectory.json",
    )
    assert "--trajectory-worker" in command
    assert "--config" in command
    assert "--synthetic" not in command


def test_calibration_receipt_is_attempt_bound_atomic_and_hashes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAMPAIGN_ATTEMPT_ID", "attempt-007")
    monkeypatch.setenv("CAMPAIGN_ATTEMPT_SHA256", "a" * 64)
    monkeypatch.setenv("CAMPAIGN_STAGE_ID", "unit_calibration")
    monkeypatch.setenv("DEPLOYMENT_RELEASE_ID", "b" * 64)
    monkeypatch.setenv("BISTKA_PREFLIGHT_SHA256", "c" * 64)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    output = tmp_path / "measurement.csv"
    output.write_text("value\n1\n", encoding="utf-8")
    receipt_path = receipt_path_for(output)

    run = CalibrationRun(kind="unit_calibration", receipt_path=receipt_path, parameters={"seed": 7})
    with run:
        run.add_output("measurement", output)
        run.succeed({"accepted": True})

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "success"
    assert receipt["campaign_attempt_id"] == "attempt-007"
    assert receipt["campaign_attempt_sha256"] == "a" * 64
    assert receipt["campaign_stage_id"] == "unit_calibration"
    assert receipt["deployment_release_id"] == "b" * 64
    assert receipt["bistka_preflight_sha256"] == "c" * 64
    assert receipt["slurm_job_id"] == "12345"
    assert receipt["outputs"][0]["sha256"]
    assert receipt["result"] == {"accepted": True}
    assert not list(tmp_path.glob(f".{receipt_path.name}.*"))


def test_slurm_calibration_without_attempt_id_writes_failed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CAMPAIGN_ATTEMPT_ID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    receipt_path = tmp_path / "failed.receipt.json"
    run = CalibrationRun(kind="unit_calibration", receipt_path=receipt_path, parameters={})

    with pytest.raises(RuntimeError, match="CAMPAIGN_ATTEMPT_ID"):
        with run:
            pass

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["slurm_job_id"] == "12345"
    assert receipt["error"]["type"] == "RuntimeError"
