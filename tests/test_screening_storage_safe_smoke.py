from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.campaigns.calibration import sha256_file
from src.experiments import run_screening_smoke as base_smoke
from src.experiments import run_screening_storage_safe_smoke as safe_smoke
from src.experiments.screening_storage import (
    RETENTION_RECEIPT,
    RETENTION_SIDECAR,
    finalize_checkpoint_free_trial,
)


def _valid_trial(run_dir: Path, algorithm: str) -> None:
    run_dir.mkdir(parents=True)
    payloads = {
        "experiment_manifest.json": b"{}\n",
        "resolved_config.json": b"{}\n",
        "split_summary.csv": b"split,rows\ntrain,1\n",
        f"{algorithm}_model.zip": b"model",
        f"{algorithm}_metrics.csv": b"policy,sharpe\npolicy,1.0\n",
        f"{algorithm}_equity_curve.csv": b"date,portfolio_value\n2026-01-01,1\n",
        f"{algorithm}_actions.csv": b"date,ticker,weight\n2026-01-01,A,1\n",
        "trial_metadata.json": b"{}\n",
        "screening_smoke_result.json": b"{}\n",
    }
    for name, payload in payloads.items():
        (run_dir / name).write_bytes(payload)
    finalize_checkpoint_free_trial(
        run_dir,
        algorithm,
        callback_instances=1,
        callback_on_step_calls=2,
    )


def _fake_summary(project_root: Path, output_path: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for algorithm in base_smoke.SCREENING_ALGORITHMS:
        run_dir = output_path / f"algorithm_{algorithm}"
        _valid_trial(run_dir, algorithm)
        artifacts = base_smoke._required_artifacts(
            run_dir, algorithm, base_smoke.DEFAULT_TIMESTEPS
        )
        assert all(path.is_file() for path in artifacts)
        runs.append(
            {
                "algorithm": algorithm,
                "run_dir": str(run_dir.relative_to(project_root)),
                "artifacts": [
                    {
                        "path": str(path.relative_to(project_root)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in artifacts
                ],
            }
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "smoke_kind": "observed_data_five_algorithm_screening",
        "algorithms": list(base_smoke.SCREENING_ALGORITHMS),
        "run_count": len(runs),
        "runs": runs,
    }


def test_required_artifacts_replace_checkpoint_with_signed_retention(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "algorithm_ppo"
    _valid_trial(run_dir, "ppo")

    artifacts = safe_smoke._storage_safe_required_artifacts(run_dir, "ppo", 2048)

    assert run_dir / RETENTION_RECEIPT in artifacts
    assert run_dir / RETENTION_SIDECAR in artifacts
    assert not any("checkpoints" in path.parts for path in artifacts)


def test_required_artifacts_fail_without_retention_receipt(tmp_path: Path) -> None:
    run_dir = tmp_path / "algorithm_ppo"
    run_dir.mkdir()
    (run_dir / "ppo_model.zip").write_bytes(b"model")

    with pytest.raises(ValueError, match="retention is invalid"):
        safe_smoke._storage_safe_required_artifacts(run_dir, "ppo", 2048)


def test_storage_safe_smoke_patches_restores_and_signs_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    original_train = base_smoke.train_with_stable_baselines
    original_required = base_smoke._required_artifacts
    saw_hooks = False

    def fake_base(**kwargs: Any) -> dict[str, Any]:
        nonlocal saw_hooks
        saw_hooks = (
            base_smoke.train_with_stable_baselines is not original_train
            and base_smoke._required_artifacts is safe_smoke._storage_safe_required_artifacts
        )
        output = tmp_path / str(kwargs["output_dir"])
        output.mkdir(parents=True)
        return _fake_summary(tmp_path, output)

    monkeypatch.setattr(base_smoke, "run_screening_smoke", fake_base)
    summary = safe_smoke.run_screening_storage_safe_smoke(
        config_path="locked.yaml",
        output_dir="results/safe_smoke_1",
    )

    assert saw_hooks is True
    assert base_smoke.train_with_stable_baselines is original_train
    assert base_smoke._required_artifacts is original_required
    assert summary["checkpoint_retention"]["policy"] == safe_smoke.STORAGE_SAFE_POLICY
    assert summary["checkpoint_retention"]["run_count"] == 5
    assert safe_smoke.validate_storage_safe_smoke_summary(
        summary, project_root=tmp_path
    ) == []
    summary_path = tmp_path / "results/safe_smoke_1/screening_smoke_summary.json"
    assert Path(str(summary_path) + ".sha256").read_text(encoding="utf-8") == (
        sha256_file(summary_path) + "\n"
    )
    validation_path = tmp_path / "results/safe_smoke_1" / safe_smoke.VALIDATION_RECEIPT
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["status"] == "pass"
    assert validation["policy"] == safe_smoke.STORAGE_SAFE_POLICY
    assert validation["summary"]["sha256"] == sha256_file(summary_path)
    assert validation["run_count"] == 5
    assert (tmp_path / "results/safe_smoke_1" / safe_smoke.VALIDATION_SIDECAR).read_text(
        encoding="utf-8"
    ) == sha256_file(validation_path) + "\n"


def test_storage_safe_smoke_restores_hooks_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    original_train = base_smoke.train_with_stable_baselines
    original_required = base_smoke._required_artifacts

    def failed_base(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("smoke failed")

    monkeypatch.setattr(base_smoke, "run_screening_smoke", failed_base)
    with pytest.raises(RuntimeError, match="smoke failed"):
        safe_smoke.run_screening_storage_safe_smoke(
            config_path="locked.yaml",
            output_dir="results/safe_smoke_failed",
        )

    assert base_smoke.train_with_stable_baselines is original_train
    assert base_smoke._required_artifacts is original_required


def test_storage_safe_summary_rejects_checkpoint_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_base(**kwargs: Any) -> dict[str, Any]:
        output = tmp_path / str(kwargs["output_dir"])
        output.mkdir(parents=True)
        return _fake_summary(tmp_path, output)

    monkeypatch.setattr(base_smoke, "run_screening_smoke", fake_base)
    summary = safe_smoke.run_screening_storage_safe_smoke(
        config_path="locked.yaml",
        output_dir="results/safe_smoke_2",
    )
    summary["runs"][0]["artifacts"].append(
        {"path": "results/safe_smoke_2/algorithm_ppo/checkpoints/ppo.zip"}
    )

    issues = safe_smoke.validate_storage_safe_smoke_summary(
        summary, project_root=tmp_path
    )
    assert "storage-safe smoke ppo inventories a checkpoint" in issues
