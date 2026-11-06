from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.validate_screening_storage_safe_smoke import validate_run
from src.campaigns.calibration import sha256_file
from src.experiments import run_screening_smoke as base_smoke
from src.experiments import run_screening_storage_safe_smoke as safe_smoke
from src.experiments.screening_storage import finalize_checkpoint_free_trial


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


def _materialize_valid_run(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.chdir(root)

    def fake_base(**kwargs: Any) -> dict[str, Any]:
        output = root / str(kwargs["output_dir"])
        output.mkdir(parents=True)
        runs: list[dict[str, Any]] = []
        for algorithm in base_smoke.SCREENING_ALGORITHMS:
            trial = output / f"algorithm_{algorithm}"
            _valid_trial(trial, algorithm)
            artifacts = base_smoke._required_artifacts(
                trial,
                algorithm,
                base_smoke.DEFAULT_TIMESTEPS,
            )
            runs.append(
                {
                    "algorithm": algorithm,
                    "run_dir": str(trial.relative_to(root)),
                    "artifacts": [
                        {
                            "path": str(path.relative_to(root)),
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

    monkeypatch.setattr(base_smoke, "run_screening_smoke", fake_base)
    output = Path("results/screening_storage_safe_smoke_123")
    safe_smoke.run_screening_storage_safe_smoke(
        config_path="locked.yaml",
        output_dir=output,
    )
    return root / output


def test_validator_accepts_complete_storage_safe_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _materialize_valid_run(tmp_path, monkeypatch)

    result = validate_run(run, project_root=tmp_path)

    assert result["status"] == "pass"
    assert result["experiment_manifests"] == 5
    assert result["retention_receipts"] == 5
    assert result["checkpoint_paths"] == 0


def test_validator_rejects_tampered_validation_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _materialize_valid_run(tmp_path, monkeypatch)
    validation_path = run / safe_smoke.VALIDATION_RECEIPT
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["status"] = "fail"
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")
    (run / safe_smoke.VALIDATION_SIDECAR).write_text(
        sha256_file(validation_path) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contract does not match"):
        validate_run(run, project_root=tmp_path)


def test_validator_rejects_any_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _materialize_valid_run(tmp_path, monkeypatch)
    checkpoint = run / "algorithm_ppo/checkpoints/ppo/unexpected.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="contains a checkpoints path"):
        validate_run(run, project_root=tmp_path)
