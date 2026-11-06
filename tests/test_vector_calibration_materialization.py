from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from src.campaigns.calibration import canonical_sha256, receipt_path_for, sha256_file
from src.campaigns.vector_calibration import (
    SELECTION_RULE,
    materialize_calibrated_screening,
)


DATA_HASH = "a" * 64
CONFIG_HASH = "b" * 64
CODE_HASH = "c" * 64


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _artifact(name: str, path: Path) -> dict:
    return {
        "name": name,
        "path": f"/lustrefs/project/test/{path.name}",
        "exists": True,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _calibration_fixture(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "project"
    campaign = project / "campaigns" / "test_campaign"
    reports = project / "reports"
    config_dir = project / "config"
    for directory in (campaign / "attempts", reports, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base = config_dir / "screening.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "preserved_project", "seed": 42},
                "paths": {"features": "data/features.csv"},
                "training": {"num_envs": 8, "total_timesteps": 20_000},
                "screening": {"algorithms": ["ppo", "a2c"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    attempt = {
        "schema_version": 1,
        "status": "rendered",
        "campaign_id": "test_campaign",
        "stage_id": "vector_benchmark",
        "attempt_id": "vector_benchmark_20260715T000000Z_deadbeef",
        "inputs": [],
    }
    attempt_path = campaign / "attempts" / "vector_benchmark_latest.json"
    _write_json(attempt_path, attempt)

    csv_path = reports / "vectorized_training_benchmark_12345.csv"
    throughputs = {1: [89.0, 90.0, 91.0], 2: [99.0, 100.0, 101.0], 4: [95.0, 96.0, 97.0]}
    fieldnames = [
        "num_envs",
        "repeat",
        "samples_per_second",
        "data_source",
        "data_sha256",
        "config_sha256",
        "code_sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for num_envs, values in throughputs.items():
            for repeat, throughput in enumerate(values, start=1):
                writer.writerow(
                    {
                        "num_envs": num_envs,
                        "repeat": repeat,
                        "samples_per_second": throughput,
                        "data_source": "yahoo_ohlcv",
                        "data_sha256": DATA_HASH,
                        "config_sha256": CONFIG_HASH,
                        "code_sha256": CODE_HASH,
                    }
                )
    metadata_path = csv_path.with_suffix(".json")
    metadata = {
        "schema_version": 2,
        "selected_num_envs": 2,
        "selection_rule": SELECTION_RULE,
        "selection_fraction_of_max": 0.95,
        "throughput_statistic": "median_samples_per_second",
        "repeats_per_env_count": 3,
        "trial_count": 9,
        "results_file": f"/lustrefs/project/test/reports/{csv_path.name}",
        "data_provenance": {
            "source": "yahoo_ohlcv",
            "synthetic": False,
            "data_sha256": DATA_HASH,
            "config_sha256": CONFIG_HASH,
        },
        "code_provenance": {"sha256": CODE_HASH, "files": []},
        "summary_by_env_count": [
            {"num_envs": num_envs, "repeats": 3, "median_samples_per_second": values[1]}
            for num_envs, values in throughputs.items()
        ],
    }
    _write_json(metadata_path, metadata)
    receipt_path = receipt_path_for(csv_path)
    receipt = {
        "schema_version": 1,
        "calibration_kind": "vectorized_training_benchmark",
        "status": "success",
        "campaign_attempt_id": attempt["attempt_id"],
        "slurm_job_id": "12345",
        "campaign_id": "test_campaign",
        "parameters": {
            "env_counts": [1, 2, 4],
            "repeats": 3,
            "synthetic": False,
        },
        "outputs": [
            _artifact("benchmark_csv", csv_path),
            _artifact("benchmark_metadata", metadata_path),
        ],
        "result": {
            "selected_num_envs": 2,
            "trial_count": 9,
            "data_sha256": DATA_HASH,
            "config_sha256": CONFIG_HASH,
            "code_sha256": CODE_HASH,
        },
    }
    _write_json(receipt_path, receipt)
    return {
        "project": project,
        "campaign": campaign,
        "base": base,
        "csv": csv_path,
        "metadata": metadata_path,
        "receipt": receipt_path,
        "attempt": attempt_path,
        "output": campaign / "derived" / "locked_screening.yaml",
    }


def _materialize(paths: dict[str, Path]) -> dict:
    return materialize_calibrated_screening(
        project_root=paths["project"],
        campaign_dir=paths["campaign"],
        base_config=paths["base"],
        output=paths["output"],
    )


def test_materialization_changes_num_envs_and_preserves_yaml_with_atomic_provenance(
    tmp_path: Path,
) -> None:
    paths = _calibration_fixture(tmp_path)

    decision = _materialize(paths)

    locked = yaml.safe_load(paths["output"].read_text(encoding="utf-8"))
    assert locked["training"]["num_envs"] == 2
    assert locked["training"]["total_timesteps"] == 20_000
    assert locked["project"] == {"name": "preserved_project", "seed": 42}
    assert locked["screening"] == {"algorithms": ["ppo", "a2c"]}
    assert decision["campaign_attempt_id"].endswith("deadbeef")
    assert decision["selection"]["previous_num_envs"] == 8
    assert decision["selection"]["selected_num_envs"] == 2
    assert decision["output"]["sha256"] == sha256_file(paths["output"])
    assert set(decision["sources"]) == {
        "base_config",
        "benchmark_csv",
        "benchmark_metadata",
        "benchmark_receipt",
        "latest_attempt",
    }
    decision_path = paths["output"].with_suffix(".decision.json")
    assert json.loads(decision_path.read_text(encoding="utf-8"))["status"] == "success"
    assert paths["output"].with_suffix(".yaml.sha256").read_text(encoding="utf-8").strip() == sha256_file(
        paths["output"]
    )
    assert decision_path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip() == sha256_file(
        decision_path
    )
    assert not list(paths["output"].parent.glob(".*.tmp"))


def test_schema_v2_native_attempt_materializes(tmp_path: Path) -> None:
    paths = _calibration_fixture(tmp_path)
    attempt = json.loads(paths["attempt"].read_text(encoding="utf-8"))
    attempt["schema_version"] = 2
    manifest = paths["campaign"] / "deployment" / "manifest.json"
    preflight = paths["project"] / "reports" / "preflight.json"
    _write_json(manifest, {"release_id": "d" * 64})
    _write_json(preflight, {"status": "pass"})
    attempt.update(
        {
            "deployment_manifest_path": str(manifest.relative_to(paths["project"])),
            "deployment_manifest_sha256": sha256_file(manifest),
            "deployment_release_id": "d" * 64,
            "preflight_receipt_path": str(preflight.relative_to(paths["project"])),
            "preflight_sha256": sha256_file(preflight),
        }
    )
    attempt["attempt_sha256"] = canonical_sha256(attempt)
    _write_json(paths["attempt"], attempt)
    _write_json(
        paths["campaign"] / "attempts" / f"{attempt['attempt_id']}.json",
        attempt,
    )

    decision = _materialize(paths)

    assert decision["evidence_mode"] == "native"
    assert decision["campaign_id"] == "test_campaign"
    assert yaml.safe_load(paths["output"].read_text(encoding="utf-8"))["training"][
        "num_envs"
    ] == 2


def test_materialization_resolves_inherited_base_configuration(tmp_path: Path) -> None:
    paths = _calibration_fixture(tmp_path)
    parent = paths["base"].with_name("screening_parent.yaml")
    paths["base"].replace(parent)
    paths["base"].write_text(
        yaml.safe_dump(
            {
                "extends": parent.name,
                "project": {"name": "inherited_v2_project"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    decision = _materialize(paths)
    locked = yaml.safe_load(paths["output"].read_text(encoding="utf-8"))

    assert decision["selection"]["previous_num_envs"] == 8
    assert locked["project"]["name"] == "inherited_v2_project"
    assert locked["project"]["seed"] == 42
    assert locked["training"]["total_timesteps"] == 20_000
    assert locked["training"]["num_envs"] == 2


def test_stale_attempt_bound_receipt_fails_without_writing_output(tmp_path: Path) -> None:
    paths = _calibration_fixture(tmp_path)
    attempt = json.loads(paths["attempt"].read_text(encoding="utf-8"))
    attempt["attempt_id"] = "vector_benchmark_20260715T010000Z_newest000"
    _write_json(paths["attempt"], attempt)

    with pytest.raises(ValueError, match="latest attempt"):
        _materialize(paths)

    assert not paths["output"].exists()
    assert not paths["output"].with_suffix(".decision.json").exists()


def test_mismatched_receipt_hash_fails_without_writing_output(tmp_path: Path) -> None:
    paths = _calibration_fixture(tmp_path)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    next(item for item in receipt["outputs"] if item["name"] == "benchmark_metadata")[
        "sha256"
    ] = "0" * 64
    _write_json(paths["receipt"], receipt)

    with pytest.raises(ValueError, match="receipt hash"):
        _materialize(paths)

    assert not paths["output"].exists()
    assert not paths["output"].with_suffix(".decision.json").exists()
