from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.campaigns.calibration import (
    code_provenance,
    load_observed_inputs,
    training_feature_subset,
)
from src.campaigns.gates import evaluate_stage_gate, sha256_file
from src.campaigns.model import CampaignDefinition


RUNTIME = {
    "python": "3.12.13",
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "gymnasium": "1.2.3",
    "stable_baselines3": "2.8.0",
    "torch": "2.10.0",
}
REMOTE_ROOT = Path("/lustrefs/project/test_campaign")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attempt_hash(value: dict[str, Any]) -> str:
    unhashed = dict(value)
    unhashed.pop("attempt_sha256", None)
    return hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_record(project: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(project)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _receipt_file_record(
    project: Path,
    *,
    name: str,
    path: Path,
) -> dict[str, Any]:
    relative = path.relative_to(project)
    return {
        "name": name,
        "path": str(REMOTE_ROOT / relative),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _campaign_stage(kind: str) -> dict[str, Any]:
    common = {
        "kind": "slurm",
        "job_script": f"slurm/{kind}.sbatch",
        "config_path": "config/screening.yaml",
        "fingerprint_paths": [
            "scripts/calibration_driver.py",
            "data/features.csv",
        ],
    }
    if kind == "vector_benchmark":
        return {
            "id": kind,
            "title": "Vector benchmark",
            **common,
            "requirements": [
                {
                    "name": "current vector receipt",
                    "pattern": "reports/**/*.receipt.json",
                    "validator": "calibration_receipt",
                    "options": {
                        "kind": "vectorized_training_benchmark",
                        "artifact_contract": "vector_benchmark",
                        "hostname_prefix": "bistka-",
                        "benchmark_options": {
                            "env_counts": [1, 2],
                            "repeats": 2,
                            "timesteps": 100,
                            "exact_completed": True,
                            "backend": "subproc",
                            "seed": 42,
                            "observed": True,
                            "data_source": "yahoo_ohlcv",
                            "training_split": "all",
                            "hostname_prefix": "bistka-",
                            "rss_measurement": "sampled_concurrent_process_tree_rss",
                            "distinct_processes": True,
                        },
                        "parameters": {
                            "env_counts": [1, 2],
                            "timesteps": 100,
                            "repeats": 2,
                            "backend": "subproc",
                            "seed": 42,
                            "synthetic": False,
                        },
                        "required_input_names": [
                            "config_file",
                            "resolved_config",
                            "feature_table",
                            "code:scripts/calibration_driver.py",
                        ],
                        "output_names": ["benchmark_csv", "benchmark_metadata"],
                        "result_sha256_fields": [
                            "data_sha256",
                            "config_sha256",
                            "code_sha256",
                        ],
                    },
                }
            ],
        }
    return {
        "id": "determinism",
        "title": "Determinism",
        **common,
        "requirements": [
            {
                "name": "current determinism receipt",
                "pattern": "reports/**/*.receipt.json",
                "validator": "calibration_receipt",
                "options": {
                    "kind": "fixed_seed_determinism",
                    "artifact_contract": "determinism",
                    "hostname_prefix": "bistka-",
                    "determinism_options": {
                        "seed": 42,
                        "observed": True,
                        "data_source": "yahoo_ohlcv",
                    },
                    "parameters": {"seed": 42, "synthetic": False},
                    "required_input_names": [
                        "config_file",
                        "resolved_config",
                        "feature_table",
                        "code:scripts/calibration_driver.py",
                    ],
                    "output_names": ["determinism_result"],
                    "result_sha256_fields": [
                        "trajectory_sha256",
                        "data_sha256",
                        "config_sha256",
                        "code_sha256",
                    ],
                    "result_values": {"deterministic": True},
                },
            }
        ],
    }


def _base_fixture(tmp_path: Path, kind: str) -> dict[str, Any]:
    project = tmp_path / "project"
    campaign_dir = project / "campaigns" / "test_campaign"
    for directory in (
        project / "src",
        project / "config",
        project / "data",
        project / "reports",
        project / "scripts",
        project / "slurm",
        campaign_dir / "attempts",
        campaign_dir / "deployment",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    features_path = project / "data" / "features.csv"
    pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "close": 100.0 + date_number + ticker_number,
                "signal": float(date_number - ticker_number),
            }
            for date_number, date in enumerate(("2024-01-02", "2024-01-03", "2024-01-04"))
            for ticker_number, ticker in enumerate(("A.BK", "B.BK"))
        ]
    ).to_csv(features_path, index=False)

    config_path = project / "config" / "screening.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"source": "yahoo_ohlcv"},
                "paths": {"features": "data/features.csv"},
                "features": {"columns": ["signal"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    code_path = project / "scripts" / "calibration_driver.py"
    code_path.write_text("CALIBRATION_PROTOCOL = 1\n", encoding="utf-8")
    job_path = project / "slurm" / f"{kind}.sbatch"
    job_path.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")

    mapping = {
        "campaign": {
            "id": "test_campaign",
            "title": "Calibration receipt gate fixture",
            "limits": {},
            "stages": [_campaign_stage(kind)],
        }
    }
    spec_path = project / "config" / "campaign.yaml"
    spec_path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    definition = CampaignDefinition.from_mapping(
        mapping,
        project_root=project,
        campaign_dir=campaign_dir,
        spec_path=spec_path,
    )
    stage = definition.stage_map[kind]

    deployment_path = campaign_dir / "deployment" / "bistka_deployment_manifest.json"
    release_id = "d" * 64
    _write_json(
        deployment_path,
        {
            "schema_version": 1,
            "release_id": release_id,
            "target_project_root": str(REMOTE_ROOT),
        },
    )
    preflight_path = project / "reports" / "bistka_calibration_preflight.json"
    _write_json(preflight_path, {"schema_version": 1, "status": "pass"})
    preflight_sha256 = sha256_file(preflight_path)

    attempt_id = f"{kind}_20260715T000000Z_deadbeef"
    attempt_inputs = [
        spec_path,
        job_path,
        config_path,
        code_path,
        features_path,
        deployment_path,
        preflight_path,
    ]
    attempt = {
        "schema_version": 1,
        "status": "rendered",
        "campaign_id": "test_campaign",
        "stage_id": kind,
        "attempt_id": attempt_id,
        "rendered_at_utc": "2026-07-15T00:00:00+00:00",
        "inputs": [_file_record(project, path) for path in sorted(attempt_inputs)],
        "deployment_release_id": release_id,
        "preflight_sha256": preflight_sha256,
    }
    attempt["attempt_sha256"] = _attempt_hash(attempt)
    immutable_attempt_path = campaign_dir / "attempts" / f"{attempt_id}.json"
    latest_attempt_path = campaign_dir / "attempts" / f"{kind}_latest.json"
    _write_json(immutable_attempt_path, attempt)
    _write_json(latest_attempt_path, attempt)

    inputs = load_observed_inputs(config_path)
    selected, split = training_feature_subset(inputs)
    code = code_provenance(project, ["scripts/calibration_driver.py"])
    current = {
        "source": inputs.data_source,
        "data_sha256": inputs.data_sha256,
        "config_sha256": inputs.config_sha256,
        "config_file_sha256": inputs.config_file_sha256,
        "code_sha256": code["sha256"],
        "code_files": code["files"],
        "feature_rows": len(selected),
        "feature_dates": selected["date"].nunique(),
        "feature_tickers": selected["ticker"].nunique(),
        "training_split": split,
    }
    receipt_inputs = [
        _receipt_file_record(project, name="config_file", path=config_path),
        {
            "name": "resolved_config",
            "path": str(REMOTE_ROOT / "config" / "screening.yaml"),
            "exists": True,
            "bytes": None,
            "sha256": current["config_sha256"],
        },
        _receipt_file_record(project, name="feature_table", path=features_path),
        _receipt_file_record(
            project,
            name="code:scripts/calibration_driver.py",
            path=code_path,
        ),
    ]
    return {
        "project": project,
        "campaign_dir": campaign_dir,
        "definition": definition,
        "stage": stage,
        "attempt": attempt,
        "latest_attempt_path": latest_attempt_path,
        "current": current,
        "receipt_inputs": receipt_inputs,
        "release_id": release_id,
        "preflight_sha256": preflight_sha256,
    }


def _base_receipt(paths: dict[str, Any], kind: str, job_id: str) -> dict[str, Any]:
    attempt = paths["attempt"]
    return {
        "schema_version": 1,
        "calibration_kind": kind,
        "status": "success",
        "campaign_attempt_id": attempt["attempt_id"],
        "campaign_attempt_sha256": attempt["attempt_sha256"],
        "campaign_stage_id": attempt["stage_id"],
        "deployment_release_id": paths["release_id"],
        "bistka_preflight_sha256": paths["preflight_sha256"],
        "slurm_job_id": job_id,
        "campaign_id": "test_campaign",
        "hostname": "bistka-compute-01",
        "process_id": 7000,
        "started_at_utc": "2026-07-15T00:00:00+00:00",
        "finished_at_utc": "2026-07-15T00:01:00+00:00",
        "elapsed_seconds": 60.0,
        "inputs": deepcopy(paths["receipt_inputs"]),
        "runtime": dict(RUNTIME),
    }


def _vector_fixture(tmp_path: Path) -> dict[str, Any]:
    paths = _base_fixture(tmp_path, "vector_benchmark")
    current = paths["current"]
    reports = paths["project"] / "reports"
    job_id = "12345"
    csv_path = reports / f"vectorized_training_benchmark_{job_id}.csv"
    fieldnames = [
        "num_envs",
        "repeat",
        "requested_timesteps",
        "completed_timesteps",
        "samples_per_second",
        "peak_rss_mib",
        "elapsed_seconds",
        "backend",
        "seed",
        "data_source",
        "data_sha256",
        "config_sha256",
        "config_file_sha256",
        "code_sha256",
        "feature_rows",
        "feature_dates",
        "feature_tickers",
        "process_id",
        "hostname",
        "training_split",
        "rss_measurement",
        "peak_self_rss_mib",
        "peak_child_rss_mib",
        "python_version",
        "numpy_version",
        "pandas_version",
        "gymnasium_version",
        "stable_baselines3_version",
        "torch_version",
    ]
    rows: list[dict[str, Any]] = []
    for num_envs, throughput in ((1, 94.0), (2, 100.0)):
        for repeat in (1, 2):
            row = {
                "num_envs": num_envs,
                "repeat": repeat,
                "requested_timesteps": 100,
                "completed_timesteps": 100,
                "samples_per_second": throughput,
                "peak_rss_mib": 100.0 + num_envs,
                "elapsed_seconds": 100.0 / throughput,
                "backend": "monitor" if num_envs == 1 else "subproc",
                "seed": 42,
                "data_source": current["source"],
                "data_sha256": current["data_sha256"],
                "config_sha256": current["config_sha256"],
                "config_file_sha256": current["config_file_sha256"],
                "code_sha256": current["code_sha256"],
                "feature_rows": current["feature_rows"],
                "feature_dates": current["feature_dates"],
                "feature_tickers": current["feature_tickers"],
                "process_id": 7100 + num_envs * 10 + repeat,
                "hostname": "bistka-compute-01",
                "training_split": current["training_split"],
                "rss_measurement": "sampled_concurrent_process_tree_rss",
                "peak_self_rss_mib": 75.0,
                "peak_child_rss_mib": 25.0 + num_envs,
                "python_version": RUNTIME["python"],
                "numpy_version": RUNTIME["numpy"],
                "pandas_version": RUNTIME["pandas"],
                "gymnasium_version": RUNTIME["gymnasium"],
                "stable_baselines3_version": RUNTIME["stable_baselines3"],
                "torch_version": RUNTIME["torch"],
            }
            rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = csv_path.with_suffix(".json")
    metadata = {
        "schema_version": 2,
        "selected_num_envs": 2,
        "selection_rule": "smallest environment count at or above 95% of maximum median throughput",
        "selection_fraction_of_max": 0.95,
        "throughput_statistic": "median_samples_per_second",
        "repeats_per_env_count": 2,
        "trial_count": 4,
        "trial_isolation": "one fresh Python subprocess per environment count and repeat",
        "results_file": str(REMOTE_ROOT / "reports" / csv_path.name),
        "data_provenance": {
            "source": current["source"],
            "synthetic": False,
            "data_sha256": current["data_sha256"],
            "config_sha256": current["config_sha256"],
            "config_file_sha256": current["config_file_sha256"],
        },
        "code_provenance": {
            "sha256": current["code_sha256"],
            "files": current["code_files"],
        },
        "runtime": dict(RUNTIME),
        "summary_by_env_count": [
            {
                "num_envs": num_envs,
                "repeats": 2,
                "median_samples_per_second": throughput,
                "min_samples_per_second": throughput,
                "max_samples_per_second": throughput,
                "max_peak_rss_mib": 100.0 + num_envs,
            }
            for num_envs, throughput in ((1, 94.0), (2, 100.0))
        ],
    }
    _write_json(metadata_path, metadata)

    receipt_path = reports / f"{csv_path.name}.receipt.json"
    receipt = _base_receipt(paths, "vectorized_training_benchmark", job_id)
    receipt.update(
        {
            "parameters": {
                "env_counts": [1, 2],
                "timesteps": 100,
                "repeats": 2,
                "backend": "subproc",
                "seed": 42,
                "synthetic": False,
            },
            "outputs": [
                _receipt_file_record(paths["project"], name="benchmark_csv", path=csv_path),
                _receipt_file_record(
                    paths["project"], name="benchmark_metadata", path=metadata_path
                ),
            ],
            "result": {
                "selected_num_envs": 2,
                "trial_count": 4,
                "data_sha256": current["data_sha256"],
                "config_sha256": current["config_sha256"],
                "code_sha256": current["code_sha256"],
            },
        }
    )
    _write_json(receipt_path, receipt)
    paths.update(
        {
            "csv_path": csv_path,
            "metadata_path": metadata_path,
            "metadata": metadata,
            "receipt_path": receipt_path,
            "receipt": receipt,
        }
    )
    return paths


def _determinism_fixture(tmp_path: Path) -> dict[str, Any]:
    paths = _base_fixture(tmp_path, "determinism")
    current = paths["current"]
    reports = paths["project"] / "reports"
    job_id = "67890"
    trajectory_sha256 = "e" * 64
    process_ids = [8101, 8102]
    provenance = {
        "source": current["source"],
        "synthetic": False,
        "data_sha256": current["data_sha256"],
        "config_sha256": current["config_sha256"],
        "config_file_sha256": current["config_file_sha256"],
        "code_sha256": current["code_sha256"],
        "feature_rows": current["feature_rows"],
        "feature_dates": current["feature_dates"],
        "feature_tickers": current["feature_tickers"],
        "training_split": current["training_split"],
    }
    result_path = reports / f"determinism_{job_id}.json"
    result = {
        "schema_version": 2,
        "deterministic": True,
        "seed": 42,
        "steps": 7,
        "first_steps": 7,
        "second_steps": 7,
        "first_sha256": trajectory_sha256,
        "second_sha256": trajectory_sha256,
        "comparison_method": "two_independent_subprocess_random_action_trajectories",
        "random_action_distribution": "numpy.default_rng(seed).normal",
        "independent_subprocess_runs": 2,
        "process_ids": process_ids,
        "provenance_matches": True,
        "runtime_matches": True,
        "data_provenance": provenance,
        "runtime": dict(RUNTIME),
        "run_provenance": [
            {
                "trajectory_sha256": trajectory_sha256,
                "steps": 7,
                "seed": 42,
                "process_id": process_id,
                "hostname": "bistka-compute-01",
                "data_sha256": current["data_sha256"],
                "config_sha256": current["config_sha256"],
                "config_file_sha256": current["config_file_sha256"],
                "code_sha256": current["code_sha256"],
            }
            for process_id in process_ids
        ],
    }
    _write_json(result_path, result)
    receipt_path = reports / f"{result_path.name}.receipt.json"
    receipt = _base_receipt(paths, "fixed_seed_determinism", job_id)
    receipt.update(
        {
            "parameters": {"seed": 42, "synthetic": False},
            "outputs": [
                _receipt_file_record(
                    paths["project"], name="determinism_result", path=result_path
                )
            ],
            "result": {
                "deterministic": True,
                "trajectory_sha256": trajectory_sha256,
                "data_sha256": current["data_sha256"],
                "config_sha256": current["config_sha256"],
                "code_sha256": current["code_sha256"],
            },
        }
    )
    _write_json(receipt_path, receipt)
    paths.update({"result_path": result_path, "receipt_path": receipt_path, "receipt": receipt})
    return paths


def _issues(paths: dict[str, Any]) -> str:
    gate = evaluate_stage_gate(paths["definition"], paths["stage"])
    return "\n".join(gate.issues)


def _enable_migrated_evidence_mode(paths: dict[str, Any], kind: str) -> None:
    mapping = yaml.safe_load(
        (paths["project"] / "config" / "campaign.yaml").read_text(encoding="utf-8")
    )
    requirement = mapping["campaign"]["stages"][0]["requirements"][0]
    requirement["validator"] = "calibration_evidence"
    requirement["patterns"] = [
        "reports/**/*.receipt.json",
        "{campaign_dir}/imports/v1_calibration_evidence.json",
    ]
    requirement.pop("pattern", None)
    definition = CampaignDefinition.from_mapping(
        mapping,
        project_root=paths["project"],
        campaign_dir=paths["campaign_dir"],
        spec_path=paths["project"] / "config" / "campaign.yaml",
    )
    paths["definition"] = definition
    paths["stage"] = definition.stage_map[kind]


def test_exact_current_vector_artifacts_and_receipt_pass(tmp_path: Path) -> None:
    paths = _vector_fixture(tmp_path)

    gate = evaluate_stage_gate(paths["definition"], paths["stage"])

    assert gate.passed, gate.issues


def test_exact_current_determinism_result_and_receipt_pass(tmp_path: Path) -> None:
    paths = _determinism_fixture(tmp_path)

    gate = evaluate_stage_gate(paths["definition"], paths["stage"])

    assert gate.passed, gate.issues


def test_schema_v2_attempt_uses_its_declared_manifest_and_preflight_paths(
    tmp_path: Path,
) -> None:
    paths = _vector_fixture(tmp_path)
    project = paths["project"]
    campaign_dir = paths["campaign_dir"]
    old_manifest = campaign_dir / "deployment" / "bistka_deployment_manifest.json"
    old_preflight = project / "reports" / "bistka_calibration_preflight.json"
    manifest = campaign_dir / "deployment" / "bistka_deployment_v2_manifest.json"
    preflight = project / "reports" / "bistka_calibration_v2_preflight.json"
    old_manifest.replace(manifest)
    old_preflight.replace(preflight)

    attempt = deepcopy(paths["attempt"])
    attempt["schema_version"] = 2
    attempt["deployment_manifest_path"] = str(manifest.relative_to(project))
    attempt["preflight_receipt_path"] = str(preflight.relative_to(project))
    attempt["inputs"] = [
        _file_record(
            project,
            (
                manifest
                if record["path"] == str(old_manifest.relative_to(project))
                else preflight
                if record["path"] == str(old_preflight.relative_to(project))
                else project / record["path"]
            ),
        )
        for record in attempt["inputs"]
    ]
    attempt["preflight_sha256"] = sha256_file(preflight)
    attempt["attempt_sha256"] = _attempt_hash(attempt)
    immutable = campaign_dir / "attempts" / f"{attempt['attempt_id']}.json"
    _write_json(immutable, attempt)
    _write_json(paths["latest_attempt_path"], attempt)

    receipt = deepcopy(paths["receipt"])
    receipt["campaign_attempt_sha256"] = attempt["attempt_sha256"]
    receipt["bistka_preflight_sha256"] = attempt["preflight_sha256"]
    _write_json(paths["receipt_path"], receipt)

    gate = evaluate_stage_gate(paths["definition"], paths["stage"])

    assert gate.passed, gate.issues
    assert str(manifest.relative_to(project)) in gate.evidence_files
    assert str(preflight.relative_to(project)) in gate.evidence_files


def test_native_and_imported_calibration_evidence_is_ambiguous(
    tmp_path: Path,
) -> None:
    paths = _vector_fixture(tmp_path)
    _enable_migrated_evidence_mode(paths, "vector_benchmark")
    _write_json(
        paths["campaign_dir"] / "imports" / "v1_calibration_evidence.json",
        {"schema_version": 1},
    )

    issues = _issues(paths)

    assert "choose exactly one evidence mode" in issues


def test_receipt_for_previous_attempt_cannot_satisfy_new_latest_attempt(tmp_path: Path) -> None:
    paths = _vector_fixture(tmp_path)
    latest = deepcopy(paths["attempt"])
    latest["attempt_id"] = "vector_benchmark_20260715T010000Z_newest00"
    latest["rendered_at_utc"] = "2026-07-15T01:00:00+00:00"
    latest["attempt_sha256"] = _attempt_hash(latest)
    _write_json(paths["latest_attempt_path"], latest)
    _write_json(
        paths["campaign_dir"] / "attempts" / f"{latest['attempt_id']}.json",
        latest,
    )

    issues = _issues(paths)

    assert "expected exactly one receipt for the latest rendered calibration attempt, found 0" in issues


def test_mismatched_returned_output_hash_fails_closed(tmp_path: Path) -> None:
    paths = _vector_fixture(tmp_path)
    receipt = deepcopy(paths["receipt"])
    next(
        item for item in receipt["outputs"] if item["name"] == "benchmark_metadata"
    )["sha256"] = "0" * 64
    _write_json(paths["receipt_path"], receipt)

    issues = _issues(paths)

    assert "returned calibration output hash or size does not match" in issues


def test_metadata_selected_count_must_follow_measured_rule(tmp_path: Path) -> None:
    paths = _vector_fixture(tmp_path)
    metadata = deepcopy(paths["metadata"])
    metadata["selected_num_envs"] = 1
    _write_json(paths["metadata_path"], metadata)
    receipt = deepcopy(paths["receipt"])
    metadata_record = next(
        item for item in receipt["outputs"] if item["name"] == "benchmark_metadata"
    )
    metadata_record["sha256"] = sha256_file(paths["metadata_path"])
    metadata_record["bytes"] = paths["metadata_path"].stat().st_size
    receipt["result"]["selected_num_envs"] = 1
    _write_json(paths["receipt_path"], receipt)

    issues = _issues(paths)

    assert "selected_num_envs does not follow the declared 95%-of-peak rule" in issues


def test_receipt_must_match_deployment_release_and_preflight(tmp_path: Path) -> None:
    paths = _determinism_fixture(tmp_path)
    receipt = deepcopy(paths["receipt"])
    receipt["deployment_release_id"] = "1" * 64
    receipt["bistka_preflight_sha256"] = "2" * 64
    _write_json(paths["receipt_path"], receipt)

    issues = _issues(paths)

    assert "calibration receipt deployment release id does not match" in issues
    assert "calibration receipt BistKA preflight hash does not match" in issues


def test_broad_glob_other_receipt_does_not_count_as_current_attempt(tmp_path: Path) -> None:
    paths = _determinism_fixture(tmp_path)
    stale = deepcopy(paths["receipt"])
    stale["campaign_attempt_id"] = "determinism_20260714T230000Z_stale000"
    other_path = paths["project"] / "reports" / "archive" / "unrelated.receipt.json"
    _write_json(other_path, stale)
    paths["receipt_path"].unlink()

    issues = _issues(paths)

    assert "expected exactly one receipt for the latest rendered calibration attempt, found 0" in issues
