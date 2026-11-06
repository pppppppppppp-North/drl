from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.campaigns.control import _render_command, build_plan, load_definition
from src.campaigns.decisions import create_protocol_lock, materialize_decisions
from src.campaigns.gates import evaluate_campaign, sha256_file, verify_stage
from src.campaigns.model import CampaignConfigurationError, CampaignDefinition, StageState
from src.campaigns.models import build_locked_stress_config
from src.campaigns.slurm import render_stage_command, validate_all_slurm_stages
from src.experiments.experiment_manifest import write_experiment_manifest
from src.data.market_data_readiness import evaluate_market_data, write_readiness_outputs
from src.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _definition(tmp_path: Path, campaign: dict) -> CampaignDefinition:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True, parents=True)
    spec = project / "config" / "campaign.yaml"
    spec.parent.mkdir(exist_ok=True)
    mapping = {"campaign": campaign}
    spec.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return CampaignDefinition.from_mapping(
        mapping,
        project_root=project,
        campaign_dir=project / "campaigns" / campaign.get("id", "test_campaign"),
        spec_path=spec,
    )


def test_default_plan_has_exact_workload_and_truthful_local_blockers(tmp_path) -> None:
    campaign_dir = PROJECT_ROOT / "campaigns" / f"pytest_{tmp_path.name}"
    try:
        _, config = load_definition(
            PROJECT_ROOT / "config" / "research_campaign.yaml",
            project_root=PROJECT_ROOT,
            campaign_dir=campaign_dir,
        )
        config = copy.deepcopy(config)
        capacity_stage = next(
            stage for stage in config["campaign"]["stages"] if stage["id"] == "capacity"
        )
        capacity_stage["requirements"][0]["pattern"] = (
            "{campaign_dir}/fixtures/bistka_capacity_*.txt"
        )
        definition = CampaignDefinition.from_mapping(
            config,
            project_root=PROJECT_ROOT,
            campaign_dir=campaign_dir,
            spec_path=PROJECT_ROOT / "config" / "research_campaign.yaml",
        )
        plan = build_plan(definition, config)
        statuses = {status.stage_id: status for status in evaluate_campaign(definition)}

        assert plan["workload"] == {
            "planned_model_runs": 503,
            "planned_training_timesteps": 29_450_240,
            "array_elements": 23,
        }
        assert plan["safety"]["render_only"] is True
        assert statuses["capacity"].state == StageState.READY
        assert statuses["screening"].state == StageState.PENDING
    finally:
        shutil.rmtree(campaign_dir, ignore_errors=True)

    missing_definition = _definition(
        tmp_path / "missing_data",
        {
            "id": "missing_data",
            "limits": {},
            "stages": [
                {
                    "id": "data_readiness",
                    "kind": "manual",
                    "missing_state": "BLOCKED",
                    "requirements": [
                        {"name": "real OHLCV prices", "pattern": "data/raw/prices_real_ohlcv.csv"}
                    ],
                }
            ],
        },
    )
    missing_status = evaluate_campaign(missing_definition)[0]
    assert missing_status.state == StageState.BLOCKED
    assert "real OHLCV prices" in missing_status.gate.issues[0]


def test_observed_market_receipt_rejects_generated_origin_and_detects_data_drift(tmp_path) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw" / "prices.csv"
    index = project / "data" / "raw" / "index.csv"
    features_path = project / "data" / "processed" / "features.csv"
    manifest = project / "data" / "data_manifest.csv"
    for path in (raw, index, features_path, manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2024-01-01", periods=8)
    stock_rows = []
    feature_rows = []
    for date_number, date in enumerate(dates):
        for ticker_number, ticker in enumerate(("A.BK", "B.BK")):
            close = 100.0 + date_number + ticker_number
            stock_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000,
                }
            )
            feature_rows.append({"date": date, "ticker": ticker, "return_1d": 0.0})
    pd.DataFrame(stock_rows).to_csv(raw, index=False)
    pd.DataFrame(
        {
            "date": dates,
            "ticker": ["^SET.BK"] * len(dates),
            "open": np.arange(len(dates)) + 1000.0,
            "high": np.arange(len(dates)) + 1001.0,
            "low": np.arange(len(dates)) + 999.0,
            "close": np.arange(len(dates)) + 1000.0,
            "volume": [1000] * len(dates),
            "index_name": ["set"] * len(dates),
        }
    ).to_csv(index, index=False)
    pd.DataFrame(feature_rows).to_csv(features_path, index=False)
    pd.DataFrame(
        [
            {"source": "yahoo_thai_ohlcv"},
            {"source": "yahoo_thai_market_indices"},
        ]
    ).to_csv(manifest, index=False)
    config = {
        "paths": {
            "raw_prices": "data/raw/prices.csv",
            "features": "data/processed/features.parquet",
            "fallback_features_csv": "data/processed/features.csv",
        },
        "data": {
            "source": "yahoo_ohlcv",
            "tickers": ["A.BK", "B.BK"],
            "start": "2024-01-01",
            "end": "2024-01-12",
        },
        "market_indices": {
            "raw_prices": "data/raw/index.csv",
            "symbols": {"set": "^SET.BK"},
        },
        "features": {"columns": ["return_1d"]},
    }
    receipt, checks, coverage, missingness = evaluate_market_data(config, project_root=project)
    receipt_path = write_readiness_outputs(
        project / "reports" / "data_quality_market",
        config_path="config/test.yaml",
        receipt=receipt,
        checks=checks,
        coverage=coverage,
        missingness=missingness,
    )
    campaign = {
        "id": "observed_data",
        "limits": {},
        "stages": [
            {
                "id": "data_readiness",
                "kind": "manual",
                "missing_state": "BLOCKED",
                "requirements": [
                    {
                        "name": "receipt",
                        "pattern": "reports/data_quality_market/market_data_readiness.json",
                        "validator": "market_data_readiness",
                    }
                ],
            }
        ],
    }
    spec = project / "config" / "campaign.yaml"
    spec.parent.mkdir(parents=True)
    mapping = {"campaign": campaign}
    spec.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    definition = CampaignDefinition.from_mapping(
        mapping,
        project_root=project,
        campaign_dir=project / "campaigns" / "observed_data",
        spec_path=spec,
    )
    assert evaluate_campaign(definition)[0].state == StageState.VERIFY_PENDING

    observed_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    observed_receipt["data_origin"] = "generated"
    receipt_path.write_text(json.dumps(observed_receipt), encoding="utf-8")
    receipt_path.with_suffix(".sha256").write_text(sha256_file(receipt_path) + "\n", encoding="utf-8")
    generated_status = evaluate_campaign(definition)[0]
    assert generated_status.state == StageState.BLOCKED
    assert "generated or unknown-origin" in " ".join(generated_status.gate.issues)

    receipt, checks, coverage, missingness = evaluate_market_data(config, project_root=project)
    write_readiness_outputs(
        receipt_path.parent,
        config_path="config/test.yaml",
        receipt=receipt,
        checks=checks,
        coverage=coverage,
        missingness=missingness,
    )
    verify_stage(definition, "data_readiness")
    raw.write_text(raw.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert evaluate_campaign(definition)[0].state == StageState.STALE


def test_default_slurm_shapes_match_campaign_grid() -> None:
    definition, _ = load_definition(PROJECT_ROOT / "config" / "research_campaign.yaml")
    resources = validate_all_slurm_stages(definition)

    assert resources["screening"]["array_tasks"] == 5
    assert resources["screening"]["cpus_per_task"] == 2
    assert resources["screening"]["memory"] == "12GB"
    assert resources["screening"]["array_concurrency"] == 2
    assert resources["budget_sweep"]["array_tasks"] == 4
    assert resources["nested_screening"]["array_tasks"] == 5
    assert resources["robustness"]["array_tasks"] == 4
    assert resources["feature_ablation"]["array_tasks"] == 5
    assert all(value["array_concurrency"] <= 2 for value in resources.values())
    screening_script = (
        PROJECT_ROOT / "slurm" / "large_scale_screening.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --requeue" in screening_script
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-results/${RUN_PREFIX}_${SLURM_ARRAY_JOB_ID}}"' in (
        screening_script
    )


def test_slurm_validation_rejects_declared_array_mismatch(tmp_path) -> None:
    config = load_config(PROJECT_ROOT / "config" / "research_campaign.yaml")
    config = copy.deepcopy(config)
    screening = next(item for item in config["campaign"]["stages"] if item["id"] == "screening")
    screening["array_tasks"] = 4
    definition = CampaignDefinition.from_mapping(
        config,
        project_root=PROJECT_ROOT,
        campaign_dir=PROJECT_ROOT / "campaigns" / f"pytest_array_{tmp_path.name}",
        spec_path=PROJECT_ROOT / "config" / "research_campaign.yaml",
    )
    with pytest.raises(CampaignConfigurationError, match="declared array_tasks=4"):
        validate_all_slurm_stages(definition)


def test_definition_rejects_unknown_dependency_duplicate_stage_and_cycle(tmp_path) -> None:
    base = {
        "id": "test_campaign",
        "limits": {},
        "stages": [
            {"id": "first", "kind": "manual", "requirements": [{"name": "x", "pattern": "x"}]},
            {"id": "second", "kind": "manual", "depends_on": ["first"], "requirements": [{"name": "y", "pattern": "y"}]},
        ],
    }
    unknown = copy.deepcopy(base)
    unknown["stages"][1]["depends_on"] = ["missing"]
    with pytest.raises(CampaignConfigurationError, match="unknown dependencies"):
        _definition(tmp_path / "unknown", unknown)

    duplicate = copy.deepcopy(base)
    duplicate["stages"][1]["id"] = "first"
    with pytest.raises(CampaignConfigurationError, match="must be unique"):
        _definition(tmp_path / "duplicate", duplicate)

    cycle = copy.deepcopy(base)
    cycle["stages"][0]["depends_on"] = ["second"]
    with pytest.raises(CampaignConfigurationError, match="contains a cycle"):
        _definition(tmp_path / "cycle", cycle)


def test_success_marker_requires_explicit_verification_and_becomes_stale(tmp_path) -> None:
    campaign = {
        "id": "test_campaign",
        "limits": {},
        "stages": [
            {
                "id": "input_gate",
                "kind": "manual",
                "missing_state": "BLOCKED",
                "requirements": [{"name": "input", "pattern": "data/input.csv"}],
            },
            {
                "id": "downstream",
                "kind": "manual",
                "depends_on": ["input_gate"],
                "requirements": [{"name": "output", "pattern": "results/output.json"}],
            },
        ],
    }
    definition = _definition(tmp_path, campaign)
    input_path = definition.project_root / "data" / "input.csv"
    input_path.parent.mkdir()
    input_path.write_text("value\n1\n", encoding="utf-8")

    states = {item.stage_id: item.state for item in evaluate_campaign(definition)}
    assert states == {"input_gate": StageState.VERIFY_PENDING, "downstream": StageState.PENDING}
    verify_stage(definition, "input_gate")
    states = {item.stage_id: item.state for item in evaluate_campaign(definition)}
    assert states["input_gate"] == StageState.VERIFIED
    assert states["downstream"] == StageState.READY

    output_path = definition.project_root / "results" / "output.json"
    output_path.parent.mkdir()
    output_path.write_text("{}\n", encoding="utf-8")
    verify_stage(definition, "downstream")
    input_path.write_text("value\n2\n", encoding="utf-8")
    states = {item.stage_id: item.state for item in evaluate_campaign(definition)}
    assert states["input_gate"] == StageState.STALE
    assert states["downstream"] == StageState.STALE


def test_exact_manifest_gate_rejects_missing_duplicate_and_failed_runs(tmp_path) -> None:
    campaign = {
        "id": "test_campaign",
        "limits": {},
        "stages": [
            {
                "id": "compute",
                "kind": "manual",
                "run_glob": "results/run_*",
                "manifest_pattern": "**/experiment_manifest.json",
                "expected_manifests": 2,
            }
        ],
    }
    definition = _definition(tmp_path, campaign)
    run = definition.project_root / "results" / "run_1"
    for index in range(2):
        path = run / f"unit_{index}" / "experiment_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"experiment_id": f"run_{index}", "status": "complete"}), encoding="utf-8"
        )
    assert evaluate_campaign(definition)[0].state == StageState.VERIFY_PENDING
    verify_stage(definition, "compute")
    assert evaluate_campaign(definition)[0].state == StageState.VERIFIED

    second = run / "unit_1" / "experiment_manifest.json"
    second.write_text(json.dumps({"experiment_id": "run_0", "status": "complete"}), encoding="utf-8")
    status = evaluate_campaign(definition)[0]
    assert status.state == StageState.STALE
    assert "duplicate experiment ids" in " ".join(status.gate.issues)

    second.write_text(json.dumps({"experiment_id": "run_1", "status": "failed"}), encoding="utf-8")
    status = evaluate_campaign(definition)[0]
    assert status.state == StageState.STALE
    assert status.gate.manifest_failed == 1


def test_capacity_benchmark_and_determinism_validators_reject_weak_evidence(tmp_path) -> None:
    campaign = {
        "id": "test_campaign",
        "limits": {},
        "stages": [
            {
                "id": "capacity",
                "kind": "manual",
                "requirements": [
                    {"name": "capacity", "pattern": "reports/capacity.txt", "validator": "capacity_snapshot"}
                ],
            },
            {
                "id": "benchmark",
                "kind": "manual",
                "requirements": [
                    {
                        "name": "benchmark",
                        "pattern": "reports/benchmark.csv",
                        "validator": "vector_benchmark",
                        "options": {"env_counts": [1, 2]},
                    }
                ],
            },
            {
                "id": "determinism",
                "kind": "manual",
                "requirements": [
                    {"name": "determinism", "pattern": "reports/determinism.json", "validator": "determinism"}
                ],
            },
        ],
    }
    definition = _definition(tmp_path, campaign)
    reports = definition.project_root / "reports"
    reports.mkdir()
    (reports / "capacity.txt").write_text(
        "# snapshot\n## mycredit \nunavailable\n## myquota \nok\n## sinfo \nok\n## squeue -u user\nok\n",
        encoding="utf-8",
    )
    (reports / "benchmark.csv").write_text(
        "num_envs,requested_timesteps,completed_timesteps,samples_per_second,peak_rss_mib\n"
        "1,64,64,10,100\n",
        encoding="utf-8",
    )
    (reports / "determinism.json").write_text(
        json.dumps(
            {
                "deterministic": False,
                "first_sha256": "a" * 64,
                "second_sha256": "b" * 64,
                "steps": 10,
            }
        ),
        encoding="utf-8",
    )
    states = {item.stage_id: item for item in evaluate_campaign(definition)}
    assert all(item.state == StageState.BLOCKED for item in states.values())

    (reports / "capacity.txt").write_text(
        "# snapshot\n## mycredit \n100\n## myquota \n10GB\n## sinfo \ncompute-devel\n## squeue -u user\nempty\n",
        encoding="utf-8",
    )
    (reports / "benchmark.csv").write_text(
        "num_envs,requested_timesteps,completed_timesteps,samples_per_second,peak_rss_mib\n"
        "1,64,64,10,100\n2,64,64,15,120\n",
        encoding="utf-8",
    )
    digest = "c" * 64
    (reports / "determinism.json").write_text(
        json.dumps(
            {
                "deterministic": True,
                "first_sha256": digest,
                "second_sha256": digest,
                "steps": 10,
            }
        ),
        encoding="utf-8",
    )
    assert all(item.state == StageState.VERIFY_PENDING for item in evaluate_campaign(definition))


def test_capacity_gate_requires_declared_live_policy_evidence(tmp_path) -> None:
    definition = _definition(
        tmp_path,
        {
            "id": "policy_gate",
            "limits": {},
            "stages": [
                {
                    "id": "capacity",
                    "kind": "manual",
                    "requirements": [
                        {
                            "name": "live capacity and policy",
                            "pattern": "reports/capacity.txt",
                            "validator": "capacity_snapshot",
                            "options": {
                                "required_fragments": [
                                    "Maximum Job Submission: 5 jobs/group",
                                    "Miniforge3/25.3.0-3",
                                ]
                            },
                        }
                    ],
                }
            ],
        },
    )
    snapshot = definition.project_root / "reports/capacity.txt"
    snapshot.parent.mkdir()
    base = (
        "# snapshot\n## mycredit\n100\n## myquota\n10GB\n"
        "## sinfo\ncompute-devel\n## squeue -u user\nempty\n"
    )
    snapshot.write_text(base, encoding="utf-8")
    assert evaluate_campaign(definition)[0].state == StageState.BLOCKED

    snapshot.write_text(
        base + "Maximum Job Submission: 5 jobs/group\nMiniforge3/25.3.0-3\n",
        encoding="utf-8",
    )
    assert evaluate_campaign(definition)[0].state == StageState.VERIFY_PENDING


def test_post_lock_manifest_must_carry_exact_protocol_hash(tmp_path) -> None:
    project = tmp_path / "project"
    locked_input = project / "config" / "science.yaml"
    locked_input.parent.mkdir(parents=True)
    locked_input.write_text("seed: 42\n", encoding="utf-8")
    campaign = {
        "id": "test_campaign",
        "limits": {},
        "protocol_lock": {
            "path": "protocol_lock.json",
            "inputs": [{"name": "science", "pattern": "config/science.yaml"}],
        },
        "stages": [
            {
                "id": "test_stage",
                "kind": "manual",
                "run_glob": "results/run_*",
                "manifest_pattern": "experiment_manifest.json",
                "expected_manifests": 1,
                "require_protocol_hash": True,
            }
        ],
    }
    spec = project / "config" / "campaign.yaml"
    mapping = {"campaign": campaign}
    spec.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    definition = CampaignDefinition.from_mapping(
        mapping, project_root=project, campaign_dir=project / "campaigns" / "test_campaign", spec_path=spec
    )
    create_protocol_lock(definition)
    manifest_path = project / "results" / "run_1" / "experiment_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"experiment_id": "one", "status": "complete"}), encoding="utf-8")
    assert evaluate_campaign(definition)[0].state == StageState.BLOCKED

    lock_hash = definition.protocol_lock_path.with_suffix(".sha256").read_text().strip()
    manifest_path.write_text(
        json.dumps({"experiment_id": "one", "status": "complete", "protocol_lock_hash": lock_hash}),
        encoding="utf-8",
    )
    assert evaluate_campaign(definition)[0].state == StageState.VERIFY_PENDING


def test_rendered_command_is_one_stage_only_and_never_executes_sbatch(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    script = project / "slurm" / "job.sbatch"
    script.parent.mkdir(parents=True)
    script.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH --partition=compute-devel",
                "#SBATCH --account=25sfcs03",
                "#SBATCH --cpus-per-task=2",
                "#SBATCH --array=0-1%2",
                "#SBATCH --output=logs/job_%A_%a.out",
                "#SBATCH --error=logs/job_%A_%a.err",
            ]
        ),
        encoding="utf-8",
    )
    campaign = {
        "id": "test_campaign",
        "limits": {
            "account": "25sfcs03",
            "allowed_partitions": ["compute-devel"],
            "max_array_tasks": 5,
            "max_concurrent_tasks": 2,
            "max_cpus_per_task": 32,
            "max_rendered_submissions": 1,
        },
        "stages": [
            {
                "id": "compute",
                "kind": "slurm",
                "job_script": "slurm/job.sbatch",
                "array_tasks": 2,
                "run_dir_template": "results/run_{job_id}",
                "requirements": [{"name": "result", "pattern": "results/result.csv"}],
            }
        ],
    }
    spec = project / "config" / "campaign.yaml"
    spec.parent.mkdir()
    mapping = {"campaign": campaign}
    spec.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    definition = CampaignDefinition.from_mapping(
        mapping, project_root=project, campaign_dir=project / "campaigns" / "test_campaign", spec_path=spec
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("rendering executed a process"))

    command = render_stage_command(definition, definition.stages[0])

    assert command.count("sbatch ") == 1
    assert "--parsable" in command
    assert '${COMPUTE_SUBMISSION%%;*}' in command
    assert "OMP_NUM_THREADS=1" in command
    assert "--export=ALL" not in command
    assert "PATH=/usr/local/bin:/usr/bin:/bin" in command


def test_decision_materialization_propagates_budget_and_per_window_parameters(tmp_path) -> None:
    project = tmp_path / "project"
    for name, value in {
        "budget.yaml": {"budget_sweep": {"algorithms": ["ppo"], "budgets": [20, 100]}},
        "nested.yaml": {"nested_screening": {"algorithms": ["ppo", "a2c"], "window_ids": [0, 1]}, "training": {}},
        "robustness.yaml": {"robustness": {"algorithms": ["ppo"], "window_ids": [0, 1]}, "training": {}},
        "ablation.yaml": {"feature_ablation": {"algorithm": "ppo", "window_ids": [0, 1]}},
    }.items():
        path = project / "config" / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
    screening = project / "results" / "screen" / "aggregate" / "selection.json"
    budget = project / "results" / "budget" / "aggregate" / "budget.json"
    nested = project / "results" / "nested" / "aggregate" / "nested.csv"
    screening.parent.mkdir(parents=True)
    budget.parent.mkdir(parents=True)
    nested.parent.mkdir(parents=True)
    screening.write_text(
        json.dumps({"selection_split": "validation", "selected_algorithms": ["ppo", "a2c"]}), encoding="utf-8"
    )
    budget.write_text(
        json.dumps({"selection_split": "validation", "selected_training_budget": 100}), encoding="utf-8"
    )
    nested.write_text(
        "algorithm,window,param_learning_rate,param_gamma\n"
        "ppo,0,0.001,0.99\nppo,1,0.002,0.98\n"
        "a2c,0,0.003,0.97\na2c,1,0.004,0.96\n",
        encoding="utf-8",
    )
    campaign = {
        "id": "test_campaign",
        "limits": {},
        "decisions": {
            "window_ids": [0, 1],
            "sources": {
                "screening_selection": "results/screen/aggregate/selection.json",
                "budget_selection": "results/budget/aggregate/budget.json",
                "nested_selection": "results/nested/aggregate/nested.csv",
            },
            "base_configs": {
                "budget": "config/budget.yaml",
                "nested": "config/nested.yaml",
                "robustness": "config/robustness.yaml",
                "ablation": "config/ablation.yaml",
            },
            "outputs": {
                "budget": "{campaign_dir}/derived/budget.yaml",
                "nested": "{campaign_dir}/derived/nested.yaml",
                "robustness": "{campaign_dir}/derived/robustness.yaml",
                "ablation": "{campaign_dir}/derived/ablation.yaml",
            },
        },
        "stages": [{"id": "gate", "kind": "manual", "requirements": [{"name": "x", "pattern": "x"}]}],
    }
    spec = project / "config" / "campaign.yaml"
    mapping = {"campaign": campaign}
    spec.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    definition = CampaignDefinition.from_mapping(
        mapping, project_root=project, campaign_dir=project / "campaigns" / "test_campaign", spec_path=spec
    )

    materialize_decisions(definition, "budget")
    materialize_decisions(definition, "nested")
    record = materialize_decisions(definition, "robustness")
    resolved = load_config(project / record["output"])

    assert resolved["robustness"]["algorithms"] == ["ppo", "a2c"]
    assert resolved["robustness"]["total_timesteps"] == 100
    assert resolved["robustness"]["window_algorithm_parameters"]["ppo"]["1"]["learning_rate"] == 0.002


def _strict_screening_decision_definition(
    tmp_path: Path,
    *,
    evidence_pattern: str = "aggregate/selection.json",
    decision_pattern: str = "results/screen_*/aggregate/selection.json",
) -> tuple[CampaignDefinition, Path]:
    project = tmp_path / "project"
    budget_config = project / "config" / "budget.yaml"
    budget_config.parent.mkdir(parents=True)
    budget_config.write_text(
        yaml.safe_dump({"budget_sweep": {"algorithms": ["ppo", "a2c"]}}),
        encoding="utf-8",
    )
    selection = project / "results" / "screen_verified" / "aggregate" / "selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(
        json.dumps(
            {"selection_split": "validation", "selected_algorithms": ["ppo"]}
        ),
        encoding="utf-8",
    )
    campaign = {
        "id": "test_campaign",
        "strict_deployment_resolution": True,
        "limits": {},
        "decisions": {
            "sources": {"screening_selection": decision_pattern},
            "source_stages": {"screening_selection": "screening_aggregate"},
            "base_configs": {"budget": "config/budget.yaml"},
            "outputs": {"budget": "{campaign_dir}/derived/budget.yaml"},
        },
        "protocol_lock": {
            "path": "protocol_lock.json",
            "inputs": [
                {
                    "name": "screening_selection",
                    "pattern": decision_pattern,
                }
            ],
        },
        "stages": [
            {
                "id": "screening",
                "kind": "manual",
                "run_glob": "results/screen_verified",
            },
            {
                "id": "screening_aggregate",
                "kind": "local",
                "depends_on": ["screening"],
                "run_source": "screening",
                "requirements": [
                    {"name": "selection", "pattern": evidence_pattern}
                ],
            },
        ],
    }
    spec = project / "config" / "campaign.yaml"
    mapping = {"campaign": campaign}
    spec.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    definition = CampaignDefinition.from_mapping(
        mapping,
        project_root=project,
        campaign_dir=project / "campaigns" / "test_campaign",
        spec_path=spec,
    )
    return definition, selection


def test_strict_decision_uses_verified_gate_evidence_not_newer_glob_match(tmp_path) -> None:
    definition, verified_selection = _strict_screening_decision_definition(tmp_path)
    verify_stage(definition, "screening")
    verify_stage(definition, "screening_aggregate")

    unverified_selection = (
        definition.project_root
        / "results"
        / "screen_unverified"
        / "aggregate"
        / "selection.json"
    )
    unverified_selection.parent.mkdir(parents=True)
    unverified_selection.write_text(
        json.dumps(
            {"selection_split": "validation", "selected_algorithms": ["a2c"]}
        ),
        encoding="utf-8",
    )
    old_ns = verified_selection.stat().st_mtime_ns
    os.utime(unverified_selection, ns=(old_ns + 1_000_000, old_ns + 1_000_000))

    record = materialize_decisions(definition, "budget")

    assert record["selected_algorithms"] == ["ppo"]
    source = record["sources"]["screening_selection"]
    status = {item.stage_id: item for item in evaluate_campaign(definition)}[
        "screening_aggregate"
    ]
    assert source["path"] == str(verified_selection.relative_to(definition.project_root))
    assert source["source_stage_id"] == "screening_aggregate"
    assert source["evidence_fingerprint"] == status.gate.evidence_fingerprint
    resolved = load_config(definition.project_root / record["output"])
    assert resolved["campaign_decision_provenance"]["sources"][
        "screening_selection"
    ] == source


def test_strict_decision_rejects_unverified_source_stage(tmp_path) -> None:
    definition, _ = _strict_screening_decision_definition(tmp_path)
    verify_stage(definition, "screening")

    with pytest.raises(ValueError, match="screening_aggregate.*VERIFIED"):
        materialize_decisions(definition, "budget")


def test_strict_decision_rejects_ambiguous_verified_gate_evidence(tmp_path) -> None:
    definition, selection = _strict_screening_decision_definition(
        tmp_path,
        evidence_pattern="aggregate/*.json",
        decision_pattern="results/screen_*/aggregate/*.json",
    )
    (selection.parent / "selection_copy.json").write_text(
        selection.read_text(encoding="utf-8"), encoding="utf-8"
    )
    verify_stage(definition, "screening")
    verify_stage(definition, "screening_aggregate")

    with pytest.raises(ValueError, match="exactly one.*found 2"):
        materialize_decisions(definition, "budget")


def test_strict_protocol_lock_excludes_newer_unverified_selection(tmp_path) -> None:
    definition, verified_selection = _strict_screening_decision_definition(tmp_path)
    verify_stage(definition, "screening")
    verify_stage(definition, "screening_aggregate")
    unverified_selection = (
        definition.project_root
        / "results"
        / "screen_unverified"
        / "aggregate"
        / "selection.json"
    )
    unverified_selection.parent.mkdir(parents=True)
    unverified_selection.write_text(
        json.dumps(
            {"selection_split": "validation", "selected_algorithms": ["a2c"]}
        ),
        encoding="utf-8",
    )
    old_ns = verified_selection.stat().st_mtime_ns
    os.utime(unverified_selection, ns=(old_ns + 1_000_000, old_ns + 1_000_000))

    lock = create_protocol_lock(definition)

    assert [item["path"] for item in lock["inputs"]] == [
        str(verified_selection.relative_to(definition.project_root))
    ]
    assert all(
        item["path"] != str(unverified_selection.relative_to(definition.project_root))
        for item in lock["inputs"]
    )


def test_protocol_lock_is_idempotent_and_refuses_scientific_drift(tmp_path) -> None:
    project = tmp_path / "project"
    config_path = project / "config" / "science.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("alpha: 1\nbeta: 2\n", encoding="utf-8")
    campaign = {
        "id": "test_campaign",
        "limits": {},
        "protocol_lock": {
            "path": "protocol_lock.json",
            "inputs": [{"name": "science", "pattern": "config/science.yaml"}],
        },
        "stages": [{"id": "gate", "kind": "manual", "requirements": [{"name": "x", "pattern": "x"}]}],
    }
    spec = project / "config" / "campaign.yaml"
    mapping = {"campaign": campaign}
    spec.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    definition = CampaignDefinition.from_mapping(
        mapping, project_root=project, campaign_dir=project / "campaigns" / "test_campaign", spec_path=spec
    )

    first = create_protocol_lock(definition)
    first_bytes = definition.protocol_lock_path.read_bytes()
    second = create_protocol_lock(definition)
    assert first == second
    assert definition.protocol_lock_path.read_bytes() == first_bytes

    config_path.write_text("alpha: 3\nbeta: 2\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        create_protocol_lock(definition)


def test_fixed_seed_registry_materializes_exact_window_grid_and_detects_model_drift(tmp_path) -> None:
    project = tmp_path / "project"
    selection = project / "results" / "screen_1" / "aggregate" / "selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(
        json.dumps({"selection_split": "validation", "selected_algorithms": ["ppo", "a2c"]}),
        encoding="utf-8",
    )
    science = project / "config" / "science.yaml"
    stress_base = project / "config" / "stress.yaml"
    science.parent.mkdir(parents=True, exist_ok=True)
    science.write_text("seed: 42\n", encoding="utf-8")
    stress_base.write_text(
        yaml.safe_dump(
            {
                "data": {"tickers": ["A", "B"]},
                "features": {"lookback": 2, "columns": ["return_1d"]},
                "execution_stress": {"require_models": False, "model_paths": [], "scenarios": [{"name": "base"}]},
            }
        ),
        encoding="utf-8",
    )
    campaign = {
        "id": "test_campaign",
        "limits": {},
        "decisions": {
            "window_ids": [0, 1],
            "stress_model_seed": 42,
            "sources": {"screening_selection": "results/screen_*/aggregate/selection.json"},
            "base_configs": {"stress": "config/stress.yaml"},
            "outputs": {
                "robustness": "{campaign_dir}/derived/robustness.yaml",
                "stress": "{campaign_dir}/derived/stress.yaml",
                "model_registry": "{campaign_dir}/derived/model_registry.json",
            },
        },
        "protocol_lock": {
            "path": "protocol_lock.json",
            "inputs": [
                {"name": "science", "pattern": "config/science.yaml"},
                {
                    "name": "screening_selection",
                    "pattern": "results/screen_*/aggregate/selection.json",
                },
            ],
        },
        "stages": [
            {"id": "robustness", "kind": "manual", "run_glob": "results/robustness_*"},
            {
                "id": "stress_config",
                "kind": "local",
                "requirements": [
                    {
                        "name": "registry",
                        "pattern": "{campaign_dir}/derived/model_registry.json",
                        "validator": "model_registry",
                    },
                    {"name": "sidecar", "pattern": "{campaign_dir}/derived/model_registry.sha256"},
                    {
                        "name": "stress config",
                        "pattern": "{campaign_dir}/derived/stress.yaml",
                        "validator": "stress_config",
                    },
                ],
            },
        ],
    }
    spec = project / "config" / "campaign.yaml"
    mapping = {"campaign": campaign}
    spec.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    definition = CampaignDefinition.from_mapping(
        mapping, project_root=project, campaign_dir=project / "campaigns" / "test_campaign", spec_path=spec
    )
    robustness_config = definition.campaign_dir / "derived" / "robustness.yaml"
    robustness_config.parent.mkdir(parents=True)
    robustness_config.write_text(
        yaml.safe_dump(
            {
                "walk_forward": {
                    "train_window": 6,
                    "validation_window": 3,
                    "test_window": 3,
                    "step_size": 3,
                    "expanding_train": False,
                }
            }
        ),
        encoding="utf-8",
    )
    create_protocol_lock(definition)
    lock_hash = definition.protocol_lock_path.with_suffix(".sha256").read_text().strip()
    newer_selection = project / "results" / "screen_2" / "aggregate" / "selection.json"
    newer_selection.parent.mkdir(parents=True)
    newer_selection.write_text(
        json.dumps({"selection_split": "validation", "selected_algorithms": ["ppo"]}),
        encoding="utf-8",
    )
    robustness_run = project / "results" / "robustness_1"
    feature_hash = "d" * 64
    for algorithm in ("ppo", "a2c"):
        for window in (0, 1):
            for seed in (42, 43):
                run = robustness_run / f"algorithm_{algorithm}" / f"window_{window}" / f"seed_{seed}"
                run.mkdir(parents=True)
                (run / f"{algorithm}_window_{window}_model.zip").write_bytes(
                    f"{algorithm}-{window}-{seed}".encode()
                )
                (run / f"{algorithm}_test_equity_curve.csv").write_text(
                    "date,portfolio_value\n2024-01-01,100\n2024-01-02,101\n", encoding="utf-8"
                )
                (run / "walk_forward_metrics.csv").write_text(
                    "window,seed,split,policy,sharpe\n"
                    f"{window},{seed},test,{algorithm},0.1\n",
                    encoding="utf-8",
                )
                (run / "resolved_config.json").write_text(
                    json.dumps(
                        {
                            "data": {"tickers": ["A", "B"]},
                            "features": {"lookback": 2, "columns": ["return_1d"]},
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "experiment_manifest.json").write_text(
                    json.dumps(
                        {
                            "experiment_id": f"{algorithm}_{window}_{seed}",
                            "status": "complete",
                            "campaign_id": "test_campaign",
                            "protocol_lock_hash": lock_hash,
                            "identity": {"algorithm": algorithm, "window": window, "seed": seed},
                            "data_files": {"features": {"sha256": feature_hash}},
                        }
                    ),
                    encoding="utf-8",
                )

    result = build_locked_stress_config(definition, robustness_run_dir=robustness_run)
    registry = json.loads((project / result["model_registry"]).read_text())
    locked_stress = load_config(project / result["locked_stress_config"])

    assert result["registered_models"] == 4
    assert registry["screening_selection_path"] == str(selection.relative_to(project))
    assert registry["screening_selection_sha256"] == sha256_file(selection)
    assert {(row["algorithm"], row["window"], row["seed"]) for row in registry["models"]} == {
        ("ppo", 0, 42),
        ("ppo", 1, 42),
        ("a2c", 0, 42),
        ("a2c", 1, 42),
    }
    assert locked_stress["execution_stress"]["require_models"] is True
    assert locked_stress["walk_forward"] == load_config(robustness_config)["walk_forward"]
    registry_path = project / result["model_registry"]
    original_registry = registry_path.read_bytes()
    seed_drift = copy.deepcopy(registry)
    seed_drift["selected_seed"] = 43
    registry_path.write_text(json.dumps(seed_drift, indent=2) + "\n", encoding="utf-8")
    registry_path.with_suffix(".sha256").write_text(sha256_file(registry_path) + "\n", encoding="utf-8")
    seed_status = evaluate_campaign(definition)[1]
    assert seed_status.state == StageState.BLOCKED
    assert "predeclared campaign stress seed" in " ".join(seed_status.gate.issues)
    registry_path.write_bytes(original_registry)
    registry_path.with_suffix(".sha256").write_text(sha256_file(registry_path) + "\n", encoding="utf-8")

    registry_path.write_text(json.dumps({"models": ["malformed"], "selected_seed": "bad"}), encoding="utf-8")
    registry_path.with_suffix(".sha256").write_text(sha256_file(registry_path) + "\n", encoding="utf-8")
    malformed_status = evaluate_campaign(definition)[1]
    assert malformed_status.state == StageState.BLOCKED
    assert "malformed" in " ".join(malformed_status.gate.issues)
    registry_path.write_bytes(original_registry)
    registry_path.with_suffix(".sha256").write_text(sha256_file(registry_path) + "\n", encoding="utf-8")

    assert evaluate_campaign(definition)[1].state == StageState.VERIFY_PENDING
    verify_stage(definition, "stress_config")

    registered_model = project / registry["models"][0]["path"]
    registered_model.write_bytes(b"drift")
    status = evaluate_campaign(definition)[1]
    assert status.state == StageState.STALE
    assert "SHA-256 mismatch" in " ".join(status.gate.issues)


def test_execution_stress_routes_models_to_matching_walk_forward_windows(tmp_path, monkeypatch) -> None:
    import src.experiments.run_execution_stress as stress_module
    from src.data.splits import generate_walk_forward_splits, walk_forward_config_from_mapping

    dates = pd.date_range("2024-01-01", periods=18, freq="D")
    features = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "close": 100.0 + index, "return_1d": 0.0}
            for index, date in enumerate(dates)
            for ticker in ("A", "B")
        ]
    )
    feature_path = tmp_path / "features.csv"
    features.to_csv(feature_path, index=False)
    feature_data_hash = sha256_file(feature_path)
    protocol_hash = "e" * 64
    model_paths = []
    windows_config = {
        "train_window": 6,
        "validation_window": 3,
        "test_window": 3,
        "step_size": 3,
        "expanding_train": False,
    }
    windows = generate_walk_forward_splits(features, walk_forward_config_from_mapping(windows_config))
    schema_hash = stress_module._feature_schema_hash(
        {"data": {"tickers": ["A", "B"]}, "features": {"lookback": 1, "columns": ["return_1d"]}}
    )
    for window in (0, 1):
        model = tmp_path / f"ppo_{window}.zip"
        model.write_bytes(f"model-{window}".encode())
        source = tmp_path / f"source_{window}.csv"
        source_dates = pd.to_datetime(windows[window]["test"]["date"].drop_duplicates().sort_values())
        pd.DataFrame(
            {"date": source_dates, "portfolio_value": 100.0 + np.arange(len(source_dates), dtype=float)}
        ).to_csv(source, index=False)
        model_paths.append(
            {
                "model_id": f"ppo_window_{window}",
                "policy": f"ppo_window_{window}",
                "algo": "ppo",
                "window": window,
                "seed": 42,
                "path": str(model),
                "sha256": sha256_file(model),
                "source_test_curve_path": str(source),
                "source_test_curve_sha256": sha256_file(source),
                "feature_schema_sha256": schema_hash,
                "feature_data_sha256": feature_data_hash,
                "protocol_lock_hash": protocol_hash,
            }
        )
    config = {
        "project": {"seed": 42},
        "paths": {"features": str(feature_path), "fallback_features_csv": str(feature_path)},
        "data": {"tickers": ["A", "B"]},
        "features": {"lookback": 1, "columns": ["return_1d"]},
        "environment": {"initial_cash": 100.0, "transaction_cost": 0.0, "reward": {}},
        "walk_forward": windows_config,
        "execution_stress": {
            "require_models": True,
            "protocol_lock_hash": protocol_hash,
            "model_paths": model_paths,
            "scenarios": [
                {"name": "base", "transaction_cost": 0.0},
                {"name": "hostile", "transaction_cost": 0.01},
            ],
        },
    }
    calls = []
    baseline_calls = []

    class FakeModelClass:
        @staticmethod
        def load(path):
            return {"path": str(path)}

    def fake_environment(_config, part):
        return {"dates": pd.to_datetime(part["date"].drop_duplicates().sort_values()).reset_index(drop=True)}

    def fake_evaluate(model, env, seed, policy_name, return_actions):
        calls.append((policy_name, tuple(env["dates"])))
        curve = pd.DataFrame(
            {
                "date": env["dates"],
                "portfolio_value": 100.0 + np.arange(len(env["dates"]), dtype=float),
                "turnover": 0.0,
            }
        )
        actions = pd.DataFrame(
            [
                {"date": date, "policy": policy_name, "ticker": "A", "weight": 1.0}
                for date in env["dates"]
            ]
        )
        return curve, actions

    def fake_baselines(_config, part, seed):
        window_dates = pd.to_datetime(part["date"].drop_duplicates().sort_values()).reset_index(drop=True)
        baseline_calls.append(tuple(window_dates))
        curve = pd.DataFrame(
            {
                "date": window_dates,
                "portfolio_value": 100.0,
                "policy": "equal_weight",
                "turnover": 0.0,
            }
        )
        metrics = pd.DataFrame([{"policy": "equal_weight", "sharpe": 0.0}])
        return curve, metrics

    monkeypatch.setenv("PROTOCOL_LOCK_HASH", protocol_hash)
    monkeypatch.setattr(stress_module, "ensure_feature_inputs", lambda _config: None)
    monkeypatch.setattr(stress_module, "load_feature_table", lambda *_args, **_kwargs: features)
    monkeypatch.setattr(stress_module, "_stable_baselines_algorithm", lambda _algo: FakeModelClass)
    monkeypatch.setattr(stress_module, "make_trading_env_from_features", fake_environment)
    monkeypatch.setattr(stress_module, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(stress_module, "run_baselines_on_features", fake_baselines)
    monkeypatch.setattr(stress_module, "_plot_execution_sensitivity", lambda *_args: None)

    metrics, readiness = stress_module.run_execution_stress(config, tmp_path / "stress")

    assert len(calls) == 4
    assert len(baseline_calls) == 4
    for policy, observed_dates in calls:
        window = int(policy.rsplit("_", 1)[1])
        expected_dates = tuple(pd.to_datetime(windows[window]["test"]["date"].drop_duplicates().sort_values()))
        assert observed_dates == expected_dates
    candidate = metrics[metrics["policy"].str.startswith("ppo_window_")]
    assert len(candidate) == 4
    assert set(candidate["window"]) == {0, 1}
    assert readiness["ready"].all()
    replay = pd.read_csv(tmp_path / "stress" / "execution_stress_base_replay.csv")
    assert replay["curve_match"].all()

    missing_binding = copy.deepcopy(config)
    del missing_binding["execution_stress"]["model_paths"][0]["feature_data_sha256"]
    with pytest.raises(ValueError, match="failed registry verification"):
        stress_module.run_execution_stress(missing_binding, tmp_path / "stress_missing_binding")


def test_experiment_manifest_hashes_the_feature_csv_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    fallback = tmp_path / "data" / "features.csv"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("date,ticker,return_1d\n2024-01-01,A,0.0\n", encoding="utf-8")
    output = tmp_path / "results" / "run"

    manifest_path = write_experiment_manifest(
        output,
        config={
            "data": {"source": "synthetic"},
            "paths": {
                "features": "data/features.parquet",
                "fallback_features_csv": "data/features.csv",
            }
        },
        identity={"experiment_id": "fallback"},
        status="complete",
        started_at="2024-01-01T00:00:00+00:00",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["data_files"]["features"]["path"] == str(fallback)
    assert manifest["data_files"]["features"]["sha256"] == sha256_file(fallback)
    assert manifest["data_provenance"] == {
        "data_origin": "generated",
        "evidence_class": "engineering_only",
        "execution_eligible": True,
        "publication_claim_eligible": False,
    }


def test_experiment_manifest_records_strict_slurm_provenance(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    expected = {
        "CAMPAIGN_ID": "strict_campaign",
        "CAMPAIGN_ATTEMPT_ID": "compute_20260716T010203Z_deadbeef",
        "CAMPAIGN_ATTEMPT_SHA256": "a" * 64,
        "CAMPAIGN_STAGE_ID": "compute",
        "DEPLOYMENT_RELEASE_ID": "b" * 64,
        "BISTKA_PREFLIGHT_SHA256": "c" * 64,
        "SLURM_JOB_ID": "50001",
        "SLURM_ARRAY_JOB_ID": "49999",
        "SLURM_ARRAY_TASK_ID": "3",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    manifest_path = write_experiment_manifest(
        tmp_path / "results" / "run" / "unit",
        config={"data": {"source": "synthetic"}},
        identity={"experiment_id": "strict-unit"},
        status="complete",
        started_at="2026-07-16T01:02:03+00:00",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["campaign_id"] == expected["CAMPAIGN_ID"]
    assert manifest["campaign_attempt_id"] == expected["CAMPAIGN_ATTEMPT_ID"]
    assert manifest["campaign_attempt_sha256"] == expected["CAMPAIGN_ATTEMPT_SHA256"]
    assert manifest["campaign_stage_id"] == expected["CAMPAIGN_STAGE_ID"]
    assert manifest["deployment_release_id"] == expected["DEPLOYMENT_RELEASE_ID"]
    assert manifest["bistka_preflight_sha256"] == expected["BISTKA_PREFLIGHT_SHA256"]
    assert manifest["runtime"]["slurm_job_id"] == expected["SLURM_JOB_ID"]
    assert manifest["runtime"]["slurm_array_job_id"] == expected["SLURM_ARRAY_JOB_ID"]


def test_calibration_attempt_fingerprints_runtime_selected_feature_table(tmp_path) -> None:
    project = tmp_path / "project"
    config_path = project / "config" / "screening.yaml"
    job_path = project / "slurm" / "calibration.sbatch"
    driver_path = project / "scripts" / "calibration_driver.py"
    selected_fallback = project / "data" / "features.csv"
    for path in (config_path, job_path, driver_path, selected_fallback):
        path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"source": "observed_fixture"},
                "paths": {
                    "features": "data/features.parquet",
                    "fallback_features_csv": "data/features.csv",
                },
                "features": {"columns": ["signal"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    selected_fallback.write_text(
        "date,ticker,close,signal\n2024-01-02,A.BK,100.0,0.25\n",
        encoding="utf-8",
    )
    job_path.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
    driver_path.write_text("CALIBRATION_PROTOCOL = 1\n", encoding="utf-8")
    campaign = {
        "id": "attempt_inputs",
        "limits": {},
        "stages": [
            {
                "id": "calibration",
                "kind": "slurm",
                "job_script": "slurm/calibration.sbatch",
                "config_path": "config/screening.yaml",
                "fingerprint_paths": ["scripts/calibration_driver.py"],
                "requirements": [
                    {"name": "future result", "pattern": "reports/calibration.json"}
                ],
            }
        ],
    }
    spec_path = project / "config" / "campaign.yaml"
    mapping = {"campaign": campaign}
    spec_path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    definition = CampaignDefinition.from_mapping(
        mapping,
        project_root=project,
        campaign_dir=project / "campaigns" / "attempt_inputs",
        spec_path=spec_path,
    )

    _render_command(definition, "calibration")

    attempt_path = definition.campaign_dir / "attempts" / "calibration_latest.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    inputs = {str(record["path"]): record for record in attempt["inputs"]}
    assert "data/features.parquet" not in inputs
    assert inputs["data/features.csv"]["sha256"] == sha256_file(selected_fallback)


def _strict_release_fixture(
    tmp_path: Path,
    *,
    include_runtime_binding: bool = True,
) -> dict[str, Path]:
    from src.campaigns.bistka_deployment import build_deployment, run_preflight

    project = tmp_path / "project"
    config_path = project / "config" / "runtime.yaml"
    feature_path = project / "data" / "features.csv"
    readiness_path = project / "reports" / "readiness.json"
    job_path = project / "slurm" / "job.sbatch"
    deployment_config_path = project / "config" / "deployment.yaml"
    manifest_path = project / "campaigns" / "strict_campaign" / "deployment" / "manifest.json"
    bundle_path = manifest_path.with_name("release.tar.gz")
    preflight_path = project / "reports" / "preflight.json"
    for path in (
        config_path,
        feature_path,
        readiness_path,
        job_path,
        deployment_config_path,
        manifest_path,
        preflight_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"source": "observed_fixture", "tickers": ["A.BK"]},
                "paths": {
                    "features": "data/features.csv",
                    "fallback_features_csv": "data/features.csv",
                },
                "features": {"columns": ["signal"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    feature_path.write_text(
        "date,ticker,close,signal\n2024-01-02,A.BK,100.0,0.25\n",
        encoding="utf-8",
    )
    readiness_path.write_text(
        json.dumps(
            {
                "data_origin": "observed_market",
                "execution_eligible": True,
                "evidence_class": "test_fixture",
            }
        ),
        encoding="utf-8",
    )
    job_path.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
    deployment_config_path.write_text(
        yaml.safe_dump(
            {
                "deployment": {
                    "campaign_id": "strict_campaign",
                    "inputs": [
                        "config/runtime.yaml",
                        "data/features.csv",
                        "reports/readiness.json",
                        "slurm/job.sbatch",
                    ],
                    "runtime_configs": (
                        ["config/runtime.yaml"] if include_runtime_binding else []
                    ),
                    "target_project_root": str(project.resolve()),
                    "required_hostname_prefix": "",
                    "required_packages": ["pyyaml"],
                    "data_readiness_path": "reports/readiness.json",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    build_deployment(
        project_root=project,
        config_path=deployment_config_path,
        output_manifest=manifest_path,
        output_bundle=bundle_path,
    )
    receipt = run_preflight(
        project_root=project,
        manifest_path=manifest_path,
        output_path=preflight_path,
        allow_non_bistka=True,
    )
    assert receipt["status"] == "pass"
    return {
        "project": project,
        "config": config_path,
        "feature": feature_path,
        "job": job_path,
        "manifest": manifest_path,
        "preflight": preflight_path,
    }


def _strict_multi_runtime_fixture(tmp_path: Path) -> dict[str, Path | list[Path]]:
    from src.campaigns.bistka_deployment import build_deployment, run_preflight

    fixture = _strict_release_fixture(tmp_path)
    project = fixture["project"]
    assert isinstance(project, Path)
    orchestrator = project / "config" / "orchestrator.yaml"
    group_configs = [project / "config" / f"group_{name}.yaml" for name in ("a", "b")]
    orchestrator.write_text(
        yaml.safe_dump({"feature_ablation": {"groups": ["a", "b"]}}, sort_keys=False),
        encoding="utf-8",
    )
    for group_path in group_configs:
        group_path.write_text(
            yaml.safe_dump(
                {
                    "data": {"source": "observed_fixture", "tickers": ["A.BK"]},
                    "paths": {
                        "features": "data/features.csv",
                        "fallback_features_csv": "data/features.csv",
                    },
                    "features": {"columns": ["signal"]},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    deployment_config = project / "config" / "deployment.yaml"
    deployment_config.write_text(
        yaml.safe_dump(
            {
                "deployment": {
                    "campaign_id": "strict_campaign",
                    "inputs": [
                        "config/orchestrator.yaml",
                        "config/group_a.yaml",
                        "config/group_b.yaml",
                        "data/features.csv",
                        "reports/readiness.json",
                        "slurm/job.sbatch",
                    ],
                    "runtime_configs": [
                        "config/group_a.yaml",
                        "config/group_b.yaml",
                    ],
                    "target_project_root": str(project.resolve()),
                    "required_hostname_prefix": "",
                    "required_packages": ["pyyaml"],
                    "data_readiness_path": "reports/readiness.json",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = fixture["manifest"]
    preflight = fixture["preflight"]
    assert isinstance(manifest, Path)
    assert isinstance(preflight, Path)
    build_deployment(
        project_root=project,
        config_path=deployment_config,
        output_manifest=manifest,
        output_bundle=manifest.with_name("release.tar.gz"),
    )
    receipt = run_preflight(
        project_root=project,
        manifest_path=manifest,
        output_path=preflight,
        allow_non_bistka=True,
    )
    assert receipt["status"] == "pass"
    fixture["config"] = orchestrator
    fixture["runtime_configs"] = group_configs
    return fixture


def _strict_definition_for_fixture(
    fixture: dict[str, Path],
    stages: list[dict],
) -> CampaignDefinition:
    project = fixture["project"]
    campaign = {
        "id": "strict_campaign",
        "strict_deployment_resolution": True,
        "limits": {},
        "stages": stages,
    }
    mapping = {"campaign": campaign}
    spec = project / "config" / "campaign.yaml"
    spec.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return CampaignDefinition.from_mapping(
        mapping,
        project_root=project,
        campaign_dir=project / "campaigns" / "strict_campaign",
        spec_path=spec,
    )


def _strict_slurm_stage(stage_id: str, **overrides) -> dict:
    stage = {
        "id": stage_id,
        "kind": "slurm",
        "job_script": "slurm/job.sbatch",
        "config_path": "config/runtime.yaml",
        "deployment_manifest": "campaigns/strict_campaign/deployment/manifest.json",
        "preflight_receipt": "reports/preflight.json",
        "requirements": [
            {"name": "future output", "pattern": f"results/{stage_id}.json"}
        ],
    }
    stage.update(overrides)
    return stage


def _strict_stale_stage_fixture(tmp_path: Path) -> dict[str, object]:
    fixture = _strict_release_fixture(tmp_path)
    definition = _strict_definition_for_fixture(
        fixture,
        [
            {
                "id": "dependency",
                "kind": "manual",
                "requirements": [
                    {"name": "dependency", "pattern": "reports/dependency.json"}
                ],
            },
            _strict_slurm_stage("compute", depends_on=["dependency"]),
        ],
    )
    dependency_path = definition.project_root / "reports" / "dependency.json"
    dependency_path.write_text('{"version": 1}\n', encoding="utf-8")
    verify_stage(definition, "dependency")
    _render_command(definition, "compute")
    old_attempt = json.loads(
        (
            definition.campaign_dir / "attempts" / "compute_latest.json"
        ).read_text(encoding="utf-8")
    )
    result_path = definition.project_root / "results" / "compute.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text('{"version": 1}\n', encoding="utf-8")
    verify_stage(definition, "compute")
    marker_path = (
        definition.campaign_dir / "stages" / "compute" / "_SUCCESS.json"
    )
    old_marker = marker_path.read_bytes()
    result_path.write_text('{"version": 2}\n', encoding="utf-8")
    assert evaluate_campaign(definition)[1].state == StageState.STALE
    return {
        "definition": definition,
        "fixture": fixture,
        "old_attempt": old_attempt,
        "marker_path": marker_path,
        "old_marker": old_marker,
    }


def test_strict_stale_slurm_stage_rerenders_as_a_new_superseding_attempt(
    tmp_path,
) -> None:
    context = _strict_stale_stage_fixture(tmp_path)
    definition = context["definition"]
    marker_path = context["marker_path"]
    old_marker = context["old_marker"]
    old_attempt = context["old_attempt"]
    assert isinstance(definition, CampaignDefinition)
    assert isinstance(marker_path, Path)
    assert isinstance(old_marker, bytes)
    assert isinstance(old_attempt, dict)

    _render_command(definition, "compute")

    new_attempt = json.loads(
        (
            definition.campaign_dir / "attempts" / "compute_latest.json"
        ).read_text(encoding="utf-8")
    )
    superseded = new_attempt["supersedes_stale_evidence"]
    old_marker_value = json.loads(old_marker)
    assert new_attempt["attempt_id"] != old_attempt["attempt_id"]
    assert superseded == {
        "stage_id": "compute",
        "success_marker_path": (
            "campaigns/strict_campaign/stages/compute/_SUCCESS.json"
        ),
        "success_marker_sha256": hashlib.sha256(old_marker).hexdigest(),
        "verified_evidence_fingerprint": old_marker_value[
            "evidence_fingerprint"
        ],
        "current_evidence_fingerprint": evaluate_campaign(definition)[1]
        .gate.evidence_fingerprint,
    }
    assert [record["stage_id"] for record in new_attempt["predecessor_gates"]] == [
        "dependency"
    ]
    assert marker_path.read_bytes() == old_marker
    assert evaluate_campaign(definition)[1].state == StageState.STALE

    verify_stage(definition, "compute")

    assert marker_path.read_bytes() != old_marker
    assert evaluate_campaign(definition)[1].state == StageState.VERIFIED


def test_strict_stale_slurm_rerender_rejects_submitted_latest_attempt(
    tmp_path,
) -> None:
    context = _strict_stale_stage_fixture(tmp_path)
    definition = context["definition"]
    old_attempt = context["old_attempt"]
    assert isinstance(definition, CampaignDefinition)
    assert isinstance(old_attempt, dict)
    latest_path = definition.campaign_dir / "attempts" / "compute_latest.json"
    old_attempt["status"] = "submitted"
    latest_path.write_text(json.dumps(old_attempt), encoding="utf-8")

    with pytest.raises(ValueError, match="latest attempt is submitted"):
        _render_command(definition, "compute")


def test_strict_stale_slurm_rerender_revalidates_deployment_context(
    tmp_path,
) -> None:
    context = _strict_stale_stage_fixture(tmp_path)
    definition = context["definition"]
    fixture = context["fixture"]
    assert isinstance(definition, CampaignDefinition)
    assert isinstance(fixture, dict)
    preflight_path = fixture["preflight"]
    assert isinstance(preflight_path, Path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["release_id"] = "0" * 64
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="preflight does not validate against its manifest",
    ):
        _render_command(definition, "compute")


def test_strict_stale_slurm_rerender_requires_verified_run_source(
    tmp_path,
) -> None:
    fixture = _strict_release_fixture(tmp_path)
    source = _strict_slurm_stage("source")
    aggregate = _strict_slurm_stage(
        "aggregate",
        run_source="source",
    )
    aggregate.pop("deployment_manifest")
    aggregate.pop("preflight_receipt")
    definition = _strict_definition_for_fixture(fixture, [source, aggregate])
    source_result = definition.project_root / "results" / "source.json"
    aggregate_result = definition.project_root / "results" / "aggregate.json"
    source_result.parent.mkdir(parents=True, exist_ok=True)
    source_result.write_text('{"version": 1}\n', encoding="utf-8")
    verify_stage(definition, "source")
    _render_command(definition, "aggregate")
    aggregate_result.write_text('{"version": 1}\n', encoding="utf-8")
    verify_stage(definition, "aggregate")
    aggregate_result.write_text('{"version": 2}\n', encoding="utf-8")
    source_result.write_text('{"version": 2}\n', encoding="utf-8")
    statuses = {
        status.stage_id: status.state for status in evaluate_campaign(definition)
    }
    assert statuses == {
        "source": StageState.STALE,
        "aggregate": StageState.STALE,
    }

    with pytest.raises(
        ValueError,
        match="run-source predecessors are VERIFIED: \\['source'\\]",
    ):
        _render_command(definition, "aggregate")


def test_stale_slurm_rerender_requires_strict_deployment_resolution(
    tmp_path,
) -> None:
    project = tmp_path / "project"
    job_path = project / "slurm" / "job.sbatch"
    job_path.parent.mkdir(parents=True)
    job_path.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
    definition = _definition(
        tmp_path,
        {
            "id": "non_strict",
            "limits": {},
            "stages": [
                {
                    "id": "compute",
                    "kind": "slurm",
                    "job_script": "slurm/job.sbatch",
                    "requirements": [
                        {"name": "result", "pattern": "results/compute.json"}
                    ],
                }
            ],
        },
    )
    result_path = definition.project_root / "results" / "compute.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text('{"version": 1}\n', encoding="utf-8")
    verify_stage(definition, "compute")
    result_path.write_text('{"version": 2}\n', encoding="utf-8")
    assert evaluate_campaign(definition)[0].state == StageState.STALE

    with pytest.raises(ValueError, match="requires strict deployment resolution"):
        _render_command(definition, "compute")


def test_strict_definition_requires_a_resolvable_deployment_pair(tmp_path) -> None:
    missing = {
        "id": "strict_missing",
        "strict_deployment_resolution": True,
        "limits": {},
        "stages": [
            {"id": "compute", "kind": "slurm", "config_path": "config/runtime.yaml"}
        ],
    }
    with pytest.raises(
        CampaignConfigurationError,
        match="execution stage compute must declare both",
    ):
        _definition(tmp_path / "missing", missing)

    partial = copy.deepcopy(missing)
    partial["id"] = "strict_partial"
    partial["stages"][0]["deployment_manifest"] = "deployment/manifest.json"
    with pytest.raises(
        CampaignConfigurationError,
        match="must declare both deployment_manifest and preflight_receipt",
    ):
        _definition(tmp_path / "partial", partial)

    empty_runtime_configs = copy.deepcopy(missing)
    empty_runtime_configs["id"] = "strict_empty_runtime_configs"
    empty_runtime_configs["stages"][0].update(
        {
            "deployment_manifest": "deployment/manifest.json",
            "preflight_receipt": "reports/preflight.json",
            "runtime_configs": [],
        }
    )
    with pytest.raises(CampaignConfigurationError, match="runtime_configs must be a non-empty list"):
        _definition(tmp_path / "empty_runtime_configs", empty_runtime_configs)

    unsafe_runtime_configs = copy.deepcopy(empty_runtime_configs)
    unsafe_runtime_configs["id"] = "strict_unsafe_runtime_configs"
    unsafe_runtime_configs["stages"][0]["runtime_configs"] = ["../outside.yaml"]
    with pytest.raises(CampaignConfigurationError, match="runtime_config must stay inside"):
        _definition(tmp_path / "unsafe_runtime_configs", unsafe_runtime_configs)

    transitive = {
        "id": "strict_transitive",
        "strict_deployment_resolution": True,
        "limits": {},
        "stages": [
            {
                "id": "compute",
                "kind": "slurm",
                "config_path": "config/runtime.yaml",
                "runtime_configs": ["config/group_a.yaml", "config/group_b.yaml"],
                "deployment_manifest": "deployment/manifest.json",
                "preflight_receipt": "reports/preflight.json",
            },
            {
                "id": "aggregate_one",
                "kind": "slurm",
                "config_path": "config/runtime.yaml",
                "run_source": "compute",
            },
            {
                "id": "aggregate_two",
                "kind": "slurm",
                "config_path": "config/runtime.yaml",
                "run_source": "aggregate_one",
            },
        ],
    }
    definition = _definition(tmp_path / "transitive", transitive)
    assert definition.resolve_deployment_pair("aggregate_two") == (
        "deployment/manifest.json",
        "reports/preflight.json",
    )
    assert definition.resolve_runtime_config_paths("aggregate_two") == (
        "config/group_a.yaml",
        "config/group_b.yaml",
    )


def test_strict_render_binds_exact_release_preflight_config_and_feature(tmp_path) -> None:
    fixture = _strict_release_fixture(tmp_path)
    definition = _strict_definition_for_fixture(
        fixture,
        [_strict_slurm_stage("compute")],
    )

    _render_command(definition, "compute")

    attempt = json.loads(
        (definition.campaign_dir / "attempts" / "compute_latest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    assert attempt["schema_version"] == 2
    assert attempt["deployment_manifest_path"] == (
        "campaigns/strict_campaign/deployment/manifest.json"
    )
    assert attempt["deployment_manifest_sha256"] == sha256_file(fixture["manifest"])
    assert attempt["deployment_release_id"] == manifest["release_id"]
    assert attempt["preflight_receipt_path"] == "reports/preflight.json"
    assert attempt["preflight_sha256"] == sha256_file(fixture["preflight"])
    assert attempt["runtime_config_path"] == "config/runtime.yaml"
    assert attempt["runtime_config_sha256"] == sha256_file(fixture["config"])
    assert attempt["runtime_feature_path"] == "data/features.csv"
    assert attempt["runtime_feature_sha256"] == sha256_file(fixture["feature"])
    assert attempt["runtime_configs"] == [
        {
            "path": "config/runtime.yaml",
            "sha256": sha256_file(fixture["config"]),
            "feature_path": "data/features.csv",
            "feature_sha256": sha256_file(fixture["feature"]),
        }
    ]
    assert attempt["predecessor_gates"] == []
    attempt_inputs = {record["path"] for record in attempt["inputs"]}
    assert {
        "campaigns/strict_campaign/deployment/manifest.json",
        "reports/preflight.json",
        "config/runtime.yaml",
        "data/features.csv",
    } <= attempt_inputs


def _strict_manifest_gate_fixture(
    tmp_path: Path,
    *,
    manifest_count: int = 2,
) -> dict[str, object]:
    fixture = _strict_release_fixture(tmp_path)
    definition = _strict_definition_for_fixture(
        fixture,
        [
            _strict_slurm_stage(
                "compute",
                requirements=[],
                run_glob="results/compute_*",
                run_dir_template="results/compute_{job_id}",
                manifest_pattern="**/experiment_manifest.json",
                expected_manifests=manifest_count,
            )
        ],
    )
    _render_command(definition, "compute")
    latest_path = definition.campaign_dir / "attempts" / "compute_latest.json"
    attempt = json.loads(latest_path.read_text(encoding="utf-8"))
    parent_job_id = "424242"
    run_dir = definition.project_root / "results" / f"compute_{parent_job_id}"
    preflight = json.loads(fixture["preflight"].read_text(encoding="utf-8"))
    manifest_paths: list[Path] = []
    for index in range(manifest_count):
        manifest_path = run_dir / f"unit_{index}" / "experiment_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "experiment_id": f"experiment-{index}",
                    "status": "complete",
                    "campaign_id": definition.campaign_id,
                    "campaign_attempt_id": attempt["attempt_id"],
                    "campaign_attempt_sha256": attempt["attempt_sha256"],
                    "campaign_stage_id": "compute",
                    "deployment_release_id": attempt["deployment_release_id"],
                    "bistka_preflight_sha256": attempt["preflight_sha256"],
                    "runtime": {
                        "slurm_array_job_id": parent_job_id,
                        "slurm_job_id": str(500000 + index),
                        "slurm_array_task_id": str(index),
                        "packages": preflight["packages"],
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest_paths.append(manifest_path)
    return {
        "definition": definition,
        "fixture": fixture,
        "attempt": attempt,
        "latest_path": latest_path,
        "manifest_paths": manifest_paths,
    }


def test_strict_stale_slurm_rerender_rejects_running_manifests(tmp_path) -> None:
    context = _strict_manifest_gate_fixture(tmp_path)
    definition = context["definition"]
    manifest_paths = context["manifest_paths"]
    assert isinstance(definition, CampaignDefinition)
    assert isinstance(manifest_paths, list)
    verify_stage(definition, "compute")
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest_paths[0].write_text(json.dumps(manifest), encoding="utf-8")
    status = evaluate_campaign(definition)[0]
    assert status.state == StageState.STALE
    assert status.gate.manifest_running == 1

    with pytest.raises(ValueError, match="manifest\\(s\\) are running"):
        _render_command(definition, "compute")


def test_strict_manifest_gate_binds_attempt_release_preflight_and_parent_job(
    tmp_path,
) -> None:
    context = _strict_manifest_gate_fixture(tmp_path)
    definition = context["definition"]
    assert isinstance(definition, CampaignDefinition)

    status = evaluate_campaign(definition)[0]

    assert status.state == StageState.VERIFY_PENDING
    assert status.gate.issues == []
    assert {
        "campaigns/strict_campaign/attempts/compute_latest.json",
        f"campaigns/strict_campaign/attempts/{context['attempt']['attempt_id']}.json",
    } <= set(status.gate.evidence_files)
    verify_stage(definition, "compute")
    assert evaluate_campaign(definition)[0].state == StageState.VERIFIED


def test_strict_manifest_gate_uses_slurm_job_id_for_non_array_parent(
    tmp_path,
) -> None:
    context = _strict_manifest_gate_fixture(tmp_path)
    definition = context["definition"]
    manifest_paths = context["manifest_paths"]
    assert isinstance(definition, CampaignDefinition)
    assert isinstance(manifest_paths, list)
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runtime"]["slurm_array_job_id"] = None
        manifest["runtime"]["slurm_job_id"] = "424242"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status = evaluate_campaign(definition)[0]

    assert status.state == StageState.VERIFY_PENDING
    assert status.gate.issues == []


@pytest.mark.parametrize(
    ("case", "expected_issue"),
    [
        ("missing_manifest_binding", "strict manifest is missing campaign_attempt_sha256"),
        ("mismatched_manifest_binding", "strict manifest deployment_release_id mismatch"),
        ("missing_parent_job", "strict manifest has no positive parent SLURM job id"),
        (
            "runtime_package_drift",
            "strict manifest runtime package pyyaml does not match the bound preflight",
        ),
        ("missing_immutable_attempt", "immutable strict SLURM attempt record is missing"),
        ("tampered_latest_attempt", "latest strict SLURM attempt is not in rendered state"),
        (
            "attempt_input_drift",
            "strict SLURM attempt input changed after command rendering: config/runtime.yaml",
        ),
    ],
)
def test_strict_manifest_gate_rejects_missing_tampered_or_drifted_provenance(
    tmp_path,
    case,
    expected_issue,
) -> None:
    context = _strict_manifest_gate_fixture(tmp_path / case)
    definition = context["definition"]
    fixture = context["fixture"]
    attempt = context["attempt"]
    latest_path = context["latest_path"]
    manifest_path = context["manifest_paths"][0]
    assert isinstance(definition, CampaignDefinition)
    assert isinstance(fixture, dict)
    assert isinstance(attempt, dict)
    assert isinstance(latest_path, Path)
    assert isinstance(manifest_path, Path)

    if case in {
        "missing_manifest_binding",
        "mismatched_manifest_binding",
        "missing_parent_job",
        "runtime_package_drift",
    }:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if case == "missing_manifest_binding":
            manifest.pop("campaign_attempt_sha256")
        elif case == "mismatched_manifest_binding":
            manifest["deployment_release_id"] = "0" * 64
        elif case == "runtime_package_drift":
            manifest["runtime"]["packages"]["pyyaml"] = "0.0.0"
        else:
            manifest["runtime"]["slurm_array_job_id"] = None
            manifest["runtime"]["slurm_job_id"] = None
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif case == "missing_immutable_attempt":
        (
            definition.campaign_dir
            / "attempts"
            / f"{attempt['attempt_id']}.json"
        ).unlink()
    elif case == "tampered_latest_attempt":
        latest_attempt = json.loads(latest_path.read_text(encoding="utf-8"))
        latest_attempt["status"] = "submitted"
        latest_path.write_text(json.dumps(latest_attempt), encoding="utf-8")
    else:
        config_path = fixture["config"]
        assert isinstance(config_path, Path)
        config_path.write_text(
            config_path.read_text(encoding="utf-8") + "\n# drift\n",
            encoding="utf-8",
        )

    status = evaluate_campaign(definition)[0]
    assert status.state == StageState.BLOCKED
    assert expected_issue in " ".join(status.gate.issues)


def test_strict_manifest_gate_rejects_duplicate_ids_and_mixed_parent_jobs(
    tmp_path,
) -> None:
    context = _strict_manifest_gate_fixture(tmp_path)
    definition = context["definition"]
    manifest_paths = context["manifest_paths"]
    assert isinstance(definition, CampaignDefinition)
    assert isinstance(manifest_paths, list)
    first = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    second = json.loads(manifest_paths[1].read_text(encoding="utf-8"))
    second["experiment_id"] = first["experiment_id"]
    second["runtime"]["slurm_array_job_id"] = "424243"
    manifest_paths[1].write_text(json.dumps(second), encoding="utf-8")

    status = evaluate_campaign(definition)[0]

    assert status.state == StageState.BLOCKED
    assert "experiment manifests contain duplicate experiment ids" in status.gate.issues
    assert "strict experiment manifests contain mixed parent SLURM jobs" in status.gate.issues
    assert any(
        "parent SLURM job does not match its run directory" in issue
        for issue in status.gate.issues
    )


def test_strict_multi_runtime_compute_and_aggregate_bind_every_effective_config(
    tmp_path,
) -> None:
    fixture = _strict_multi_runtime_fixture(tmp_path)
    runtime_configs = ["config/group_a.yaml", "config/group_b.yaml"]
    compute = _strict_slurm_stage(
        "compute",
        config_path="config/orchestrator.yaml",
        runtime_configs=runtime_configs,
        requirements=[],
    )
    aggregate = _strict_slurm_stage(
        "aggregate",
        config_path="config/orchestrator.yaml",
        depends_on=["compute"],
        run_source="compute",
    )
    aggregate.pop("deployment_manifest")
    aggregate.pop("preflight_receipt")
    definition = _strict_definition_for_fixture(fixture, [compute, aggregate])

    assert definition.stage_map["compute"].runtime_config_paths == tuple(runtime_configs)
    assert definition.resolve_runtime_config_paths("aggregate") == tuple(runtime_configs)
    verify_stage(definition, "compute")
    _render_command(definition, "aggregate")

    attempt = json.loads(
        (definition.campaign_dir / "attempts" / "aggregate_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert [record["path"] for record in attempt["runtime_configs"]] == runtime_configs
    assert all(record["feature_path"] == "data/features.csv" for record in attempt["runtime_configs"])
    assert all(
        record["feature_sha256"] == sha256_file(fixture["feature"])
        for record in attempt["runtime_configs"]
    )
    assert attempt["runtime_config_path"] == "config/orchestrator.yaml"
    assert attempt["runtime_config_sha256"] == sha256_file(fixture["config"])
    assert attempt["runtime_feature_path"] is None
    assert attempt["runtime_feature_sha256"] is None
    attempt_inputs = {record["path"] for record in attempt["inputs"]}
    assert {
        "config/orchestrator.yaml",
        "config/group_a.yaml",
        "config/group_b.yaml",
        "data/features.csv",
    } <= attempt_inputs


def test_strict_render_rejects_missing_runtime_binding_and_preflight_drift(tmp_path) -> None:
    unbound_fixture = _strict_release_fixture(
        tmp_path / "unbound",
        include_runtime_binding=False,
    )
    unbound_definition = _strict_definition_for_fixture(
        unbound_fixture,
        [_strict_slurm_stage("compute")],
    )
    with pytest.raises(ValueError, match="runtime feature binding does not match"):
        _render_command(unbound_definition, "compute")

    drift_fixture = _strict_release_fixture(tmp_path / "drift")
    drift_definition = _strict_definition_for_fixture(
        drift_fixture,
        [_strict_slurm_stage("compute")],
    )
    receipt = json.loads(drift_fixture["preflight"].read_text(encoding="utf-8"))
    receipt["release_id"] = "0" * 64
    drift_fixture["preflight"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="preflight does not validate against its manifest"):
        _render_command(drift_definition, "compute")


def test_strict_compute_rejects_zero_observed_bindings_and_synthetic_runtime_config(
    tmp_path,
) -> None:
    zero_fixture = _strict_multi_runtime_fixture(tmp_path / "zero")
    zero_definition = _strict_definition_for_fixture(
        zero_fixture,
        [
            _strict_slurm_stage(
                "compute",
                config_path="config/orchestrator.yaml",
                runtime_configs=["config/orchestrator.yaml"],
            )
        ],
    )
    with pytest.raises(ValueError, match="zero observed runtime feature bindings"):
        _render_command(zero_definition, "compute")

    from src.campaigns.bistka_deployment import build_deployment, run_preflight

    synthetic_fixture = _strict_multi_runtime_fixture(tmp_path / "synthetic")
    project = synthetic_fixture["project"]
    assert isinstance(project, Path)
    synthetic_group = project / "config" / "group_a.yaml"
    synthetic_value = load_config(synthetic_group)
    synthetic_value["data"]["source"] = "synthetic"
    synthetic_group.write_text(
        yaml.safe_dump(synthetic_value, sort_keys=False),
        encoding="utf-8",
    )
    manifest = synthetic_fixture["manifest"]
    preflight = synthetic_fixture["preflight"]
    assert isinstance(manifest, Path)
    assert isinstance(preflight, Path)
    build_deployment(
        project_root=project,
        config_path=project / "config" / "deployment.yaml",
        output_manifest=manifest,
        output_bundle=manifest.with_name("release.tar.gz"),
    )
    run_preflight(
        project_root=project,
        manifest_path=manifest,
        output_path=preflight,
        allow_non_bistka=True,
    )
    synthetic_definition = _strict_definition_for_fixture(
        synthetic_fixture,
        [
            _strict_slurm_stage(
                "compute",
                config_path="config/orchestrator.yaml",
                runtime_configs=["config/group_a.yaml", "config/group_b.yaml"],
            )
        ],
    )
    with pytest.raises(ValueError, match="uses prohibited synthetic data"):
        _render_command(synthetic_definition, "compute")


def test_strict_aggregate_inherits_pair_transitively_and_fingerprints_predecessor(
    tmp_path,
) -> None:
    fixture = _strict_release_fixture(tmp_path)
    compute = _strict_slurm_stage("compute", requirements=[])
    aggregate_one = _strict_slurm_stage(
        "aggregate_one",
        depends_on=["compute"],
        run_source="compute",
    )
    aggregate_one.pop("deployment_manifest")
    aggregate_one.pop("preflight_receipt")
    aggregate_two = _strict_slurm_stage(
        "aggregate_two",
        depends_on=["aggregate_one"],
        run_source="aggregate_one",
    )
    aggregate_two.pop("deployment_manifest")
    aggregate_two.pop("preflight_receipt")
    definition = _strict_definition_for_fixture(
        fixture,
        [compute, aggregate_one, aggregate_two],
    )
    verify_stage(definition, "compute")

    _render_command(definition, "aggregate_one")
    first_attempt = json.loads(
        (definition.campaign_dir / "attempts" / "aggregate_one_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert first_attempt["deployment_manifest_path"] == (
        "campaigns/strict_campaign/deployment/manifest.json"
    )
    assert [record["stage_id"] for record in first_attempt["predecessor_gates"]] == [
        "compute"
    ]
    predecessor = first_attempt["predecessor_gates"][0]
    assert predecessor["gate_report_sha256"] == sha256_file(
        definition.campaign_dir / "stages" / "compute" / "gate_report.json"
    )
    assert predecessor["success_marker_sha256"] == sha256_file(
        definition.campaign_dir / "stages" / "compute" / "_SUCCESS.json"
    )

    aggregate_one_result = definition.project_root / "results" / "aggregate_one.json"
    aggregate_one_result.parent.mkdir(parents=True)
    aggregate_one_result.write_text("{}\n", encoding="utf-8")
    verify_stage(definition, "aggregate_one")
    _render_command(definition, "aggregate_two")
    second_attempt = json.loads(
        (definition.campaign_dir / "attempts" / "aggregate_two_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert second_attempt["deployment_release_id"] == first_attempt[
        "deployment_release_id"
    ]
    assert [record["stage_id"] for record in second_attempt["predecessor_gates"]] == [
        "aggregate_one"
    ]
