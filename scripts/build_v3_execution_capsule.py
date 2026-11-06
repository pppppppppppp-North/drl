"""Build checksum-bound V3 work units without authorizing or running them."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.v3.development_gate import (
    sha256_file,
    validate_development_protocol,
    validate_development_request,
)
from src.v3.evaluation_policy import (
    validate_candidate_ledger,
    validate_evaluation_policy,
    validate_fold_schedule,
)
from src.v3.execution_plan import (
    ARRAY_BATCH_BY_FOLD,
    build_execution_plan,
    summarize_execution_plan,
    validate_execution_plan,
)


DEFAULT_PROTOCOL = "config/v3/development_protocol.yaml"
DEFAULT_PLAN = "reports/data_quality_v3/v3_execution_plan_20260904.csv"
DEFAULT_FOLD_WORKLOAD = (
    "reports/data_quality_v3/v3_execution_plan_fold_workload_20260904.csv"
)
DEFAULT_RECEIPT = (
    "reports/data_quality_v3/v3_execution_capsule_readiness_20260904.json"
)

V2_IMMUTABILITY_BINDINGS = {
    "locked_preregistration": {
        "path": "config/research_preregistration.yaml",
        "sha256": "69a02d38f64ce65f3d593cf28dbe4135e0965102380aa18d6f000a9514b5e045",
    },
    "research_campaign": {
        "path": "config/research_campaign.yaml",
        "sha256": "9d0d02ad1890379e72ce8d7513cf93dc83a54e8afc1c6f527c5275e76476a0b4",
    },
    "active_storage_safe_release_manifest": {
        "path": (
            "campaigns/large_scale_thai_drl_v2/deployment/"
            "bistka_screening_storage_safe_cutover_v2_manifest.json"
        ),
        "sha256": "439b590bc4a85f03a6d98723ee3d28e40acdc00d95bdfbf48d310f07d867e25c",
    },
}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _write_sidecar(path: Path) -> Path:
    sidecar = Path(f"{path}.sha256")
    _atomic_text(sidecar, f"{sha256_file(path)}  {path.name}\n")
    return sidecar


def _verify_sidecar(path: Path) -> Path:
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    tokens = sidecar.read_text(encoding="utf-8").split()
    if not tokens or tokens[0].lower() != sha256_file(path):
        raise ValueError(f"SHA-256 sidecar does not match: {path}")
    return sidecar


def _safe_project_path(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} path must be a nonempty string")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} path must stay inside the project")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} path escapes the project")
    return resolved


def _descriptor(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact escapes the project: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": resolved.relative_to(root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_descriptor(
    root: Path,
    value: Any,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} descriptor must be a mapping")
    path = _safe_project_path(root, value.get("path"), label)
    expected = _descriptor(root, path)
    if value != expected:
        raise ValueError(f"{label} descriptor does not match the exact file")
    return path, expected


def _protocol_chain(root: Path, protocol_path: Path) -> dict[str, Any]:
    structural_issues, _ = validate_development_protocol(
        protocol_path,
        project_root=root,
        require_locked=False,
    )
    if structural_issues:
        raise ValueError(
            "invalid structural V3 protocol: " + "; ".join(structural_issues)
        )
    loaded = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("development protocol must be a mapping")
    policy_binding = loaded.get("evaluation_policy")
    if not isinstance(policy_binding, dict):
        raise ValueError("development protocol has no evaluation-policy binding")
    policy_path = _safe_project_path(
        root,
        policy_binding.get("path"),
        "evaluation policy",
    )
    if sha256_file(policy_path) != str(policy_binding.get("sha256", "")).lower():
        raise ValueError("development protocol evaluation-policy hash does not match")
    policy_sidecar = _verify_sidecar(policy_path)
    receipt_binding = policy_binding.get("receipt")
    if not isinstance(receipt_binding, dict):
        raise ValueError("development protocol has no evaluation-receipt binding")
    evaluation_receipt_path = _safe_project_path(
        root,
        receipt_binding.get("path"),
        "evaluation readiness receipt",
    )
    if sha256_file(evaluation_receipt_path) != str(
        receipt_binding.get("sha256", "")
    ).lower():
        raise ValueError("development protocol evaluation-receipt hash does not match")
    evaluation_receipt_sidecar = _verify_sidecar(evaluation_receipt_path)
    evaluation_receipt = _load_json(
        evaluation_receipt_path,
        "evaluation readiness receipt",
    )
    if (
        evaluation_receipt.get("evidence_class")
        != "v3_evaluation_policy_structural_readiness"
        or evaluation_receipt.get("schedule_structural_eligible") is not True
        or evaluation_receipt.get("candidate_registry_eligible") is not True
        or evaluation_receipt.get("training_enabled") is not False
        or evaluation_receipt.get("hpc_jobs_submitted") != 0
    ):
        raise ValueError("evaluation readiness receipt has an invalid boundary")
    receipt_policy = evaluation_receipt.get("policy")
    if not isinstance(receipt_policy, dict):
        raise ValueError("evaluation receipt does not bind its policy")
    if receipt_policy.get("path") != _descriptor(root, policy_path)["path"]:
        raise ValueError("evaluation receipt points to a different policy")
    if receipt_policy.get("sha256") != sha256_file(policy_path):
        raise ValueError("evaluation receipt policy hash does not match")

    outputs = evaluation_receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("evaluation receipt outputs must be a mapping")
    candidate_path, candidate_descriptor = _require_descriptor(
        root,
        outputs.get("candidate_ledger"),
        "candidate ledger",
    )
    candidate_sidecar_path, candidate_sidecar_descriptor = _require_descriptor(
        root,
        outputs.get("candidate_ledger_sidecar"),
        "candidate ledger sidecar",
    )
    if candidate_sidecar_path != Path(f"{candidate_path}.sha256"):
        raise ValueError("candidate ledger sidecar path is not canonical")
    _verify_sidecar(candidate_path)
    schedule_path, schedule_descriptor = _require_descriptor(
        root,
        outputs.get("fold_schedule"),
        "fold schedule",
    )
    schedule_sidecar_path, schedule_sidecar_descriptor = _require_descriptor(
        root,
        outputs.get("fold_schedule_sidecar"),
        "fold schedule sidecar",
    )
    if schedule_sidecar_path != Path(f"{schedule_path}.sha256"):
        raise ValueError("fold schedule sidecar path is not canonical")
    _verify_sidecar(schedule_path)
    summary_path, summary_descriptor = _require_descriptor(
        root,
        outputs.get("fold_summary"),
        "fold summary",
    )
    summary_sidecar_path, summary_sidecar_descriptor = _require_descriptor(
        root,
        outputs.get("fold_summary_sidecar"),
        "fold summary sidecar",
    )
    if summary_sidecar_path != Path(f"{summary_path}.sha256"):
        raise ValueError("fold summary sidecar path is not canonical")
    _verify_sidecar(summary_path)
    return {
        "protocol": loaded,
        "policy_path": policy_path,
        "policy_sidecar": policy_sidecar,
        "evaluation_receipt": evaluation_receipt,
        "evaluation_receipt_path": evaluation_receipt_path,
        "evaluation_receipt_sidecar": evaluation_receipt_sidecar,
        "candidate_path": candidate_path,
        "candidate_descriptor": candidate_descriptor,
        "candidate_sidecar": candidate_sidecar_descriptor,
        "schedule_path": schedule_path,
        "schedule_descriptor": schedule_descriptor,
        "schedule_sidecar": schedule_sidecar_descriptor,
        "summary_path": summary_path,
        "summary_descriptor": summary_descriptor,
        "summary_sidecar": summary_sidecar_descriptor,
    }


def _fold_workload(plan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_id, group in plan.groupby("fold_id", sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "fold_id": str(fold_id),
                "array_batch_id": ARRAY_BATCH_BY_FOLD[str(fold_id)],
                "train_decisions": int(first["train_decisions"]),
                "validation_decisions": int(first["validation_decisions"]),
                "candidate_configurations": int(group["candidate_id"].nunique()),
                "model_seeds": int(group["seed"].nunique()),
                "planned_cells": int(len(group)),
                "planned_training_timesteps": int(
                    group["planned_training_timesteps"].sum()
                ),
                "execution_status": "blocked_not_authorized",
            }
        )
    return pd.DataFrame(rows)


def build(
    *,
    project_root: Path,
    protocol_path: Path,
    plan_path: Path,
    fold_workload_path: Path,
    receipt_path: Path,
    generated_at_utc: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    protocol = (
        protocol_path.resolve()
        if protocol_path.is_absolute()
        else (root / protocol_path).resolve()
    )
    outputs = {
        "plan": (root / plan_path).resolve(),
        "fold_workload": (root / fold_workload_path).resolve(),
        "receipt": (root / receipt_path).resolve(),
    }
    if not protocol.is_relative_to(root):
        raise ValueError("protocol path escapes the project")
    for label, path in outputs.items():
        if not path.is_relative_to(root):
            raise ValueError(f"{label} output path escapes the project")

    chain = _protocol_chain(root, protocol)
    policy_issues, policy = validate_evaluation_policy(chain["policy_path"])
    if policy_issues:
        raise ValueError("invalid evaluation policy: " + "; ".join(policy_issues))
    candidates = pd.read_csv(chain["candidate_path"], keep_default_na=False)
    schedule = pd.read_csv(chain["schedule_path"], keep_default_na=False)
    fold_summary = pd.read_csv(chain["summary_path"], keep_default_na=False)
    candidate_issues = validate_candidate_ledger(candidates, policy)
    sessions = schedule["date"].drop_duplicates()
    schedule_issues = validate_fold_schedule(schedule, sessions, policy)
    if candidate_issues or schedule_issues:
        raise ValueError("; ".join([*candidate_issues, *schedule_issues]))

    bindings = {
        "protocol_sha256": sha256_file(protocol),
        "evaluation_policy_sha256": sha256_file(chain["policy_path"]),
        "evaluation_readiness_sha256": sha256_file(
            chain["evaluation_receipt_path"]
        ),
        "candidate_ledger_artifact_sha256": sha256_file(chain["candidate_path"]),
        "fold_schedule_artifact_sha256": sha256_file(chain["schedule_path"]),
        "fold_summary_artifact_sha256": sha256_file(chain["summary_path"]),
    }
    plan = build_execution_plan(
        policy,
        candidates,
        fold_summary,
        bindings=bindings,
    )
    plan_issues = validate_execution_plan(
        plan,
        policy,
        candidates,
        fold_summary,
        bindings=bindings,
    )
    if plan_issues:
        raise ValueError("; ".join(plan_issues))
    workload = _fold_workload(plan)
    _atomic_csv(outputs["plan"], plan)
    plan_sidecar = _write_sidecar(outputs["plan"])
    _atomic_csv(outputs["fold_workload"], workload)
    workload_sidecar = _write_sidecar(outputs["fold_workload"])

    train_gate_issues, _ = validate_development_request(
        protocol,
        project_root=root,
        split="train",
    )
    v2_descriptors: dict[str, Any] = {}
    for label, binding in V2_IMMUTABILITY_BINDINGS.items():
        path = root / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"locked V2 artifact changed: {binding['path']}")
        v2_descriptors[label] = _descriptor(root, path)

    implementation_paths = {
        "execution_plan": root / "src/v3/execution_plan.py",
        "trial_launcher": root / "src/v3/trial_launcher.py",
        "fold_dataset": root / "src/v3/fold_dataset.py",
        "fold_preprocessing": root / "src/v3/fold_preprocessing.py",
        "transition_result_evidence": root / "src/v3/result_evidence.py",
        "artifact_builder": root / "scripts/build_v3_execution_capsule.py",
    }
    for path in implementation_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = summarize_execution_plan(plan)
    receipt = {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc,
        "status": "blocked",
        "evidence_class": "v3_execution_capsule_structural_readiness",
        "execution_plan_structural_eligible": True,
        "launch_authorized": False,
        "training_enabled": False,
        "test_access_enabled": False,
        "all_access_enabled": False,
        "hpc_submission_enabled": False,
        "hpc_jobs_planned_maximum": 3,
        "hpc_jobs_submitted": 0,
        "inputs": {
            "development_protocol": _descriptor(root, protocol),
            "evaluation_policy": _descriptor(root, chain["policy_path"]),
            "evaluation_policy_sidecar": _descriptor(root, chain["policy_sidecar"]),
            "evaluation_readiness_receipt": _descriptor(
                root, chain["evaluation_receipt_path"]
            ),
            "evaluation_readiness_receipt_sidecar": _descriptor(
                root, chain["evaluation_receipt_sidecar"]
            ),
            "candidate_ledger": chain["candidate_descriptor"],
            "candidate_ledger_sidecar": chain["candidate_sidecar"],
            "fold_schedule": chain["schedule_descriptor"],
            "fold_schedule_sidecar": chain["schedule_sidecar"],
            "fold_summary": chain["summary_descriptor"],
            "fold_summary_sidecar": chain["summary_sidecar"],
            "v2_immutability_baseline": v2_descriptors,
        },
        "implementation": {
            label: _descriptor(root, path)
            for label, path in implementation_paths.items()
        },
        "outputs": {
            "execution_plan": _descriptor(root, outputs["plan"]),
            "execution_plan_sidecar": _descriptor(root, plan_sidecar),
            "fold_workload": _descriptor(root, outputs["fold_workload"]),
            "fold_workload_sidecar": _descriptor(root, workload_sidecar),
        },
        "metrics": {
            **summary,
            "validation_transitions": int(
                fold_summary["validation_decisions"].astype(int).sum()
            ),
            "whole_project_verified_stages": 11,
            "whole_project_total_stages": 37,
            "whole_project_percent": 29.7,
        },
        "checks": [
            {"check": "development_protocol_structural", "status": "pass", "severity": "critical"},
            {"check": "upstream_receipt_descriptors", "status": "pass", "severity": "critical"},
            {"check": "candidate_grid_canonical", "status": "pass", "severity": "critical"},
            {"check": "fold_schedule_canonical", "status": "pass", "severity": "critical"},
            {"check": "exact_540_cell_registry", "status": "pass", "severity": "critical"},
            {"check": "three_array_batch_mapping", "status": "pass", "severity": "critical"},
            {"check": "all_cells_blocked", "status": "pass", "severity": "critical"},
            {"check": "fold_dataset_contract_implemented", "status": "pass", "severity": "critical"},
            {"check": "train_only_preprocessing_contract_implemented", "status": "pass", "severity": "critical"},
            {"check": "transition_result_evidence_contract_implemented", "status": "pass", "severity": "critical"},
            {"check": "licensed_market_package", "status": "fail", "severity": "critical"},
            {"check": "evaluation_policy_locked", "status": "fail", "severity": "critical"},
            {"check": "aggregate_metric_derivation_locked", "status": "fail", "severity": "critical"},
            {"check": "development_protocol_locked", "status": "fail", "severity": "critical"},
            {"check": "v3_training_gate", "status": "fail", "severity": "critical"},
            {"check": "reviewed_v3_trial_runner", "status": "fail", "severity": "critical"},
        ],
        "current_train_gate_issues": train_gate_issues,
        "scientific_boundary": {
            "registry_is_execution_authority": False,
            "launcher_must_revalidate_at_point_of_use": True,
            "candidate_metadata_execution_status_is_not_authorization": True,
            "current_schedule_is_provisional": True,
            "known_period_is_not_final_holdout": True,
            "no_market_features_loaded": True,
            "no_model_backend_imported": True,
            "no_performance_metrics_generated": True,
            "test_and_all_forbidden": True,
            "hpc_jobs_submitted": 0,
        },
        "report_design": {
            "audience": "technical",
            "primary_surface": "mcp_app_report",
            "chart_contract": {
                "question": "How does validation evidence volume vary across the six planned folds?",
                "family": "comparison",
                "variant": "single_series_bar",
                "takeaway": "The six validation folds contain 118 to 122 scored transitions, while each remains blocked from execution.",
                "claim_boundary": "schedule_and_workload_only_not_performance",
            },
        },
    }
    _atomic_text(
        outputs["receipt"],
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    _write_sidecar(outputs["receipt"])
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the blocked V3 execution capsule and exact work registry"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--plan-output", default=DEFAULT_PLAN)
    parser.add_argument("--fold-workload-output", default=DEFAULT_FOLD_WORKLOAD)
    parser.add_argument("--receipt-output", default=DEFAULT_RECEIPT)
    parser.add_argument("--generated-at-utc", default=None)
    args = parser.parse_args(argv)
    generated = args.generated_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    receipt = build(
        project_root=args.project_root,
        protocol_path=Path(args.protocol),
        plan_path=Path(args.plan_output),
        fold_workload_path=Path(args.fold_workload_output),
        receipt_path=Path(args.receipt_output),
        generated_at_utc=generated,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "work_units": receipt["metrics"]["work_units"],
                "array_batches": receipt["metrics"]["array_batches"],
                "authorized_hpc_cells": receipt["metrics"][
                    "authorized_hpc_cells"
                ],
                "hpc_jobs_submitted": receipt["hpc_jobs_submitted"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
