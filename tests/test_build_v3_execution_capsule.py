from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.v3.evaluation_policy import validate_evaluation_policy
from src.v3.execution_plan import BINDING_COLUMNS, validate_execution_plan


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = (
    ROOT
    / "reports/data_quality_v3/v3_execution_capsule_readiness_20260907.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_descriptor_tree(value: object) -> None:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            path = ROOT / str(value["path"])
            assert path.is_file()
            assert path.stat().st_size == value["bytes"]
            assert _sha256(path) == value["sha256"]
            return
        for child in value.values():
            _assert_descriptor_tree(child)
    elif isinstance(value, list):
        for child in value:
            _assert_descriptor_tree(child)


def test_execution_capsule_receipt_is_bound_and_fail_closed() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    _assert_descriptor_tree(receipt)
    sidecar = Path(f"{RECEIPT_PATH}.sha256")
    digest, name = sidecar.read_text(encoding="utf-8").split()
    assert name == RECEIPT_PATH.name
    assert digest == _sha256(RECEIPT_PATH)

    assert receipt["status"] == "blocked"
    assert receipt["execution_plan_structural_eligible"] is True
    assert receipt["launch_authorized"] is False
    assert receipt["training_enabled"] is False
    assert receipt["test_access_enabled"] is False
    assert receipt["all_access_enabled"] is False
    assert receipt["hpc_submission_enabled"] is False
    assert receipt["hpc_jobs_submitted"] == 0
    assert receipt["metrics"]["work_units"] == 540
    assert receipt["metrics"]["blocked_cells"] == 540
    assert receipt["metrics"]["authorized_hpc_cells"] == 0
    assert receipt["metrics"]["planned_training_timesteps"] == 63_000_000
    assert receipt["metrics"]["whole_project_percent"] == 29.7
    assert "V3 protocol is not locked" in receipt["current_train_gate_issues"]

    checks = {row["check"]: row["status"] for row in receipt["checks"]}
    assert checks["exact_540_cell_registry"] == "pass"
    assert checks["fold_dataset_contract_implemented"] == "pass"
    assert checks["train_only_preprocessing_contract_implemented"] == "pass"
    assert checks["transition_result_evidence_contract_implemented"] == "pass"
    assert checks["licensed_market_package"] == "fail"
    assert checks["aggregate_metric_derivation_locked"] == "fail"
    assert checks["reviewed_v3_trial_runner"] == "fail"


def test_saved_execution_plan_revalidates_against_bound_upstream_artifacts() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    inputs = receipt["inputs"]
    plan = pd.read_csv(ROOT / receipt["outputs"]["execution_plan"]["path"])
    candidates = pd.read_csv(ROOT / inputs["candidate_ledger"]["path"], keep_default_na=False)
    summary = pd.read_csv(ROOT / inputs["fold_summary"]["path"], keep_default_na=False)
    issues, policy = validate_evaluation_policy(ROOT / inputs["evaluation_policy"]["path"])
    assert issues == []
    bindings = {column: str(plan[column].iloc[0]) for column in BINDING_COLUMNS}

    assert validate_execution_plan(
        plan,
        policy,
        candidates,
        summary,
        bindings=bindings,
    ) == []
    assert plan["work_unit_id"].nunique() == 540
    assert plan["cell_sha256"].nunique() == 540
    assert set(plan.groupby("array_batch_id").size()) == {180}
    assert plan["requested_roles"].eq("train_then_validation").all()
    assert plan["execution_status"].eq("blocked_not_authorized").all()
    assert not plan["hpc_submission_authorized"].any()


def test_execution_capsule_contains_no_performance_claims() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    metrics = receipt["metrics"]
    forbidden = {
        "sharpe",
        "net_return",
        "cumulative_return",
        "max_drawdown",
        "profit",
        "model_rank",
        "winning_candidate",
    }
    assert forbidden.isdisjoint(metrics)
    assert receipt["scientific_boundary"]["no_performance_metrics_generated"] is True
    assert receipt["scientific_boundary"]["known_period_is_not_final_holdout"] is True
