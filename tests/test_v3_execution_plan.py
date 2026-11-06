from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.v3.evaluation_policy import validate_evaluation_policy
from src.v3.execution_plan import (
    BINDING_COLUMNS,
    EXECUTION_PLAN_COLUMNS,
    build_execution_plan,
    summarize_execution_plan,
    validate_execution_plan,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config/v3/evaluation_policy_draft.yaml"
CANDIDATES = (
    ROOT / "reports/data_quality_v3/v3_candidate_ledger_20260820.csv"
)
FOLD_SUMMARY = (
    ROOT / "reports/data_quality_v3/v3_evaluation_fold_summary_20260820.csv"
)


def _inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    issues, policy = validate_evaluation_policy(POLICY)
    assert issues == []
    candidates = pd.read_csv(CANDIDATES, keep_default_na=False)
    summary = pd.read_csv(FOLD_SUMMARY, keep_default_na=False)
    bindings = {
        key: hashlib.sha256(f"test:{key}".encode()).hexdigest()
        for key in BINDING_COLUMNS
    }
    return policy, candidates, summary, bindings


def test_exact_execution_plan_is_blocked_and_budget_reconciles() -> None:
    policy, candidates, summary, bindings = _inputs()
    plan = build_execution_plan(
        policy,
        candidates,
        summary,
        bindings=bindings,
    )

    assert list(plan.columns) == list(EXECUTION_PLAN_COLUMNS)
    assert len(plan) == 540
    assert plan["work_unit_id"].nunique() == 540
    assert plan["cell_sha256"].nunique() == 540
    assert plan["execution_status"].eq("blocked_not_authorized").all()
    assert not plan["hpc_submission_authorized"].any()
    assert set(plan.groupby("array_batch_id").size()) == {180}
    assert plan["planned_training_timesteps"].sum() == 63_000_000
    assert validate_execution_plan(
        plan,
        policy,
        candidates,
        summary,
        bindings=bindings,
    ) == []
    assert summarize_execution_plan(plan) == {
        "work_units": 540,
        "candidates": 30,
        "folds": 6,
        "seeds": 3,
        "array_batches": 3,
        "maximum_cells_per_array_batch": 180,
        "planned_training_timesteps": 63_000_000,
        "blocked_cells": 540,
        "authorized_hpc_cells": 0,
    }


def test_execution_plan_is_invariant_to_input_row_order() -> None:
    policy, candidates, summary, bindings = _inputs()
    first = build_execution_plan(
        policy,
        candidates,
        summary,
        bindings=bindings,
    )
    second = build_execution_plan(
        policy,
        candidates.sample(frac=1.0, random_state=71),
        summary.sample(frac=1.0, random_state=73),
        bindings=bindings,
    )
    pd.testing.assert_frame_equal(first, second)


def test_execution_plan_rejects_any_authorization_or_cell_tamper() -> None:
    policy, candidates, summary, bindings = _inputs()
    plan = build_execution_plan(
        policy,
        candidates,
        summary,
        bindings=bindings,
    )
    authorized = plan.copy()
    authorized.loc[0, "hpc_submission_authorized"] = True
    issues = validate_execution_plan(
        authorized,
        policy,
        candidates,
        summary,
        bindings=bindings,
    )
    assert "the structural execution plan cannot authorize HPC submission" in issues
    assert "execution plan differs from the exact canonical registry" in issues

    moved = plan.copy()
    moved.loc[0, "array_batch_id"] = "array_batch_03"
    issues = validate_execution_plan(
        moved,
        policy,
        candidates,
        summary,
        bindings=bindings,
    )
    assert issues == ["execution plan differs from the exact canonical registry"]

    rehashed = plan.copy()
    rehashed.loc[0, "cell_sha256"] = "0" * 64
    issues = validate_execution_plan(
        rehashed,
        policy,
        candidates,
        summary,
        bindings=bindings,
    )
    assert issues == ["execution plan differs from the exact canonical registry"]


def test_execution_plan_rejects_binding_and_fold_drift() -> None:
    policy, candidates, summary, bindings = _inputs()
    invalid_bindings = dict(bindings)
    invalid_bindings["protocol_sha256"] = "not-a-hash"
    try:
        build_execution_plan(
            policy,
            candidates,
            summary,
            bindings=invalid_bindings,
        )
    except ValueError as exc:
        assert "invalid SHA-256" in str(exc)
    else:
        raise AssertionError("invalid bindings were accepted")

    boundary_owned = summary.copy()
    boundary_owned.loc[0, "boundary_transition_owned"] = True
    try:
        build_execution_plan(
            policy,
            candidates,
            boundary_owned,
            bindings=bindings,
        )
    except ValueError as exc:
        assert "boundary transition" in str(exc)
    else:
        raise AssertionError("owned cross-boundary transition was accepted")
