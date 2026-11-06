"""Deterministic, non-executable V3 candidate/fold/seed work registry.

The registry is deliberately downstream of the development protocol and the
evaluation-policy readiness receipt.  It enumerates future work without
authorizing training, importing a trainer, reading a holdout, or submitting an
HPC job.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.v3.evaluation_policy import (
    FOLD_IDS,
    MODEL_SEEDS,
    validate_candidate_ledger,
)


EXECUTION_PLAN_SCHEMA_VERSION = 1
EXECUTION_STATUS = "blocked_not_authorized"
ARRAY_BATCH_BY_FOLD = {
    "fold_01": "array_batch_01",
    "fold_02": "array_batch_01",
    "fold_03": "array_batch_02",
    "fold_04": "array_batch_02",
    "fold_05": "array_batch_03",
    "fold_06": "array_batch_03",
}

BINDING_COLUMNS = (
    "protocol_sha256",
    "evaluation_policy_sha256",
    "evaluation_readiness_sha256",
    "candidate_ledger_artifact_sha256",
    "fold_schedule_artifact_sha256",
    "fold_summary_artifact_sha256",
)

EXECUTION_PLAN_COLUMNS = (
    "schema_version",
    "work_unit_id",
    "global_cell_index",
    "array_batch_id",
    "array_task_index",
    "candidate_id",
    "candidate_sha256",
    "algorithm",
    "parameter_profile",
    "timestep_budget",
    "hyperparameters_json",
    "fold_id",
    "fold_schedule_sha256",
    "train_decision_start",
    "train_decision_end",
    "train_decisions",
    "validation_start",
    "validation_decision_end",
    "validation_terminal_outcome",
    "validation_decisions",
    "seed",
    "planned_training_timesteps",
    "requested_roles",
    "execution_status",
    "hpc_submission_authorized",
    *BINDING_COLUMNS,
    "cell_sha256",
)

_INTEGER_COLUMNS = (
    "schema_version",
    "global_cell_index",
    "array_task_index",
    "timestep_budget",
    "train_decisions",
    "validation_decisions",
    "seed",
    "planned_training_timesteps",
)


def _valid_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_bindings(bindings: Mapping[str, Any]) -> dict[str, str]:
    if set(bindings) != set(BINDING_COLUMNS):
        missing = sorted(set(BINDING_COLUMNS) - set(bindings))
        extra = sorted(set(bindings) - set(BINDING_COLUMNS))
        raise ValueError(
            f"execution-plan bindings differ from the exact schema; "
            f"missing={missing}, extra={extra}"
        )
    normalized = {key: str(bindings[key]).lower() for key in BINDING_COLUMNS}
    invalid = [key for key, value in normalized.items() if not _valid_sha256(value)]
    if invalid:
        raise ValueError(f"invalid SHA-256 execution-plan bindings: {invalid}")
    return normalized


def _validate_fold_summary(summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fold_id",
        "train_decision_start",
        "train_decision_end",
        "train_decisions",
        "validation_start",
        "validation_decision_end",
        "validation_terminal_outcome",
        "validation_decisions",
        "boundary_transition_owned",
        "fold_schedule_sha256",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"fold summary is missing columns: {missing}")
    if summary["fold_id"].duplicated().any():
        raise ValueError("fold summary has duplicate fold identifiers")
    indexed = summary.set_index(summary["fold_id"].astype(str), drop=False)
    observed_ids = set(indexed.index)
    if observed_ids != set(FOLD_IDS):
        raise ValueError("fold summary is not the exact six-fold registry")
    ordered = indexed.loc[list(FOLD_IDS)].reset_index(drop=True).copy()

    for column in (
        "train_decision_start",
        "train_decision_end",
        "validation_start",
        "validation_decision_end",
        "validation_terminal_outcome",
    ):
        parsed = pd.to_datetime(ordered[column], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"fold summary contains invalid {column}")
        ordered[column] = parsed.dt.date.astype(str)
    for column in ("train_decisions", "validation_decisions"):
        parsed = pd.to_numeric(ordered[column], errors="coerce")
        if parsed.isna().any() or (parsed <= 0).any() or (parsed % 1 != 0).any():
            raise ValueError(f"fold summary contains invalid {column}")
        ordered[column] = parsed.astype(int)

    boundary = ordered["boundary_transition_owned"]
    if boundary.dtype == object:
        boundary = boundary.map(
            {True: True, False: False, "True": True, "False": False}
        )
    if boundary.isna().any() or boundary.astype(bool).any():
        raise ValueError("every train/validation boundary transition must be unowned")

    for row in ordered.itertuples(index=False):
        train_start = pd.Timestamp(row.train_decision_start)
        train_end = pd.Timestamp(row.train_decision_end)
        validation_start = pd.Timestamp(row.validation_start)
        validation_end = pd.Timestamp(row.validation_decision_end)
        terminal = pd.Timestamp(row.validation_terminal_outcome)
        if not train_start <= train_end < validation_start <= validation_end < terminal:
            raise ValueError(f"{row.fold_id} has invalid chronological boundaries")
        if not _valid_sha256(row.fold_schedule_sha256):
            raise ValueError(f"{row.fold_id} has an invalid schedule SHA-256")
    return ordered


def _cell_hash(row: Mapping[str, Any]) -> str:
    return _canonical_hash({key: row[key] for key in EXECUTION_PLAN_COLUMNS[:-1]})


def build_execution_plan(
    policy: Mapping[str, Any],
    candidates: pd.DataFrame,
    fold_summary: pd.DataFrame,
    *,
    bindings: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the exact 30 x 6 x 3 registry with all cells blocked."""

    candidate_issues = validate_candidate_ledger(candidates, policy)
    if candidate_issues:
        raise ValueError("; ".join(candidate_issues))
    binding_values = _validate_bindings(bindings)
    folds = _validate_fold_summary(fold_summary)
    seeds = tuple(policy["candidate_budget"]["model_seeds"])
    if seeds != MODEL_SEEDS or any(type(seed) is not int for seed in seeds):
        raise ValueError("model seeds differ from the exact registered tuple")

    candidate_rows = candidates.sort_values(
        "candidate_id", kind="mergesort"
    ).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    batch_counts = {batch: 0 for batch in sorted(set(ARRAY_BATCH_BY_FOLD.values()))}
    global_index = 0
    for candidate in candidate_rows.itertuples(index=False):
        for fold in folds.itertuples(index=False):
            batch_id = ARRAY_BATCH_BY_FOLD[str(fold.fold_id)]
            for seed in seeds:
                row: dict[str, Any] = {
                    "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
                    "work_unit_id": (
                        f"v3__{candidate.candidate_id}__{fold.fold_id}__s{seed}"
                    ),
                    "global_cell_index": global_index,
                    "array_batch_id": batch_id,
                    "array_task_index": batch_counts[batch_id],
                    "candidate_id": str(candidate.candidate_id),
                    "candidate_sha256": str(candidate.candidate_sha256),
                    "algorithm": str(candidate.algorithm),
                    "parameter_profile": str(candidate.parameter_profile),
                    "timestep_budget": int(candidate.timestep_budget),
                    "hyperparameters_json": str(candidate.hyperparameters_json),
                    "fold_id": str(fold.fold_id),
                    "fold_schedule_sha256": str(fold.fold_schedule_sha256),
                    "train_decision_start": str(fold.train_decision_start),
                    "train_decision_end": str(fold.train_decision_end),
                    "train_decisions": int(fold.train_decisions),
                    "validation_start": str(fold.validation_start),
                    "validation_decision_end": str(fold.validation_decision_end),
                    "validation_terminal_outcome": str(
                        fold.validation_terminal_outcome
                    ),
                    "validation_decisions": int(fold.validation_decisions),
                    "seed": int(seed),
                    "planned_training_timesteps": int(candidate.timestep_budget),
                    "requested_roles": "train_then_validation",
                    "execution_status": EXECUTION_STATUS,
                    "hpc_submission_authorized": False,
                    **binding_values,
                }
                row["cell_sha256"] = _cell_hash(row)
                rows.append(row)
                batch_counts[batch_id] += 1
                global_index += 1

    plan = pd.DataFrame(rows, columns=EXECUTION_PLAN_COLUMNS)
    expected_cells = int(policy["candidate_budget"]["planned_model_fits"])
    if len(plan) != expected_cells:
        raise ValueError(
            f"execution plan has {len(plan)} cells; expected {expected_cells}"
        )
    expected_timesteps = int(
        policy["candidate_budget"]["planned_training_timesteps"]
    )
    if int(plan["planned_training_timesteps"].sum()) != expected_timesteps:
        raise ValueError("execution plan does not reconcile the registered timestep budget")
    if set(batch_counts.values()) != {180}:
        raise ValueError("each future array batch must contain exactly 180 cells")
    return plan


def _normalize_observed(plan: pd.DataFrame) -> pd.DataFrame:
    if list(plan.columns) != list(EXECUTION_PLAN_COLUMNS):
        raise ValueError("execution plan columns do not match the canonical schema")
    normalized = plan.copy()
    for column in _INTEGER_COLUMNS:
        parsed = pd.to_numeric(normalized[column], errors="coerce")
        if parsed.isna().any() or (parsed % 1 != 0).any():
            raise ValueError(f"execution plan contains invalid integer column {column}")
        normalized[column] = parsed.astype(int)
    authorization = normalized["hpc_submission_authorized"]
    if authorization.dtype == object:
        authorization = authorization.map(
            {True: True, False: False, "True": True, "False": False}
        )
    if authorization.isna().any():
        raise ValueError("execution plan contains invalid HPC authorization flags")
    normalized["hpc_submission_authorized"] = authorization.astype(bool)
    string_columns = [
        column
        for column in EXECUTION_PLAN_COLUMNS
        if column not in _INTEGER_COLUMNS and column != "hpc_submission_authorized"
    ]
    for column in string_columns:
        normalized[column] = normalized[column].astype(str)
    return normalized


def validate_execution_plan(
    plan: pd.DataFrame,
    policy: Mapping[str, Any],
    candidates: pd.DataFrame,
    fold_summary: pd.DataFrame,
    *,
    bindings: Mapping[str, Any],
) -> list[str]:
    """Compare a work registry with a fresh canonical materialization."""

    try:
        observed = _normalize_observed(plan)
        expected = build_execution_plan(
            policy,
            candidates,
            fold_summary,
            bindings=bindings,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [str(exc)]
    issues: list[str] = []
    if observed["work_unit_id"].duplicated().any():
        issues.append("execution plan has duplicate work-unit identifiers")
    if observed["cell_sha256"].duplicated().any():
        issues.append("execution plan has duplicate cell hashes")
    if not observed["execution_status"].eq(EXECUTION_STATUS).all():
        issues.append("every execution-plan cell must remain blocked")
    if observed["hpc_submission_authorized"].any():
        issues.append("the structural execution plan cannot authorize HPC submission")
    if not observed.equals(expected):
        issues.append("execution plan differs from the exact canonical registry")
    return list(dict.fromkeys(issues))


def summarize_execution_plan(plan: pd.DataFrame) -> dict[str, Any]:
    """Return bounded structural counts; never return performance metrics."""

    normalized = _normalize_observed(plan)
    return {
        "work_units": int(len(normalized)),
        "candidates": int(normalized["candidate_id"].nunique()),
        "folds": int(normalized["fold_id"].nunique()),
        "seeds": int(normalized["seed"].nunique()),
        "array_batches": int(normalized["array_batch_id"].nunique()),
        "maximum_cells_per_array_batch": int(
            normalized.groupby("array_batch_id").size().max()
        ),
        "planned_training_timesteps": int(
            normalized["planned_training_timesteps"].sum()
        ),
        "blocked_cells": int(
            normalized["execution_status"].eq(EXECUTION_STATUS).sum()
        ),
        "authorized_hpc_cells": int(
            normalized["hpc_submission_authorized"].sum()
        ),
    }
