"""Fail-closed V3 temporal evaluation and model-selection primitives.

This module is isolated from the shared V2 splitter.  It materializes explicit
session roles, owns each close-to-close transition, freezes a finite candidate
grid, and refuses incomplete selection evidence.  It does not authorize V3
training or holdout access.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


POLICY_SCHEMA_VERSION = 1
POLICY_ID = "thai_drl_v3_evaluation_selection"
POLICY_STATUSES = {"draft_not_approved", "locked"}
ALGORITHMS = ("ppo", "a2c", "ddpg", "td3", "sac")
PROFILES = ("stable_default", "low_lr_regularized")
TIMESTEP_BUDGETS = (50_000, 100_000, 200_000)
MODEL_SEEDS = (42, 43, 44)
FOLD_IDS = tuple(f"fold_{index:02d}" for index in range(1, 7))

SCHEDULE_COLUMNS = (
    "fold_id",
    "date",
    "global_session_index",
    "train_context",
    "train_decision",
    "train_outcome",
    "validation_context",
    "validation_decision",
    "validation_outcome",
    "preprocessing_fit_allowed",
    "model_update_allowed",
    "validation_scoring_allowed",
    "unused_boundary_transition_start",
    "transition_owner",
    "transition_outcome_date",
    "holdout_access_allowed",
)

CANDIDATE_COLUMNS = (
    "candidate_id",
    "algorithm",
    "parameter_profile",
    "timestep_budget",
    "design_seed",
    "hyperparameters_json",
    "candidate_sha256",
    "execution_status",
)

METRIC_KEY_COLUMNS = ("candidate_id", "fold_id", "seed")
METRIC_COLUMNS = (
    *METRIC_KEY_COLUMNS,
    "status",
    "net_cumulative_return",
    "net_log_return",
    "sharpe",
    "max_drawdown",
    "turnover",
    "equal_weight_net_log_return",
    "equal_weight_max_drawdown",
    "set_index_net_log_return",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str, issues: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        issues.append(f"{label} must be a mapping")
        return {}
    return value


def _require_exact(
    mapping: Mapping[str, Any],
    key: str,
    expected: Any,
    *,
    label: str,
    issues: list[str],
) -> None:
    if mapping.get(key) != expected:
        issues.append(f"{label}.{key} must be {expected!r}")


def validate_evaluation_policy(
    path: Path,
    *,
    require_locked: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Validate the complete V3 evaluation contract without unlocking it."""

    policy_path = Path(path)
    if not policy_path.is_file():
        return [f"V3 evaluation policy is missing: {policy_path}"], {}
    try:
        loaded = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot read V3 evaluation policy: {exc}"], {}
    issues: list[str] = []
    policy = dict(_mapping(loaded, "policy", issues))
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        issues.append(f"policy schema_version must be {POLICY_SCHEMA_VERSION}")

    registration = _mapping(policy.get("registration"), "registration", issues)
    _require_exact(registration, "id", POLICY_ID, label="registration", issues=issues)
    status = registration.get("status")
    if status not in POLICY_STATUSES:
        issues.append("registration.status must be draft_not_approved or locked")
    _require_exact(
        registration,
        "replaces_v2",
        False,
        label="registration",
        issues=issues,
    )

    boundary = _mapping(
        policy.get("scientific_boundary"),
        "scientific_boundary",
        issues,
    )
    for key, expected in {
        "development_only": True,
        "training_enabled": False,
        "publication_claim_enabled": False,
        "known_2021_2024_period_usable_as_new_final_holdout": False,
        "allowed_splits": ["train", "validation"],
        "forbidden_splits": ["test", "all"],
    }.items():
        _require_exact(
            boundary,
            key,
            expected,
            label="scientific_boundary",
            issues=issues,
        )

    session = _mapping(policy.get("session_contract"), "session_contract", issues)
    for key, expected in {
        "timezone": "Asia/Bangkok",
        "required_final_source": "licensed_authoritative_set_session_calendar",
        "current_membership_session_role": "provisional_schedule_geometry_only",
        "input_order_rule": "sort_then_validate",
        "duplicate_session_rule": "reject",
        "invalid_or_intraday_session_rule": "reject",
        "future_suffix_rule": "fixed_fold_count_preserves_existing_schedule",
    }.items():
        _require_exact(session, key, expected, label="session_contract", issues=issues)

    folds = _mapping(policy.get("fold_geometry"), "fold_geometry", issues)
    for key, expected in {
        "kind": "expanding_half_year_validation",
        "fold_count": 6,
        "initial_train_context_sessions": 30,
        "validation_context_sessions": 30,
        "transition_horizon_sessions": 1,
        "expanding_train": True,
        "minimum_train_decision_sessions": 200,
        "unused_boundary_rule": (
            "train_end_to_validation_start_transition_is_unassigned"
        ),
        "cross_fold_reuse_rule": (
            "permitted_only_in_a_later_chronological_fold"
        ),
    }.items():
        _require_exact(folds, key, expected, label="fold_geometry", issues=issues)
    periods = folds.get("validation_periods")
    expected_periods = [
        {
            "fold_id": fold_id,
            "start_on_or_after": f"{year}-{month:02d}-01",
            "end_before": f"{end_year}-{end_month:02d}-01",
        }
        for fold_id, year, month, end_year, end_month in (
            ("fold_01", 2022, 1, 2022, 7),
            ("fold_02", 2022, 7, 2023, 1),
            ("fold_03", 2023, 1, 2023, 7),
            ("fold_04", 2023, 7, 2024, 1),
            ("fold_05", 2024, 1, 2024, 7),
            ("fold_06", 2024, 7, 2025, 1),
        )
    ]
    if periods != expected_periods:
        issues.append("fold_geometry.validation_periods must be the six fixed half-years")

    ownership = _mapping(
        policy.get("transition_ownership"),
        "transition_ownership",
        issues,
    )
    for key, expected in {
        "decision_and_outcome_must_share_scored_split": True,
        "cross_boundary_transition": "forbidden",
        "terminal_outcome_session_required": True,
        "validation_context_triggers_new_fit": False,
        "portfolio_and_model_state_reset_at_validation_start": True,
    }.items():
        _require_exact(
            ownership,
            key,
            expected,
            label="transition_ownership",
            issues=issues,
        )

    context = _mapping(policy.get("context_only"), "context_only", issues)
    for key, expected in {
        "scope": "per_fold_and_role",
        "may_supply_lagged_features": True,
        "may_be_decision_or_outcome_target": False,
        "preprocessing_fit_allowed": False,
        "scoring_allowed": False,
        "model_update_allowed": False,
        "validation_context_may_overlap_prior_training_role": True,
    }.items():
        _require_exact(context, key, expected, label="context_only", issues=issues)

    preprocessing = _mapping(
        policy.get("preprocessing"),
        "preprocessing",
        issues,
    )
    for key, expected in {
        "fit_scope": "train_decision_dates_only",
        "fit_separately_per_fold": True,
        "validation_fit_allowed": False,
        "context_fit_allowed": False,
        "reuse_fitted_state_across_folds": False,
        "outcome_derived_feature_fit_allowed": False,
    }.items():
        _require_exact(
            preprocessing,
            key,
            expected,
            label="preprocessing",
            issues=issues,
        )

    budget = _mapping(policy.get("candidate_budget"), "candidate_budget", issues)
    for key, expected in {
        "algorithms": list(ALGORITHMS),
        "parameter_profiles": list(PROFILES),
        "timestep_budgets": list(TIMESTEP_BUDGETS),
        "configurations_per_algorithm": 6,
        "total_candidate_configurations": 30,
        "candidate_design": "exact_profile_by_budget_grid_v1",
        "candidate_design_seed": 20260820,
        "model_seeds": list(MODEL_SEEDS),
        "planned_model_fits": 540,
        "planned_training_timesteps": 63_000_000,
        "maximum_hpc_array_jobs": 3,
        "maximum_concurrent_array_jobs": 2,
        "array_job_mapping": "two_folds_per_array_job",
        "adaptive_candidate_expansion": "forbidden",
        "unregistered_retries_count_as_new_candidates": True,
    }.items():
        _require_exact(budget, key, expected, label="candidate_budget", issues=issues)

    selection = _mapping(policy.get("selection"), "selection", issues)
    for key, expected in {
        "selection_unit": "candidate_configuration",
        "required_fold_seed_cells_per_candidate": 18,
        "seed_aggregation_within_fold": "arithmetic_mean",
        "primary_metric": (
            "median_fold_annualized_excess_net_log_return_vs_active_equal_weight"
        ),
        "primary_direction": "maximize",
        "net_of_transaction_cost_and_slippage": True,
        "deterministic_tie_breakers": [
            "worst_fold_mean_annualized_excess_net_log_return_maximize",
            "median_validation_excess_max_drawdown_maximize",
            "median_validation_turnover_minimize",
            "smaller_timestep_budget_preferred",
            "candidate_id_lexicographic_ascending",
        ],
        "numeric_tie_rounding_decimals": 12,
        "incomplete_or_nonfinite_candidate": "reject",
        "selection_adaptation_between_folds": "forbidden",
        "seed_runs_are_independent_market_periods": False,
        "nonpositive_best_primary_result": "select_active_equal_weight_no_drl",
        "required_baselines": ["active_equal_weight", "set_index_buy_and_hold"],
        "baseline_dates": "identical_validation_transitions",
    }.items():
        _require_exact(selection, key, expected, label="selection", issues=issues)

    result = _mapping(policy.get("result_ledger"), "result_ledger", issues)
    for key, expected in {
        "exact_cartesian_product_required": True,
        "required_numeric_fields": [
            "net_cumulative_return",
            "net_log_return",
            "sharpe",
            "max_drawdown",
            "turnover",
            "equal_weight_net_log_return",
            "equal_weight_max_drawdown",
            "set_index_net_log_return",
        ],
        "maximum_drawdown_range": [-1.0, 0.0],
        "negative_turnover_allowed": False,
    }.items():
        _require_exact(result, key, expected, label="result_ledger", issues=issues)

    holdout = _mapping(policy.get("final_holdout"), "final_holdout", issues)
    for key, expected in {
        "status": "absent",
        "required_start_on_or_after": "2025-01-01",
        "test_access_enabled": False,
        "all_access_enabled": False,
        "changed_policy_requires_new_untouched_holdout": True,
    }.items():
        _require_exact(holdout, key, expected, label="final_holdout", issues=issues)

    decisions = policy.get("author_decisions_required_before_lock")
    if not isinstance(decisions, list) or not all(
        isinstance(value, str) and value.strip() for value in decisions
    ):
        issues.append("author_decisions_required_before_lock must be a string list")
        decisions = []
    if status == "draft_not_approved" and not decisions:
        issues.append("draft evaluation policy must disclose unresolved decisions")

    if require_locked:
        if status != "locked":
            issues.append("V3 evaluation policy is not locked")
        if registration.get("approved_by_authors") is not True:
            issues.append("V3 evaluation policy is not approved by both authors")
        if decisions:
            issues.append("locked evaluation policy cannot retain unresolved decisions")
        sidecar = Path(f"{policy_path}.sha256")
        if not sidecar.is_file():
            issues.append("locked V3 evaluation policy sidecar is missing")
        else:
            tokens = sidecar.read_text(encoding="utf-8").split()
            if not tokens or tokens[0].lower() != sha256_file(policy_path):
                issues.append("locked V3 evaluation policy sidecar does not match")

    return list(dict.fromkeys(issues)), policy


def normalize_session_calendar(values: Any) -> pd.DatetimeIndex:
    """Return unique ordered midnight sessions, rejecting ambiguous input."""

    if isinstance(values, pd.DataFrame):
        if set(values.columns) != {"date"}:
            raise ValueError("session frame must contain exactly one date column")
        raw = values["date"]
    elif isinstance(values, (pd.Series, pd.Index, Sequence)) and not isinstance(
        values, (str, bytes)
    ):
        raw = list(values)
    else:
        raise ValueError("sessions must be a date sequence or one-column frame")
    if len(raw) == 0:
        raise ValueError("session calendar must not be empty")
    parsed = pd.to_datetime(raw, errors="coerce")
    index = pd.DatetimeIndex(parsed)
    if index.isna().any():
        raise ValueError("session calendar contains invalid dates")
    if index.tz is not None:
        raise ValueError("session calendar must use timezone-free exchange dates")
    if not index.equals(index.normalize()):
        raise ValueError("session calendar contains intraday timestamps")
    if index.duplicated().any():
        raise ValueError("session calendar contains duplicate dates")
    return index.sort_values()


def _hash_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_evaluation_schedule(
    sessions: Any,
    policy: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize explicit per-fold session roles and transition ownership."""

    dates = normalize_session_calendar(sessions)
    folds = policy["fold_geometry"]
    initial_context_count = int(folds["initial_train_context_sessions"])
    validation_context_count = int(folds["validation_context_sessions"])
    minimum_train = int(folds["minimum_train_decision_sessions"])
    global_index = {date: index for index, date in enumerate(dates)}
    schedule_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    validation_decision_sets: list[set[pd.Timestamp]] = []

    for period in folds["validation_periods"]:
        fold_id = str(period["fold_id"])
        start = pd.Timestamp(period["start_on_or_after"])
        end = pd.Timestamp(period["end_before"])
        validation = dates[(dates >= start) & (dates < end)]
        if len(validation) < 2:
            raise ValueError(f"{fold_id} needs at least two validation sessions")
        training = dates[dates < validation[0]]
        if len(training) < initial_context_count + minimum_train + 1:
            raise ValueError(f"{fold_id} has insufficient training sessions")

        train_context = set(training[:initial_context_count])
        train_decisions = set(training[initial_context_count:-1])
        train_outcomes = set(training[initial_context_count + 1 :])
        validation_context = set(training[-validation_context_count:])
        validation_decisions = set(validation[:-1])
        validation_outcomes = set(validation[1:])
        if len(validation_context) != validation_context_count:
            raise ValueError(f"{fold_id} lacks complete validation context")
        if len(train_decisions) < minimum_train:
            raise ValueError(f"{fold_id} has too few training decisions")
        if any(validation_decisions & prior for prior in validation_decision_sets):
            raise ValueError("validation decision sets must be globally disjoint")
        validation_decision_sets.append(validation_decisions)

        fold_dates = dates[dates <= validation[-1]]
        rows: list[dict[str, Any]] = []
        train_next = {
            training[index]: training[index + 1]
            for index in range(initial_context_count, len(training) - 1)
        }
        validation_next = {
            validation[index]: validation[index + 1]
            for index in range(len(validation) - 1)
        }
        for date in fold_dates:
            is_train_decision = date in train_decisions
            is_validation_decision = date in validation_decisions
            owner = (
                "train"
                if is_train_decision
                else "validation"
                if is_validation_decision
                else ""
            )
            outcome = train_next.get(date)
            if outcome is None:
                outcome = validation_next.get(date)
            rows.append(
                {
                    "fold_id": fold_id,
                    "date": date.date().isoformat(),
                    "global_session_index": int(global_index[date]),
                    "train_context": date in train_context,
                    "train_decision": is_train_decision,
                    "train_outcome": date in train_outcomes,
                    "validation_context": date in validation_context,
                    "validation_decision": is_validation_decision,
                    "validation_outcome": date in validation_outcomes,
                    "preprocessing_fit_allowed": is_train_decision,
                    "model_update_allowed": is_train_decision,
                    "validation_scoring_allowed": is_validation_decision,
                    "unused_boundary_transition_start": date == training[-1],
                    "transition_owner": owner,
                    "transition_outcome_date": (
                        outcome.date().isoformat() if outcome is not None else ""
                    ),
                    "holdout_access_allowed": False,
                }
            )
        fold_frame = pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
        schedule_parts.append(fold_frame)
        summaries.append(
            {
                "fold_id": fold_id,
                "train_context_start": training[0].date().isoformat(),
                "train_context_end": training[initial_context_count - 1]
                .date()
                .isoformat(),
                "train_decision_start": min(train_decisions).date().isoformat(),
                "train_decision_end": max(train_decisions).date().isoformat(),
                "train_decisions": len(train_decisions),
                "train_terminal_outcome": training[-1].date().isoformat(),
                "validation_context_start": training[-validation_context_count]
                .date()
                .isoformat(),
                "validation_context_end": training[-1].date().isoformat(),
                "validation_start": validation[0].date().isoformat(),
                "validation_decision_end": validation[-2].date().isoformat(),
                "validation_decisions": len(validation_decisions),
                "validation_terminal_outcome": validation[-1].date().isoformat(),
                "boundary_transition_owned": False,
                "fold_schedule_sha256": _hash_frame(fold_frame),
            }
        )

    schedule = pd.concat(schedule_parts, ignore_index=True)
    summary = pd.DataFrame(summaries)
    if tuple(summary["fold_id"]) != FOLD_IDS:
        raise ValueError("fold identifiers changed")
    return schedule, summary


def validate_fold_schedule(
    schedule: pd.DataFrame,
    sessions: Any,
    policy: Mapping[str, Any],
) -> list[str]:
    """Reject schedule drift by comparing with a fresh canonical materialization."""

    issues: list[str] = []
    if list(schedule.columns) != list(SCHEDULE_COLUMNS):
        return ["fold schedule columns do not match the canonical schema"]
    expected, _ = build_evaluation_schedule(sessions, policy)
    observed = schedule.copy()
    for column in SCHEDULE_COLUMNS:
        if column in {
            "train_context",
            "train_decision",
            "train_outcome",
            "validation_context",
            "validation_decision",
            "validation_outcome",
            "preprocessing_fit_allowed",
            "model_update_allowed",
            "validation_scoring_allowed",
            "unused_boundary_transition_start",
            "holdout_access_allowed",
        }:
            observed[column] = observed[column].astype(bool)
    if not observed.equals(expected):
        issues.append("fold schedule differs from the canonical policy materialization")
    return issues


def _profile_parameters(algorithm: str, profile: str) -> dict[str, Any]:
    if algorithm == "ppo":
        return (
            {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
                "n_steps": 2048,
                "batch_size": 64,
                "clip_range": 0.2,
            }
            if profile == "stable_default"
            else {
                "learning_rate": 0.0001,
                "gamma": 0.995,
                "gae_lambda": 0.9,
                "ent_coef": 0.001,
                "n_steps": 1024,
                "batch_size": 128,
                "clip_range": 0.15,
            }
        )
    if algorithm == "a2c":
        return (
            {
                "learning_rate": 0.0007,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.0,
                "n_steps": 5,
            }
            if profile == "stable_default"
            else {
                "learning_rate": 0.0003,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "ent_coef": 0.001,
                "n_steps": 20,
            }
        )
    common = (
        {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 256,
            "train_freq": 1,
            "gradient_steps": 1,
        }
        if profile == "stable_default"
        else {
            "learning_rate": 0.0003,
            "gamma": 0.995,
            "tau": 0.01,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 128,
            "train_freq": 4,
            "gradient_steps": 4,
        }
    )
    if algorithm == "td3":
        common = {**common, "policy_delay": 2}
    if algorithm == "sac":
        common = {**common, "ent_coef": "auto"}
    return common


def build_candidate_ledger(policy: Mapping[str, Any]) -> pd.DataFrame:
    """Materialize the exact finite candidate registry before any training."""

    budget = policy["candidate_budget"]
    rows: list[dict[str, Any]] = []
    for algorithm in budget["algorithms"]:
        for profile in budget["parameter_profiles"]:
            params = _profile_parameters(str(algorithm), str(profile))
            params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
            for timesteps in budget["timestep_budgets"]:
                candidate_id = (
                    f"{algorithm}_{profile}_{int(timesteps) // 1000:03d}k"
                )
                canonical = {
                    "algorithm": algorithm,
                    "candidate_id": candidate_id,
                    "design_seed": int(budget["candidate_design_seed"]),
                    "hyperparameters": params,
                    "parameter_profile": profile,
                    "timestep_budget": int(timesteps),
                }
                digest = hashlib.sha256(
                    json.dumps(
                        canonical,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "algorithm": algorithm,
                        "parameter_profile": profile,
                        "timestep_budget": int(timesteps),
                        "design_seed": int(budget["candidate_design_seed"]),
                        "hyperparameters_json": params_json,
                        "candidate_sha256": digest,
                        "execution_status": "draft_not_executable",
                    }
                )
    frame = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    return frame.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)


def validate_candidate_ledger(
    ledger: pd.DataFrame,
    policy: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    if list(ledger.columns) != list(CANDIDATE_COLUMNS):
        return ["candidate ledger columns do not match the canonical schema"]
    expected = build_candidate_ledger(policy)
    observed = ledger.copy().sort_values("candidate_id", kind="mergesort").reset_index(
        drop=True
    )
    for column in ("timestep_budget", "design_seed"):
        observed[column] = pd.to_numeric(observed[column], errors="coerce")
    if not observed.equals(expected):
        issues.append("candidate ledger differs from the exact preregistered grid")
    return issues


def validate_candidate_metric_ledger(
    metrics: pd.DataFrame,
    policy: Mapping[str, Any],
    candidates: pd.DataFrame,
) -> list[str]:
    """Require the exact candidate × fold × seed result grid."""

    issues: list[str] = []
    missing_columns = sorted(set(METRIC_COLUMNS) - set(metrics.columns))
    if missing_columns:
        return [f"candidate metric ledger is missing columns: {missing_columns}"]
    if metrics.duplicated(list(METRIC_KEY_COLUMNS)).any():
        issues.append("candidate metric ledger has duplicate candidate/fold/seed keys")
    expected_keys = {
        (candidate_id, fold_id, seed)
        for candidate_id in candidates["candidate_id"].astype(str)
        for fold_id in FOLD_IDS
        for seed in policy["candidate_budget"]["model_seeds"]
    }
    observed_keys = set(
        metrics[list(METRIC_KEY_COLUMNS)].itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        issues.append("candidate metric ledger is not the exact preregistered grid")
    if not metrics["status"].eq("complete").all():
        issues.append("every preregistered fold/seed cell must be complete")
    numeric_columns = list(policy["result_ledger"]["required_numeric_fields"])
    numeric = metrics[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        issues.append("candidate metric ledger contains nonfinite required metrics")
        return list(dict.fromkeys(issues))
    cumulative = numeric["net_cumulative_return"].to_numpy(dtype=np.float64)
    if (cumulative <= -1.0).any():
        issues.append("net cumulative return must be greater than -100 percent")
    else:
        expected_log = np.log1p(cumulative)
        if not np.allclose(
            expected_log,
            numeric["net_log_return"].to_numpy(dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        ):
            issues.append("net log return does not reconcile with cumulative return")
    drawdown = numeric["max_drawdown"].to_numpy(dtype=np.float64)
    baseline_drawdown = numeric["equal_weight_max_drawdown"].to_numpy(
        dtype=np.float64
    )
    if ((drawdown < -1.0) | (drawdown > 0.0)).any() or (
        (baseline_drawdown < -1.0) | (baseline_drawdown > 0.0)
    ).any():
        issues.append("maximum drawdown must remain inside [-1, 0]")
    if (numeric["turnover"] < 0.0).any():
        issues.append("turnover must be nonnegative")
    return list(dict.fromkeys(issues))


def rank_candidates(
    metrics: pd.DataFrame,
    policy: Mapping[str, Any],
    candidates: pd.DataFrame,
    fold_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the frozen seed-within-fold aggregation and tie hierarchy."""

    issues = validate_candidate_metric_ledger(metrics, policy, candidates)
    if issues:
        raise ValueError("; ".join(issues))
    counts = dict(
        zip(
            fold_summary["fold_id"].astype(str),
            fold_summary["validation_decisions"].astype(int),
            strict=True,
        )
    )
    values = metrics.copy()
    values["validation_decisions"] = values["fold_id"].map(counts)
    if values["validation_decisions"].isna().any():
        raise ValueError("fold summary is missing a required validation count")
    values["annualized_excess_net_log_return"] = (
        252.0
        / values["validation_decisions"].astype(float)
        * (
            values["net_log_return"].astype(float)
            - values["equal_weight_net_log_return"].astype(float)
        )
    )
    values["excess_max_drawdown"] = (
        values["max_drawdown"].astype(float)
        - values["equal_weight_max_drawdown"].astype(float)
    )
    fold_values = (
        values.groupby(["candidate_id", "fold_id"], as_index=False)
        .agg(
            fold_mean_annualized_excess_net_log_return=(
                "annualized_excess_net_log_return",
                "mean",
            ),
            fold_mean_excess_max_drawdown=("excess_max_drawdown", "mean"),
            fold_mean_turnover=("turnover", "mean"),
        )
        .sort_values(["candidate_id", "fold_id"], kind="mergesort")
    )
    ranking = (
        fold_values.groupby("candidate_id", as_index=False)
        .agg(
            primary_score=(
                "fold_mean_annualized_excess_net_log_return",
                "median",
            ),
            worst_fold_score=(
                "fold_mean_annualized_excess_net_log_return",
                "min",
            ),
            median_excess_max_drawdown=(
                "fold_mean_excess_max_drawdown",
                "median",
            ),
            median_turnover=("fold_mean_turnover", "median"),
        )
        .merge(
            candidates[["candidate_id", "algorithm", "timestep_budget"]],
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
    )
    decimals = int(policy["selection"]["numeric_tie_rounding_decimals"])
    for column in (
        "primary_score",
        "worst_fold_score",
        "median_excess_max_drawdown",
        "median_turnover",
    ):
        ranking[column] = ranking[column].round(decimals)
    ranking = ranking.sort_values(
        [
            "primary_score",
            "worst_fold_score",
            "median_excess_max_drawdown",
            "median_turnover",
            "timestep_budget",
            "candidate_id",
        ],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=int))
    best = ranking.iloc[0]
    use_drl = float(best["primary_score"]) > 0.0
    decision = {
        "selected_policy": (
            str(best["candidate_id"]) if use_drl else "active_equal_weight_no_drl"
        ),
        "drl_selected": use_drl,
        "best_candidate_id": str(best["candidate_id"]),
        "best_primary_score": float(best["primary_score"]),
        "selection_cells": int(len(values)),
    }
    return ranking, decision
