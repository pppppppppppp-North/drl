"""Transition-level evidence required before V3 aggregate model metrics.

The evaluation-policy module validates an aggregate candidate ledger.  This
module provides the missing evidence layer beneath it: one exact row for every
authorized validation transition, checksum-bound to the candidate, fold data,
preprocessing state, and model.  Aggregate selection metrics are derived only
after the transition ledger reconciles.

The current draft policy intentionally lacks two final metric definitions
(Sharpe and turnover aggregation).  Consequently, it can validate structural
transition evidence but cannot yet produce a selection-ready aggregate row.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


RESULT_EVIDENCE_SCHEMA_VERSION = 1
RESULT_SPLIT = "validation"
METRIC_DERIVATION_CONTRACT = {
    "periods_per_year": 252,
    "sharpe_definition": (
        "annualized_compound_return_divided_by_population_return_volatility"
    ),
    "turnover_aggregation": "mean_security_turnover_per_transition",
    "drawdown_convention": "negative_underwater_return",
}

IDENTITY_COLUMNS = (
    "candidate_id",
    "candidate_sha256",
    "fold_id",
    "fold_schedule_sha256",
    "seed",
    "dataset_manifest_sha256",
    "preprocessing_state_sha256",
    "model_sha256",
    "split",
)

TRANSITION_RESULT_COLUMNS = (
    "schema_version",
    *IDENTITY_COLUMNS,
    "transition_index",
    "decision_date",
    "outcome_date",
    "portfolio_value_before",
    "gross_return",
    "turnover",
    "transaction_cost_paid",
    "slippage_paid",
    "net_return",
    "portfolio_value_after",
    "self_financing_reconciliation_error",
    "equal_weight_value_before",
    "equal_weight_net_return",
    "equal_weight_value_after",
    "set_index_value_before",
    "set_index_net_return",
    "set_index_value_after",
    "result_row_sha256",
)

_INTEGER_COLUMNS = ("schema_version", "seed", "transition_index")
_FLOAT_COLUMNS = (
    "portfolio_value_before",
    "gross_return",
    "turnover",
    "transaction_cost_paid",
    "slippage_paid",
    "net_return",
    "portfolio_value_after",
    "self_financing_reconciliation_error",
    "equal_weight_value_before",
    "equal_weight_net_return",
    "equal_weight_value_after",
    "set_index_value_before",
    "set_index_net_return",
    "set_index_value_after",
)


@dataclass(frozen=True)
class TransitionResultBinding:
    """Expected immutable identity for one candidate/fold/seed result."""

    candidate_id: str
    candidate_sha256: str
    fold_id: str
    fold_schedule_sha256: str
    seed: int
    dataset_manifest_sha256: str
    preprocessing_state_sha256: str
    model_sha256: str
    split: str = RESULT_SPLIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_sha256": self.candidate_sha256,
            "fold_id": self.fold_id,
            "fold_schedule_sha256": self.fold_schedule_sha256,
            "seed": self.seed,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "preprocessing_state_sha256": self.preprocessing_state_sha256,
            "model_sha256": self.model_sha256,
            "split": self.split,
        }


def _valid_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def validate_result_binding(binding: TransitionResultBinding) -> list[str]:
    issues: list[str] = []
    if not isinstance(binding.candidate_id, str) or not binding.candidate_id:
        issues.append("result binding candidate_id must be nonblank")
    if not isinstance(binding.fold_id, str) or not binding.fold_id:
        issues.append("result binding fold_id must be nonblank")
    if type(binding.seed) is not int:
        issues.append("result binding seed must be an integer, not a Boolean")
    if binding.split != RESULT_SPLIT:
        issues.append("selection evidence may contain only validation transitions")
    for field in (
        "candidate_sha256",
        "fold_schedule_sha256",
        "dataset_manifest_sha256",
        "preprocessing_state_sha256",
        "model_sha256",
    ):
        if not _valid_sha256(getattr(binding, field)):
            issues.append(f"result binding {field} is not a valid SHA-256")
    return issues


def _normalize_results(results: pd.DataFrame) -> pd.DataFrame:
    if list(results.columns) != list(TRANSITION_RESULT_COLUMNS):
        raise ValueError("transition result columns do not match the canonical schema")
    normalized = results.copy()
    for column in _INTEGER_COLUMNS:
        values = pd.to_numeric(normalized[column], errors="coerce")
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f"transition results contain invalid integer {column}")
        normalized[column] = values.astype(int)
    for column in _FLOAT_COLUMNS:
        values = pd.to_numeric(normalized[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"transition results contain nonfinite {column}")
        normalized[column] = values.astype(float)
    string_columns = [
        column
        for column in TRANSITION_RESULT_COLUMNS
        if column not in _INTEGER_COLUMNS and column not in _FLOAT_COLUMNS
    ]
    for column in string_columns:
        normalized[column] = normalized[column].astype(str)
    return normalized.sort_values("transition_index", kind="mergesort").reset_index(
        drop=True
    )


def _hash_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("cannot hash nonfinite transition evidence")
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return value


def _result_row_sha256(row: Mapping[str, Any]) -> str:
    payload = {
        column: _hash_value(row[column])
        for column in TRANSITION_RESULT_COLUMNS[:-1]
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_transition_results(results: pd.DataFrame) -> pd.DataFrame:
    """Normalize and hash an unsealed or already shaped result ledger."""

    required_without_hash = list(TRANSITION_RESULT_COLUMNS[:-1])
    if list(results.columns) == required_without_hash:
        work = results.copy()
        work["result_row_sha256"] = ""
    elif list(results.columns) == list(TRANSITION_RESULT_COLUMNS):
        work = results.copy()
    else:
        raise ValueError("unsealed transition results do not match the canonical schema")
    normalized = _normalize_results(work)
    normalized["result_row_sha256"] = [
        _result_row_sha256(row)
        for row in normalized.to_dict(orient="records")
    ]
    return normalized


def _expected_transition_frame(transitions: pd.DataFrame) -> pd.DataFrame:
    required = {"transition_index", "decision_date", "outcome_date"}
    missing = sorted(required - set(transitions.columns))
    if missing:
        raise ValueError(f"expected transition ledger is missing columns: {missing}")
    expected = transitions[
        ["transition_index", "decision_date", "outcome_date"]
    ].copy()
    expected["transition_index"] = pd.to_numeric(
        expected["transition_index"], errors="coerce"
    )
    if expected["transition_index"].isna().any() or (
        expected["transition_index"] % 1 != 0
    ).any():
        raise ValueError("expected transition indices are invalid")
    expected["transition_index"] = expected["transition_index"].astype(int)
    expected["decision_date"] = expected["decision_date"].astype(str)
    expected["outcome_date"] = expected["outcome_date"].astype(str)
    expected = expected.sort_values(
        "transition_index", kind="mergesort"
    ).reset_index(drop=True)
    if expected["transition_index"].tolist() != list(range(len(expected))):
        raise ValueError("expected transition indices must be contiguous from zero")
    if expected.duplicated(["decision_date"]).any():
        raise ValueError("expected transition ledger has duplicate decision dates")
    if expected.duplicated(["outcome_date"]).any():
        raise ValueError("expected transition ledger has duplicate outcome dates")
    return expected


def _close(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.isclose(left, right, atol=1e-10, rtol=1e-12)


def _check_value_chain(
    results: pd.DataFrame,
    *,
    before: str,
    step_return: str,
    after: str,
    label: str,
) -> list[str]:
    issues: list[str] = []
    before_values = results[before].to_numpy(dtype=np.float64)
    returns = results[step_return].to_numpy(dtype=np.float64)
    after_values = results[after].to_numpy(dtype=np.float64)
    if (before_values <= 0.0).any() or (after_values <= 0.0).any():
        issues.append(f"{label} values must remain positive")
    if (returns <= -1.0).any():
        issues.append(f"{label} returns must be greater than -100 percent")
    expected_after = before_values * (1.0 + returns)
    if not _close(after_values, expected_after).all():
        issues.append(f"{label} value changes do not reconcile with returns")
    if len(results) > 1 and not _close(before_values[1:], after_values[:-1]).all():
        issues.append(f"{label} value chain is discontinuous")
    return issues


def validate_transition_results(
    results: pd.DataFrame,
    expected_transitions: pd.DataFrame,
    binding: TransitionResultBinding,
) -> list[str]:
    """Validate exact transition identity, accounting, costs, and baselines."""

    issues = validate_result_binding(binding)
    try:
        normalized = _normalize_results(results)
        expected = _expected_transition_frame(expected_transitions)
    except (KeyError, TypeError, ValueError) as exc:
        return list(dict.fromkeys([*issues, str(exc)]))
    if normalized.empty:
        issues.append("transition result ledger must not be empty")
        return list(dict.fromkeys(issues))
    if normalized["transition_index"].duplicated().any():
        issues.append("transition result ledger has duplicate indices")
    if normalized["transition_index"].tolist() != list(range(len(normalized))):
        issues.append("transition result indices must be contiguous from zero")
    observed_transitions = normalized[
        ["transition_index", "decision_date", "outcome_date"]
    ]
    if not observed_transitions.equals(expected):
        issues.append("transition results are not the exact authorized fold ledger")

    expected_identity = binding.to_dict()
    for column in IDENTITY_COLUMNS:
        expected_value = expected_identity[column]
        observed = normalized[column]
        if column == "seed":
            matches = observed.eq(int(expected_value))
        else:
            matches = observed.eq(str(expected_value))
        if not matches.all():
            issues.append(f"transition results contain a foreign {column}")
    if not normalized["schema_version"].eq(RESULT_EVIDENCE_SCHEMA_VERSION).all():
        issues.append("transition result schema version is unsupported")

    expected_hashes = [
        _result_row_sha256(row)
        for row in normalized.to_dict(orient="records")
    ]
    if normalized["result_row_sha256"].tolist() != expected_hashes:
        issues.append("transition result row SHA-256 does not match")

    if (normalized["turnover"] < 0.0).any():
        issues.append("transition turnover must be nonnegative")
    if (normalized["transaction_cost_paid"] < 0.0).any():
        issues.append("transaction costs must be nonnegative")
    if (normalized["slippage_paid"] < 0.0).any():
        issues.append("slippage must be nonnegative")
    expected_net = (
        normalized["gross_return"]
        - normalized["transaction_cost_paid"]
        - normalized["slippage_paid"]
    ).to_numpy(dtype=np.float64)
    if not _close(
        normalized["net_return"].to_numpy(dtype=np.float64),
        expected_net,
    ).all():
        issues.append("net returns do not reconcile with gross return and costs")
    if (
        np.abs(
            normalized["self_financing_reconciliation_error"].to_numpy(
                dtype=np.float64
            )
        )
        > 1e-10
    ).any():
        issues.append("self-financing reconciliation error exceeds tolerance")

    issues.extend(
        _check_value_chain(
            normalized,
            before="portfolio_value_before",
            step_return="net_return",
            after="portfolio_value_after",
            label="candidate portfolio",
        )
    )
    issues.extend(
        _check_value_chain(
            normalized,
            before="equal_weight_value_before",
            step_return="equal_weight_net_return",
            after="equal_weight_value_after",
            label="active equal-weight baseline",
        )
    )
    issues.extend(
        _check_value_chain(
            normalized,
            before="set_index_value_before",
            step_return="set_index_net_return",
            after="set_index_value_after",
            label="SET index baseline",
        )
    )
    return list(dict.fromkeys(issues))


def _max_drawdown(initial: float, ending_values: pd.Series) -> float:
    values = np.concatenate(
        ([float(initial)], ending_values.to_numpy(dtype=np.float64))
    )
    running_peak = np.maximum.accumulate(values)
    return float(np.min(values / running_peak - 1.0))


def derive_transition_summary(
    results: pd.DataFrame,
    expected_transitions: pd.DataFrame,
    binding: TransitionResultBinding,
) -> dict[str, Any]:
    """Derive audit metrics that do not depend on unresolved policy choices."""

    issues = validate_transition_results(results, expected_transitions, binding)
    if issues:
        raise ValueError("; ".join(issues))
    values = _normalize_results(results)
    candidate_returns = values["net_return"].to_numpy(dtype=np.float64)
    equal_weight_returns = values["equal_weight_net_return"].to_numpy(
        dtype=np.float64
    )
    set_returns = values["set_index_net_return"].to_numpy(dtype=np.float64)
    return {
        **binding.to_dict(),
        "status": "transition_evidence_validated",
        "validation_transitions": int(len(values)),
        "net_cumulative_return": float(
            values["portfolio_value_after"].iloc[-1]
            / values["portfolio_value_before"].iloc[0]
            - 1.0
        ),
        "net_log_return": float(np.log1p(candidate_returns).sum()),
        "max_drawdown": _max_drawdown(
            values["portfolio_value_before"].iloc[0],
            values["portfolio_value_after"],
        ),
        "mean_turnover": float(values["turnover"].mean()),
        "total_turnover": float(values["turnover"].sum()),
        "total_transaction_cost_fraction": float(
            values["transaction_cost_paid"].sum()
        ),
        "total_slippage_fraction": float(values["slippage_paid"].sum()),
        "equal_weight_net_log_return": float(np.log1p(equal_weight_returns).sum()),
        "equal_weight_max_drawdown": _max_drawdown(
            values["equal_weight_value_before"].iloc[0],
            values["equal_weight_value_after"],
        ),
        "set_index_net_log_return": float(np.log1p(set_returns).sum()),
    }


def _require_metric_derivation(policy: Mapping[str, Any]) -> None:
    result_ledger = policy.get("result_ledger")
    if not isinstance(result_ledger, Mapping):
        raise ValueError("evaluation policy result_ledger must be a mapping")
    observed = result_ledger.get("metric_derivation")
    if observed != METRIC_DERIVATION_CONTRACT:
        raise ValueError(
            "evaluation policy must register the exact Sharpe, turnover, and "
            "drawdown derivation contract before aggregate metrics are eligible"
        )


def derive_candidate_metric_row(
    results: pd.DataFrame,
    expected_transitions: pd.DataFrame,
    binding: TransitionResultBinding,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Create one selection-ledger row only under an explicit metric contract."""

    _require_metric_derivation(policy)
    summary = derive_transition_summary(results, expected_transitions, binding)
    values = _normalize_results(results)
    periods = int(METRIC_DERIVATION_CONTRACT["periods_per_year"])
    count = len(values)
    cumulative = float(summary["net_cumulative_return"])
    annualized_return = (1.0 + cumulative) ** (periods / count) - 1.0
    volatility = float(values["net_return"].std(ddof=0)) * np.sqrt(periods)
    sharpe = annualized_return / volatility if volatility > 0.0 else 0.0
    return {
        "candidate_id": binding.candidate_id,
        "fold_id": binding.fold_id,
        "seed": binding.seed,
        "status": "complete",
        "net_cumulative_return": cumulative,
        "net_log_return": float(summary["net_log_return"]),
        "sharpe": float(sharpe),
        "max_drawdown": float(summary["max_drawdown"]),
        "turnover": float(summary["mean_turnover"]),
        "equal_weight_net_log_return": float(
            summary["equal_weight_net_log_return"]
        ),
        "equal_weight_max_drawdown": float(
            summary["equal_weight_max_drawdown"]
        ),
        "set_index_net_log_return": float(summary["set_index_net_log_return"]),
    }
