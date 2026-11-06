from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v3.evaluation_policy import METRIC_COLUMNS, validate_evaluation_policy
from src.v3.result_evidence import (
    METRIC_DERIVATION_CONTRACT,
    TRANSITION_RESULT_COLUMNS,
    TransitionResultBinding,
    derive_candidate_metric_row,
    derive_transition_summary,
    seal_transition_results,
    validate_transition_results,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config/v3/evaluation_policy_draft.yaml"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding() -> TransitionResultBinding:
    return TransitionResultBinding(
        candidate_id="ppo_stable_default_050k",
        candidate_sha256=_digest("candidate"),
        fold_id="fold_01",
        fold_schedule_sha256=_digest("fold"),
        seed=42,
        dataset_manifest_sha256=_digest("dataset"),
        preprocessing_state_sha256=_digest("preprocessor"),
        model_sha256=_digest("model"),
    )


def _expected_transitions() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    return pd.DataFrame(
        {
            "transition_index": range(4),
            "decision_date": [value.date().isoformat() for value in dates[:-1]],
            "outcome_date": [value.date().isoformat() for value in dates[1:]],
        }
    )


def _chain(initial: float, returns: list[float]) -> tuple[list[float], list[float]]:
    before: list[float] = []
    after: list[float] = []
    current = initial
    for value in returns:
        before.append(current)
        current *= 1.0 + value
        after.append(current)
    return before, after


def _results() -> pd.DataFrame:
    binding = _binding()
    transitions = _expected_transitions()
    gross = [0.011, -0.018, 0.032, 0.004]
    costs = [0.001, 0.001, 0.001, 0.001]
    slippage = [0.0002, 0.0002, 0.0002, 0.0002]
    net = [g - c - s for g, c, s in zip(gross, costs, slippage, strict=True)]
    equal_returns = [0.008, -0.010, 0.015, 0.002]
    set_returns = [0.005, -0.020, 0.010, 0.001]
    before, after = _chain(1.0, net)
    equal_before, equal_after = _chain(1.0, equal_returns)
    set_before, set_after = _chain(1.0, set_returns)
    identity = binding.to_dict()
    rows = []
    for index, transition in transitions.iterrows():
        rows.append(
            {
                "schema_version": 1,
                **identity,
                "transition_index": int(index),
                "decision_date": transition["decision_date"],
                "outcome_date": transition["outcome_date"],
                "portfolio_value_before": before[index],
                "gross_return": gross[index],
                "turnover": [0.4, 0.2, 0.1, 0.3][index],
                "transaction_cost_paid": costs[index],
                "slippage_paid": slippage[index],
                "net_return": net[index],
                "portfolio_value_after": after[index],
                "self_financing_reconciliation_error": 0.0,
                "equal_weight_value_before": equal_before[index],
                "equal_weight_net_return": equal_returns[index],
                "equal_weight_value_after": equal_after[index],
                "set_index_value_before": set_before[index],
                "set_index_net_return": set_returns[index],
                "set_index_value_after": set_after[index],
            }
        )
    return seal_transition_results(
        pd.DataFrame(rows, columns=TRANSITION_RESULT_COLUMNS[:-1])
    )


def test_exact_transition_evidence_reconciles_and_derives_audit_metrics() -> None:
    results = _results()
    transitions = _expected_transitions()
    binding = _binding()

    assert list(results.columns) == list(TRANSITION_RESULT_COLUMNS)
    assert results["result_row_sha256"].nunique() == len(results)
    assert validate_transition_results(results, transitions, binding) == []
    summary = derive_transition_summary(results, transitions, binding)
    assert summary["status"] == "transition_evidence_validated"
    assert summary["validation_transitions"] == 4
    assert np.isclose(summary["mean_turnover"], 0.25)
    assert np.isclose(summary["total_turnover"], 1.0)
    assert np.isclose(
        summary["net_cumulative_return"],
        results["portfolio_value_after"].iloc[-1] - 1.0,
    )
    assert -1.0 <= summary["max_drawdown"] <= 0.0
    assert np.isclose(
        summary["net_log_return"],
        np.log1p(results["net_return"]).sum(),
    )


def test_missing_duplicate_or_foreign_transitions_fail_closed() -> None:
    results = _results()
    transitions = _expected_transitions()
    binding = _binding()

    issues = validate_transition_results(results.iloc[:-1], transitions, binding)
    assert "transition results are not the exact authorized fold ledger" in issues

    duplicate = pd.concat([results.iloc[[0]], results.iloc[[0]], results.iloc[1:]])
    issues = validate_transition_results(duplicate, transitions, binding)
    assert "transition result ledger has duplicate indices" in issues

    foreign = results.copy()
    foreign["model_sha256"] = _digest("another model")
    foreign = seal_transition_results(foreign)
    issues = validate_transition_results(foreign, transitions, binding)
    assert "transition results contain a foreign model_sha256" in issues


def test_tampered_rows_fail_hash_and_rehashed_accounting_tamper_fails_reconciliation() -> None:
    results = _results()
    transitions = _expected_transitions()
    binding = _binding()

    tampered = results.copy()
    tampered.loc[1, "net_return"] += 0.01
    issues = validate_transition_results(tampered, transitions, binding)
    assert "transition result row SHA-256 does not match" in issues
    assert "net returns do not reconcile with gross return and costs" in issues

    rehashed = seal_transition_results(tampered)
    issues = validate_transition_results(rehashed, transitions, binding)
    assert "transition result row SHA-256 does not match" not in issues
    assert "net returns do not reconcile with gross return and costs" in issues
    assert "candidate portfolio value changes do not reconcile with returns" in issues


def test_both_baselines_must_share_the_exact_value_chain() -> None:
    results = _results()
    broken = results.copy()
    broken.loc[2, "equal_weight_value_before"] += 0.1
    broken.loc[2, "set_index_value_after"] += 0.2
    broken = seal_transition_results(broken)
    issues = validate_transition_results(
        broken,
        _expected_transitions(),
        _binding(),
    )
    assert "active equal-weight baseline value chain is discontinuous" in issues
    assert "SET index baseline value changes do not reconcile with returns" in issues


def test_current_policy_refuses_aggregate_metric_conversion_until_definitions_lock() -> None:
    issues, policy = validate_evaluation_policy(POLICY_PATH)
    assert issues == []
    with pytest.raises(ValueError, match="must register the exact Sharpe"):
        derive_candidate_metric_row(
            _results(),
            _expected_transitions(),
            _binding(),
            policy,
        )

    explicit = copy.deepcopy(policy)
    explicit["result_ledger"]["metric_derivation"] = copy.deepcopy(
        METRIC_DERIVATION_CONTRACT
    )
    row = derive_candidate_metric_row(
        _results(),
        _expected_transitions(),
        _binding(),
        explicit,
    )
    assert tuple(row) == METRIC_COLUMNS
    assert row["status"] == "complete"
    assert np.isfinite(list(row.values())[4:]).all()
    assert row["turnover"] == pytest.approx(0.25)
    assert -1.0 <= row["max_drawdown"] <= 0.0
