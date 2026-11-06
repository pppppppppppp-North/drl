from __future__ import annotations

import numpy as np
import pytest

from src.v3.execution_ledger import (
    execute_self_financing_step,
    security_turnover,
)


def _execute(**overrides):
    arguments = {
        "current_weights": np.array([0.0, 0.0]),
        "current_cash_weight": 1.0,
        "desired_weights": np.array([0.5, 0.5]),
        "asset_returns": np.array([0.0, 0.0]),
        "buy_allowed": np.array([True, True]),
        "sell_allowed": np.array([True, True]),
        "forced_exit_required": np.array([False, False]),
        "transaction_cost_rate": 0.0,
        "slippage_rate": 0.0,
        "max_turnover": None,
    }
    arguments.update(overrides)
    return execute_self_financing_step(**arguments)


def test_security_turnover_excludes_cash_settlement_leg() -> None:
    assert security_turnover(
        np.array([0.0, 0.0]),
        np.array([0.5, 0.0]),
    ) == pytest.approx(0.5)
    assert security_turnover(
        np.array([0.5, 0.0]),
        np.array([0.0, 0.5]),
    ) == pytest.approx(1.0)


def test_costs_are_funded_without_negative_cash() -> None:
    result = _execute(
        desired_weights=np.array([1.0, 0.0]),
        transaction_cost_rate=0.01,
    )

    assert result.turnover == pytest.approx(1.0 / 1.01)
    assert result.transaction_cost_paid == pytest.approx(0.01 / 1.01)
    assert result.pre_return_cash_fraction == pytest.approx(0.0)
    assert result.executed_target_weights.sum() == pytest.approx(1.0 / 1.01)
    assert result.ending_weights.sum() + result.ending_cash_weight == pytest.approx(
        1.0
    )


def test_cross_sectional_returns_drift_ending_weights() -> None:
    result = _execute(asset_returns=np.array([1.0, 0.0]))

    assert result.gross_return == pytest.approx(0.5)
    assert result.net_return == pytest.approx(0.5)
    assert result.ending_weights.tolist() == pytest.approx(
        [2.0 / 3.0, 1.0 / 3.0]
    )
    assert result.ending_cash_weight == pytest.approx(0.0)
    assert result.reconciliation_error == pytest.approx(0.0, abs=1e-12)


def test_untradeable_forced_exit_freezes_holding_without_cost() -> None:
    result = _execute(
        current_weights=np.array([0.5, 0.0]),
        current_cash_weight=0.5,
        desired_weights=np.array([0.0, 0.0]),
        buy_allowed=np.array([False, True]),
        sell_allowed=np.array([False, True]),
        forced_exit_required=np.array([True, False]),
        transaction_cost_rate=0.01,
    )

    assert result.executed_target_weights.tolist() == pytest.approx([0.5, 0.0])
    assert result.turnover == pytest.approx(0.0)
    assert result.transaction_cost_paid == pytest.approx(0.0)
    assert result.forced_exit_pending.tolist() == [True, False]


def test_resumed_trading_executes_pending_exit_once() -> None:
    result = _execute(
        current_weights=np.array([0.5, 0.0]),
        current_cash_weight=0.5,
        desired_weights=np.array([0.0, 0.0]),
        buy_allowed=np.array([False, True]),
        sell_allowed=np.array([True, True]),
        forced_exit_required=np.array([True, False]),
        transaction_cost_rate=0.01,
    )

    assert result.executed_target_weights.tolist() == pytest.approx([0.0, 0.0])
    assert result.forced_exit_turnover == pytest.approx(0.5)
    assert result.turnover == pytest.approx(0.5)
    assert result.transaction_cost_paid == pytest.approx(0.005)
    assert result.forced_exit_pending.tolist() == [False, False]
    assert result.ending_cash_weight == pytest.approx(1.0)


def test_mandatory_exit_overrides_discretionary_turnover_cap() -> None:
    result = _execute(
        current_weights=np.array([0.8, 0.0]),
        current_cash_weight=0.2,
        desired_weights=np.array([0.0, 1.0]),
        buy_allowed=np.array([False, True]),
        sell_allowed=np.array([True, True]),
        forced_exit_required=np.array([True, False]),
        max_turnover=0.5,
    )

    assert result.forced_exit_turnover == pytest.approx(0.8)
    assert result.discretionary_turnover == pytest.approx(0.0)
    assert result.turnover_limit_overridden is True
    assert result.executed_target_weights.tolist() == pytest.approx([0.0, 0.0])


def test_invalid_held_return_fails_closed() -> None:
    with pytest.raises(ValueError, match="held assets"):
        _execute(asset_returns=np.array([np.nan, 0.0]))


def test_current_wealth_must_reconcile_before_execution() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        _execute(
            current_weights=np.array([0.5, 0.0]),
            current_cash_weight=0.4,
        )
