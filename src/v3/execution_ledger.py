"""Self-financing execution accounting for the isolated V3 development path.

The locked V2 environment is intentionally untouched.  This module separates
security notional from the cash balancing leg, preserves untradeable holdings,
funds costs from cash, and carries post-return weights into the next decision.
It is a structural execution primitive, not permission to train V3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_TOLERANCE = 1e-10


@dataclass(frozen=True)
class ExecutionStepResult:
    """Auditable output of one close-to-close portfolio transition."""

    executed_target_weights: np.ndarray
    trade_deltas: np.ndarray
    ending_weights: np.ndarray
    ending_cash_weight: float
    turnover: float
    discretionary_turnover: float
    forced_exit_turnover: float
    transaction_cost_paid: float
    slippage_paid: float
    pre_return_cash_fraction: float
    gross_return: float
    net_return: float
    portfolio_multiplier: float
    forced_exit_pending: np.ndarray
    turnover_limit_overridden: bool
    reconciliation_error: float


def _float_vector(value: np.ndarray, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite one-dimensional vector")
    return vector.copy()


def _boolean_vector(
    value: np.ndarray,
    *,
    label: str,
    size: int,
) -> np.ndarray:
    vector = np.asarray(value, dtype=bool)
    if vector.shape != (size,):
        raise ValueError(f"{label} must have shape {(size,)}, got {vector.shape}")
    return vector.copy()


def security_turnover(
    previous_weights: np.ndarray,
    target_weights: np.ndarray,
) -> float:
    """Return gross traded security notional as a fraction of starting wealth.

    Cash is the settlement account, not a second trade.  A 50% purchase from
    cash therefore has 50% turnover, while rotating 50% from one security into
    another has 100% turnover because both the sale and purchase are charged.
    """

    previous = _float_vector(previous_weights, label="previous_weights")
    target = _float_vector(target_weights, label="target_weights")
    if previous.shape != target.shape:
        raise ValueError("previous_weights and target_weights must have equal shape")
    return float(np.abs(target - previous).sum())


def execute_self_financing_step(
    *,
    current_weights: np.ndarray,
    current_cash_weight: float,
    desired_weights: np.ndarray,
    asset_returns: np.ndarray,
    buy_allowed: np.ndarray,
    sell_allowed: np.ndarray,
    forced_exit_required: np.ndarray | None = None,
    transaction_cost_rate: float = 0.0,
    slippage_rate: float = 0.0,
    max_turnover: float | None = None,
) -> ExecutionStepResult:
    """Execute one constrained rebalance and reconcile ending wealth exactly.

    Inputs and outputs are fractions of wealth at the start of the decision.
    Costs are charged on the absolute security trades only.  Mandatory exits
    override a discretionary turnover cap, while an exit that cannot trade is
    carried as a pending frozen holding.
    """

    current = _float_vector(current_weights, label="current_weights")
    desired = _float_vector(desired_weights, label="desired_weights")
    returns = np.asarray(asset_returns, dtype=np.float64)
    if returns.ndim != 1:
        raise ValueError("asset_returns must be a one-dimensional vector")
    returns = returns.copy()
    if current.shape != desired.shape or current.shape != returns.shape:
        raise ValueError("weight and return vectors must have equal shape")
    size = len(current)
    can_buy = _boolean_vector(buy_allowed, label="buy_allowed", size=size)
    can_sell = _boolean_vector(sell_allowed, label="sell_allowed", size=size)
    forced = (
        np.zeros(size, dtype=bool)
        if forced_exit_required is None
        else _boolean_vector(
            forced_exit_required,
            label="forced_exit_required",
            size=size,
        )
    )

    cash = float(current_cash_weight)
    transaction_rate = float(transaction_cost_rate)
    slip_rate = float(slippage_rate)
    if not np.isfinite(cash) or cash < -_TOLERANCE:
        raise ValueError("current_cash_weight must be finite and non-negative")
    if (current < -_TOLERANCE).any() or (desired < -_TOLERANCE).any():
        raise ValueError("long-only weights must be non-negative")
    if not np.isclose(current.sum() + cash, 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("current security and cash weights must sum to one")
    if desired.sum() > 1.0 + 1e-8:
        raise ValueError("desired_weights cannot exceed total wealth")
    if transaction_rate < 0.0 or slip_rate < 0.0:
        raise ValueError("cost and slippage rates must be non-negative")
    if transaction_rate + slip_rate >= 1.0:
        raise ValueError("combined execution rates must be below one")
    if max_turnover is not None:
        max_turnover = float(max_turnover)
        if not np.isfinite(max_turnover) or max_turnover <= 0.0:
            raise ValueError("max_turnover must be finite and positive")

    current = np.maximum(current, 0.0)
    desired = np.maximum(desired, 0.0)
    pending = forced & ~can_sell

    permitted = desired.copy()
    blocked_buys = ~can_buy & (permitted > current)
    blocked_sells = ~can_sell & (permitted < current)
    permitted[blocked_buys] = current[blocked_buys]
    permitted[blocked_sells] = current[blocked_sells]
    permitted[pending] = current[pending]
    executable_forced = forced & can_sell
    permitted[executable_forced] = 0.0

    mandatory_target = current.copy()
    mandatory_target[executable_forced] = 0.0
    mandatory_turnover = security_turnover(current, mandatory_target)
    discretionary_delta = permitted - mandatory_target
    discretionary_turnover_requested = float(
        np.abs(discretionary_delta).sum()
    )
    discretionary_scale = 1.0
    if max_turnover is not None:
        remaining_budget = max(max_turnover - mandatory_turnover, 0.0)
        if discretionary_turnover_requested > remaining_budget + _TOLERANCE:
            discretionary_scale = (
                remaining_budget / discretionary_turnover_requested
                if discretionary_turnover_requested
                else 0.0
            )
    target = mandatory_target + discretionary_scale * discretionary_delta

    # Costs must be paid from actual cash.  Preserve all sales and frozen
    # holdings, then proportionally reduce purchases only when the requested
    # target would otherwise create negative cash.
    delta = target - current
    sales = np.minimum(delta, 0.0)
    purchases = np.maximum(delta, 0.0)
    sale_notional = float(-sales.sum())
    purchase_notional = float(purchases.sum())
    total_rate = transaction_rate + slip_rate
    base_after_sales = current + sales
    cash_after_sales_and_cost = (
        1.0 - float(base_after_sales.sum()) - total_rate * sale_notional
    )
    if cash_after_sales_and_cost < -1e-9:
        raise ValueError("sale costs cannot be funded without borrowing")
    purchase_scale = 1.0
    purchase_cash_need = purchase_notional * (1.0 + total_rate)
    if purchase_cash_need > max(cash_after_sales_and_cost, 0.0) + _TOLERANCE:
        purchase_scale = (
            max(cash_after_sales_and_cost, 0.0) / purchase_cash_need
            if purchase_cash_need
            else 0.0
        )
    target = base_after_sales + purchase_scale * purchases

    trade_deltas = target - current
    turnover = float(np.abs(trade_deltas).sum())
    forced_exit_turnover = float(
        np.abs(trade_deltas[executable_forced]).sum()
    )
    discretionary_turnover = max(turnover - forced_exit_turnover, 0.0)
    transaction_cost = transaction_rate * turnover
    slippage = slip_rate * turnover
    pre_return_cash = (
        1.0 - float(target.sum()) - transaction_cost - slippage
    )
    if pre_return_cash < -1e-9:
        raise ValueError("execution produced negative cash")
    pre_return_cash = max(pre_return_cash, 0.0)

    held = target > _TOLERANCE
    if (~np.isfinite(returns[held])).any() or (returns[held] <= -1.0).any():
        raise ValueError("held assets require finite returns greater than -100%")
    safe_returns = np.where(np.isfinite(returns), returns, 0.0)
    ending_asset_values = target * (1.0 + safe_returns)
    portfolio_multiplier = float(ending_asset_values.sum() + pre_return_cash)
    if not np.isfinite(portfolio_multiplier) or portfolio_multiplier <= 0.0:
        raise ValueError("portfolio wealth must remain finite and positive")
    ending_weights = ending_asset_values / portfolio_multiplier
    ending_cash = pre_return_cash / portfolio_multiplier
    gross_return = float(np.dot(target, safe_returns))
    net_return = portfolio_multiplier - 1.0
    expected_multiplier = (
        1.0 + gross_return - transaction_cost - slippage
    )
    reconciliation_error = float(portfolio_multiplier - expected_multiplier)
    if abs(reconciliation_error) > 1e-10:
        raise ValueError("self-financing portfolio reconciliation failed")
    if abs(float(ending_weights.sum()) + ending_cash - 1.0) > 1e-10:
        raise ValueError("ending weights do not reconcile to one")

    return ExecutionStepResult(
        executed_target_weights=target,
        trade_deltas=trade_deltas,
        ending_weights=ending_weights,
        ending_cash_weight=float(ending_cash),
        turnover=turnover,
        discretionary_turnover=discretionary_turnover,
        forced_exit_turnover=forced_exit_turnover,
        transaction_cost_paid=float(transaction_cost),
        slippage_paid=float(slippage),
        pre_return_cash_fraction=float(pre_return_cash),
        gross_return=gross_return,
        net_return=float(net_return),
        portfolio_multiplier=portfolio_multiplier,
        forced_exit_pending=pending,
        turnover_limit_overridden=bool(
            max_turnover is not None
            and mandatory_turnover > max_turnover + _TOLERANCE
        ),
        reconciliation_error=reconciliation_error,
    )
