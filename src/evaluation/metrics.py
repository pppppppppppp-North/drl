from __future__ import annotations

import numpy as np
import pandas as pd


def equity_metrics(
    equity_curve: pd.DataFrame,
    periods_per_year: int = 252,
    benchmark_curve: pd.DataFrame | pd.Series | None = None,
) -> dict[str, float]:
    values = equity_curve["portfolio_value"].astype(float)
    returns = values.pct_change().fillna(0.0)
    cumulative_return = values.iloc[-1] / values.iloc[0] - 1.0
    volatility = returns.std(ddof=0) * np.sqrt(periods_per_year)
    annualized_return = (1.0 + cumulative_return) ** (periods_per_year / max(len(returns), 1)) - 1.0
    downside = returns.where(returns < 0, 0.0)
    downside_deviation = np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year)
    sharpe = annualized_return / volatility if volatility else 0.0
    sortino = annualized_return / downside_deviation if downside_deviation else 0.0
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annualized_return / abs(max_drawdown) if max_drawdown else 0.0
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() else 0.0
    value_at_risk = float(returns.quantile(0.05))
    tail_returns = returns[returns <= value_at_risk]
    conditional_value_at_risk = float(tail_returns.mean()) if not tail_returns.empty else 0.0
    turnover = equity_curve["turnover"].astype(float) if "turnover" in equity_curve else pd.Series(0.0, index=equity_curve.index)
    transaction_cost_paid = (
        equity_curve["transaction_cost_paid"].astype(float).fillna(0.0)
        if "transaction_cost_paid" in equity_curve
        else pd.Series(0.0, index=equity_curve.index)
    )
    slippage_paid = (
        equity_curve["slippage_paid"].astype(float).fillna(0.0)
        if "slippage_paid" in equity_curve
        else pd.Series(0.0, index=equity_curve.index)
    )
    cash_weight = (
        equity_curve["cash_weight"].astype(float)
        if "cash_weight" in equity_curve
        else pd.Series(np.nan, index=equity_curve.index)
    )

    metrics = {
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "win_rate": float((returns > 0).mean()),
        "profit_factor": float(profit_factor),
        "downside_deviation": float(downside_deviation),
        "value_at_risk_95": value_at_risk,
        "conditional_value_at_risk_95": conditional_value_at_risk,
        "avg_turnover": float(turnover.mean()),
        "total_turnover": float(turnover.sum()),
        "total_transaction_cost": float(transaction_cost_paid.sum()),
        "total_slippage": float(slippage_paid.sum()),
        "active_trading_steps": float((turnover > 1e-6).sum()),
        "avg_cash_weight": float(cash_weight.mean()) if cash_weight.notna().any() else 0.0,
        "final_portfolio_value": float(values.iloc[-1]),
    }
    if benchmark_curve is not None:
        metrics.update(benchmark_relative_metrics(equity_curve, benchmark_curve, periods_per_year=periods_per_year))
    return metrics


def _value_series(curve: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(curve, pd.Series):
        series = curve.astype(float)
    else:
        series = curve["portfolio_value"].astype(float)
        if "date" in curve:
            series.index = pd.to_datetime(curve["date"]).to_numpy()
    return series


def benchmark_relative_metrics(
    equity_curve: pd.DataFrame | pd.Series,
    benchmark_curve: pd.DataFrame | pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Beta, alpha, tracking error, and information ratio against a benchmark value curve.

    Both inputs are value curves (a ``portfolio_value`` column or a value Series). When both carry
    a ``date`` column they are aligned on date; otherwise they are aligned by position and must have
    equal length.
    """

    portfolio = _value_series(equity_curve)
    benchmark = _value_series(benchmark_curve)
    has_dates = isinstance(portfolio.index, pd.DatetimeIndex) and isinstance(benchmark.index, pd.DatetimeIndex)
    if has_dates:
        joined = pd.concat({"portfolio": portfolio, "benchmark": benchmark}, axis=1, join="inner").dropna()
    else:
        if len(portfolio) != len(benchmark):
            raise ValueError("portfolio and benchmark curves must have equal length when no dates are given")
        joined = pd.DataFrame(
            {"portfolio": portfolio.to_numpy(), "benchmark": benchmark.to_numpy()}
        )
    if len(joined) < 3:
        raise ValueError("benchmark-relative metrics need at least three aligned observations")

    returns = joined.pct_change().dropna()
    portfolio_returns = returns["portfolio"]
    benchmark_returns = returns["benchmark"]
    active = portfolio_returns - benchmark_returns
    benchmark_variance = float(benchmark_returns.var(ddof=0))
    covariance = float(((portfolio_returns - portfolio_returns.mean()) * (benchmark_returns - benchmark_returns.mean())).mean())
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
    alpha_daily = float(portfolio_returns.mean() - beta * benchmark_returns.mean())
    tracking_error = float(active.std(ddof=0) * np.sqrt(periods_per_year))
    n = len(returns)
    portfolio_annual = float((1.0 + portfolio_returns).prod() ** (periods_per_year / n) - 1.0)
    benchmark_annual = float((1.0 + benchmark_returns).prod() ** (periods_per_year / n) - 1.0)
    excess_annual = portfolio_annual - benchmark_annual
    information_ratio = excess_annual / tracking_error if tracking_error > 0 else 0.0
    correlation = float(portfolio_returns.corr(benchmark_returns)) if n > 1 else 0.0
    up = benchmark_returns > 0
    down = benchmark_returns < 0
    up_capture = (
        float(portfolio_returns[up].mean() / benchmark_returns[up].mean()) if up.any() and benchmark_returns[up].mean() else 0.0
    )
    down_capture = (
        float(portfolio_returns[down].mean() / benchmark_returns[down].mean())
        if down.any() and benchmark_returns[down].mean()
        else 0.0
    )
    return {
        "beta": float(beta),
        "alpha_annualized": float(alpha_daily * periods_per_year),
        "tracking_error": tracking_error,
        "information_ratio": float(information_ratio),
        "excess_annualized_return": float(excess_annual),
        "benchmark_correlation": correlation if np.isfinite(correlation) else 0.0,
        "up_capture": up_capture,
        "down_capture": down_capture,
    }
