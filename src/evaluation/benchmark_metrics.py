"""Benchmark-relative evaluation metrics (milestone M7), additive to the locked ``metrics.py``.

``equity_metrics_with_benchmark`` returns everything ``src.evaluation.metrics.equity_metrics``
returns, plus beta, alpha, tracking error, information ratio, excess annualized return,
correlation, and up/down capture against a benchmark value curve (for example the SET50 ETF).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import equity_metrics


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

    Both inputs are value curves (a ``portfolio_value`` column or a value Series). When both carry a
    ``date`` column they are aligned on date; otherwise they are aligned by position and must have
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
        joined = pd.DataFrame({"portfolio": portfolio.to_numpy(), "benchmark": benchmark.to_numpy()})
    if len(joined) < 3:
        raise ValueError("benchmark-relative metrics need at least three aligned observations")

    returns = joined.pct_change().dropna()
    portfolio_returns = returns["portfolio"]
    benchmark_returns = returns["benchmark"]
    active = portfolio_returns - benchmark_returns
    benchmark_variance = float(benchmark_returns.var(ddof=0))
    covariance = float(
        ((portfolio_returns - portfolio_returns.mean()) * (benchmark_returns - benchmark_returns.mean())).mean()
    )
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
        float(portfolio_returns[up].mean() / benchmark_returns[up].mean())
        if up.any() and benchmark_returns[up].mean()
        else 0.0
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


def equity_metrics_with_benchmark(
    equity_curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame | pd.Series | None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """``equity_metrics`` plus benchmark-relative metrics (when a benchmark is given)."""

    metrics = equity_metrics(equity_curve, periods_per_year=periods_per_year)
    if benchmark_curve is not None:
        metrics.update(benchmark_relative_metrics(equity_curve, benchmark_curve, periods_per_year=periods_per_year))
    return metrics
