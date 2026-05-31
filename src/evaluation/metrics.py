from __future__ import annotations

import numpy as np
import pandas as pd


def equity_metrics(equity_curve: pd.DataFrame, periods_per_year: int = 252) -> dict[str, float]:
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
    cash_weight = (
        equity_curve["cash_weight"].astype(float)
        if "cash_weight" in equity_curve
        else pd.Series(np.nan, index=equity_curve.index)
    )

    return {
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
        "active_trading_steps": float((turnover > 1e-6).sum()),
        "avg_cash_weight": float(cash_weight.mean()) if cash_weight.notna().any() else 0.0,
        "final_portfolio_value": float(values.iloc[-1]),
    }
