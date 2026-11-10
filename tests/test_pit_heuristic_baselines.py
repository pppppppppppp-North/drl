from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.pit_heuristic_baselines import History, min_variance_weights, momentum_policy


class _Env:
    """The parts of PointInTimeSlotEnv the heuristic policies read."""

    def __init__(self, dates, tickers, step: int) -> None:
        self.dates = list(dates)
        self.tickers = list(tickers)
        self.num_slots = len(tickers) + 1  # one empty slot
        self.slots = np.tile(np.r_[np.arange(len(tickers)), -1], (len(self.dates), 1))
        self.current_step = step
        self.steps_since_rebalance = 0
        self.rebalance_every = 21


def _panel(prices: dict[str, np.ndarray]) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=len(next(iter(prices.values()))))
    return pd.DataFrame([{"date": d, "ticker": t, "close": p[i]} for t, p in prices.items() for i, d in enumerate(dates)])


def test_momentum_ranks_by_the_12_1_month_return_and_ignores_the_decision_session() -> None:
    n = 300
    flat = np.full(n, 100.0)
    winner = np.linspace(50.0, 100.0, n)  # rises steadily over the ranking window
    jumper = flat.copy()
    jumper[-1] = 1_000.0  # jumps on the decision session itself, which the policy must not see
    panel = _panel({"A": flat, "B": winner, "C": jumper})
    history = History(panel)
    env = _Env(sorted(panel["date"].unique()), ["A", "B", "C"], step=n - 1)
    weights = momentum_policy(history, k=1)(None, env, 0)
    np.testing.assert_allclose(weights, [0.0, 1.0, 0.0, 0.0])


def test_momentum_skips_members_without_a_year_of_history() -> None:
    n = 300
    young = np.r_[np.full(100, np.nan), np.linspace(10.0, 100.0, n - 100)]
    panel = _panel({"A": np.linspace(100.0, 110.0, n), "B": young}).dropna()
    history = History(panel)
    env = _Env(history.closes.index, ["A", "B"], step=n - 1)
    weights = momentum_policy(history, k=2)(None, env, 0)
    np.testing.assert_allclose(weights, [1.0, 0.0, 0.0])


def test_min_variance_prefers_the_quiet_stock_within_the_cap() -> None:
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({"quiet": rng.normal(0, 0.005, 252), "loud": rng.normal(0, 0.03, 252),
                            "new": np.r_[np.full(200, np.nan), rng.normal(0, 0.001, 52)]})
    weights, solved = min_variance_weights(returns, cap=0.9, shrink=0.5)
    assert solved
    assert weights.sum() == pytest.approx(1.0)
    assert weights[2] == 0.0  # too few returns to estimate
    assert 0.85 < weights[0] <= 0.9 + 1e-9 and weights[1] > 0.0
    capped, _ = min_variance_weights(returns[["quiet", "loud"]], cap=0.6, shrink=0.5)
    assert capped[0] == pytest.approx(0.6, abs=1e-6)
