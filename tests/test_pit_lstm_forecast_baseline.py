from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.pit_lstm_forecast_baseline import (
    PitLSTMForecaster,
    forecast_policy,
    information_coefficients,
    training_samples,
    windows_at,
)
from src.envs.long_term_env import LongTermRewardConfig
from src.envs.point_in_time_env import PointInTimeSlotEnv

STOCK = ["return_1d", "volatility_20d"]
MARKET = ["mkt_return_1d"]


def _env(n_dates: int = 120, spans=None, rebalance_every: int = 1) -> PointInTimeSlotEnv:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for i, ticker in enumerate(("A.BK", "B.BK", "C.BK")):
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n_dates))
        for t, date in enumerate(dates):
            rows.append({"date": date, "ticker": ticker, "close": close[t], "history_sessions": t + 1,
                         "rank_value": 10.0 - i, "return_1d": rng.normal(), "volatility_20d": abs(rng.normal()),
                         "mkt_return_1d": float(t) / 100})
    panel = pd.DataFrame(rows)
    spans = spans or {"A.BK": (0, n_dates), "B.BK": (0, n_dates), "C.BK": (60, n_dates)}
    members = pd.DataFrame([{"date": dates[t], "ticker": k, "price_ticker": k}
                            for k, (first, last) in spans.items() for t in range(first, last)])
    return PointInTimeSlotEnv(panel, members, STOCK, MARKET, num_slots=3, lookback=5, min_history_sessions=1,
                              transaction_cost=0.001, slippage=0.0, rebalance_every=rebalance_every,
                              reward_config=LongTermRewardConfig(type="log_return", turnover_penalty=0.0,
                                                                 drawdown_penalty=0.0))


def test_samples_use_only_tradable_members_past_rows_and_targets_inside_the_window() -> None:
    env = _env()
    x, y, decision, target = training_samples(env, horizon=7, stride=3)
    assert x.shape[1:] == (5, 3)  # lookback x (2 stock + 1 market) features
    assert target.max() <= np.datetime64(env.dates[-1])
    step = env.lookback  # first decision session: A and B only
    np.testing.assert_allclose(x[0], np.c_[env.stock_tensor[0:5, 0, :], env.market_tensor[0:5]])
    expected = env.close_prices[step + 7, 0] / env.close_prices[step, 0] - 1.0
    assert y[0] == pytest.approx(expected, rel=1e-5)
    per_date = pd.Series(decision).value_counts()
    joined = pd.Timestamp(env.dates[61])  # C is listed from session 60, so it is tradable from 61
    assert per_date[per_date.index < joined].max() == 2
    assert per_date[per_date.index >= joined].max() == 3


def test_windows_end_the_session_before_the_decision() -> None:
    env = _env()
    window = windows_at(env, 40, np.array([1]))
    np.testing.assert_allclose(window[0, -1, :2], env.stock_tensor[39, 1])
    np.testing.assert_allclose(window[0, :, 2], env.market_tensor[35:40, 0])


class _Fixed:
    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=float)

    def predict(self, windows: np.ndarray) -> np.ndarray:
        return self.values[: windows.shape[0]]


def test_policy_buys_the_top_positive_forecasts_on_decision_sessions_only() -> None:
    env = _env(rebalance_every=5)
    env.reset(seed=0)
    log: list = []
    policy = forecast_policy(_Fixed([0.02, -0.01, 0.05]), k=1, log=log)
    weights = policy(None, env, 0)
    seated = env.slots[env.current_step]
    assert weights.sum() == pytest.approx(1.0) and weights[np.argmax(weights)] == 1.0
    assert seated[np.argmax(weights)] >= 0
    env.steps_since_rebalance = 1
    assert not policy(None, env, 1).any()
    assert len(log) == 1


class _Prices:
    """Six stocks with constant daily growth from 1% to 6%."""

    def __init__(self) -> None:
        self.dates = list(range(30))
        growth = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        self.close_prices = 100 * (1 + growth[None, :]) ** np.arange(30)[:, None]


def test_information_coefficient_skips_horizons_that_end_after_the_window() -> None:
    log = [(3, np.arange(6), np.arange(6, dtype=float)), (25, np.arange(6), np.zeros(6))]
    assert information_coefficients(_Prices(), log, horizon=7) == [pytest.approx(1.0)]


def test_forecaster_fits_on_a_training_environment() -> None:
    env = _env(n_dates=200)
    forecaster = PitLSTMForecaster(STOCK + MARKET, lookback=5, horizon=7, epochs=2, stride=2, hidden_size=4)
    forecaster.fit_env(env)
    assert forecaster.n_train_ > 0 and forecaster.n_validation_ > 0
    prediction = forecaster.predict(windows_at(env, 150, np.array([0, 1, 2])))
    assert prediction.shape == (3,) and np.isfinite(prediction).all()
