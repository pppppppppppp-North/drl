from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.predictive_baselines import RidgeForecaster, forecaster_policy, top_k_allocation
from src.envs.trading_env import ThaiStockTradingEnv


def _frame(n_dates: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for t, drift in (("UP", 0.003), ("FLAT", 0.0), ("DOWN", -0.003)):
        close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.002, n_dates))
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t, "close": close[i], "signal": drift * 100 + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def test_forecaster_learns_signal_and_never_uses_future_targets() -> None:
    frame = _frame()
    model = RidgeForecaster(["signal"], horizon=5, alpha=1.0).fit(frame)
    preds = model.predict(np.array([[0.3], [0.0], [-0.3]]))
    assert preds[0] > preds[1] > preds[2]
    # The last `horizon` sessions per ticker have no in-sample forward return and are dropped,
    # so no target is ever filled from data beyond the training frame.
    assert model.n_train_ == len(frame) - 5 * frame["ticker"].nunique()
    # Appending later (validation) rows to the data must not change a fit on the training rows.
    train_end = sorted(frame["date"].unique())[89]
    train = frame[frame["date"] <= train_end]
    a = RidgeForecaster(["signal"], horizon=5).fit(train)
    extended = frame.copy()
    extended.loc[extended["date"] > train_end, "close"] *= 10
    b = RidgeForecaster(["signal"], horizon=5).fit(extended[extended["date"] <= train_end])
    assert np.allclose(a.coef_, b.coef_)


def test_top_k_allocation_and_cash_fallback() -> None:
    assert top_k_allocation(np.array([0.02, -0.01, 0.03]), k=2).tolist() == [0.5, 0.0, 0.5]
    assert top_k_allocation(np.array([0.02, -0.01, -0.03]), k=2).tolist() == [1.0, 0.0, 0.0]
    assert top_k_allocation(np.array([-0.02, -0.01]), k=2).sum() == 0.0


def test_forecaster_policy_uses_latest_completed_row() -> None:
    frame = _frame()
    model = RidgeForecaster(["signal"], horizon=5).fit(frame)
    env = ThaiStockTradingEnv(frame, ["signal"], lookback=3, accounting="self_financing")
    env.reset()
    weights = forecaster_policy(model, k=1)(None, env, 0)
    assert weights[env.tickers.index("UP")] == pytest.approx(1.0)
