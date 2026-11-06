from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.predictive_baselines import RidgeForecaster, forecaster_policy, top_k_allocation
from src.envs.long_term_env import LongTermTradingEnv


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
    env = LongTermTradingEnv(frame, ["signal"], lookback=3, accounting="self_financing")
    env.reset()
    weights = forecaster_policy(model, k=1)(None, env, 0)
    assert weights[env.tickers.index("UP")] == pytest.approx(1.0)


def test_lstm_forecaster_learns_signal_and_purges_overlapping_targets() -> None:
    pytest.importorskip("torch")
    from src.agents.predictive_baselines import LSTMForecaster

    frame = _frame(160)
    model = LSTMForecaster(["signal"], lookback=5, horizon=5, hidden_size=8, epochs=25, patience=25, seed=0).fit(frame)
    # Purge: every training target is realized before the early-stopping block starts.
    assert model.validation_start_ is not None
    assert model.n_train_ > 0 and model.n_validation_ > 0
    total = 3 * (160 - 5 - 5)
    assert model.n_train_ + model.n_validation_ < total
    windows = np.stack([np.full((5, 1), v) for v in (0.3, 0.0, -0.3)])
    preds = model.predict(windows)
    assert preds.shape == (3,)
    assert preds[0] > preds[2]


def test_sequence_policy_reads_only_past_rows() -> None:
    pytest.importorskip("torch")
    from src.agents.predictive_baselines import LSTMForecaster, sequence_forecaster_policy

    frame = _frame(160)
    model = LSTMForecaster(["signal"], lookback=5, horizon=5, hidden_size=8, epochs=3, seed=0).fit(frame)
    env = LongTermTradingEnv(frame, ["signal"], lookback=5, accounting="self_financing")
    env.reset()
    for _ in range(10):
        env.step(np.array([1 / 3, 1 / 3, 1 / 3]))
    policy = sequence_forecaster_policy(model, k=1)
    before = policy(None, env, 0)
    env.feature_tensor[env.current_step :] = 99.0  # future rows must not matter
    after = policy(None, env, 0)
    assert np.array_equal(before, after)
