from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.pit_cash_cap_binding import binding, cash_weights, recording_policy
from scripts.run_pit_walk_forward import rollout
from src.envs.long_term_env import LongTermRewardConfig
from src.envs.point_in_time_env import PointInTimeSlotEnv

STOCK = ["return_1d"]
MARKET = ["bot_policy_rate"]
LOOKBACK, SLOTS, CAP = 5, 3, 0.2


def _make(max_cash_weight: float = CAP):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=60)
    rows = []
    for i, ticker in enumerate(("A.BK", "B.BK", "C.BK")):
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, len(dates)))
        for t, date in enumerate(dates):
            rows.append({"date": date, "ticker": ticker, "close": close[t], "history_sessions": t + 1,
                         "rank_value": 10.0 - i, "return_1d": rng.normal(),
                         "bot_policy_rate": 1.0 if t < len(dates) // 2 else 3.0})
    panel = pd.DataFrame(rows)
    members = pd.DataFrame([{"date": d, "ticker": t, "price_ticker": t} for d in dates for t in ("A.BK", "B.BK", "C.BK")])
    return PointInTimeSlotEnv(panel, members, STOCK, MARKET, num_slots=SLOTS, lookback=LOOKBACK, min_history_sessions=1,
                              transaction_cost=0.001, slippage=0.0, rebalance_every=5, max_cash_weight=max_cash_weight,
                              reward_config=LongTermRewardConfig(type="log_return", turnover_penalty=0.0,
                                                                 drawdown_penalty=0.0))


class RateReader:
    """All cash when the latest policy rate in its window is above 2, equal weight otherwise."""

    def predict(self, obs, deterministic=True):
        width = len(STOCK) + len(MARKET) + 1
        rate = obs[: LOOKBACK * SLOTS * width].reshape(LOOKBACK, SLOTS, width)[-1, 0, len(STOCK)]
        logits = np.r_[np.full(SLOTS, -10.0), 10.0] if rate > 2.0 else np.r_[np.zeros(SLOTS), -10.0]
        return logits, None


def test_the_cap_binds_only_when_the_policy_wants_more_cash() -> None:
    env = _make()
    records: list = []
    rollout(env, recording_policy(RateReader(), None, records), "test")
    own, capped = np.asarray(records).T
    wants_cash = own > 0.5
    assert wants_cash.any() and (~wants_cash).any()  # both regimes occur
    np.testing.assert_allclose(capped[wants_cash], CAP, atol=1e-9)
    np.testing.assert_allclose(capped[~wants_cash], own[~wants_cash], atol=1e-12)
    assert env.max_cash_weight == CAP  # restored after every decision
    row = binding(records, CAP)
    assert row["decisions"] == len(records)
    assert row["share_binding"] == pytest.approx(wants_cash.mean())
    assert row["mean_capped_cash"] <= CAP + 1e-9 and row["max_own_cash"] > 0.99


def test_recording_does_not_change_the_portfolio() -> None:
    def plain(obs, env, step):
        if env.steps_since_rebalance % env.rebalance_every:
            return np.zeros(env.num_slots + 1)
        return RateReader().predict(obs)[0]

    recorded, _ = rollout(_make(), recording_policy(RateReader(), None, []), "recorded")
    unrecorded, _ = rollout(_make(), plain, "plain")
    np.testing.assert_array_equal(recorded["portfolio_value"].to_numpy(), unrecorded["portfolio_value"].to_numpy())


def test_without_a_cap_both_weights_agree() -> None:
    env = _make(max_cash_weight=1.0)
    env.reset()
    own, capped = cash_weights(env, np.r_[np.full(SLOTS, -10.0), 10.0])
    assert own == pytest.approx(capped) and own > 0.99
