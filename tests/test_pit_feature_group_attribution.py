from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.pit_feature_group_attribution import attribute, input_groups, run_once, training_means
from src.envs.long_term_env import LongTermRewardConfig
from src.envs.point_in_time_env import PointInTimeSlotEnv

STOCK = ["return_1d", "volatility_20d"]
MARKET = ["set50_etf_return_1d", "bot_policy_rate"]
LOOKBACK, SLOTS = 5, 3


def _panel(n_dates: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for i, ticker in enumerate(("A.BK", "B.BK", "C.BK")):
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n_dates))
        for t, date in enumerate(dates):
            rows.append({"date": date, "ticker": ticker, "close": close[t], "history_sessions": t + 1,
                         "rank_value": 10.0 - i, "return_1d": rng.normal(), "volatility_20d": abs(rng.normal()),
                         "set50_etf_return_1d": rng.normal(), "bot_policy_rate": 1.0 if t < n_dates // 2 else 3.0})
    return pd.DataFrame(rows)


def _make(panel: pd.DataFrame):
    dates = sorted(panel["date"].unique())
    members = pd.DataFrame([{"date": d, "ticker": t, "price_ticker": t} for d in dates for t in ("A.BK", "B.BK", "C.BK")])

    def make() -> PointInTimeSlotEnv:
        return PointInTimeSlotEnv(panel, members, STOCK, MARKET, num_slots=SLOTS, lookback=LOOKBACK,
                                  min_history_sessions=1, transaction_cost=0.001, slippage=0.0, rebalance_every=5,
                                  reward_config=LongTermRewardConfig(type="log_return", turnover_penalty=0.0,
                                                                     drawdown_penalty=0.0))

    return make


class RateReader:
    """Holds cash when the latest policy rate in its window is above 2, equal weight otherwise."""

    def predict(self, obs, deterministic=True):
        width = len(STOCK) + len(MARKET) + 1
        window = obs[: LOOKBACK * SLOTS * width].reshape(LOOKBACK, SLOTS, width)
        rate = window[-1, 0, len(STOCK) + 1]
        logits = np.r_[np.full(SLOTS, -10.0), 10.0] if rate > 2.0 else np.r_[np.zeros(SLOTS), -10.0]
        return logits, None


def test_groups_follow_the_column_prefixes() -> None:
    groups = input_groups(["return_1d", "rsi_14"], ["set50_etf_return_1d", "bot_policy_rate", "usdthb_return_5d", "x"])
    assert groups == {"technical": ("stock", [0, 1]), "market": ("market", [0]), "policy_rate": ("market", [1]),
                      "macro": ("market", [2]), "other_market": ("market", [3])}


def test_decisions_are_recorded_only_on_decision_days() -> None:
    make = _make(_panel())
    curve, targets = run_once(make, RateReader(), None)
    steps = len(curve) - 1  # the first row is the starting value
    assert len(targets) == -(-steps // 5)  # a decision every fifth session, starting with the first
    assert targets.shape[1] == make().num_series


def test_only_the_group_the_model_reads_moves_its_portfolio() -> None:
    panel = _panel()
    make = _make(panel)
    groups = input_groups(STOCK, MARKET)
    means = training_means(panel, STOCK, MARKET, groups)
    assert means["policy_rate"][0] == pytest.approx(panel.drop_duplicates("date")["bot_policy_rate"].mean())
    means["policy_rate"] = np.array([1.0])  # a training mean below the model's threshold
    detail = attribute(make, RateReader(), None, groups, means, ["shuffle", "mean"], repeats=3, seed=0)
    moved = detail.groupby(["group", "mode"])[["action_shift", "delta_return", "delta_cash"]].mean()
    for group in ("technical", "market"):
        assert moved.loc[(group, "shuffle"), "action_shift"] == 0.0
        assert moved.loc[(group, "mean"), "delta_return"] == 0.0
    assert moved.loc[("policy_rate", "shuffle"), "action_shift"] > 0.0
    assert moved.loc[("policy_rate", "mean"), "delta_cash"] < -0.3  # held at 1.0, the model never goes to cash
    assert (detail["base_cash"] > 0.3).all()


def test_changes_leave_the_next_environment_intact() -> None:
    make = _make(_panel())
    groups = input_groups(STOCK, MARKET)
    before = make().market_tensor.copy()
    attribute(make, RateReader(), None, groups, {g: np.zeros(len(p)) for g, (_, p) in groups.items()},
              ["shuffle", "mean"], repeats=1, seed=0)
    np.testing.assert_array_equal(make().market_tensor, before)
