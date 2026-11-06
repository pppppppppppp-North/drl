from __future__ import annotations

import numpy as np
import pandas as pd

from src.envs.long_term_env import LongTermTradingEnv
from src.evaluation.attribution import default_feature_groups, permutation_attribution


def test_default_feature_groups_by_prefix() -> None:
    columns = ["return_1d", "rsi_14", "set50_etf_return_1d", "usdthb_return_5d", "us10y_change_1d", "fundamental_available", "flow_foreign_net_share"]
    groups = default_feature_groups(columns)
    assert groups == {"technical": [0, 1], "market": [2], "macro": [3, 4], "fundamental": [5], "flow": [6]}


def _frame(n_dates: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    rows = []
    for t in ("AAA", "BBB", "CCC"):
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_dates))
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t, "close": close[i], "signal": rng.normal(), "usdthb_return_5d": rng.normal()})
    return pd.DataFrame(rows)


def test_permutation_attribution_detects_only_the_used_group() -> None:
    frame = _frame()
    columns = ["signal", "usdthb_return_5d"]
    lookback, n_assets = 3, 3

    def make_env():
        return LongTermTradingEnv(frame, columns, lookback=lookback, accounting="self_financing")

    def policy(obs):
        window = obs[: lookback * n_assets * len(columns)].reshape(lookback, n_assets, len(columns))
        weights = np.zeros(n_assets)
        weights[int(np.argmax(window[-1, :, 0]))] = 1.0  # uses only the technical "signal" column
        return weights

    result = permutation_attribution(make_env, policy, default_feature_groups(columns), repeats=3, seed=1)
    shift = result.groupby("group")["action_shift"].mean()
    assert shift["technical"] > 0.3
    assert shift["macro"] == 0.0
    assert (result.loc[result["group"] == "macro", "delta_cumulative_return"] == 0.0).all()
