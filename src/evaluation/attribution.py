"""Permutation attribution of data-source groups for a trained policy (milestone M7 / RQ3).

For each feature group (technical, market, macro, ...), the group's columns are shuffled along the
time axis of the evaluation environment. That breaks the link between the group and the market
while keeping its distribution. The trained policy is then re-run unchanged. The size of the
resulting change measures how much the policy relies on that group:

- ``delta_cumulative_return`` / ``delta_sharpe``: performance change versus the intact run;
- ``action_shift``: mean half-L1 distance between intact and permuted target weights at each
  decision (0 = the policy ignores the group, 1 = completely different portfolios).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.evaluation.metrics import equity_metrics


def default_feature_groups(feature_columns: list[str]) -> dict[str, list[int]]:
    """Group columns by data source using the project's column-name prefixes."""

    macro_prefixes = ("usdthb_", "brent_", "wti_", "gold_", "us10y_", "bot_", "macro_")
    groups: dict[str, list[int]] = {"technical": [], "market": [], "macro": [], "fundamental": [], "sentiment": [], "flow": []}
    for i, column in enumerate(feature_columns):
        if column.startswith(("set_", "set50_", "set50_etf_", "market_")):
            groups["market"].append(i)
        elif column.startswith(macro_prefixes):
            groups["macro"].append(i)
        elif column.startswith("fundamental_"):
            groups["fundamental"].append(i)
        elif column.startswith(("sentiment_", "news_")):
            groups["sentiment"].append(i)
        elif column.startswith("flow_"):
            groups["flow"].append(i)
        else:
            groups["technical"].append(i)
    return {name: idx for name, idx in groups.items() if idx}


def rollout_with_targets(env, policy: Callable) -> tuple[pd.DataFrame, np.ndarray]:
    obs, info = env.reset(seed=0)
    values, targets = [info["portfolio_value"]], []
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy(obs))
        values.append(info["portfolio_value"])
        if info.get("rebalanced", True):
            targets.append(np.asarray(info.get("target_weights", info["weights"]), dtype=np.float64))
    return pd.DataFrame({"portfolio_value": values}), np.asarray(targets)


def permutation_attribution(
    make_env: Callable,
    policy: Callable,
    groups: dict[str, list[int]],
    repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per group x repeat. ``make_env()`` must return a fresh evaluation environment."""

    rng = np.random.default_rng(seed)
    base_curve, base_targets = rollout_with_targets(make_env(), policy)
    base = equity_metrics(base_curve)
    rows = []
    for group, indices in groups.items():
        for repeat in range(repeats):
            env = make_env()
            permutation = rng.permutation(env.feature_tensor.shape[0])
            shuffled = env.feature_tensor[permutation][:, :, indices]
            env.feature_tensor[:, :, indices] = shuffled
            curve, targets = rollout_with_targets(env, policy)
            metrics = equity_metrics(curve)
            n = min(len(targets), len(base_targets))
            shift = float(0.5 * np.abs(targets[:n] - base_targets[:n]).sum(axis=1).mean()) if n else 0.0
            rows.append(
                {
                    "group": group,
                    "repeat": repeat,
                    "features": len(indices),
                    "delta_cumulative_return": metrics["cumulative_return"] - base["cumulative_return"],
                    "delta_sharpe": metrics["sharpe"] - base["sharpe"],
                    "action_shift": shift,
                    "base_cumulative_return": base["cumulative_return"],
                }
            )
    return pd.DataFrame(rows)
