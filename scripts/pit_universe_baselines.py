"""Survivorship comparison for the point-in-time walk-forward (milestone M3).

On the same sessions, costs and accounting as ``scripts.run_pit_walk_forward``, monthly equal
weight over three universes:

- ``pit_equal_weight_monthly``: the SET50 membership in force on each session (members with prices);
- ``hindsight_set50_equal_weight_monthly``: the 50 names of the last 2024 list, held on every
  session they have prices, whether or not they were in the index then. This is the classic
  survivorship-biased backtest universe;
- ``starter5_equal_weight_monthly``: the five hand-picked stocks of the earlier experiments.

It also runs ``pit_ridge_forecast_top10_monthly``, the supervised baseline of the five-stock
study scaled to the index. A ridge regression of 21-session forward returns is fitted on training
rows where the stock was a SET50 member, and every 21 sessions it equal-weights the ten highest
positive forecasts among the tradable members, using the same completed feature row the agent
sees.

The gaps between them measure how much choosing stocks with hindsight inflates a backtest.
Output: ``<run-dir>/fold_<Y>/baselines_universes/validation_equity_curves.csv``, merged by
``scripts.aggregate_walk_forward_reward``.

    python -m scripts.pit_universe_baselines --config config/pit_walk_forward_experiment.yaml \\
        --run-dir results/pit_walk_forward --years 2016 2017 2018 2019 2020 2021 2022
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.agents.predictive_baselines import RidgeForecaster, top_k_allocation

from scripts.run_pit_walk_forward import (
    EQUAL_WEIGHT,
    _deep_merge,
    baseline_curves,
    equal_weight_policy,
    load_inputs,
    make_env,
    rollout,
    window,
)

STARTER5 = ("ADVANC.BK", "AOT.BK", "CPALL.BK", "KBANK.BK", "PTT.BK")


def fixed_universe_mask(panel: pd.DataFrame, tickers: set[str]) -> pd.DataFrame:
    """Treat ``tickers`` as members on every session on which they have prices."""
    rows = panel.loc[panel["ticker"].isin(tickers), ["date", "ticker"]]
    return rows.assign(price_ticker=rows["ticker"], in_set50=True).reset_index(drop=True)


def ridge_policy(forecaster: RidgeForecaster, k: int):
    def _policy(obs, env, step):
        slots = env.slots[env.current_step]
        valid = slots >= 0
        weights = np.zeros(env.num_slots)
        if valid.any():
            row = env.current_step - 1  # the latest completed session, as in the agent's window
            x = np.hstack([env.stock_tensor[row, slots[valid]],
                           np.repeat(env.market_tensor[row][None, :], int(valid.sum()), axis=0)])
            weights[valid] = top_k_allocation(forecaster.predict(x), k=k)
        return weights

    return _policy


def fit_ridge(base: dict, train_panel: pd.DataFrame, mask: pd.DataFrame) -> RidgeForecaster:
    columns = list(base["features"]["stock_columns"]) + list(base["features"]["market_columns"])
    members = mask[["date", "price_ticker"]].rename(columns={"price_ticker": "ticker"}).drop_duplicates()
    rows = train_panel.sort_values(["ticker", "date"]).copy()
    # Forward returns are computed on each series' full training history, then only member rows are kept.
    forecaster = RidgeForecaster(columns, horizon=21)
    rows["target"] = rows.groupby("ticker")["close"].transform(lambda s: s.shift(-21) / s - 1.0)
    rows = rows.merge(members, on=["date", "ticker"], how="inner").dropna(subset=["target", *columns])
    x, y = rows[columns].to_numpy(dtype=float), rows["target"].to_numpy(dtype=float)
    forecaster.mean_, forecaster.scale_ = x.mean(axis=0), x.std(axis=0)
    forecaster.scale_[forecaster.scale_ == 0] = 1.0
    xs = (x - forecaster.mean_) / forecaster.scale_
    forecaster.intercept_ = float(y.mean())
    forecaster.coef_ = np.linalg.solve(xs.T @ xs + forecaster.alpha * np.eye(xs.shape[1]), xs.T @ (y - forecaster.intercept_))
    forecaster.n_train_ = int(len(rows))
    return forecaster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    spec = yaml.safe_load(Path(args.config).read_text())
    base = _deep_merge({"features": spec["features"], "universe": spec["universe"]}, spec["base"])
    mask, panel, _ = load_inputs(spec, root)
    membership = pd.read_csv(root / spec["data"]["membership"], dtype=str)
    last = membership[membership["record_id"] == membership["record_id"].iloc[-1]]
    hindsight = set(last.loc[last["in_set50"] == "True", "ticker"]) & set(panel["ticker"])
    universes = {
        "hindsight_set50_equal_weight_monthly": fixed_universe_mask(panel, hindsight),
        "starter5_equal_weight_monthly": fixed_universe_mask(panel, set(STARTER5)),
    }
    limit = pd.Timestamp(spec["splits"]["validation_end"])
    lookback = int(spec["features"]["lookback"])
    for year in args.years:
        end = min(pd.Timestamp(f"{year}-12-31"), limit).date().isoformat()
        val = window(panel, f"{year}-01-01", end, lookback)
        curves = baseline_curves(base, val, mask)
        forecaster = fit_ridge(base, window(panel, None, f"{year - 1}-12-31", lookback), mask)
        env = make_env(base, val, mask, training=False, rebalance_every=21)
        curves.append(rollout(env, ridge_policy(forecaster, k=10), "pit_ridge_forecast_top10_monthly")[0])
        for name, universe_mask in universes.items():
            env = make_env(base, val, universe_mask, training=False, rebalance_every=21)
            curves.append(rollout(env, equal_weight_policy, name)[0])
        out = Path(args.run_dir) / f"fold_{year}" / "baselines_universes"
        out.mkdir(parents=True, exist_ok=True)
        frame = pd.concat(curves, ignore_index=True)
        frame.to_csv(out / "validation_equity_curves.csv", index=False)
        finals = frame.groupby("policy")["portfolio_value"].agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
        print(year, {k: round(v, 4) for k, v in finals.items()}, flush=True)
    print(f"hindsight universe: {len(hindsight)} names of {last['record_id'].iloc[0]}; reference {EQUAL_WEIGHT}")


if __name__ == "__main__":
    main()
