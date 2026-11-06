"""Evaluate the LSTM forecast-then-allocate baseline on walk-forward folds (milestone M6).

For each validation year it trains ``LSTMForecaster`` on that fold's training dates only, then
runs the monthly top-k allocation on the validation year under the same self-financing accounting
and costs as the DRL agents. The curves go to ``<run-dir>/fold_<Y>/forecast_baselines/``, where
``scripts.aggregate_walk_forward_reward`` picks them up with the other policies.

    python -m scripts.evaluate_forecast_baselines --config config/long_term_reward_experiment.yaml \
        --run-dir results/walk_forward_reward_<date> --years 2016 2017 2018 2019 2020 2021 2022
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_reward_variant_experiment import (
    _deep_merge,
    _eval_env,
    _rollout,
    build_feature_table,
    split_with_context,
)
from src.agents.predictive_baselines import LSTMForecaster, sequence_forecaster_policy


def fold_splits(year: int, validation_end_limit: str) -> dict[str, str]:
    if pd.Timestamp(f"{year}-12-31") > pd.Timestamp(validation_end_limit):
        raise SystemExit(f"fold {year} would read past the configured validation end {validation_end_limit}")
    return {"train_end": f"{year - 1}-12-31", "validation_start": f"{year}-01-01", "validation_end": f"{year}-12-31"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    lookback = int(spec["features"]["lookback"])
    features = build_feature_table(spec, Path(args.root))
    base = _deep_merge({"features": spec["features"]}, spec["base"])
    columns = list(base["features"]["columns"])
    for year in args.years:
        splits = fold_splits(year, spec["splits"]["validation_end"])
        train = split_with_context(features, None, splits["train_end"], lookback)
        val = split_with_context(features, splits["validation_start"], splits["validation_end"], lookback)
        curves = []
        for seed in args.seeds:
            started = time.time()
            model = LSTMForecaster(columns, lookback=lookback, horizon=21, seed=seed).fit(train)
            env = _eval_env(base, val, action_transform="clip", rebalance_every=21)
            name = f"lstm_forecast_top{args.top_k}_monthly_seed{seed}"
            curve, _ = _rollout(env, sequence_forecaster_policy(model, k=args.top_k), name)
            curves.append(curve)
            print(f"{year} {name}: final={curve['portfolio_value'].iloc[-1]:,.0f} "
                  f"train={model.n_train_} val={model.n_validation_} ({time.time() - started:.0f}s)", flush=True)
        out = Path(args.run_dir) / f"fold_{year}" / "forecast_baselines"
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(curves, ignore_index=True).to_csv(out / "validation_equity_curves.csv", index=False)


if __name__ == "__main__":
    main()
