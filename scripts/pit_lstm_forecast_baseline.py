"""LSTM forecast-then-allocate benchmark on the point-in-time SET50 (milestone M6).

The supervised sequence baseline of the five-stock study (``LSTMForecaster`` in
``src/agents/predictive_baselines.py``) scaled to the index, as ``scripts.pit_universe_baselines``
does for the ridge forecaster:

- **Samples:** for each fold, a Bi-LSTM is fitted on the training years only (2012 through December
  of the year before the validation year). A sample is a (member, decision session) pair on which the
  member was tradable, exactly as the agent sees it. The input is the 30 completed sessions before
  the decision: the stock's 11 technical features and the 5 SET50 ETF market features. The target is
  the return from the decision session's close to the close 21 sessions later. Decision sessions are
  taken every 5 sessions, and no target extends past the training end. Early stopping uses the last
  15% of the training decision sessions, and samples whose target reaches into that block are purged.
- **Allocation:** every 21 sessions, it equal-weights the ten highest positive forecasts among the
  tradable members (``top_k_allocation``; cash if none is positive), like the ridge baseline, with the
  same costs and accounting as the agent.
- **Fixed in advance:** the network and its training settings are the ``LSTMForecaster`` defaults of
  the five-stock study. The sampling stride (5), the ten holdings and the three seeds (42, 43, 44, as
  in the DRL runs) were set before any result was seen.

It also records each forecast's information coefficient: the rank correlation, across the members,
between the forecasts and the realized 21-session returns, on the validation decision sessions whose
21 sessions end inside the validation year.

Output: ``<run-dir>/fold_<Y>/baselines_lstm_forecast/validation_equity_curves.csv``, merged by
``scripts.aggregate_walk_forward_reward``, plus ``<run-dir>/forecaster_fits.csv`` (one row per fold
and seed) and ``<run-dir>/receipt.json``. Prices after ``splits.validation_end`` are dropped as soon
as they are loaded, so the 2023-2024 test years and the sealed holdouts are never used.

    python -m scripts.pit_lstm_forecast_baseline --config config/pit_walk_forward_experiment.yaml \\
        --run-dir results/pit_lstm_forecast --years 2016 2017 2018 2019 2020 2021 2022
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.run_pit_walk_forward import _deep_merge, baseline_curves, load_inputs, make_env, rollout, window
from src.agents.predictive_baselines import LSTMForecaster, top_k_allocation

NAME = "pit_lstm_forecast_top10_monthly"
HORIZON = 21


def windows_at(env, step: int, assets: np.ndarray) -> np.ndarray:
    """Raw inputs for a decision at ``step``: rows ``step - lookback`` .. ``step - 1``, as (assets, lookback, features)."""
    rows = slice(step - env.lookback, step)
    stock = env.stock_tensor[rows][:, assets, :]
    market = np.repeat(env.market_tensor[rows][:, None, :], len(assets), axis=1)
    return np.transpose(np.concatenate([stock, market], axis=-1), (1, 0, 2))


def training_samples(env, horizon: int = HORIZON, stride: int = 5):
    """Windows, targets, decision dates and target dates of the tradable members of a training window."""
    xs, ys, decision, target = [], [], [], []
    last = len(env.dates) - 1
    for t in range(env.lookback, last - horizon + 1, stride):
        assets = np.flatnonzero(env.tradable[t])
        start = env.close_prices[t, assets]
        end = env.close_prices[t + horizon, assets]
        keep = np.isfinite(start) & np.isfinite(end) & (start > 0)
        assets = assets[keep]
        if assets.size == 0:
            continue
        xs.append(windows_at(env, t, assets))
        ys.append(end[keep] / start[keep] - 1.0)
        decision += [env.dates[t]] * assets.size
        target += [env.dates[t + horizon]] * assets.size
    if not xs:
        raise ValueError("no tradable member has a complete window and a target in this training window")
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
            pd.DatetimeIndex(decision).to_numpy(), pd.DatetimeIndex(target).to_numpy())


@dataclass
class PitLSTMForecaster(LSTMForecaster):
    """``LSTMForecaster`` whose samples are the tradable members of a point-in-time training environment."""

    stride: int = 5

    def fit_env(self, env) -> "PitLSTMForecaster":
        self._cached = training_samples(env, self.horizon, self.stride)
        windows = self._cached[0]
        return self.fit(pd.DataFrame(windows.reshape(-1, windows.shape[-1]), columns=self.feature_columns))

    def _samples(self, frame: pd.DataFrame):
        windows, targets, decision, target = self._cached
        scaled = ((windows.astype(np.float64) - self.mean_) / self.scale_).astype(np.float32)
        return scaled, targets, decision, target


def deciding(env) -> bool:
    return env.steps_since_rebalance % env.rebalance_every == 0


def forecast_policy(forecaster: LSTMForecaster, k: int = 10, log: list | None = None):
    def policy(obs, env, step):
        weights = np.zeros(env.num_slots)
        if not deciding(env):
            return weights
        slots = env.slots[env.current_step]
        valid = slots >= 0
        if valid.any():
            assets = slots[valid]
            forecast = forecaster.predict(windows_at(env, env.current_step, assets))
            weights[valid] = top_k_allocation(forecast, k=k)
            if log is not None:
                log.append((env.current_step, assets, forecast))
        return weights
    return policy


def information_coefficients(env, log: list, horizon: int = HORIZON) -> list[float]:
    """Spearman correlation of forecasts with realized returns, per decision whose horizon ends in the window."""
    values = []
    for step, assets, forecast in log:
        if step + horizon > len(env.dates) - 1:
            continue
        realized = env.close_prices[step + horizon, assets] / env.close_prices[step, assets] - 1.0
        frame = pd.DataFrame({"forecast": forecast, "realized": realized}).dropna()
        if len(frame) >= 5:
            values.append(float(frame["forecast"].corr(frame["realized"], method="spearman")))
    return values


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="default: the config's seeds")
    parser.add_argument("--k", type=int, default=10, help="holdings")
    parser.add_argument("--stride", type=int, default=5, help="sessions between training decision sessions")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(1)
    spec = yaml.safe_load(Path(args.config).read_text())
    limit = pd.Timestamp(spec["splits"]["validation_end"])
    if any(pd.Timestamp(f"{y}-01-01") > limit for y in args.years):
        raise SystemExit(f"a year starts after the configured validation end {limit.date()}")
    seeds = args.seeds or [int(s) for s in spec["seeds"]]
    base = _deep_merge({"features": spec["features"], "universe": spec["universe"]}, spec["base"])
    columns = list(base["features"]["stock_columns"]) + list(base["features"]["market_columns"])
    lookback = int(base["features"]["lookback"])
    mask, panel, _ = load_inputs(spec, Path(args.root))
    panel = panel[panel["date"] <= limit].reset_index(drop=True)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    fits = []
    for year in args.years:
        train_env = make_env(base, window(panel, None, f"{year - 1}-12-31", lookback), mask, training=False)
        if train_env.dates[-1] >= pd.Timestamp(f"{year}-01-01"):
            raise AssertionError("training window reaches into the validation year")
        end = min(pd.Timestamp(f"{year}-12-31"), limit).date().isoformat()
        val = window(panel, f"{year}-01-01", end, lookback)
        curves = baseline_curves(base, val, mask)
        for seed in seeds:
            started = time.time()
            forecaster = PitLSTMForecaster(columns, lookback=lookback, horizon=HORIZON, seed=seed, stride=args.stride)
            forecaster.fit_env(train_env)
            env = make_env(base, val, mask, training=False, rebalance_every=21)
            log: list = []
            curve, _ = rollout(env, forecast_policy(forecaster, k=args.k, log=log), f"{NAME}_seed{seed}")
            curves.append(curve)
            ics = information_coefficients(env, log)
            cash = float(np.mean([top_k_allocation(f, k=args.k).sum() == 0 for _, _, f in log])) if log else float("nan")
            fits.append({"fold": year, "seed": seed, "train_samples": forecaster.n_train_,
                         "early_stopping_samples": forecaster.n_validation_,
                         "early_stopping_start": str(forecaster.validation_start_.date()),
                         "best_early_stopping_loss": forecaster.best_validation_loss_,
                         "decisions": len(log), "all_cash_decisions": cash,
                         "mean_information_coefficient": float(np.mean(ics)) if ics else float("nan"),
                         "ic_decisions": len(ics), "annual_return": curve["portfolio_value"].iloc[-1] / curve["portfolio_value"].iloc[0] - 1.0,
                         "seconds": round(time.time() - started, 1)})
            print(json.dumps(fits[-1]), flush=True)
        out = run_dir / f"fold_{year}" / "baselines_lstm_forecast"
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(curves, ignore_index=True).to_csv(out / "validation_equity_curves.csv", index=False)
        pd.DataFrame(fits).to_csv(run_dir / "forecaster_fits.csv", index=False)

    receipt = {"config": args.config, "years": args.years, "seeds": seeds, "k": args.k, "stride": args.stride,
               "horizon": HORIZON, "feature_columns": columns, "validation_end": str(limit.date()),
               "code_commit": _commit()}
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
