"""Heuristic benchmarks on the point-in-time SET50 (milestone M6).

On the same sessions, members, costs and accounting as ``scripts.run_pit_walk_forward``, rebalanced
every 21 sessions like the equal-weight baseline, and using only closing prices up to the session
before each decision (the agent's information lag):

- ``pit_momentum_top10_monthly``: equal weight in the ten tradable members with the highest
  12-1 month momentum, the return from 252 to 21 sessions before the decision (cross-sectional
  momentum). Members with less history are not ranked.
- ``pit_inverse_volatility_monthly``: every tradable member, weighted by the inverse of the
  standard deviation of its last 60 daily returns.
- ``pit_min_variance_monthly``: the long-only minimum-variance portfolio of the tradable members,
  from their last 252 daily returns. The sample covariance is shrunk halfway toward its diagonal,
  and each weight is capped at 10%. Members with fewer than 126 returns in that window are left
  out, and if the optimizer fails the portfolio falls back to inverse-variance weights.

The parameters are common textbook choices, fixed before any result was seen and not tuned. Output:
``<run-dir>/fold_<Y>/baselines_heuristic/validation_equity_curves.csv``, merged by
``scripts.aggregate_walk_forward_reward``. Prices after ``splits.validation_end`` are dropped as
soon as they are loaded, so the 2023-2024 test years and the sealed holdouts are never used.

    python -m scripts.pit_heuristic_baselines --config config/pit_walk_forward_experiment.yaml \\
        --run-dir results/pit_heuristic_baselines --years 2016 2017 2018 2019 2020 2021 2022
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from scripts.run_pit_walk_forward import _deep_merge, baseline_curves, load_inputs, make_env, rollout, window

MOMENTUM = "pit_momentum_top10_monthly"
INVERSE_VOL = "pit_inverse_volatility_monthly"
MIN_VARIANCE = "pit_min_variance_monthly"


def deciding(env) -> bool:
    return env.steps_since_rebalance % env.rebalance_every == 0


class History:
    """Forward-filled closes and daily returns of every series, by date, up to the validation end."""

    def __init__(self, panel: pd.DataFrame) -> None:
        closes = panel.pivot(index="date", columns="ticker", values="close").sort_index()
        self.observed = closes.notna()
        self.closes = closes.ffill()
        self.returns = self.closes.pct_change().where(self.observed)
        self.position = {pd.Timestamp(d): i for i, d in enumerate(self.closes.index)}

    def last_completed(self, env) -> int:
        """Row of the session before the decision session."""
        return self.position[pd.Timestamp(env.dates[env.current_step])] - 1

    def seated(self, env) -> tuple[np.ndarray, list[str]]:
        slots = env.slots[env.current_step]
        valid = slots >= 0
        return valid, [env.tickers[u] for u in slots[valid]]


def _to_slots(env, valid: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.zeros(env.num_slots)
    out[valid] = weights
    return out


def momentum_policy(history: History, k: int = 10, lookback: int = 252, skip: int = 21):
    def policy(obs, env, step):
        if not deciding(env):
            return np.zeros(env.num_slots)
        valid, tickers = history.seated(env)
        last = history.last_completed(env)
        weights = np.zeros(len(tickers))
        if tickers and last - lookback >= 0:
            start = history.closes.iloc[last - lookback][tickers].to_numpy(dtype=float)
            end = history.closes.iloc[last - skip][tickers].to_numpy(dtype=float)
            had_history = history.observed.iloc[: last - lookback + 1][tickers].any().to_numpy()
            score = np.where(had_history & (start > 0), end / start - 1.0, np.nan)
            ranked = [i for i in np.argsort(-np.nan_to_num(score, nan=-np.inf)) if np.isfinite(score[i])][:k]
            weights[ranked] = 1.0 / len(ranked) if ranked else 0.0
        if weights.sum() == 0 and tickers:
            weights[:] = 1.0 / len(tickers)
        return _to_slots(env, valid, weights)
    return policy


def inverse_volatility_policy(history: History, window_size: int = 60):
    def policy(obs, env, step):
        if not deciding(env):
            return np.zeros(env.num_slots)
        valid, tickers = history.seated(env)
        last = history.last_completed(env)
        recent = history.returns.iloc[max(0, last - window_size + 1) : last + 1][tickers]
        vol = recent.std().to_numpy(dtype=float)
        inverse = np.where(np.isfinite(vol) & (vol > 0), 1.0 / vol, 0.0)
        weights = inverse / inverse.sum() if inverse.sum() > 0 else np.full(len(tickers), 1.0 / max(1, len(tickers)))
        return _to_slots(env, valid, weights)
    return policy


def min_variance_weights(returns: pd.DataFrame, cap: float = 0.10, shrink: float = 0.5,
                         min_observations: int = 126) -> tuple[np.ndarray, bool]:
    """Long-only minimum variance with a weight cap; returns (weights over the columns, solved)."""
    counts = returns.notna().sum().to_numpy()
    usable = counts >= min_observations
    weights = np.zeros(returns.shape[1])
    if usable.sum() == 0:
        return weights, False
    cov = returns.loc[:, usable].cov(min_periods=min_observations // 2).to_numpy(dtype=float)
    variance = np.diag(cov).copy()
    cov = np.nan_to_num(cov, nan=0.0)
    cov = (1.0 - shrink) * cov + shrink * np.diag(variance)
    n = int(usable.sum())
    upper = max(cap, 1.0 / n)
    start = np.full(n, 1.0 / n)
    result = minimize(lambda w: float(w @ cov @ w), start, jac=lambda w: 2.0 * cov @ w, method="SLSQP",
                      bounds=[(0.0, upper)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                      options={"maxiter": 500, "ftol": 1e-12})
    solved = bool(result.success) and np.isfinite(result.x).all()
    if solved:
        chosen = np.clip(result.x, 0.0, None)
    else:
        inverse = np.where(variance > 0, 1.0 / variance, 0.0)
        chosen = inverse / inverse.sum()
    weights[usable] = chosen / chosen.sum()
    return weights, solved


def min_variance_policy(history: History, window_size: int = 252, failures: list | None = None):
    def policy(obs, env, step):
        if not deciding(env):
            return np.zeros(env.num_slots)
        valid, tickers = history.seated(env)
        last = history.last_completed(env)
        recent = history.returns.iloc[max(0, last - window_size + 1) : last + 1][tickers]
        weights, solved = min_variance_weights(recent)
        if not solved and failures is not None:
            failures.append(str(pd.Timestamp(env.dates[env.current_step]).date()))
        if weights.sum() == 0 and tickers:
            weights = np.full(len(tickers), 1.0 / len(tickers))
        return _to_slots(env, valid, weights)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    limit = pd.Timestamp(spec["splits"]["validation_end"])
    if any(pd.Timestamp(f"{y}-01-01") > limit for y in args.years):
        raise SystemExit(f"a year starts after the configured validation end {limit.date()}")
    base = _deep_merge({"features": spec["features"], "universe": spec["universe"]}, spec["base"])
    mask, panel, _ = load_inputs(spec, Path(args.root))
    panel = panel[panel["date"] <= limit].reset_index(drop=True)
    history = History(panel)
    lookback = int(spec["features"]["lookback"])
    failures: list[str] = []
    for year in args.years:
        end = min(pd.Timestamp(f"{year}-12-31"), limit).date().isoformat()
        val = window(panel, f"{year}-01-01", end, lookback)
        curves = baseline_curves(base, val, mask)
        for name, policy in ((MOMENTUM, momentum_policy(history)), (INVERSE_VOL, inverse_volatility_policy(history)),
                             (MIN_VARIANCE, min_variance_policy(history, failures=failures))):
            env = make_env(base, val, mask, training=False, rebalance_every=21)
            curves.append(rollout(env, policy, name)[0])
        out = Path(args.run_dir) / f"fold_{year}" / "baselines_heuristic"
        out.mkdir(parents=True, exist_ok=True)
        frame = pd.concat(curves, ignore_index=True)
        frame.to_csv(out / "validation_equity_curves.csv", index=False)
        finals = frame.groupby("policy")["portfolio_value"].agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
        print(year, {k: round(v, 4) for k, v in finals.items()}, flush=True)
    print(f"minimum-variance decisions that fell back to inverse variance: {len(failures)} {failures[:5]}")


if __name__ == "__main__":
    main()
