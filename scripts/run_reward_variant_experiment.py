"""Milestone M1 experiment: compare reward/accounting designs on a multi-year validation window.

Every policy, including the legacy-trained one, is *evaluated* under correct self-financing
accounting so the equity curves are comparable real P&L. Only the validation window is read; the
test years and the sealed holdout are never loaded.

Usage::

    python -m scripts.run_reward_variant_experiment --config config/long_term_reward_experiment.yaml \
        --output-dir results/long_term_reward_experiment_20261106 [--variants log_weekly ...] [--timesteps N]
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.envs.long_term_env import LongTermTradingEnv
from src.evaluation.benchmark_metrics import equity_metrics_with_benchmark
from src.evaluation.significance import significance_table
from src.features.macro_context import merge_macro_context
from src.features.market_context import build_index_context
from src.features.technical import add_technical_features


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_feature_table(config: dict, root: Path) -> pd.DataFrame:
    data_cfg = config["data"]
    prices = pd.read_csv(root / data_cfg["prices"], parse_dates=["date"])
    if data_cfg.get("drop_all_zero_volume_dates", True):
        closed = prices.groupby("date")["volume"].apply(lambda s: bool((s <= 0).all()))
        prices = prices[~prices["date"].isin(closed[closed].index)]
    features = add_technical_features(prices[["date", "ticker", "open", "high", "low", "close", "volume"]])
    index = pd.read_csv(root / data_cfg["index"], parse_dates=["date"])
    index = index.assign(index_name=data_cfg["index_name"])
    context = build_index_context(index, data_cfg["index_name"], target_dates=features["date"])
    merged = features.merge(context, on="date", how="left")
    merged["benchmark_close"] = merged["date"].map(index.drop_duplicates("date").set_index("date")["close"])
    macro_path = data_cfg.get("macro")
    if macro_path:
        # Daily macro proxies merged backward-only onto Thai trading dates (src.features.macro_context);
        # the environment's observation window adds one more session of lag before any decision.
        macro = pd.read_csv(root / macro_path, parse_dates=["date"])
        names = data_cfg.get("macro_names", {})
        macro = macro.assign(macro_name=macro["ticker"].map(lambda t: names.get(t, str(t).lower())))
        merged = merge_macro_context(merged, macro[["date", "macro_name", "close"]])
    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)


def split_with_context(features: pd.DataFrame, start: str | None, end: str, lookback: int) -> pd.DataFrame:
    dates = pd.Index(sorted(features["date"].unique()))
    end_ts = pd.Timestamp(end)
    if start is None:
        keep = dates[dates <= end_ts]
    else:
        start_pos = int(dates.searchsorted(pd.Timestamp(start)))
        if start_pos < lookback:
            raise ValueError("not enough history before the split start for the lookback context")
        keep = dates[start_pos - lookback :]
        keep = keep[keep <= end_ts]
    return features[features["date"].isin(keep)].reset_index(drop=True)


def _eval_env(config: dict, features: pd.DataFrame, *, action_transform: str, rebalance_every: int) -> LongTermTradingEnv:
    env_cfg = config["environment"]
    return LongTermTradingEnv(
        features=features,
        feature_columns=config["features"]["columns"],
        lookback=int(config["features"]["lookback"]),
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        slippage=env_cfg.get("slippage", 0.0),
        accounting="self_financing",
        rebalance_every=rebalance_every,
        action_transform=action_transform,
    )


def _rollout(env: LongTermTradingEnv, policy, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs, info = env.reset(seed=0)
    rows, actions = [dict(date=info["date"], policy=name, portfolio_value=info["portfolio_value"], turnover=0.0,
                          transaction_cost_paid=0.0, slippage_paid=0.0, cash_weight=info["cash_weight"])], []
    done, step = False, 0
    while not done:
        action = policy(obs, env, step)
        obs, _, done, _, info = env.step(action)
        rows.append(dict(date=info["date"], policy=name, portfolio_value=info["portfolio_value"], turnover=info["turnover"],
                         transaction_cost_paid=info["transaction_cost_paid"], slippage_paid=info["slippage_paid"],
                         cash_weight=info["cash_weight"]))
        actions.extend(dict(date=info["date"], policy=name, ticker=t, weight=float(w)) for t, w in zip(env.tickers, info["weights"]))
        step += 1
    return pd.DataFrame(rows), pd.DataFrame(actions)


def baseline_curves(config: dict, features: pd.DataFrame, train_features: pd.DataFrame | None = None) -> list[pd.DataFrame]:
    curves = []
    if train_features is not None:
        from src.agents.predictive_baselines import RidgeForecaster, forecaster_policy

        forecaster = RidgeForecaster(list(config["features"]["columns"]), horizon=21).fit(train_features)
        env = _eval_env(config, features, action_transform="clip", rebalance_every=21)
        curves.append(_rollout(env, forecaster_policy(forecaster, k=2), "ridge_forecast_top2_monthly")[0])
    env = _eval_env(config, features, action_transform="clip", rebalance_every=1)
    n = env.num_assets
    equal = np.full(n, 1.0 / n)
    curves.append(_rollout(env, lambda o, e, s: equal if s == 0 else e.weights, "equal_weight_buy_hold")[0])
    env = _eval_env(config, features, action_transform="clip", rebalance_every=21)
    curves.append(_rollout(env, lambda o, e, s: equal, "equal_weight_monthly")[0])
    for i, ticker in enumerate(env.tickers):
        one = np.zeros(n)
        one[i] = 1.0
        env = _eval_env(config, features, action_transform="clip", rebalance_every=1)
        curves.append(_rollout(env, lambda o, e, s, one=one: one if s == 0 else e.weights, f"buy_hold_{ticker}")[0])
    # Investable market benchmark: buy-and-hold the SET50 ETF over the same decision dates.
    env = _eval_env(config, features, action_transform="clip", rebalance_every=1)
    dates = pd.to_datetime(pd.Series(env.dates[env.lookback :]))
    bench = features.drop_duplicates("date").set_index("date")["benchmark_close"].reindex(dates).ffill().to_numpy()
    initial = config["environment"]["initial_cash"] * (1 - config["environment"]["transaction_cost"] - config["environment"].get("slippage", 0.0))
    curves.append(pd.DataFrame({"date": dates.dt.date.astype(str), "policy": "set50_etf_buy_hold",
                                "portfolio_value": np.r_[config["environment"]["initial_cash"], initial * bench[1:] / bench[0]],
                                "turnover": np.r_[0.0, 1.0, np.zeros(len(dates) - 2)]}))
    return curves


def train_variant(config: dict, train_features: pd.DataFrame, seed: int, out_dir: Path):
    from src.train import _algorithm_kwargs
    from src.train_long_term import build_long_term_training_env, policy_kwargs_from_config
    from stable_baselines3 import PPO

    training = config["training"]
    env = build_long_term_training_env(config, train_features, seed)
    kwargs = _algorithm_kwargs("ppo", int(training["total_timesteps"]), training)
    policy_kwargs = policy_kwargs_from_config(config, train_features)
    if policy_kwargs:
        kwargs["policy_kwargs"] = policy_kwargs
    model = PPO("MlpPolicy", env, seed=seed, verbose=int(training.get("verbose", 0)), device="cpu", **kwargs)
    model.learn(total_timesteps=int(training["total_timesteps"]))
    normalizer = None
    if hasattr(env, "obs_rms"):
        env.training = False
        normalizer = env.normalize_obs
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "ppo_model")
    if hasattr(env, "save"):
        env.save(str(out_dir / "vecnormalize.pkl"))
    return model, normalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--aggregate-dirs", nargs="*", default=None, help="merge finished per-variant output dirs")
    parser.add_argument("--baselines-only", action="store_true", help="evaluate only the non-learning baselines")
    parser.add_argument("--fold-year", type=int, default=None,
                        help="walk-forward fold: train through Dec 31 of the previous year, validate on this calendar year")
    args = parser.parse_args()

    if args.aggregate_dirs:
        aggregate_dirs([Path(d) for d in args.aggregate_dirs], Path(args.output_dir))
        return
    root = Path(args.root)
    spec = yaml.safe_load(Path(args.config).read_text())
    if args.fold_year is not None:
        limit = pd.Timestamp(spec["splits"]["validation_end"])
        if pd.Timestamp(f"{args.fold_year}-01-01") > limit:
            raise SystemExit(f"--fold-year {args.fold_year} starts after the configured validation end {limit.date()}")
        # The fold never reads past the configured validation end (the last session before the test years).
        fold_end = min(pd.Timestamp(f"{args.fold_year}-12-31"), limit)
        spec["splits"] = {
            "train_end": f"{args.fold_year - 1}-12-31",
            "validation_start": f"{args.fold_year}-01-01",
            "validation_end": fold_end.date().isoformat(),
        }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lookback = int(spec["features"]["lookback"])
    features = build_feature_table(spec, root)
    train = split_with_context(features, None, spec["splits"]["train_end"], lookback)
    val = split_with_context(features, spec["splits"]["validation_start"], spec["splits"]["validation_end"], lookback)
    if pd.to_datetime(val["date"]).max() > pd.Timestamp(spec["splits"]["validation_end"]):
        raise AssertionError("validation window leaked past its end date")
    split_info = {
        "train": [str(train["date"].min().date()), str(train["date"].max().date()), int(train["date"].nunique())],
        "validation_with_context": [str(val["date"].min().date()), str(val["date"].max().date()), int(val["date"].nunique())],
        "lookback_context_sessions": lookback,
        "tickers": sorted(features["ticker"].unique().tolist()),
    }
    (out / "splits.json").write_text(json.dumps(split_info, indent=2) + "\n")
    print(json.dumps(split_info))

    base = _deep_merge({"features": spec["features"]}, spec["base"])
    curves = baseline_curves(base, val, train_features=train)
    if args.baselines_only:
        aggregate(pd.concat(curves, ignore_index=True), out, n_trials=1)
        return
    seeds = args.seeds or spec["seeds"]
    selected = [v for v in spec["variants"] if not args.variants or v["name"] in args.variants]
    timing = []
    for variant in selected:
        config = _deep_merge(base, {k: v for k, v in variant.items() if k in {"environment", "training", "features"}})
        if args.timesteps:
            config["training"]["total_timesteps"] = args.timesteps
        env_cfg = config["environment"]
        for seed in seeds:
            started = time.time()
            run_dir = out / "runs" / f"{variant['name']}_seed{seed}"
            model, normalizer = train_variant(config, train, seed, run_dir)
            eval_env = _eval_env(
                config, val, action_transform=env_cfg.get("action_transform", "clip"), rebalance_every=int(env_cfg.get("rebalance_every", 1))
            )

            def policy(obs, env, step, model=model, normalizer=normalizer):
                prepared = normalizer(obs) if normalizer is not None else obs
                return model.predict(prepared, deterministic=True)[0]

            curve, actions = _rollout(eval_env, policy, f"{variant['name']}_seed{seed}")
            curve.to_csv(run_dir / "validation_equity_curve.csv", index=False)
            actions.to_csv(run_dir / "validation_actions.csv", index=False)
            curves.append(curve)
            elapsed = time.time() - started
            timing.append({"variant": variant["name"], "seed": seed, "seconds": round(elapsed, 1),
                           "timesteps": int(config["training"]["total_timesteps"])})
            print(f"{variant['name']} seed={seed} done in {elapsed:.0f}s final={curve['portfolio_value'].iloc[-1]:,.0f}", flush=True)

    all_curves = pd.concat(curves, ignore_index=True)
    pd.DataFrame(timing).to_csv(out / "timing.csv", index=False)
    aggregate(all_curves, out, n_trials=max(1, len(selected)))


def aggregate(all_curves: pd.DataFrame, out: Path, n_trials: int) -> None:
    all_curves.to_csv(out / "validation_equity_curves.csv", index=False)
    bench = all_curves[all_curves["policy"] == "set50_etf_buy_hold"][["date", "portfolio_value"]]
    rows = []
    for name, curve in all_curves.groupby("policy", sort=False):
        m = {"policy": name}
        m.update(equity_metrics_with_benchmark(curve.reset_index(drop=True), bench.reset_index(drop=True)))
        rows.append(m)
    metrics = pd.DataFrame(rows)
    metrics["variant"] = metrics["policy"].str.replace(r"_seed\d+$", "", regex=True)
    metrics.to_csv(out / "validation_metrics.csv", index=False)
    cols = ["cumulative_return", "annualized_return", "sharpe", "sortino", "max_drawdown", "calmar", "avg_turnover",
            "avg_cash_weight", "beta", "information_ratio"]
    summary = metrics.groupby("variant", sort=False)[cols].agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out / "validation_summary.csv", index=False)
    sig = significance_table(all_curves, reference="equal_weight_buy_hold", n_trials=n_trials, n_boot=1000)
    sig.to_csv(out / "validation_significance.csv", index=False)
    print(summary[[c for c in summary.columns if c.endswith("_mean")]].round(4).to_string())


def aggregate_dirs(dirs: list[Path], out: Path) -> None:
    """Merge per-variant output directories (run in parallel) into one result set."""

    frames, timing, variants = [], [], set()
    for d in dirs:
        curves = pd.read_csv(d / "validation_equity_curves.csv")
        frames.append(curves)
        if (d / "timing.csv").exists():
            t = pd.read_csv(d / "timing.csv")
            timing.append(t)
            variants.update(t["variant"].unique())
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "policy"], keep="first")
    out.mkdir(parents=True, exist_ok=True)
    if timing:
        pd.concat(timing, ignore_index=True).to_csv(out / "timing.csv", index=False)
    aggregate(merged, out, n_trials=max(1, len(variants)))


if __name__ == "__main__":
    main()
