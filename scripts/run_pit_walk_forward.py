"""Point-in-time SET50 walk-forward (milestone M3).

One fold trains on 2012 through December of year Y-1 and validates on year Y. The investable set
on every session is the SET50 membership in force at that session, from the official 2012-2024
archive, restricted to members with Yahoo prices. Baselines on the same sessions:

- ``pit_equal_weight_monthly``: equal weight across the tradable members, rebalanced every 21
  sessions;
- ``set50_etf_buy_hold``: the SET50 ETF (TDEX), which tracks the full index, including members
  that Yahoo has no prices for.

With ``data.policy_rate`` set, the official BOT policy-rate features are added as market-level
inputs (see ``add_policy_rate``); a variant uses them by listing them in its market columns. With
``data.macro`` set, the daily macro proxies (USD/THB, oil, gold, the US 10-year yield) are added the
same way (see ``add_macro_proxies``). With ``data.sector: true``, four sector-relative stock inputs from
the archive's dated SET sectors are added (``src.features.pit_sector_context``); a variant uses them by
listing them in its stock columns.

Validation never reads past ``splits.validation_end`` (2022-12-30), so the 2023-2024 test years
and the sealed 2025+ holdouts are never loaded. The output layout matches
``scripts.aggregate_walk_forward_reward`` (``<run>/fold_<Y>/<part>/validation_equity_curves.csv``).

Usage::

    python -m scripts.run_pit_walk_forward --config config/pit_walk_forward_experiment.yaml \\
        --output-dir results/pit_walk_forward/fold_2019/pit_plan_weekly_lstm_seed42 \\
        --fold-year 2019 --variants pit_plan_weekly_lstm --seeds 42 [--timesteps N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.run_reward_variant_experiment import _deep_merge
from src.data.point_in_time_panel import build_point_in_time_inputs
from src.envs.long_term_env import LongTermRewardConfig
from src.envs.point_in_time_env import PointInTimeSlotEnv
from src.evaluation.benchmark_metrics import equity_metrics_with_benchmark
from src.evaluation.significance import significance_table

BENCHMARK = "set50_etf_buy_hold"
EQUAL_WEIGHT = "pit_equal_weight_monthly"


def load_inputs(spec: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = spec["data"]
    mask, panel, receipt = build_point_in_time_inputs(
        root=root,
        membership_path=data["membership"],
        price_map_path=data["price_map"],
        price_paths=list(data["prices"]),
        index_path=data["index"],
        index_name=data.get("index_name", "set50_etf"),
        # Prices after 2024 exist only in the sealed holdout; only the holdout stage's config sets data.end
        # (scripts/run_holdout_evaluation.py). Every other run keeps the 2024-12-31 default.
        end=str(data.get("end", "2024-12-31")),
    )
    if data.get("sector"):
        # Peers come from the full SET100 list of each session, so this runs before the SET50 filter.
        from src.features.pit_sector_context import add_sector_context

        membership = pd.read_csv(root / data["membership"], dtype=str, keep_default_na=False)
        panel = add_sector_context(panel, mask, membership)
        receipt = {**receipt, "sector": f"dated SET sectors from {data['membership']}"}
    if data.get("universe", "set50") == "set50":
        mask = mask[mask["in_set50"]]
    if data.get("policy_rate"):
        history = pd.read_csv(root / data["policy_rate"])
        panel = add_policy_rate(panel, history, spec["splits"]["validation_end"])
        receipt = {**receipt, "policy_rate": str(data["policy_rate"])}
    if data.get("macro"):
        macro = pd.read_csv(root / data["macro"], parse_dates=["date"])
        panel = add_macro_proxies(panel, macro, data.get("macro_names", {}), spec["splits"]["validation_end"])
        receipt = {**receipt, "macro": str(data["macro"])}
    return mask.reset_index(drop=True), panel, receipt


def add_policy_rate(panel: pd.DataFrame, history: pd.DataFrame, validation_end: str) -> pd.DataFrame:
    """Add the official BOT policy-rate features (``src.features.policy_rate_context``) to every row.

    They are market-level: one value per session, the same for every stock. A decision becomes
    available from the session after its announcement. Decisions announced after ``validation_end``
    are dropped first, so nothing from the test years or the holdout enters.
    """
    from src.features.policy_rate_context import merge_policy_rate_context

    history = history[pd.to_datetime(history["announcement_date"]) <= pd.Timestamp(validation_end)]
    return merge_policy_rate_context(panel, history).drop(columns=["bot_source_announcement"])


def add_macro_proxies(panel: pd.DataFrame, macro: pd.DataFrame, names: dict, validation_end: str) -> pd.DataFrame:
    """Add the daily macro-proxy features (``src.features.macro_context``) to every row.

    They are market-level: one value per session, the same for every stock. Each proxy's latest close
    on or before a Thai session is carried onto it, and a decision reads only the sessions before it
    (the environment's information lag), so a close from the US evening is used from the next Thai
    session on. Prices after ``validation_end`` are dropped first, so nothing from the test years or
    the holdout enters. ``names`` maps Yahoo tickers to feature prefixes, for example ``{"BZ=F": "brent"}``.
    """
    from src.features.macro_context import merge_macro_context

    macro = macro[pd.to_datetime(macro["date"]) <= pd.Timestamp(validation_end)]
    macro = macro.assign(macro_name=macro["ticker"].map(lambda t: names.get(t, str(t).lower())))
    return merge_macro_context(panel, macro[["date", "macro_name", "close"]])


def window(panel: pd.DataFrame, start: str | None, end: str, lookback: int) -> pd.DataFrame:
    """Sessions up to ``end``; from ``start`` with ``lookback`` sessions of earlier context."""
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    end_ts = pd.Timestamp(end)
    if start is None:
        keep = dates[dates <= end_ts]
    else:
        position = int(dates.searchsorted(pd.Timestamp(start)))
        if position < lookback:
            raise ValueError("not enough sessions before the window start for the lookback")
        keep = dates[position - lookback :]
        keep = keep[keep <= end_ts]
    return panel[panel["date"].isin(keep)].reset_index(drop=True)


def make_env(config: dict, panel: pd.DataFrame, mask: pd.DataFrame, *, training: bool,
             rebalance_every: int | None = None) -> PointInTimeSlotEnv:
    env_cfg, feat = config["environment"], config["features"]
    episode = env_cfg.get("train_episode_length") if training else None
    return PointInTimeSlotEnv(
        panel,
        mask,
        feat["stock_columns"],
        feat["market_columns"],
        num_slots=int(config["universe"]["num_slots"]),
        lookback=int(feat["lookback"]),
        min_history_sessions=int(config["universe"].get("min_history_sessions", 60)),
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        slippage=env_cfg.get("slippage", 0.0),
        reward_config=LongTermRewardConfig(**env_cfg.get("reward", {})),
        rebalance_every=int(rebalance_every or env_cfg.get("rebalance_every", 1)),
        random_start=bool(env_cfg.get("train_random_start", False)) if training else False,
        episode_length=int(episode) if episode is not None else None,
        slot_order=str(config["universe"].get("slot_order", "traded_value")),
        no_trade_band=float(env_cfg.get("no_trade_band", 0.0)),
        max_cash_weight=float(env_cfg.get("max_cash_weight", 1.0)),
    )


def rollout(env: PointInTimeSlotEnv, policy, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs, info = env.reset(seed=0)
    rows = [{"date": info["date"], "policy": name, "portfolio_value": info["portfolio_value"], "turnover": 0.0,
             "transaction_cost_paid": 0.0, "slippage_paid": 0.0, "cash_weight": info["cash_weight"],
             "forced_exit_turnover": 0.0, "tradable_count": int(env.tradable[env.current_step].sum())}]
    holdings = []
    done, step = False, 0
    while not done:
        obs, _, done, _, info = env.step(policy(obs, env, step))
        rows.append({key: info[key] for key in ("date", "portfolio_value", "turnover", "transaction_cost_paid",
                                                "slippage_paid", "cash_weight", "forced_exit_turnover", "tradable_count")}
                    | {"policy": name})
        holdings.extend({"date": info["date"], "policy": name, "ticker": t, "weight": w} for t, w in info["holdings"].items())
        step += 1
    return pd.DataFrame(rows), pd.DataFrame(holdings)


def equal_weight_policy(obs, env: PointInTimeSlotEnv, step: int) -> np.ndarray:
    valid = env.slots[env.current_step] >= 0
    return valid / max(1, int(valid.sum()))


def baseline_curves(config: dict, val: pd.DataFrame, mask: pd.DataFrame) -> list[pd.DataFrame]:
    env = make_env(config, val, mask, training=False, rebalance_every=21)
    curve, _ = rollout(env, equal_weight_policy, EQUAL_WEIGHT)
    dates = pd.to_datetime(pd.Series(env.dates[env.lookback :]))
    bench = val.drop_duplicates("date").set_index("date")["benchmark_close"].reindex(dates).ffill().to_numpy()
    env_cfg = config["environment"]
    initial = env_cfg["initial_cash"] * (1 - env_cfg["transaction_cost"] - env_cfg.get("slippage", 0.0))
    etf = pd.DataFrame({"date": dates.dt.date.astype(str), "policy": BENCHMARK,
                        "portfolio_value": np.r_[env_cfg["initial_cash"], initial * bench[1:] / bench[0]],
                        "turnover": np.r_[0.0, 1.0, np.zeros(len(dates) - 2)]})
    return [curve, etf]


def train(config: dict, train_panel: pd.DataFrame, mask: pd.DataFrame, seed: int, out_dir: Path):
    import stable_baselines3 as sb3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from src.models.shared_asset_policy import feature_statistics, policy_from_config
    from src.train import _algorithm_kwargs

    training = config["training"]
    algo = str(training.get("algo", "ppo")).lower()
    if algo not in {"ppo", "a2c"}:
        raise ValueError("the point-in-time runner supports the on-policy algorithms ppo and a2c")

    def factory():
        env = make_env(config, train_panel, mask, training=True)
        env.reset(seed=seed)
        return Monitor(env)

    vec_env = DummyVecEnv([factory])
    if training.get("normalize_observations", False):
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    probe = vec_env.envs[0] if hasattr(vec_env, "envs") else vec_env.venv.envs[0]
    feature_columns = probe.unwrapped.feature_columns
    kwargs = _algorithm_kwargs(algo, int(training["total_timesteps"]), training)
    policy_config = training.get("policy")
    actor = str((policy_config or {}).get("actor", "dense"))
    # The shared-asset actor standardizes features with statistics from the training window only.
    stats = (feature_statistics(train_panel, config["features"]["stock_columns"], config["features"]["market_columns"])
             if actor == "shared_asset" else None)
    policy, policy_kwargs = policy_from_config(
        policy_config, lookback=int(config["features"]["lookback"]), num_assets=int(config["universe"]["num_slots"]),
        feature_columns=feature_columns, feature_stats=stats,
        normalize_observations=bool(training.get("normalize_observations", False)))
    if policy_kwargs:
        kwargs["policy_kwargs"] = policy_kwargs
    model_cls = getattr(sb3, algo.upper())
    model = model_cls(policy, vec_env, seed=seed, verbose=int(training.get("verbose", 0)), device="cpu", **kwargs)
    model.learn(total_timesteps=int(training["total_timesteps"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / f"{algo}_model")
    (out_dir / "model_info.json").write_text(json.dumps({"algo": algo, "actor": actor,
                                                          "timesteps": int(training["total_timesteps"]),
                                                          "feature_columns": feature_columns}) + "\n")
    normalizer = None
    if isinstance(vec_env, VecNormalize):
        vec_env.training = False
        vec_env.save(str(out_dir / "vecnormalize.pkl"))
        normalizer = vec_env.normalize_obs
    return model, normalizer


def summarize(curves: pd.DataFrame, out: Path, n_trials: int) -> pd.DataFrame:
    bench = curves[curves["policy"] == BENCHMARK][["date", "portfolio_value"]].reset_index(drop=True)
    rows = []
    for name, curve in curves.groupby("policy", sort=False):
        rows.append({"policy": name, **equity_metrics_with_benchmark(curve.reset_index(drop=True), bench)})
    metrics = pd.DataFrame(rows)
    metrics["variant"] = metrics["policy"].str.replace(r"_seed\d+$", "", regex=True)
    metrics.to_csv(out / "validation_metrics.csv", index=False)
    if EQUAL_WEIGHT in set(curves["policy"]):
        significance_table(curves, reference=EQUAL_WEIGHT, n_trials=n_trials, n_boot=1000).to_csv(
            out / "validation_significance.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--fold-year", type=int, required=True)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    spec = yaml.safe_load(Path(args.config).read_text())
    limit = pd.Timestamp(spec["splits"]["validation_end"])
    if pd.Timestamp(f"{args.fold_year}-01-01") > limit:
        raise SystemExit(f"--fold-year {args.fold_year} starts after the configured validation end {limit.date()}")
    fold_end = min(pd.Timestamp(f"{args.fold_year}-12-31"), limit).date().isoformat()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mask, panel, receipt = load_inputs(spec, root)
    lookback = int(spec["features"]["lookback"])
    train_panel = window(panel, None, f"{args.fold_year - 1}-12-31", lookback)
    val = window(panel, f"{args.fold_year}-01-01", fold_end, lookback)
    if val["date"].max() > limit:
        raise AssertionError("validation window leaked past its end date")
    base = _deep_merge({"features": spec["features"], "universe": spec["universe"]}, spec["base"])
    probe = make_env(base, val, mask, training=False)
    split_info = {
        "fold_year": args.fold_year,
        "train": [str(train_panel["date"].min().date()), str(train_panel["date"].max().date()),
                  int(train_panel["date"].nunique())],
        "validation_with_context": [str(val["date"].min().date()), str(val["date"].max().date()), int(val["date"].nunique())],
        "validation_series": probe.num_series,
        "tradable_members_per_session": {
            "min": int(probe.tradable[probe.lookback:-1].sum(axis=1).min()),
            "median": float(np.median(probe.tradable[probe.lookback:-1].sum(axis=1))),
        },
        "inputs": receipt["inputs"],
    }
    (out / "splits.json").write_text(json.dumps(split_info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in split_info.items() if k != "inputs"}), flush=True)

    curves = baseline_curves(base, val, mask)
    timing, holdings_frames = [], []
    seeds = args.seeds or spec["seeds"]
    selected = [] if args.baselines_only else [v for v in spec["variants"] if not args.variants or v["name"] in args.variants]
    for variant in selected:
        config = _deep_merge(base, {k: v for k, v in variant.items() if k in {"environment", "training", "features", "universe"}})
        if args.timesteps:
            config["training"]["total_timesteps"] = args.timesteps
        for seed in seeds:
            started = time.time()
            run_dir = out / "runs" / f"{variant['name']}_seed{seed}"
            model, normalizer = train(config, train_panel, mask, seed, run_dir)
            env = make_env(config, val, mask, training=False)

            def policy(obs, env, step, model=model, normalizer=normalizer):
                prepared = normalizer(obs) if normalizer is not None else obs
                return model.predict(prepared, deterministic=True)[0]

            name = f"{variant['name']}_seed{seed}"
            curve, holdings = rollout(env, policy, name)
            curve.to_csv(run_dir / "validation_equity_curve.csv", index=False)
            holdings.to_csv(run_dir / "validation_holdings.csv", index=False)
            curves.append(curve)
            holdings_frames.append(holdings)
            elapsed = time.time() - started
            timing.append({"variant": variant["name"], "seed": seed, "seconds": round(elapsed, 1),
                           "timesteps": int(config["training"]["total_timesteps"])})
            print(f"{name} fold={args.fold_year} done in {elapsed:.0f}s final={curve['portfolio_value'].iloc[-1]:,.0f}",
                  flush=True)

    all_curves = pd.concat(curves, ignore_index=True)
    all_curves.to_csv(out / "validation_equity_curves.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "timing.csv", index=False)
    metrics = summarize(all_curves, out, n_trials=max(1, len(selected)))
    cols = ["policy", "cumulative_return", "sharpe", "max_drawdown", "avg_turnover", "avg_cash_weight"]
    print(metrics[[c for c in cols if c in metrics.columns]].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
