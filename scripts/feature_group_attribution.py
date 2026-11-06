"""Which data sources do the trained walk-forward agents rely on? (milestone M7 / RQ3)

Loads every saved PPO model (and its frozen VecNormalize statistics) for one variant in a
walk-forward run directory, and runs ``src.evaluation.attribution.permutation_attribution`` on that
fold's validation year. Writes ``attribution_<variant>.csv`` (every fold x seed x group x repeat) and
``attribution_<variant>_summary.csv`` (mean per group) into the run directory.

    python -m scripts.feature_group_attribution --config config/source_ablation_experiment.yaml \
        --run-dir results/source_ablation_<date> --variant src_technical_market_macro
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_reward_variant_experiment import _deep_merge, _eval_env, build_feature_table, split_with_context
from src.evaluation.attribution import default_feature_groups, permutation_attribution


def load_policy(run_dir: Path, env):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(run_dir / "ppo_model.zip", device="cpu")
    normalizer = None
    stats = run_dir / "vecnormalize.pkl"
    if stats.exists():
        vec = VecNormalize.load(str(stats), DummyVecEnv([lambda: env]))
        vec.training = False
        normalizer = vec.normalize_obs

    def policy(obs):
        return model.predict(normalizer(obs) if normalizer else obs, deterministic=True)[0]

    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    variant = next(v for v in spec["variants"] if v["name"] == args.variant)
    base = _deep_merge({"features": spec["features"]}, spec["base"])
    config = _deep_merge(base, {k: v for k, v in variant.items() if k in {"environment", "training", "features"}})
    env_cfg = config["environment"]
    lookback = int(config["features"]["lookback"])
    features = build_feature_table(spec, Path(args.root))
    groups = default_feature_groups(list(config["features"]["columns"]))
    run_root = Path(args.run_dir)
    rows = []
    for model_dir in sorted(run_root.glob(f"fold_*/{args.variant}_seed*/runs/{args.variant}_seed*")):
        if not (model_dir / "ppo_model.zip").exists():
            continue
        year = int(model_dir.parts[-4].split("_")[1])
        seed = int(model_dir.name.rsplit("seed", 1)[1])
        val = split_with_context(features, f"{year}-01-01", f"{year}-12-31", lookback)

        def make_env():
            return _eval_env(config, val, action_transform=env_cfg.get("action_transform", "clip"),
                             rebalance_every=int(env_cfg.get("rebalance_every", 1)))

        policy = load_policy(model_dir, make_env())
        frame = permutation_attribution(make_env, policy, groups, repeats=args.repeats, seed=seed)
        rows.append(frame.assign(fold=year, seed=seed))
        print(f"{year} seed {seed}: " + ", ".join(
            f"{g}: shift={d['action_shift'].mean():.3f} dRet={d['delta_cumulative_return'].mean():+.3f}"
            for g, d in frame.groupby("group")), flush=True)
    if not rows:
        raise SystemExit(f"no saved models for {args.variant} under {run_root}")
    detail = pd.concat(rows, ignore_index=True)
    detail.to_csv(run_root / f"attribution_{args.variant}.csv", index=False)
    summary = (
        detail.groupby("group")
        .agg(features=("features", "first"), mean_action_shift=("action_shift", "mean"),
             mean_delta_return=("delta_cumulative_return", "mean"), mean_delta_sharpe=("delta_sharpe", "mean"),
             runs=("fold", "size"))
        .reset_index()
        .sort_values("mean_action_shift", ascending=False)
    )
    summary.to_csv(run_root / f"attribution_{args.variant}_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
