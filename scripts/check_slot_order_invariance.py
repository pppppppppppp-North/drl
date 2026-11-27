"""Check whether a trained point-in-time policy depends on the slot order.

Rolls each trained model out deterministically on one validation year twice, once with sticky slots
and once with slots ordered by traded value, and reports on how many sessions the two orders seat
members differently and how far apart the two equity curves end. A policy that scores every stock
with shared weights (``actor: shared_asset``) gives the same curve up to rounding; the dense head
does not. Like the walk-forward runner, it never reads past ``splits.validation_end``.

    python -m scripts.check_slot_order_invariance --config config/pit_walk_forward_experiment.yaml \\
        --fold-year 2019 --run-dirs results/<run>/fold_2019/<variant>_seed42/runs/<variant>_seed42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.run_pit_walk_forward import load_inputs, make_env, rollout, window
from scripts.run_reward_variant_experiment import _deep_merge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold-year", type=int, required=True)
    parser.add_argument("--run-dirs", nargs="+", required=True, help="folders holding <algo>_model.zip and model_info.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    import stable_baselines3 as sb3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    spec = yaml.safe_load(Path(args.config).read_text())
    limit = pd.Timestamp(spec["splits"]["validation_end"])
    if pd.Timestamp(f"{args.fold_year}-01-01") > limit:
        raise SystemExit(f"--fold-year {args.fold_year} starts after the configured validation end {limit.date()}")
    mask, panel, _ = load_inputs(spec, Path(args.root))
    lookback = int(spec["features"]["lookback"])
    fold_end = min(pd.Timestamp(f"{args.fold_year}-12-31"), limit).date().isoformat()
    val = window(panel, f"{args.fold_year}-01-01", fold_end, lookback)
    base = _deep_merge({"features": spec["features"], "universe": spec["universe"]}, spec["base"])
    envs = {order: make_env(_deep_merge(base, {"universe": {"slot_order": order}}), val, mask, training=False)
            for order in ("sticky", "traded_value")}
    sticky, traded = envs["sticky"], envs["traded_value"]
    differ = int((sticky.slots[lookback:-1] != traded.slots[lookback:-1]).any(axis=1).sum())

    rows = []
    for run_dir in map(Path, args.run_dirs):
        algo = json.loads((run_dir / "model_info.json").read_text())["algo"]
        model = getattr(sb3, algo.upper()).load(run_dir / f"{algo}_model", device="cpu")
        normalize = None
        if (run_dir / "vecnormalize.pkl").exists():
            probe = make_env(base, val, mask, training=False)
            normalize = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), DummyVecEnv([lambda: probe])).normalize_obs

        def policy(obs, env, step, model=model, normalize=normalize):
            return model.predict(normalize(obs) if normalize else obs, deterministic=True)[0]

        finals = {}
        for order, env in envs.items():
            curve, _ = rollout(env, policy, order)
            finals[order] = curve["portfolio_value"].to_numpy()
        gap = float(np.max(np.abs(finals["sticky"] / finals["traded_value"] - 1.0)))
        rows.append({"run": run_dir.name, "sessions_seated_differently": differ,
                     "final_sticky": round(float(finals["sticky"][-1]), 2),
                     "final_traded_value": round(float(finals["traded_value"][-1]), 2), "max_relative_gap": gap})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
