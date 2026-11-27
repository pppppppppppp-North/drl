"""How often did the cash cap bind? (milestone M5)

For each saved model of a point-in-time variant with ``environment.max_cash_weight`` below 1, replays the
fold's validation year and records, at every decision, two cash weights:

- ``own``: the policy's own cash weight, from the same action with the cap switched off;
- ``capped``: the cash weight of the target the environment trades towards.

The cap binds on a decision when the policy's own cash weight is above it. The actions, and so the
portfolio, are those of the capped environment; before anything is recorded, every model must replay its
saved validation curve. Writes ``cash_cap_binding_<variant>.csv`` (one row per model) into
``--output-dir`` (default: the run directory).

    python -m scripts.pit_cash_cap_binding --config config/pit_walk_forward_experiment.yaml \
        --run-dir results/pit_walk_forward_cash_cap --variant pit_plan_weekly_lstm_shared_cashcap
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.evaluate_pit_seed_ensemble import _predict, deciding, load_members
from scripts.run_pit_walk_forward import load_inputs, make_env, rollout, window
from scripts.run_reward_variant_experiment import _deep_merge


def cash_weights(env, action: np.ndarray) -> tuple[float, float]:
    """(policy's own cash weight, capped cash weight) of one action on the current session."""

    slots = env.slots[env.current_step]
    cap = env.max_cash_weight
    try:
        env.max_cash_weight = 1.0
        own = 1.0 - float(env._slot_weights(action, slots).sum())
    finally:
        env.max_cash_weight = cap
    capped = 1.0 - float(env._slot_weights(action, slots).sum())
    return own, capped


def recording_policy(model, normalize, records: list) -> Callable:
    """The model's own decisions, recording both cash weights of each decision."""

    def policy(obs, env, step):
        if not deciding(env):
            return np.zeros(env.num_slots + 1)
        action = _predict(model, normalize, obs)
        records.append(cash_weights(env, action))
        return action

    return policy


def binding(records: list[tuple[float, float]], cap: float) -> dict:
    own, capped = np.asarray(records, dtype=np.float64).reshape(-1, 2).T
    return {
        "decisions": int(own.size),
        "share_binding": float((own > cap + 1e-9).mean()) if own.size else 0.0,
        "mean_own_cash": float(own.mean()) if own.size else 0.0,
        "max_own_cash": float(own.max()) if own.size else 0.0,
        "mean_capped_cash": float(capped.mean()) if capped.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True, help="walk-forward run folder holding fold_<Y>/<variant>_seed<S>")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", default=None, help="where to write results (default: --run-dir)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--fold-years", nargs="*", type=int, default=None)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    entry = next((v for v in spec["variants"] if v["name"] == args.variant), None)
    if entry is None:
        raise SystemExit(f"variant {args.variant!r} is not in {args.config}")
    limit = pd.Timestamp(spec["splits"]["validation_end"])
    source = Path(args.run_dir)
    years = args.fold_years or sorted(int(p.name.split("_")[1]) for p in source.glob("fold_*") if p.is_dir())
    if any(pd.Timestamp(f"{y}-01-01") > limit for y in years):
        raise SystemExit(f"a fold year starts after the configured validation end {limit.date()}")
    seeds = args.seeds or [int(s) for s in spec["seeds"]]
    out = Path(args.output_dir or args.run_dir)
    out.mkdir(parents=True, exist_ok=True)

    mask, panel, _ = load_inputs(spec, Path(args.root))
    lookback = int(spec["features"]["lookback"])
    base = _deep_merge({"features": spec["features"], "universe": spec["universe"]}, spec["base"])
    config = _deep_merge(base, {k: v for k, v in entry.items() if k in {"environment", "training", "features", "universe"}})

    rows = []
    for year in years:
        end = min(pd.Timestamp(f"{year}-12-31"), limit).date().isoformat()
        val = window(panel, f"{year}-01-01", end, lookback)
        if val["date"].max() > limit:
            raise AssertionError("validation window leaked past its end date")

        def make(val=val):
            return make_env(config, val, mask, training=False)

        probe = make()
        if probe.max_cash_weight >= 1.0:
            raise SystemExit(f"{args.variant} has no cash cap (environment.max_cash_weight is 1)")
        for seed, model, normalize, run_dir in load_members(source, year, args.variant, seeds, probe):
            records: list[tuple[float, float]] = []
            env = make()
            curve, _ = rollout(env, recording_policy(model, normalize, records), "cash_cap_binding")
            saved = pd.read_csv(run_dir / "validation_equity_curve.csv")["portfolio_value"].to_numpy()
            gap = float(np.max(np.abs(curve["portfolio_value"].to_numpy() / saved - 1.0)))
            if gap > 1e-6:
                raise SystemExit(f"fold {year} seed {seed}: replay differs from the saved curve (gap {gap:.2e})")
            rows.append({"fold": year, "seed": seed, "cap": env.max_cash_weight,
                         **binding(records, env.max_cash_weight), "replay_gap": gap})
            print(rows[-1], flush=True)
    if not rows:
        raise SystemExit(f"no saved models for {args.variant} under {source}")
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f"cash_cap_binding_{args.variant}.csv", index=False)
    overall = float((frame["share_binding"] * frame["decisions"]).sum() / frame["decisions"].sum())
    pd.set_option("display.width", 200)
    print(frame.round(3).to_string(index=False))
    print(f"the cap bound on {overall:.1%} of {int(frame['decisions'].sum())} decisions, "
          f"in {int((frame['share_binding'] > 0).sum())} of {len(frame)} models")


if __name__ == "__main__":
    main()
