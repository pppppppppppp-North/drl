"""Evaluate trained point-in-time models as a seed ensemble and with a no-trade band (no training).

For every fold, this loads the trained models of one variant (one per seed) from a walk-forward run
and rolls them out deterministically on that fold's validation year:

- ``<variant>_ensemble``: on every decision session, each seed's model scores the same observation
  (the ensemble's own holdings), and the portfolio takes the mean of their target weights, cash
  included;
- ``<variant>_evalband_seed<S>``: each seed's model alone, with a no-trade band applied at
  evaluation only (the models were trained without it);
- ``<variant>_ensemble_band``: the ensemble with the same band.

A variant trained with a band (``environment.no_trade_band``) is evaluated in its own environment,
band included, so only ``--modes ensemble`` applies to it.

Each seed's model is also replayed without changes, and the run stops if that replay does not match
the validation curve saved by the walk-forward run. The baselines are recomputed on the same
sessions. Output: ``<out>/fold_<Y>/<mode>/validation_equity_curves.csv`` and
``<out>/fold_<Y>/<mode>/runs/<policy>/validation_holdings.csv``, the layout read by
``scripts.aggregate_walk_forward_reward`` and ``scripts.walk_forward_agent_diagnostics``. Like the
walk-forward runner, it never reads past ``splits.validation_end``, so the 2023-2024 test years and
the sealed holdouts are never loaded.

    python -m scripts.evaluate_pit_seed_ensemble --config config/pit_walk_forward_experiment.yaml \\
        --source-run results/pit_walk_forward_shared_20261110 --variant pit_plan_weekly_lstm_shared \\
        --output-dir results/pit_seed_ensemble_20261110 --band 0.005
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.run_pit_walk_forward import baseline_curves, load_inputs, make_env, rollout, window
from scripts.run_reward_variant_experiment import _deep_merge

MODES = ("ensemble", "evalband", "ensemble_band")


def deciding(env) -> bool:
    """True on the sessions where the environment uses the action (it ignores it in between)."""
    return env.steps_since_rebalance % env.rebalance_every == 0


def load_members(source: Path, year: int, variant: str, seeds: list[int], probe) -> list[tuple]:
    import stable_baselines3 as sb3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    members = []
    for seed in seeds:
        run_dir = source / f"fold_{year}" / f"{variant}_seed{seed}" / "runs" / f"{variant}_seed{seed}"
        algo = json.loads((run_dir / "model_info.json").read_text())["algo"]
        model = getattr(sb3, algo.upper()).load(run_dir / f"{algo}_model", device="cpu")
        normalize = None
        if (run_dir / "vecnormalize.pkl").exists():
            normalize = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), DummyVecEnv([lambda: probe])).normalize_obs
        members.append((seed, model, normalize, run_dir))
    return members


def _predict(model, normalize, obs) -> np.ndarray:
    return model.predict(normalize(obs) if normalize is not None else obs, deterministic=True)[0]


def seed_policy(model, normalize):
    def policy(obs, env, step):
        if not deciding(env):
            return np.zeros(env.num_slots + 1)
        return _predict(model, normalize, obs)
    return policy


def ensemble_policy(members):
    def policy(obs, env, step):
        if not deciding(env):
            return np.zeros(env.num_slots)
        slots = env.slots[env.current_step]
        targets = [env._slot_weights(_predict(model, normalize, obs), slots) for _, model, normalize, _ in members]
        return np.mean(targets, axis=0)  # slot weights; the environment keeps 1 - sum as cash
    return policy


def _save(out: Path, mode: str, baselines: list[pd.DataFrame], results: list[tuple[str, pd.DataFrame, pd.DataFrame]]) -> None:
    folder = out / mode
    for name, curve, holdings in results:
        run_dir = folder / "runs" / name
        run_dir.mkdir(parents=True, exist_ok=True)
        curve.to_csv(run_dir / "validation_equity_curve.csv", index=False)
        holdings.to_csv(run_dir / "validation_holdings.csv", index=False)
    pd.concat([*baselines, *[curve for _, curve, _ in results]], ignore_index=True).to_csv(
        folder / "validation_equity_curves.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-run", required=True, help="walk-forward run folder holding fold_<Y>/<variant>_seed<S>")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--fold-years", nargs="*", type=int, default=None)
    parser.add_argument("--band", type=float, default=0.005, help="no-trade band for the band modes")
    parser.add_argument("--modes", nargs="*", default=list(MODES), choices=MODES)
    parser.add_argument("--suffix", default="", help="appended to the new policies' names, for example _b0100 in a band sensitivity run")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    entry = next((v for v in spec["variants"] if v["name"] == args.variant), None)
    if entry is None:
        raise SystemExit(f"variant {args.variant!r} is not in {args.config}")
    limit = pd.Timestamp(spec["splits"]["validation_end"])
    source = Path(args.source_run)
    years = args.fold_years or sorted(int(p.name.split("_")[1]) for p in source.glob("fold_*") if p.is_dir())
    if any(pd.Timestamp(f"{y}-01-01") > limit for y in years):
        raise SystemExit(f"a fold year starts after the configured validation end {limit.date()}")
    seeds = args.seeds or [int(s) for s in spec["seeds"]]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mask, panel, _ = load_inputs(spec, Path(args.root))
    lookback = int(spec["features"]["lookback"])
    base = _deep_merge({"features": spec["features"], "universe": spec["universe"]}, spec["base"])
    config = _deep_merge(base, {k: v for k, v in entry.items() if k in {"environment", "training", "features", "universe"}})
    banded = _deep_merge(config, {"environment": {"no_trade_band": float(args.band)}})
    if float(config["environment"].get("no_trade_band", 0.0)) > 0 and {"evalband", "ensemble_band"} & set(args.modes):
        raise SystemExit("the source variant already trades with a band, so the band modes would not change it; "
                         "pass --modes ensemble")

    replay_gaps = []
    for year in years:
        end = min(pd.Timestamp(f"{year}-12-31"), limit).date().isoformat()
        val = window(panel, f"{year}-01-01", end, lookback)
        if val["date"].max() > limit:
            raise AssertionError("validation window leaked past its end date")
        baselines = baseline_curves(base, val, mask)
        members = load_members(source, year, args.variant, seeds, make_env(config, val, mask, training=False))
        fold_out = out / f"fold_{year}"

        for seed, model, normalize, run_dir in members:  # the unchanged models must replay their saved curves
            curve, _ = rollout(make_env(config, val, mask, training=False), seed_policy(model, normalize), "replay")
            saved = pd.read_csv(run_dir / "validation_equity_curve.csv")["portfolio_value"].to_numpy()
            gap = float(np.max(np.abs(curve["portfolio_value"].to_numpy() / saved - 1.0)))
            replay_gaps.append({"fold": year, "seed": seed, "max_relative_gap": gap})
            if gap > 1e-6:
                raise SystemExit(f"fold {year} seed {seed}: replay differs from the saved curve (gap {gap:.2e})")

        if "ensemble" in args.modes:
            name = f"{args.variant}_ensemble{args.suffix}"
            curve, holdings = rollout(make_env(config, val, mask, training=False), ensemble_policy(members), name)
            _save(fold_out, "ensemble", baselines, [(name, curve, holdings)])
        if "evalband" in args.modes:
            results = []
            for seed, model, normalize, _ in members:
                name = f"{args.variant}_evalband{args.suffix}_seed{seed}"
                curve, holdings = rollout(make_env(banded, val, mask, training=False), seed_policy(model, normalize), name)
                results.append((name, curve, holdings))
            _save(fold_out, "evalband", baselines, results)
        if "ensemble_band" in args.modes:
            name = f"{args.variant}_ensemble_band{args.suffix}"
            curve, holdings = rollout(make_env(banded, val, mask, training=False), ensemble_policy(members), name)
            _save(fold_out, "ensemble_band", baselines, [(name, curve, holdings)])

        written = pd.concat([pd.read_csv(f) for f in sorted(fold_out.glob("*/validation_equity_curves.csv"))])
        written = written.drop_duplicates(["date", "policy"])
        finals = {p: c["portfolio_value"].iloc[-1] / c["portfolio_value"].iloc[0] - 1.0 for p, c in written.groupby("policy")}
        print(f"fold {year}: " + ", ".join(f"{p} {r:+.2%}" for p, r in sorted(finals.items())), flush=True)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    receipt = {"source_run": str(source), "variant": args.variant, "seeds": seeds, "fold_years": years,
               "band": float(args.band), "modes": list(args.modes), "suffix": args.suffix, "code_commit": commit,
               "max_replay_gap": max(g["max_relative_gap"] for g in replay_gaps)}
    (out / "ensemble_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    pd.DataFrame(replay_gaps).to_csv(out / "replay_check.csv", index=False)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
