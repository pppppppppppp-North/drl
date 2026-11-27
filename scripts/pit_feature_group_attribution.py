"""Which inputs does a trained point-in-time agent use? (milestones M7 and M4 / RQ3)

For each saved model of one variant in a point-in-time walk-forward run, re-runs the fold's validation
year with one group of inputs changed and the model unchanged:

- ``shuffle``: the group's values are permuted across sessions (``--repeats`` times). That breaks their
  link with the market but keeps their distribution, as in ``scripts.feature_group_attribution`` for the
  five-stock study.
- ``mean``: the group is held at its training-window mean, which the shared-weight actor standardizes to
  zero, so the model gets no information from it. This also removes values outside the training range.

Groups follow the column prefixes: ``technical`` (every stock column), ``market`` (``set50_etf_``),
``policy_rate`` (``bot_``) and ``macro`` (``usdthb_``, ``brent_``, ``wti_``, ``gold_``, ``us10y_``).
Only groups among the variant's inputs are tested. Each row compares a changed run with the intact one:

- ``action_shift``: mean half-L1 distance between the intact and changed stock weights at each decision,
  as in the five-stock study (0 = the same portfolio);
- ``delta_return``, ``delta_sharpe``: change in the year's net return and Sharpe ratio;
- ``delta_cash``: change in the mean cash target over the decisions.

Only the observations change: membership, slots, prices and costs are fixed when the environment is
built. Before changing anything, every model must replay its saved validation curve. Writes
``attribution_<variant>.csv`` and ``attribution_<variant>_summary.csv`` into ``--output-dir`` (default:
the run directory).

    python -m scripts.pit_feature_group_attribution --config config/pit_walk_forward_policy_rate.yaml \
        --run-dir results/pit_walk_forward_policy --variant pit_plan_weekly_lstm_shared_policy
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.evaluate_pit_seed_ensemble import _predict, deciding, load_members
from scripts.run_pit_walk_forward import load_inputs, make_env, rollout, window
from scripts.run_reward_variant_experiment import _deep_merge
from src.evaluation.metrics import equity_metrics

MACRO_PREFIXES = ("usdthb_", "brent_", "wti_", "gold_", "us10y_")
MODES = ("shuffle", "mean")


def input_groups(stock_columns: list[str], market_columns: list[str]) -> dict[str, tuple[str, list[int]]]:
    """Group name -> (``stock`` or ``market`` tensor, column positions in that tensor)."""

    groups: dict[str, tuple[str, list[int]]] = {"technical": ("stock", list(range(len(stock_columns))))}
    rules = (
        ("market", lambda c: c.startswith("set50_etf_")),
        ("policy_rate", lambda c: c.startswith("bot_")),
        ("macro", lambda c: c.startswith(MACRO_PREFIXES)),
    )
    claimed: set[int] = set()
    for name, rule in rules:
        positions = [i for i, column in enumerate(market_columns) if rule(column)]
        if positions:
            groups[name] = ("market", positions)
            claimed.update(positions)
    rest = [i for i in range(len(market_columns)) if i not in claimed]
    if rest:
        groups["other_market"] = ("market", rest)
    return groups


def recording_policy(model, normalize, targets: list) -> Callable:
    """The model's own decisions, recording each decision's stock target weights (before any band)."""

    def policy(obs, env, step):
        if not deciding(env):
            return np.zeros(env.num_slots + 1)
        action = _predict(model, normalize, obs)
        slots = env.slots[env.current_step]
        seated = slots >= 0
        weights = env._slot_weights(action, slots)
        target = np.zeros(env.num_series)
        target[slots[seated]] = weights[seated]
        targets.append(target)
        return action

    return policy


def shuffle_change(tensor: str, positions: list[int], rng: np.random.Generator) -> Callable:
    def change(env) -> None:
        values = env.stock_tensor if tensor == "stock" else env.market_tensor
        order = rng.permutation(values.shape[0])
        values[..., positions] = values[order][..., positions]

    return change


def mean_change(tensor: str, positions: list[int], means: np.ndarray) -> Callable:
    def change(env) -> None:
        values = env.stock_tensor if tensor == "stock" else env.market_tensor
        values[..., positions] = means.astype(values.dtype)

    return change


def run_once(make: Callable, model, normalize, change: Callable | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    env = make()
    if change is not None:
        change(env)
    targets: list[np.ndarray] = []
    curve, _ = rollout(env, recording_policy(model, normalize, targets), "attribution")
    return curve, np.asarray(targets)


def compare(curve: pd.DataFrame, targets: np.ndarray, base_curve: pd.DataFrame, base_targets: np.ndarray) -> dict:
    metrics, base = equity_metrics(curve[["portfolio_value"]]), equity_metrics(base_curve[["portfolio_value"]])
    n = min(len(targets), len(base_targets))
    shift = float(0.5 * np.abs(targets[:n] - base_targets[:n]).sum(axis=1).mean()) if n else 0.0
    cash = float((1.0 - targets[:n].sum(axis=1)).mean()) if n else 0.0
    base_cash = float((1.0 - base_targets[:n].sum(axis=1)).mean()) if n else 0.0
    return {
        "action_shift": shift,
        "delta_return": metrics["cumulative_return"] - base["cumulative_return"],
        "delta_sharpe": metrics["sharpe"] - base["sharpe"],
        "delta_cash": cash - base_cash,
        "base_return": base["cumulative_return"],
        "base_cash": base_cash,
    }


def attribute(make: Callable, model, normalize, groups: dict, means: dict[str, np.ndarray], modes: list[str],
              repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_curve, base_targets = run_once(make, model, normalize)
    rows = []
    for name, (tensor, positions) in groups.items():
        for mode in modes:
            changes = ([shuffle_change(tensor, positions, rng) for _ in range(repeats)] if mode == "shuffle"
                       else [mean_change(tensor, positions, means[name])])
            for repeat, change in enumerate(changes):
                curve, targets = run_once(make, model, normalize, change)
                rows.append({"group": name, "mode": mode, "repeat": repeat, "features": len(positions),
                             **compare(curve, targets, base_curve, base_targets)})
    return pd.DataFrame(rows)


def training_means(train_panel: pd.DataFrame, stock_columns: list[str], market_columns: list[str],
                   groups: dict) -> dict[str, np.ndarray]:
    """Each group's training-window means, as the shared-weight actor computes them (``feature_statistics``)."""

    from src.models.shared_asset_policy import feature_statistics

    stats = feature_statistics(train_panel, stock_columns, market_columns)
    columns = {"stock": stock_columns, "market": market_columns}
    return {name: np.array([stats[columns[tensor][i]][0] for i in positions], dtype=np.float64)
            for name, (tensor, positions) in groups.items()}


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(["group", "mode"])
        .agg(features=("features", "first"), mean_action_shift=("action_shift", "mean"),
             mean_delta_return=("delta_return", "mean"), mean_abs_delta_return=("delta_return", lambda s: s.abs().mean()),
             mean_delta_sharpe=("delta_sharpe", "mean"), mean_delta_cash=("delta_cash", "mean"),
             runs=("fold", "size"))
        .reset_index()
        .sort_values(["mode", "mean_action_shift"], ascending=[False, False])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True, help="walk-forward run folder holding fold_<Y>/<variant>_seed<S>")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", default=None, help="where to write results (default: --run-dir)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--fold-years", nargs="*", type=int, default=None)
    parser.add_argument("--modes", nargs="*", default=list(MODES), choices=MODES)
    parser.add_argument("--repeats", type=int, default=5, help="permutations per group in the shuffle mode")
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
    stock_columns = list(config["features"]["stock_columns"])
    market_columns = list(config["features"]["market_columns"])
    groups = input_groups(stock_columns, market_columns)

    rows, replay_gaps = [], []
    for year in years:
        end = min(pd.Timestamp(f"{year}-12-31"), limit).date().isoformat()
        val = window(panel, f"{year}-01-01", end, lookback)
        if val["date"].max() > limit:
            raise AssertionError("validation window leaked past its end date")
        train_panel = window(panel, None, f"{year - 1}-12-31", lookback)
        means = training_means(train_panel, stock_columns, market_columns, groups)

        def make(val=val):
            return make_env(config, val, mask, training=False)

        for seed, model, normalize, run_dir in load_members(source, year, args.variant, seeds, make()):
            curve, _ = run_once(make, model, normalize)
            saved = pd.read_csv(run_dir / "validation_equity_curve.csv")["portfolio_value"].to_numpy()
            gap = float(np.max(np.abs(curve["portfolio_value"].to_numpy() / saved - 1.0)))
            replay_gaps.append(gap)
            if gap > 1e-6:
                raise SystemExit(f"fold {year} seed {seed}: replay differs from the saved curve (gap {gap:.2e})")
            frame = attribute(make, model, normalize, groups, means, list(args.modes), args.repeats, seed)
            rows.append(frame.assign(fold=year, seed=seed))
            print(f"{year} seed {seed}: " + ", ".join(
                f"{g}/{m}: shift={d['action_shift'].mean():.3f} dRet={d['delta_return'].mean():+.3f} "
                f"dCash={d['delta_cash'].mean():+.3f}" for (g, m), d in frame.groupby(["group", "mode"], sort=False)),
                flush=True)
    if not rows:
        raise SystemExit(f"no saved models for {args.variant} under {source}")
    detail = pd.concat(rows, ignore_index=True)
    detail.to_csv(out / f"attribution_{args.variant}.csv", index=False)
    summary = summarize(detail)
    summary.to_csv(out / f"attribution_{args.variant}_summary.csv", index=False)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    receipt = {"run_dir": str(source), "variant": args.variant, "seeds": seeds, "fold_years": years,
               "modes": list(args.modes), "repeats": args.repeats,
               "groups": {name: [(stock_columns if t == "stock" else market_columns)[i] for i in pos]
                          for name, (t, pos) in groups.items()},
               "max_replay_gap": max(replay_gaps), "code_commit": commit}
    (out / f"attribution_{args.variant}_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
