"""Compare the same trained policy from two run folders, fold by fold (reproducibility checks).

Each run folder holds ``fold_<year>/<variant>_seed<seed>/runs/<variant>_seed<seed>/`` with
``validation_equity_curve.csv`` and ``validation_holdings.csv`` (the walk-forward task pool's layout). For
every fold and seed present in both, this reports how far apart the two models' validation results are:
the largest relative gap in portfolio value on any session, the final values and their gap, the largest
gap in any holding's weight, and the turnover of each. Identical files give zero on every column.

Usage (BistKA's retrained models against the Mac's):
    python -m scripts.compare_model_runs --left ~/drl_runs/pit_walk_forward_cash_cap_20261119 \
        --right ~/drl_runs/pit_cash_cap_100k_replication_bistka_20261126 \
        --variant pit_plan_weekly_lstm_shared_cashcap --seeds 42 --output comparison.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _policy_dir(run: Path, year: int, variant: str, seed: int) -> Path:
    name = f"{variant}_seed{seed}"
    return run / f"fold_{year}" / name / "runs" / name


def compare_policy(left: Path, right: Path) -> dict:
    """Gaps between two policy folders' validation curves and holdings."""
    curves = [pd.read_csv(d / "validation_equity_curve.csv", parse_dates=["date"]).set_index("date") for d in (left, right)]
    if not curves[0].index.equals(curves[1].index):
        raise ValueError(f"the two curves cover different sessions: {left} and {right}")
    a, b = (c["portfolio_value"].astype(float) for c in curves)
    holdings = []
    for d in (left, right):
        h = pd.read_csv(d / "validation_holdings.csv", parse_dates=["date"])
        holdings.append(h.pivot_table(index="date", columns="ticker", values="weight", aggfunc="sum"))
    weights = holdings[0].align(holdings[1], fill_value=0.0)
    return {
        "sessions": int(len(a)),
        "final_left": float(a.iloc[-1] / a.iloc[0]),
        "final_right": float(b.iloc[-1] / b.iloc[0]),
        "final_gap": float(b.iloc[-1] / b.iloc[0] - a.iloc[-1] / a.iloc[0]),
        "max_relative_value_gap": float(np.max(np.abs(b.to_numpy() / a.to_numpy() - 1.0))),
        "max_weight_gap": float(np.nanmax(np.abs(weights[0].fillna(0.0) - weights[1].fillna(0.0)).to_numpy()))
        if len(weights[0]) else 0.0,
        "turnover_left": float(curves[0]["turnover"].sum()),
        "turnover_right": float(curves[1]["turnover"].sum()),
    }


def compare_runs(left: Path, right: Path, variant: str, seeds: list[int], years: list[int] | None = None) -> pd.DataFrame:
    if years is None:
        years = sorted(int(p.name.split("_")[1]) for p in left.glob("fold_*") if p.is_dir())
    rows = []
    for year in years:
        for seed in seeds:
            dirs = [_policy_dir(run, year, variant, seed) for run in (left, right)]
            if not all((d / "validation_equity_curve.csv").is_file() for d in dirs):
                continue
            rows.append({"fold": year, "seed": seed, **compare_policy(*dirs)})
    if not rows:
        raise ValueError(f"no fold and seed of {variant} is in both {left} and {right}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--years", type=int, nargs="*")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    table = compare_runs(args.left.expanduser(), args.right.expanduser(), args.variant, args.seeds, args.years)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
