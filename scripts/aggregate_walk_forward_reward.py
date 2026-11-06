"""Aggregate a walk-forward reward experiment (milestone M6 / RQ5).

Expected layout, produced by ``scripts.run_reward_variant_experiment --fold-year Y``::

    <run-dir>/fold_<Y>/<part>/validation_equity_curves.csv

where each ``<part>`` holds baselines or one DRL variant for that validation year. Outputs:

- ``walk_forward_metrics.csv``: one row per fold x policy (every seed);
- ``walk_forward_fold_returns.csv``: seed-averaged annual return per fold x variant;
- ``walk_forward_summary.csv``: across-fold summary per variant (mean return, mean excess vs the SET50
  ETF, folds beating the ETF, worst-fold excess, mean drawdown and turnover, seed dispersion);
- ``walk_forward_chained_equity.png``: out-of-sample equity chained across validation years.

    python -m scripts.aggregate_walk_forward_reward --run-dir results/walk_forward_reward_20261106
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.benchmark_metrics import equity_metrics_with_benchmark

BENCHMARK = "set50_etf_buy_hold"


def load_fold(fold_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(fold_dir.glob("*/validation_equity_curves.csv"))]
    if not frames:
        raise FileNotFoundError(f"no validation_equity_curves.csv under {fold_dir}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "policy"], keep="first")


def fold_metrics(curves: pd.DataFrame, year: int) -> pd.DataFrame:
    bench = curves[curves["policy"] == BENCHMARK][["date", "portfolio_value"]].reset_index(drop=True)
    rows = []
    for policy, curve in curves.groupby("policy", sort=False):
        metrics = equity_metrics_with_benchmark(curve.reset_index(drop=True), bench if not bench.empty else None)
        rows.append({"fold": year, "policy": policy, **metrics})
    frame = pd.DataFrame(rows)
    frame["variant"] = frame["policy"].str.replace(r"_seed\d+$", "", regex=True)
    frame["seed"] = frame["policy"].str.extract(r"_seed(\d+)$")[0]
    bench_return = frame.loc[frame["policy"] == BENCHMARK, "cumulative_return"]
    frame["excess_vs_benchmark"] = frame["cumulative_return"] - (float(bench_return.iloc[0]) if len(bench_return) else np.nan)
    return frame


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_fold = (
        metrics.groupby(["variant", "fold"])
        .agg(
            cumulative_return=("cumulative_return", "mean"),
            excess_vs_benchmark=("excess_vs_benchmark", "mean"),
            max_drawdown=("max_drawdown", "mean"),
            avg_turnover=("avg_turnover", "mean"),
            seed_std=("cumulative_return", "std"),
            seeds=("cumulative_return", "size"),
        )
        .reset_index()
    )
    summary = (
        per_fold.groupby("variant")
        .agg(
            folds=("fold", "nunique"),
            mean_annual_return=("cumulative_return", "mean"),
            mean_excess_vs_set50_etf=("excess_vs_benchmark", "mean"),
            median_excess_vs_set50_etf=("excess_vs_benchmark", "median"),
            folds_beating_set50_etf=("excess_vs_benchmark", lambda s: int((s > 0).sum())),
            worst_fold_excess=("excess_vs_benchmark", "min"),
            mean_max_drawdown=("max_drawdown", "mean"),
            mean_turnover=("avg_turnover", "mean"),
            mean_seed_std=("seed_std", "mean"),
            seeds_per_fold=("seeds", "max"),
        )
        .reset_index()
        .sort_values("mean_excess_vs_set50_etf", ascending=False)
    )
    return per_fold, summary


def chained_equity(fold_curves: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Chain each policy's fold curves (each fold restarts at its own initial value) into one OOS path."""

    pieces = []
    level: dict[str, float] = {}
    for year in sorted(fold_curves):
        curves = fold_curves[year].copy()
        curves["date"] = pd.to_datetime(curves["date"])
        for policy, curve in curves.groupby("policy", sort=False):
            curve = curve.sort_values("date")
            rel = curve["portfolio_value"].to_numpy() / curve["portfolio_value"].iloc[0]
            start = level.get(policy, 1.0)
            pieces.append(pd.DataFrame({"date": curve["date"].to_numpy(), "policy": policy, "value": start * rel}))
            level[policy] = start * rel[-1]
    chained = pd.concat(pieces, ignore_index=True)
    chained["variant"] = chained["policy"].str.replace(r"_seed\d+$", "", regex=True)
    return chained


def plot(chained: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#d1495b", "#00798c", "#edae49", "#30638e", "#66a182"]
    baselines = {
        BENCHMARK: ("SET50 ETF buy & hold (TDEX)", "#111111", "-"),
        "ridge_forecast_top2_monthly": ("Ridge forecast top-2, monthly", "#7a7a7a", "--"),
        "equal_weight_monthly": ("Equal weight, monthly", "#7a7a7a", ":"),
    }
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    learned = [v for v in chained["variant"].unique() if v not in baselines and chained.loc[chained["variant"] == v, "policy"].str.contains("_seed").any()]
    for color, variant in zip(palette, learned):
        wide = chained[chained["variant"] == variant].pivot_table(index="date", columns="policy", values="value")
        ax.fill_between(wide.index, wide.min(axis=1), wide.max(axis=1), color=color, alpha=0.15, linewidth=0)
        ax.plot(wide.index, wide.mean(axis=1), color=color, linewidth=1.6, label=f"{variant} (mean of {wide.shape[1]} seeds)")
    for policy, (label, color, style) in baselines.items():
        part = chained[chained["policy"] == policy]
        if not part.empty:
            ax.plot(part["date"], part["value"], color=color, linestyle=style, linewidth=1.4, label=label)
    ax.axhline(1.0, color="#bbbbbb", linewidth=0.8)
    ax.set_ylabel("Chained out-of-sample value (start = 1.0)")
    ax.set_title("Walk-forward validation years, chained (each year trained only on earlier data; net of costs)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fold_dirs = sorted(p for p in run_dir.glob("fold_*") if p.is_dir())
    if not fold_dirs:
        raise SystemExit(f"no fold_<year> directories under {run_dir}")
    fold_curves = {int(p.name.split("_")[1]): load_fold(p) for p in fold_dirs}
    metrics = pd.concat([fold_metrics(curves, year) for year, curves in fold_curves.items()], ignore_index=True)
    metrics.to_csv(run_dir / "walk_forward_metrics.csv", index=False)
    per_fold, summary = summarize(metrics)
    per_fold.to_csv(run_dir / "walk_forward_fold_returns.csv", index=False)
    summary.to_csv(run_dir / "walk_forward_summary.csv", index=False)
    chained = chained_equity(fold_curves)
    chained.to_csv(run_dir / "walk_forward_chained_equity.csv", index=False)
    plot(chained, run_dir / "walk_forward_chained_equity.png")
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print()
    print(per_fold.pivot_table(index="fold", columns="variant", values="cumulative_return").round(3).to_string())


if __name__ == "__main__":
    main()
