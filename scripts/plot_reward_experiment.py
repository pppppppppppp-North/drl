"""Plot validation equity curves from scripts.run_reward_variant_experiment output.

DRL variants are drawn as the across-seed mean with a min-max band; baselines as single lines.

    python -m scripts.plot_reward_experiment --run-dir results/long_term_reward_experiment_20261106
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BASELINES = {
    "set50_etf_buy_hold": ("SET50 ETF buy & hold (TDEX)", "#111111", "-"),
    "ridge_forecast_top2_monthly": ("Ridge forecast top-2, monthly", "#7a7a7a", "--"),
    "equal_weight_monthly": ("Equal weight, monthly", "#7a7a7a", ":"),
}
PALETTE = ["#d1495b", "#edae49", "#00798c", "#30638e", "#66a182", "#8e5572"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    curves = pd.read_csv(run_dir / "validation_equity_curves.csv", parse_dates=["date"])
    curves["value"] = curves.groupby("policy")["portfolio_value"].transform(lambda s: s / s.iloc[0])
    curves["variant"] = curves["policy"].str.replace(r"_seed\d+$", "", regex=True)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    learned = [v for v in curves["variant"].unique() if v not in BASELINES and curves.loc[curves["variant"] == v, "policy"].str.contains("_seed").any()]
    for color, variant in zip(PALETTE, learned):
        wide = curves[curves["variant"] == variant].pivot_table(index="date", columns="policy", values="value")
        ax.fill_between(wide.index, wide.min(axis=1), wide.max(axis=1), color=color, alpha=0.15, linewidth=0)
        ax.plot(wide.index, wide.mean(axis=1), color=color, linewidth=1.6, label=f"{variant} (mean of {wide.shape[1]} seeds)")
    for policy, (label, color, style) in BASELINES.items():
        part = curves[curves["policy"] == policy]
        if not part.empty:
            ax.plot(part["date"], part["value"], color=color, linestyle=style, linewidth=1.4, label=label)
    ax.axhline(1.0, color="#bbbbbb", linewidth=0.8)
    ax.set_ylabel("Portfolio value (start = 1.0)")
    ax.set_title("Validation 2020-2022: reward and rebalancing designs vs baselines (net of costs)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left", ncol=2, frameon=False)
    fig.tight_layout()
    output = Path(args.output) if args.output else run_dir / "validation_equity_curves.png"
    fig.savefig(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
