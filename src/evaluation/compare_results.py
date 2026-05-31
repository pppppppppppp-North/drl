from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "policy",
    "cumulative_return",
    "annualized_return",
    "sharpe",
    "sortino",
    "max_drawdown",
    "avg_turnover",
    "final_portfolio_value",
]


def load_comparison(run_dir: Path) -> pd.DataFrame:
    model_paths = sorted(
        path
        for path in run_dir.glob("*_metrics.csv")
        if path.name not in {"comparison_metrics.csv", "baselines_metrics.csv"}
    )
    baseline_path = run_dir / "baselines" / "baselines_metrics.csv"
    missing = []
    if not model_paths:
        missing.append(str(run_dir / "*_metrics.csv"))
    if not baseline_path.exists():
        missing.append(str(baseline_path))
    if missing:
        raise FileNotFoundError(f"missing metric files: {missing}")

    metrics = pd.concat(
        [*(pd.read_csv(path) for path in model_paths), pd.read_csv(baseline_path)],
        ignore_index=True,
    )
    return metrics.sort_values(["sharpe", "cumulative_return"], ascending=False).reset_index(drop=True)


def summarize_actions(run_dir: Path) -> pd.DataFrame:
    action_paths = sorted(run_dir.glob("*_actions.csv"))
    if not action_paths:
        raise FileNotFoundError(str(run_dir / "*_actions.csv"))

    actions = pd.concat((pd.read_csv(path) for path in action_paths), ignore_index=True)
    return (
        actions.groupby(["policy", "ticker"], as_index=False)
        .agg(
            mean_weight=("weight", "mean"),
            max_weight=("weight", "max"),
            active_days=("weight", lambda values: int((values > 1e-6).sum())),
        )
        .sort_values(["policy", "mean_weight"], ascending=[True, False])
        .reset_index(drop=True)
    )


def write_markdown_summary(metrics: pd.DataFrame, output_path: Path) -> None:
    columns = [column for column in SUMMARY_COLUMNS if column in metrics.columns]
    table = metrics[columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text(
        "# Run Comparison\n\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics = load_comparison(run_dir)
    metrics.to_csv(run_dir / "comparison_metrics.csv", index=False)
    write_markdown_summary(metrics, run_dir / "comparison_summary.md")

    action_summary = summarize_actions(run_dir)
    action_summary.to_csv(run_dir / "action_summary.csv", index=False)
    ppo_summary = action_summary[action_summary["policy"].eq("ppo")]
    if not ppo_summary.empty:
        ppo_summary.to_csv(run_dir / "ppo_action_summary.csv", index=False)

    print(metrics[[column for column in SUMMARY_COLUMNS if column in metrics.columns]].to_string(index=False))
    print(action_summary.to_string(index=False))


if __name__ == "__main__":
    main()
