from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _find_metrics_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("walk_forward_metrics.csv"))


def _find_test_curve_files(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.rglob("*_test_equity_curve.csv") if "baselines_" not in path.name)


def aggregate_walk_forward(run_dir: Path, output_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    for path in _find_metrics_files(run_dir):
        frame = pd.read_csv(path)
        frame.insert(0, "source_path", str(path.relative_to(run_dir)))
        metric_frames.append(frame)
    if not metric_frames:
        raise FileNotFoundError(f"no walk_forward_metrics.csv files found under {run_dir}")

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(output_dir / "walk_forward_all_metrics.csv", index=False)

    summary = (
        metrics.groupby(["split", "policy"], as_index=False)
        .agg(
            windows=("window", "nunique"),
            mean_cumulative_return=("cumulative_return", "mean"),
            mean_annualized_return=("annualized_return", "mean"),
            mean_sharpe=("sharpe", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
            mean_final_portfolio_value=("final_portfolio_value", "mean"),
        )
        .sort_values(["split", "mean_sharpe"], ascending=[True, False])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "walk_forward_policy_summary.csv", index=False)

    test_curves = []
    for path in _find_test_curve_files(run_dir):
        frame = pd.read_csv(path)
        parent = path.parent
        metrics_path = parent / "walk_forward_metrics.csv"
        if metrics_path.exists():
            metric_frame = pd.read_csv(metrics_path)
            window_values = metric_frame["window"].dropna().unique()
            if len(window_values):
                frame.insert(0, "window", int(window_values[0]))
        frame.insert(0, "source_path", str(path.relative_to(run_dir)))
        test_curves.append(frame)
    if test_curves:
        pd.concat(test_curves, ignore_index=True).to_csv(output_dir / "walk_forward_test_equity_curves.csv", index=False)

    return metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    metrics, summary = aggregate_walk_forward(
        Path(args.run_dir),
        Path(args.output_dir) if args.output_dir else None,
    )
    print(f"aggregated {len(metrics)} metric rows")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
