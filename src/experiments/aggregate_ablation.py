from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _feature_group_from_path(run_dir: Path, metrics_path: Path) -> str:
    try:
        return metrics_path.parent.relative_to(run_dir).parts[0]
    except (IndexError, ValueError):
        return metrics_path.parent.name


def aggregate_ablation(run_dir: Path, output_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_paths = sorted(run_dir.rglob("comparison_metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"no comparison_metrics.csv files found under {run_dir}")

    frames = []
    for path in metric_paths:
        frame = pd.read_csv(path)
        frame.insert(0, "feature_group", _feature_group_from_path(run_dir, path))
        frame.insert(0, "source_path", str(path.relative_to(run_dir)))
        frames.append(frame)

    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(output_dir / "ablation_metrics.csv", index=False)

    summary = (
        metrics.sort_values(["feature_group", "sharpe", "cumulative_return"], ascending=[True, False, False])
        .groupby("feature_group", as_index=False)
        .first()
        .sort_values(["sharpe", "cumulative_return"], ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "ablation_best_policy_summary.csv", index=False)

    ppo = metrics[metrics["policy"].eq("ppo")].copy()
    if not ppo.empty:
        ppo = ppo.sort_values(["sharpe", "cumulative_return"], ascending=False).reset_index(drop=True)
        ppo.to_csv(output_dir / "ablation_ppo_summary.csv", index=False)
        write_ablation_bar_chart(ppo, output_dir / "ablation_ppo_sharpe.png")

    return metrics, summary


def write_ablation_bar_chart(ppo_metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_data = ppo_metrics.sort_values("sharpe", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(plot_data["feature_group"], plot_data["sharpe"], color="#3b6ea8")
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Validation Sharpe")
    ax.set_ylabel("Feature group")
    ax.set_title("PPO Ablation Sharpe")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    metrics, summary = aggregate_ablation(
        Path(args.run_dir),
        Path(args.output_dir) if args.output_dir else None,
    )
    print(f"aggregated {len(metrics)} metric rows")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
