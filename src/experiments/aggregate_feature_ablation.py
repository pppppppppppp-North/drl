from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.statistics import holm_adjust, sign_flip_p_value
from src.experiments.aggregate_robustness import _bootstrap_mean_ci
from src.experiments.experiment_manifest import collect_experiment_manifests
from src.experiments.run_feature_ablation import feature_ablation_groups
from src.utils.config import load_config


def aggregate_feature_ablation(
    run_dir: Path,
    *,
    config: dict[str, Any],
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = output_dir or run_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation = config.get("feature_ablation", {})
    algorithm = str(ablation.get("algorithm", "ppo")).lower()
    seeds = [int(value) for value in ablation.get("seeds", [42, 43, 44, 45, 46])]
    windows = [int(value) for value in ablation.get("window_ids", [0, 1])]
    groups = [group.name for group in feature_ablation_groups(config)]
    confidence = float(ablation.get("confidence", 0.95))
    resamples = int(ablation.get("bootstrap_resamples", 10_000))
    bootstrap_seed = int(ablation.get("bootstrap_seed", 20260712))
    randomization_resamples = int(ablation.get("randomization_resamples", 50_000))

    rows = []
    for path in sorted(run_dir.rglob("walk_forward_metrics.csv")):
        manifest_path = path.parent / "experiment_manifest.json"
        if not manifest_path.exists():
            continue
        identity = json.loads(manifest_path.read_text(encoding="utf-8")).get("identity", {})
        feature_group = identity.get("feature_group")
        metrics = pd.read_csv(path)
        candidate = metrics[(metrics["split"].eq("test")) & (metrics["policy"].eq(algorithm))].copy()
        if candidate.empty:
            continue
        candidate.insert(0, "feature_group", feature_group)
        candidate.insert(0, "source_path", str(path.relative_to(run_dir)))
        rows.append(candidate)
    if not rows:
        raise FileNotFoundError(f"no tracked test metrics found under {run_dir}")
    metrics = pd.concat(rows, ignore_index=True)

    expected = pd.DataFrame(
        itertools.product(groups, seeds, windows),
        columns=["feature_group", "seed", "window"],
    )
    actual = metrics[["feature_group", "seed", "window"]].drop_duplicates()
    completeness = expected.merge(actual.assign(present=True), on=["feature_group", "seed", "window"], how="left")
    completeness["present"] = completeness["present"].eq(True)
    if not completeness["present"].all():
        missing = completeness[~completeness["present"]].to_dict("records")
        raise ValueError(f"incomplete feature-ablation grid: {missing[:10]}")

    summary_rows = []
    for group, part in metrics.groupby("feature_group", sort=True):
        row = {"feature_group": group, "runs": len(part), "seeds": part["seed"].nunique(), "windows": part["window"].nunique()}
        for metric_index, metric in enumerate(("cumulative_return", "sharpe", "max_drawdown")):
            values = pd.to_numeric(part[metric], errors="coerce").dropna().to_numpy(dtype=float)
            low, high = _bootstrap_mean_ci(
                values,
                confidence=confidence,
                resamples=resamples,
                seed=bootstrap_seed + metric_index,
            )
            row.update(
                {
                    f"{metric}_mean": float(values.mean()),
                    f"{metric}_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    f"{metric}_ci_low": low,
                    f"{metric}_ci_high": high,
                }
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("sharpe_mean", ascending=False).reset_index(drop=True)
    control_group = str(ablation.get("control_group", "technical"))
    if control_group not in groups:
        control_group = groups[0]
    comparisons = _paired_feature_comparisons(
        metrics,
        control_group=control_group,
        resamples=randomization_resamples,
        seed=bootstrap_seed,
    )
    manifests = pd.DataFrame(collect_experiment_manifests(run_dir))
    if not manifests.empty and not manifests["status"].eq("complete").all():
        raise ValueError("feature-ablation manifests include non-complete runs")

    metrics.to_csv(output_dir / "feature_ablation_test_metrics.csv", index=False)
    summary.to_csv(output_dir / "feature_ablation_summary.csv", index=False)
    completeness.to_csv(output_dir / "feature_ablation_completeness.csv", index=False)
    comparisons.to_csv(output_dir / "feature_ablation_paired_comparisons.csv", index=False)
    if not manifests.empty:
        manifests.to_csv(output_dir / "experiment_manifest.csv", index=False)
    _plot_ablation(summary, output_dir / "feature_ablation_confidence_intervals.png")
    return metrics, summary


def _paired_feature_comparisons(
    metrics: pd.DataFrame,
    *,
    control_group: str,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    metric_names = [name for name in ("cumulative_return", "sharpe", "max_drawdown") if name in metrics]
    control = metrics[metrics["feature_group"].eq(control_group)][["seed", "window", *metric_names]].copy()
    if control.empty:
        raise ValueError(f"feature-ablation control group is missing: {control_group}")
    control = control.rename(columns={name: f"control_{name}" for name in metric_names})
    rows = []
    candidates = sorted(set(metrics["feature_group"]) - {control_group})
    for group_index, group in enumerate(candidates):
        candidate = metrics[metrics["feature_group"].eq(group)][["seed", "window", *metric_names]]
        paired = candidate.merge(control, on=["seed", "window"], validate="one_to_one")
        row: dict[str, float | int | str] = {
            "feature_group": group,
            "control_group": control_group,
            "paired_runs": len(paired),
        }
        for metric_index, metric in enumerate(metric_names):
            differences = pd.to_numeric(paired[metric], errors="coerce") - pd.to_numeric(
                paired[f"control_{metric}"], errors="coerce"
            )
            row[f"excess_{metric}_mean"] = float(differences.mean())
            row[f"excess_{metric}_p_value"] = sign_flip_p_value(
                differences,
                resamples=resamples,
                seed=seed + group_index * 100 + metric_index,
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    for metric in metric_names:
        column = f"excess_{metric}_p_value"
        result[f"excess_{metric}_holm_p_value"] = holm_adjust(result[column])
    return result


def _plot_ablation(summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = summary["feature_group"].tolist()
    means = summary["sharpe_mean"].to_numpy(dtype=float)
    low = summary["sharpe_ci_low"].to_numpy(dtype=float)
    high = summary["sharpe_ci_high"].to_numpy(dtype=float)
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(positions, means, yerr=np.vstack([means - low, high - means]), fmt="o", capsize=5)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions, labels, rotation=45, ha="right")
    ax.set_ylabel("Test Sharpe")
    ax.set_title("Feature Ablation: Mean Test Sharpe and 95% Bootstrap Interval")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="config/large_scale_feature_ablation.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    _, summary = aggregate_feature_ablation(
        Path(args.run_dir),
        config=config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
