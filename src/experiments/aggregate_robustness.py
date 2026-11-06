from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.statistics import holm_adjust, sign_flip_p_value
from src.experiments.experiment_manifest import collect_experiment_manifests
from src.utils.config import load_config


ROBUSTNESS_METRICS = ("cumulative_return", "sharpe", "max_drawdown")
KEY_COLUMNS = ("window", "seed", "split", "policy")


def _find_metrics_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("walk_forward_metrics.csv"))


def _deduplicate_repeated_baselines(metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse identical baseline rows emitted by each algorithm task.

    PPO and A2C runs deliberately evaluate the same baseline set for the same
    seed/window. Candidate-policy rows are unique, while deterministic and
    seeded baseline rows repeat. Conflicting repeats are rejected.
    """
    key_columns = list(KEY_COLUMNS)
    duplicate_mask = metrics.duplicated(key_columns, keep=False)
    if not duplicate_mask.any():
        return metrics

    value_columns = [column for column in metrics.columns if column not in {"source_path", *key_columns}]
    for key, part in metrics.loc[duplicate_mask].groupby(key_columns, dropna=False, sort=False):
        first = part.iloc[0]
        for column in value_columns:
            values = part[column]
            if pd.api.types.is_numeric_dtype(values):
                if not np.allclose(
                    pd.to_numeric(values, errors="coerce").to_numpy(dtype=float),
                    float(first[column]),
                    equal_nan=True,
                ):
                    raise ValueError(f"conflicting repeated metric rows for key {key}: column {column}")
            elif values.astype(str).nunique(dropna=False) > 1:
                raise ValueError(f"conflicting repeated metric rows for key {key}: column {column}")
    return metrics.drop_duplicates(key_columns, keep="first").reset_index(drop=True)


def _bootstrap_mean_ci(
    values: Iterable[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
    clusters: Iterable[object] | None = None,
) -> tuple[float, float]:
    raw_values = np.asarray(list(values), dtype=float)
    valid = np.isfinite(raw_values)
    clean = raw_values[valid]
    if not len(clean):
        return float("nan"), float("nan")
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    clean_clusters = None
    if clusters is not None:
        raw_clusters = np.asarray(list(clusters), dtype=object)
        if len(raw_clusters) != len(raw_values):
            raise ValueError("clusters must have the same length as values")
        clean_clusters = raw_clusters[valid]
    rng = np.random.default_rng(seed)
    sample_means = np.empty(resamples, dtype=float)
    chunk_size = min(1000, resamples)
    completed = 0
    while completed < resamples:
        count = min(chunk_size, resamples - completed)
        if clean_clusters is None:
            indices = rng.integers(0, len(clean), size=(count, len(clean)))
            sample_means[completed : completed + count] = clean[indices].mean(axis=1)
        else:
            unique_clusters = pd.unique(clean_clusters)
            if not len(unique_clusters):
                return float("nan"), float("nan")
            for offset in range(count):
                chosen = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
                sampled = np.concatenate([clean[clean_clusters == cluster] for cluster in chosen])
                sample_means[completed + offset] = sampled.mean()
        completed += count
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(sample_means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _summarize_groups(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    metrics: Iterable[str],
    confidence: float,
    resamples: int,
    bootstrap_seed: int,
    bootstrap_unit: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for group_key, part in frame.groupby(group_columns, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_columns, key_values))
        row.update(
            {
                "runs": int(len(part)),
                "seeds": int(part["seed"].nunique()),
                "windows": int(part["window"].nunique()),
            }
        )
        for metric_index, metric in enumerate(metrics):
            values = pd.to_numeric(part[metric], errors="coerce").dropna().to_numpy(dtype=float)
            low, high = _bootstrap_mean_ci(
                values,
                confidence=confidence,
                resamples=resamples,
                seed=bootstrap_seed + metric_index,
                clusters=part.loc[pd.to_numeric(part[metric], errors="coerce").notna(), "window"]
                if bootstrap_unit == "window"
                else None,
            )
            row.update(
                {
                    f"{metric}_mean": float(np.mean(values)) if len(values) else float("nan"),
                    f"{metric}_median": float(np.median(values)) if len(values) else float("nan"),
                    f"{metric}_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    f"{metric}_ci_low": low,
                    f"{metric}_ci_high": high,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_excess(
    test_metrics: pd.DataFrame,
    *,
    candidate_policies: list[str],
    primary_baselines: list[str],
) -> pd.DataFrame:
    comparison_metrics = [metric for metric in ROBUSTNESS_METRICS if metric in test_metrics.columns]
    candidates = test_metrics[test_metrics["policy"].isin(candidate_policies)].copy()
    rows: list[pd.DataFrame] = []
    for baseline in primary_baselines:
        baseline_rows = test_metrics[test_metrics["policy"].eq(baseline)][
            ["window", "seed", *comparison_metrics]
        ].copy()
        if baseline_rows.empty:
            raise ValueError(f"primary baseline is missing from test metrics: {baseline}")
        baseline_rows = baseline_rows.rename(columns={metric: f"baseline_{metric}" for metric in comparison_metrics})
        paired = candidates.merge(baseline_rows, on=["window", "seed"], how="inner", validate="many_to_one")
        paired.insert(3, "baseline", baseline)
        for metric in comparison_metrics:
            paired[f"excess_{metric}"] = paired[metric] - paired[f"baseline_{metric}"]
        rows.append(paired)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["policy", "baseline", "window", "seed"]).reset_index(drop=True)


def _paired_summary(
    paired: pd.DataFrame,
    *,
    confidence: float,
    resamples: int,
    bootstrap_seed: int,
    bootstrap_unit: str,
    randomization_resamples: int = 50_000,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    excess_metrics = [f"excess_{metric}" for metric in ROBUSTNESS_METRICS if f"excess_{metric}" in paired]
    summary = _summarize_groups(
        paired,
        group_columns=["policy", "baseline"],
        metrics=excess_metrics,
        confidence=confidence,
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_unit=bootstrap_unit,
    )
    for metric in excess_metrics:
        win_rates = (
            paired.assign(_win=pd.to_numeric(paired[metric], errors="coerce").gt(0.0))
            .groupby(["policy", "baseline"])["_win"]
            .mean()
        )
        summary[f"{metric}_positive_rate"] = [
            float(win_rates.loc[(row.policy, row.baseline)]) for row in summary.itertuples()
        ]
        raw_p_values = []
        for row_index, row in summary.iterrows():
            values = paired[
                paired["policy"].eq(row["policy"]) & paired["baseline"].eq(row["baseline"])
            ][metric]
            raw_p_values.append(
                sign_flip_p_value(
                    values,
                    resamples=randomization_resamples,
                    seed=bootstrap_seed + 10_000 + row_index,
                )
            )
        summary[f"{metric}_p_value"] = raw_p_values
        summary[f"{metric}_holm_p_value"] = holm_adjust(raw_p_values)
    return summary


def _policy_summary_input(
    test_metrics: pd.DataFrame,
    *,
    candidate_policies: list[str],
    stochastic_baselines: list[str],
) -> pd.DataFrame:
    """Avoid treating deterministic baseline copies across seeds as new evidence."""
    stochastic = set(candidate_policies) | set(stochastic_baselines)
    stochastic_rows = test_metrics[test_metrics["policy"].isin(stochastic)]
    deterministic_rows = test_metrics[~test_metrics["policy"].isin(stochastic)].drop_duplicates(
        ["window", "policy"], keep="first"
    )
    return pd.concat([stochastic_rows, deterministic_rows], ignore_index=True)


def _validate_complete_candidate_grid(
    test_metrics: pd.DataFrame,
    *,
    algorithms: list[str],
    seeds: list[int],
    windows: list[int],
) -> pd.DataFrame:
    expected = pd.DataFrame(
        itertools.product(algorithms, seeds, windows),
        columns=["policy", "seed", "window"],
    )
    actual = test_metrics[test_metrics["policy"].isin(algorithms)][["policy", "seed", "window"]].drop_duplicates()
    report = expected.merge(actual.assign(present=True), on=["policy", "seed", "window"], how="left")
    report["present"] = report["present"].eq(True)
    missing = report[~report["present"]]
    unexpected = actual.merge(expected, on=["policy", "seed", "window"], how="left", indicator=True)
    unexpected = unexpected[unexpected["_merge"].eq("left_only")]
    if not missing.empty or not unexpected.empty:
        raise ValueError(
            "incomplete robustness candidate grid: "
            f"missing={missing[['policy', 'seed', 'window']].to_dict('records')[:10]} "
            f"unexpected={unexpected[['policy', 'seed', 'window']].to_dict('records')[:10]}"
        )
    return report


def _plot_confidence_intervals(summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ROBUSTNESS_METRICS), figsize=(15, 5))
    if len(ROBUSTNESS_METRICS) == 1:
        axes = [axes]
    policies = summary["policy"].tolist()
    positions = np.arange(len(policies))
    for axis, metric in zip(axes, ROBUSTNESS_METRICS):
        means = summary[f"{metric}_mean"].to_numpy(dtype=float)
        low = summary[f"{metric}_ci_low"].to_numpy(dtype=float)
        high = summary[f"{metric}_ci_high"].to_numpy(dtype=float)
        errors = np.vstack([means - low, high - means])
        axis.errorbar(positions, means, yerr=errors, fmt="o", capsize=5)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xticks(positions, policies, rotation=45, ha="right")
        axis.set_title(metric.replace("_", " ").title())
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Test-Only Robustness: Mean and Bootstrap Confidence Interval")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_seed_window_heatmap(test_metrics: pd.DataFrame, candidate_policies: list[str], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = test_metrics[test_metrics["policy"].isin(candidate_policies)].copy()
    subset["policy_seed"] = subset["policy"] + " / seed " + subset["seed"].astype(int).astype(str)
    matrix = subset.pivot_table(index="policy_seed", columns="window", values="sharpe", aggfunc="mean").sort_index()
    fig_height = max(4.0, 0.45 * len(matrix.index) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(np.arange(len(matrix.columns)), [str(value) for value in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_xlabel("Walk-forward test window")
    ax.set_title("Test Sharpe by Training Seed and Window")
    fig.colorbar(image, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    table = frame[columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in row)
            + " |"
        )
    return "\n".join(lines)


def _write_summary_markdown(
    policy_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    output_path: Path,
    *,
    confidence: float,
    bootstrap_unit: str,
) -> None:
    policy_columns = [
        "policy",
        "runs",
        "seeds",
        "windows",
        "cumulative_return_mean",
        "cumulative_return_ci_low",
        "cumulative_return_ci_high",
        "sharpe_mean",
        "max_drawdown_mean",
    ]
    lines = [
        "# Repeated-Seed Walk-Forward Robustness Results",
        "",
        "Only chronological `test` rows are used for the research conclusion. Validation rows remain available for diagnostics.",
        f"Intervals are percentile bootstrap intervals for the mean at {confidence:.0%} confidence using `{bootstrap_unit}` resampling.",
        "",
        "## Policy Summary",
        "",
        _markdown_table(policy_summary, policy_columns),
    ]
    if not paired_summary.empty:
        paired_columns = [
            "policy",
            "baseline",
            "runs",
            "excess_cumulative_return_mean",
            "excess_cumulative_return_ci_low",
            "excess_cumulative_return_ci_high",
            "excess_cumulative_return_positive_rate",
            "excess_cumulative_return_holm_p_value",
            "excess_max_drawdown_mean",
            "excess_max_drawdown_holm_p_value",
        ]
        lines.extend(["", "## Paired Excess Performance", "", _markdown_table(paired_summary, paired_columns)])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_robustness(
    run_dir: Path,
    *,
    config: dict,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = output_dir or run_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    for path in _find_metrics_files(run_dir):
        frame = pd.read_csv(path)
        frame.insert(0, "source_path", str(path.relative_to(run_dir)))
        metric_frames.append(frame)
    if not metric_frames:
        raise FileNotFoundError(f"no walk_forward_metrics.csv files found under {run_dir}")
    metrics = pd.concat(metric_frames, ignore_index=True)
    missing = set(KEY_COLUMNS) - set(metrics.columns)
    if missing:
        raise ValueError(f"robustness metrics are missing columns: {sorted(missing)}")
    metrics = _deduplicate_repeated_baselines(metrics)

    robustness_cfg = config.get("robustness", {})
    candidate_policies = [str(value).lower() for value in robustness_cfg.get("algorithms", ["ppo", "a2c"])]
    primary_baselines = [str(value) for value in robustness_cfg.get("primary_baselines", ["equal_weight"])]
    stochastic_baselines = [str(value) for value in robustness_cfg.get("stochastic_baselines", ["random"])]
    confidence = float(robustness_cfg.get("confidence", 0.95))
    resamples = int(robustness_cfg.get("bootstrap_resamples", 10000))
    bootstrap_seed = int(robustness_cfg.get("bootstrap_seed", 20260712))
    randomization_resamples = int(robustness_cfg.get("randomization_resamples", 50_000))
    bootstrap_unit = str(robustness_cfg.get("bootstrap_unit", "observation")).lower()
    if bootstrap_unit not in {"observation", "window"}:
        raise ValueError("robustness.bootstrap_unit must be 'observation' or 'window'")

    test_metrics = metrics[metrics["split"].eq("test")].copy()
    if test_metrics.empty:
        raise ValueError("no test metrics were found; robustness conclusions must be test-only")
    expected_seeds = [int(value) for value in robustness_cfg.get("seeds", sorted(test_metrics["seed"].unique()))]
    expected_windows = [int(value) for value in robustness_cfg.get("window_ids", sorted(test_metrics["window"].unique()))]
    completeness = _validate_complete_candidate_grid(
        test_metrics,
        algorithms=candidate_policies,
        seeds=expected_seeds,
        windows=expected_windows,
    )
    summary_metrics = [metric for metric in ROBUSTNESS_METRICS if metric in test_metrics.columns]
    summary_input = _policy_summary_input(
        test_metrics,
        candidate_policies=candidate_policies,
        stochastic_baselines=stochastic_baselines,
    )
    policy_summary = _summarize_groups(
        summary_input,
        group_columns=["policy"],
        metrics=summary_metrics,
        confidence=confidence,
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_unit=bootstrap_unit,
    ).sort_values(["sharpe_mean", "cumulative_return_mean"], ascending=False).reset_index(drop=True)
    paired = _paired_excess(
        test_metrics,
        candidate_policies=candidate_policies,
        primary_baselines=primary_baselines,
    )
    paired_summary = _paired_summary(
        paired,
        confidence=confidence,
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_unit=bootstrap_unit,
        randomization_resamples=randomization_resamples,
    )

    manifests = pd.DataFrame(collect_experiment_manifests(run_dir))
    if not manifests.empty and not manifests["status"].eq("complete").all():
        failed = manifests[~manifests["status"].eq("complete")][
            ["experiment_id", "status", "error"]
        ].to_dict("records")
        raise ValueError(f"experiment manifests include non-complete runs: {failed[:10]}")

    metrics.to_csv(output_dir / "robustness_all_metrics.csv", index=False)
    test_metrics.to_csv(output_dir / "robustness_test_metrics.csv", index=False)
    policy_summary.to_csv(output_dir / "robustness_policy_summary.csv", index=False)
    paired.to_csv(output_dir / "robustness_paired_excess.csv", index=False)
    paired_summary.to_csv(output_dir / "robustness_paired_summary.csv", index=False)
    completeness.to_csv(output_dir / "robustness_completeness.csv", index=False)
    if not manifests.empty:
        manifests.to_csv(output_dir / "experiment_manifest.csv", index=False)
    _plot_confidence_intervals(policy_summary, output_dir / "robustness_confidence_intervals.png")
    _plot_seed_window_heatmap(test_metrics, candidate_policies, output_dir / "robustness_sharpe_heatmap.png")
    _write_summary_markdown(
        policy_summary,
        paired_summary,
        output_dir / "robustness_summary.md",
        confidence=confidence,
        bootstrap_unit=bootstrap_unit,
    )
    return metrics, test_metrics, policy_summary, paired_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="config/robustness_real_ohlcv_market.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    _, test_metrics, policy_summary, paired_summary = aggregate_robustness(
        Path(args.run_dir),
        config=config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"aggregated {len(test_metrics)} test metric rows")
    print(policy_summary.to_string(index=False))
    if not paired_summary.empty:
        print(paired_summary.to_string(index=False))


if __name__ == "__main__":
    main()
