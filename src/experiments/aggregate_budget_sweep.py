from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.experiment_manifest import collect_experiment_manifests
from src.experiments.run_budget_sweep import budget_values
from src.experiments.run_screening import screening_algorithms
from src.utils.config import load_config


def aggregate_budget_sweep(
    run_dir: Path,
    *,
    config: dict[str, Any],
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    output_dir = output_dir or run_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = config.get("budget_sweep", {})
    budgets = budget_values(config)
    algorithms = screening_algorithms({"screening": {"algorithms": sweep.get("algorithms", ["ppo", "a2c"])}})
    seeds = [int(value) for value in sweep.get("seeds", [42, 43, 44])]
    windows = [int(value) for value in sweep.get("window_ids", [0, 1])]
    objective = str(sweep.get("objective_metric", "sharpe"))
    tolerance = float(sweep.get("plateau_tolerance", 0.05))
    if tolerance < 0.0:
        raise ValueError("budget_sweep.plateau_tolerance must be nonnegative")

    rows = []
    for path in sorted(run_dir.rglob("budget_trial_result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("selection_split") != "validation":
            raise ValueError(f"budget result is not validation-only: {path}")
        row = {
            "source_path": str(path.relative_to(run_dir)),
            "algorithm": data["algorithm"],
            "window": int(data["window"]),
            "seed": int(data["seed"]),
            "training_budget": int(data["training_budget"]),
        }
        row.update(data["metrics"])
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no budget_trial_result.json files found under {run_dir}")
    results = pd.DataFrame(rows)
    if objective not in results:
        raise KeyError(f"budget objective metric is missing: {objective}")

    expected = pd.DataFrame(
        itertools.product(budgets, algorithms, windows, seeds),
        columns=["training_budget", "algorithm", "window", "seed"],
    )
    actual = results[["training_budget", "algorithm", "window", "seed"]].drop_duplicates()
    duplicate_runs = results.duplicated(["training_budget", "algorithm", "window", "seed"], keep=False)
    if duplicate_runs.any():
        raise ValueError(
            "duplicate training-budget runs: "
            f"{results.loc[duplicate_runs, ['training_budget', 'algorithm', 'window', 'seed']].to_dict('records')[:10]}"
        )
    completeness = expected.merge(
        actual.assign(present=True),
        on=["training_budget", "algorithm", "window", "seed"],
        how="left",
    )
    completeness["present"] = completeness["present"].eq(True)
    if not completeness["present"].all():
        missing = completeness[~completeness["present"]].to_dict("records")
        raise ValueError(f"incomplete training-budget grid: {missing[:10]}")

    manifests = pd.DataFrame(collect_experiment_manifests(run_dir))
    if not manifests.empty and not manifests["status"].eq("complete").all():
        raise ValueError("training-budget manifests include non-complete runs")
    summary = (
        results.groupby(["training_budget", "algorithm"], as_index=False)
        .agg(
            runs=(objective, "size"),
            objective_mean=(objective, "mean"),
            objective_median=(objective, "median"),
            objective_std=(objective, "std"),
        )
        .sort_values(["algorithm", "training_budget"])
        .reset_index(drop=True)
    )
    pooled = results.groupby("training_budget")[objective].agg(["mean", "median", "std"]).reset_index()
    best_mean = float(pooled["mean"].max())
    eligible = pooled[pooled["mean"] >= best_mean - tolerance].sort_values("training_budget")
    selected_budget = int(eligible.iloc[0]["training_budget"])
    selection = {
        "selection_split": "validation",
        "objective_metric": objective,
        "plateau_tolerance": tolerance,
        "best_validation_mean": best_mean,
        "selected_training_budget": selected_budget,
        "selection_rule": "smallest common budget within plateau_tolerance of the best pooled validation mean",
    }
    results.to_csv(output_dir / "training_budget_validation_results.csv", index=False)
    summary.to_csv(output_dir / "training_budget_summary.csv", index=False)
    pooled.to_csv(output_dir / "training_budget_pooled_summary.csv", index=False)
    completeness.to_csv(output_dir / "training_budget_completeness.csv", index=False)
    if not manifests.empty:
        manifests.to_csv(output_dir / "experiment_manifest.csv", index=False)
    (output_dir / "selected_training_budget.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    return results, summary, selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="config/training_budget_sweep.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    _, summary, selection = aggregate_budget_sweep(
        Path(args.run_dir),
        config=config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(summary.to_string(index=False))
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
