from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.aggregate_trials import aggregate_trials
from src.experiments.experiment_manifest import collect_experiment_manifests
from src.experiments.run_screening import screening_algorithms
from src.utils.config import load_config


def aggregate_screening(
    run_dir: Path,
    *,
    config: dict[str, Any],
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = output_dir or run_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    screening = config.get("screening", {})
    algorithms = screening_algorithms(config)
    trials_per_algorithm = int(screening.get("trials_per_algorithm", 30))
    objective_metric = str(screening.get("objective_metric", "sharpe"))
    results = aggregate_trials(run_dir, objective_metric=objective_metric)
    duplicate_trials = results.duplicated(["trial_id", "trial_number", "algorithm"], keep=False)
    if duplicate_trials.any():
        raise ValueError(
            "duplicate screening trials: "
            f"{results.loc[duplicate_trials, ['trial_id', 'trial_number', 'algorithm']].to_dict('records')[:10]}"
        )

    expected_rows = []
    for algorithm_index, algorithm in enumerate(algorithms):
        for trial_number in range(trials_per_algorithm):
            expected_rows.append(
                {
                    "trial_id": algorithm_index * trials_per_algorithm + trial_number,
                    "trial_number": trial_number,
                    "algorithm": algorithm,
                }
            )
    expected = pd.DataFrame(expected_rows)
    completeness = expected.merge(
        results[["trial_id", "trial_number", "algorithm"]].assign(present=True),
        on=["trial_id", "trial_number", "algorithm"],
        how="left",
    )
    completeness["present"] = completeness["present"].eq(True)
    if not completeness["present"].all():
        missing = completeness[~completeness["present"]].to_dict("records")
        raise ValueError(f"incomplete screening grid: {missing[:10]}")

    manifests = pd.DataFrame(collect_experiment_manifests(run_dir))
    if not manifests.empty and not manifests["status"].eq("complete").all():
        failed = manifests[~manifests["status"].eq("complete")][
            ["experiment_id", "status", "error"]
        ].to_dict("records")
        raise ValueError(f"screening manifests include non-complete runs: {failed[:10]}")

    top_n = int(screening.get("top_n_per_algorithm", 5))
    top = (
        results.sort_values(["algorithm", "objective_value"], ascending=[True, False])
        .groupby("algorithm", as_index=False, group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    ranking = (
        top.groupby("algorithm", as_index=False)
        .agg(
            top_trials=("objective_value", "size"),
            validation_objective_mean=("objective_value", "mean"),
            validation_objective_median=("objective_value", "median"),
            validation_objective_std=("objective_value", "std"),
        )
        .sort_values(
            ["validation_objective_median", "validation_objective_mean", "algorithm"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    selection_count = int(screening.get("selected_algorithm_count", 2))
    if selection_count <= 0 or selection_count > len(algorithms):
        raise ValueError("screening.selected_algorithm_count must be between 1 and the algorithm count")
    ranking["selected"] = False
    ranking.loc[: selection_count - 1, "selected"] = True
    selected_algorithms = ranking.loc[ranking["selected"], "algorithm"].tolist()
    results.to_csv(output_dir / "screening_trial_results.csv", index=False)
    top.to_csv(output_dir / "screening_top_trials.csv", index=False)
    ranking.to_csv(output_dir / "screening_algorithm_ranking.csv", index=False)
    (output_dir / "screening_selected_algorithms.json").write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "ranking_rule": "median objective across each algorithm's top validation trials; mean breaks ties",
                "selected_algorithms": selected_algorithms,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    completeness.to_csv(output_dir / "screening_completeness.csv", index=False)
    if not manifests.empty:
        manifests.to_csv(output_dir / "experiment_manifest.csv", index=False)
    return results, top


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="config/large_scale_screening.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results, top = aggregate_screening(
        Path(args.run_dir),
        config=config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"aggregated {len(results)} complete screening trials")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
