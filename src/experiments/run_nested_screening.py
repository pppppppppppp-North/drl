from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.experiment_manifest import tracked_experiment
from src.experiments.run_screening import screening_algorithms
from src.experiments.run_trial import apply_trial_config, suggest_hyperparameters
from src.experiments.run_walk_forward import run_walk_forward_window
from src.utils.config import load_config


def run_nested_screening_task(
    config: dict[str, Any],
    *,
    task_id: int,
    output_dir: Path,
) -> str:
    nested = config.get("nested_screening", {})
    algorithms = screening_algorithms({"screening": {"algorithms": nested.get("algorithms", ["ppo", "a2c"])}})
    if task_id < 0 or task_id >= len(algorithms):
        raise ValueError(f"task_id must be between 0 and {len(algorithms) - 1}; got {task_id}")
    algorithm = algorithms[task_id]
    windows = [int(value) for value in nested.get("window_ids", [0, 1])]
    trials_per_window = int(nested.get("trials_per_algorithm_window", 10))
    total_timesteps = int(nested.get("total_timesteps", 20_000))
    base_seed = int(config.get("project", {}).get("seed", 42))
    if trials_per_window <= 0:
        raise ValueError("nested_screening.trials_per_algorithm_window must be positive")

    for window in windows:
        for trial_number in range(trials_per_window):
            trial_id = (task_id * len(windows) + windows.index(window)) * trials_per_window + trial_number
            params = suggest_hyperparameters(
                algorithm,
                trial_number,
                base_seed + window * 10_000,
                require_optuna=True,
            )
            trial_config = apply_trial_config(
                config,
                algo=algorithm,
                trial_id=trial_id,
                params=params,
                total_timesteps=total_timesteps,
            )
            trial_dir = output_dir / f"algorithm_{algorithm}" / f"window_{window}" / f"trial_{trial_number}"
            identity = {
                "experiment_id": f"nested_{algorithm}_window_{window}_trial_{trial_number}",
                "algorithm": algorithm,
                "window": window,
                "seed": int(trial_config["training"]["seed"]),
                "feature_group": trial_config.get("project", {}).get("name"),
                "cost_scenario": "base",
                "selection_split": "validation",
                "trial_id": trial_id,
                "trial_number": trial_number,
            }
            with tracked_experiment(trial_dir, config=trial_config, identity=identity):
                metrics = run_walk_forward_window(
                    trial_config,
                    window_id=window,
                    output_dir=trial_dir,
                    algo=algorithm,
                    total_timesteps=total_timesteps,
                    seed=int(trial_config["training"]["seed"]),
                    evaluation_splits=("validation",),
                )
                candidate = metrics[
                    (metrics["split"].eq("validation")) & (metrics["policy"].eq(algorithm))
                ].iloc[0]
                (trial_dir / "nested_trial_result.json").write_text(
                    json.dumps(
                        {
                            "trial_id": trial_id,
                            "trial_number": trial_number,
                            "algorithm": algorithm,
                            "window": window,
                            "seed": int(trial_config["training"]["seed"]),
                            "selection_split": "validation",
                            "objective_metric": str(nested.get("objective_metric", "sharpe")),
                            "metrics": candidate.to_dict(),
                            "sampled_params": params,
                        },
                        indent=2,
                        default=float,
                    ),
                    encoding="utf-8",
                )
    return algorithm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/nested_walk_forward_screening.yaml")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    algorithm = run_nested_screening_task(config, task_id=args.task_id, output_dir=Path(args.output_dir))
    print(json.dumps({"task_id": args.task_id, "algorithm": algorithm}, indent=2))


if __name__ == "__main__":
    main()
