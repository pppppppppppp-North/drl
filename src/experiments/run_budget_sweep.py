from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from src.experiments.experiment_manifest import tracked_experiment
from src.experiments.run_screening import screening_algorithms
from src.experiments.run_walk_forward import run_walk_forward_window
from src.utils.config import load_config


def budget_values(config: dict[str, Any]) -> list[int]:
    values = [int(value) for value in config.get("budget_sweep", {}).get("budgets", [])]
    if not values or any(value <= 0 for value in values):
        raise ValueError("budget_sweep.budgets must contain positive timestep counts")
    if len(set(values)) != len(values):
        raise ValueError("budget_sweep.budgets must be unique")
    return values


def run_budget_sweep_task(config: dict[str, Any], *, task_id: int, output_dir: Path) -> int:
    sweep = config.get("budget_sweep", {})
    budgets = budget_values(config)
    if task_id < 0 or task_id >= len(budgets):
        raise ValueError(f"task_id must be between 0 and {len(budgets) - 1}; got {task_id}")
    budget = budgets[task_id]
    algorithms = screening_algorithms({"screening": {"algorithms": sweep.get("algorithms", ["ppo", "a2c"])}})
    seeds = [int(value) for value in sweep.get("seeds", [42, 43, 44])]
    windows = [int(value) for value in sweep.get("window_ids", [0, 1])]
    parameter_map = dict(sweep.get("algorithm_parameters", {}))

    for algorithm in algorithms:
        for window in windows:
            for seed in seeds:
                run_config = copy.deepcopy(config)
                training = run_config.setdefault("training", {})
                training.update(dict(parameter_map.get(algorithm, {})))
                training.update({"algo": algorithm, "seed": seed, "total_timesteps": budget})
                run_config.setdefault("project", {})["seed"] = seed
                run_dir = (
                    output_dir
                    / f"budget_{budget}"
                    / f"algorithm_{algorithm}"
                    / f"window_{window}"
                    / f"seed_{seed}"
                )
                identity = {
                    "experiment_id": f"budget_{budget}_{algorithm}_window_{window}_seed_{seed}",
                    "algorithm": algorithm,
                    "window": window,
                    "seed": seed,
                    "training_budget": budget,
                    "selection_split": "validation",
                }
                with tracked_experiment(run_dir, config=run_config, identity=identity):
                    metrics = run_walk_forward_window(
                        run_config,
                        window_id=window,
                        output_dir=run_dir,
                        algo=algorithm,
                        total_timesteps=budget,
                        seed=seed,
                        evaluation_splits=("validation",),
                    )
                    candidate = metrics[
                        metrics["split"].eq("validation") & metrics["policy"].eq(algorithm)
                    ].iloc[0]
                    (run_dir / "budget_trial_result.json").write_text(
                        json.dumps(
                            {
                                "algorithm": algorithm,
                                "window": window,
                                "seed": seed,
                                "training_budget": budget,
                                "selection_split": "validation",
                                "metrics": candidate.to_dict(),
                            },
                            indent=2,
                            default=float,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
    return budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training_budget_sweep.yaml")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-task-grid", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.print_task_grid:
        print(json.dumps({index: budget for index, budget in enumerate(budget_values(config))}, indent=2))
        return
    budget = run_budget_sweep_task(config, task_id=args.task_id, output_dir=Path(args.output_dir))
    print(json.dumps({"task_id": args.task_id, "training_budget": budget}, indent=2))


if __name__ == "__main__":
    main()
