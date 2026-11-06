from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.experiments.experiment_manifest import tracked_experiment
from src.experiments.run_walk_forward import run_walk_forward_window
from src.utils.config import load_config


@dataclass(frozen=True)
class RobustnessTask:
    task_id: int
    algorithm: str
    window: int


def robustness_task_grid(config: dict[str, Any]) -> list[RobustnessTask]:
    robustness = config.get("robustness", {})
    algorithms = [str(value).lower() for value in robustness.get("algorithms", ["ppo", "a2c"])]
    windows = [int(value) for value in robustness.get("window_ids", [0, 1])]
    if not algorithms:
        raise ValueError("robustness.algorithms must not be empty")
    if not windows:
        raise ValueError("robustness.window_ids must not be empty")
    return [
        RobustnessTask(task_id=index, algorithm=algorithm, window=window)
        for index, (algorithm, window) in enumerate(
            (algorithm, window) for algorithm in algorithms for window in windows
        )
    ]


def robustness_seeds(config: dict[str, Any]) -> list[int]:
    seeds = [int(value) for value in config.get("robustness", {}).get("seeds", [42, 43, 44, 45, 46])]
    if not seeds:
        raise ValueError("robustness.seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("robustness.seeds must contain unique values")
    return seeds


def run_robustness_task(
    config: dict[str, Any],
    *,
    task_id: int,
    output_dir: Path,
    total_timesteps: int | None = None,
) -> RobustnessTask:
    tasks = robustness_task_grid(config)
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"task_id must be between 0 and {len(tasks) - 1}; got {task_id}")
    task = tasks[task_id]
    seeds = robustness_seeds(config)
    robustness_cfg = config.get("robustness", {})
    algorithm_parameters = dict(robustness_cfg.get("algorithm_parameters", {}).get(task.algorithm, {}))
    window_parameters = (
        robustness_cfg.get("window_algorithm_parameters", {})
        .get(task.algorithm, {})
        .get(str(task.window), {})
    )
    algorithm_parameters.update(dict(window_parameters))
    total_timesteps = int(
        total_timesteps
        if total_timesteps is not None
        else robustness_cfg.get("total_timesteps", config.get("training", {}).get("total_timesteps", 20000))
    )

    task_dir = output_dir / f"algorithm_{task.algorithm}" / f"window_{task.window}"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task_manifest.json").write_text(
        json.dumps(
            {
                **asdict(task),
                "seeds": seeds,
                "total_timesteps": total_timesteps,
                "claim_split": "test",
                "algorithm_parameters": algorithm_parameters,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    task_config = copy.deepcopy(config)
    task_config.setdefault("training", {}).update(algorithm_parameters)
    for seed in seeds:
        seed_dir = task_dir / f"seed_{seed}"
        seed_config = copy.deepcopy(task_config)
        seed_config.setdefault("project", {})["seed"] = seed
        seed_config.setdefault("training", {})["seed"] = seed
        seed_config["training"]["algo"] = task.algorithm
        seed_config["training"]["total_timesteps"] = total_timesteps
        identity = {
            "experiment_id": f"{task.algorithm}_window_{task.window}_seed_{seed}",
            "algorithm": task.algorithm,
            "window": task.window,
            "seed": seed,
            "feature_group": seed_config.get("project", {}).get("name"),
            "cost_scenario": "base",
            "claim_split": "test",
        }
        with tracked_experiment(seed_dir, config=seed_config, identity=identity):
            run_walk_forward_window(
                seed_config,
                window_id=task.window,
                output_dir=seed_dir,
                algo=task.algorithm,
                total_timesteps=total_timesteps,
                seed=seed,
            )
    return task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/robustness_real_ohlcv_market.yaml")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--print-task-grid", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.print_task_grid:
        print(json.dumps([asdict(task) for task in robustness_task_grid(config)], indent=2))
        return
    task = run_robustness_task(
        config,
        task_id=args.task_id,
        output_dir=Path(args.output_dir),
        total_timesteps=args.total_timesteps,
    )
    print(json.dumps(asdict(task), indent=2))


if __name__ == "__main__":
    main()
