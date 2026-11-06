from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.experiment_manifest import tracked_experiment
from src.experiments.run_walk_forward import run_walk_forward_window
from src.utils.config import load_config


@dataclass(frozen=True)
class FeatureAblationGroup:
    name: str
    config_path: str


def feature_ablation_groups(config: dict[str, Any]) -> list[FeatureAblationGroup]:
    groups = [
        FeatureAblationGroup(name=str(item["name"]), config_path=str(item["config"]))
        for item in config.get("feature_ablation", {}).get("groups", [])
    ]
    if not groups:
        raise ValueError("feature_ablation.groups must not be empty")
    if len({group.name for group in groups}) != len(groups):
        raise ValueError("feature ablation group names must be unique")
    return groups


def groups_for_task(config: dict[str, Any], task_id: int) -> list[FeatureAblationGroup]:
    task_count = int(config.get("feature_ablation", {}).get("task_count", 5))
    if task_count <= 0:
        raise ValueError("feature_ablation.task_count must be positive")
    if task_id < 0 or task_id >= task_count:
        raise ValueError(f"task_id must be between 0 and {task_count - 1}; got {task_id}")
    return feature_ablation_groups(config)[task_id::task_count]


def run_feature_ablation_task(
    orchestrator_config: dict[str, Any],
    *,
    task_id: int,
    output_dir: Path,
) -> list[str]:
    ablation = orchestrator_config.get("feature_ablation", {})
    seeds = [int(value) for value in ablation.get("seeds", [42, 43, 44, 45, 46])]
    windows = [int(value) for value in ablation.get("window_ids", [0, 1])]
    algorithm = str(ablation.get("algorithm", "ppo")).lower()
    total_timesteps = int(ablation.get("total_timesteps", 50_000))
    training_overrides = dict(ablation.get("training_overrides", {}))
    algorithm_parameters = dict(ablation.get("algorithm_parameters", {}))
    completed_groups = []

    for group in groups_for_task(orchestrator_config, task_id):
        group_config = load_config(group.config_path)
        group_config = copy.deepcopy(group_config)
        group_config.setdefault("training", {}).update(training_overrides)
        group_config["training"].update(algorithm_parameters)
        group_config["training"]["algo"] = algorithm
        group_config["training"]["total_timesteps"] = total_timesteps
        for window in windows:
            window_parameters = dict(ablation.get("window_algorithm_parameters", {}).get(str(window), {}))
            for seed in seeds:
                run_config = copy.deepcopy(group_config)
                run_config.setdefault("training", {}).update(window_parameters)
                run_config.setdefault("project", {})["seed"] = seed
                run_config.setdefault("training", {})["seed"] = seed
                run_dir = output_dir / f"feature_{group.name}" / f"window_{window}" / f"seed_{seed}"
                identity = {
                    "experiment_id": f"ablation_{group.name}_{algorithm}_window_{window}_seed_{seed}",
                    "algorithm": algorithm,
                    "window": window,
                    "seed": seed,
                    "feature_group": group.name,
                    "cost_scenario": "base",
                    "claim_split": "test",
                }
                with tracked_experiment(run_dir, config=run_config, identity=identity):
                    run_walk_forward_window(
                        run_config,
                        window_id=window,
                        output_dir=run_dir,
                        algo=algorithm,
                        total_timesteps=total_timesteps,
                        seed=seed,
                    )
        completed_groups.append(group.name)
    return completed_groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/large_scale_feature_ablation.yaml")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-task-grid", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.print_task_grid:
        task_count = int(config.get("feature_ablation", {}).get("task_count", 5))
        print(json.dumps({task: [group.name for group in groups_for_task(config, task)] for task in range(task_count)}, indent=2))
        return
    groups = run_feature_ablation_task(config, task_id=args.task_id, output_dir=Path(args.output_dir))
    print(json.dumps({"task_id": args.task_id, "completed_groups": groups}, indent=2))


if __name__ == "__main__":
    main()
