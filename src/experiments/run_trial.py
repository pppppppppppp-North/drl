from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

from src.train import ensure_feature_inputs, train_with_stable_baselines, write_split_summary
from src.utils.config import ensure_dirs, load_config


DEFAULT_ALGORITHMS = ("ppo", "a2c")
OBJECTIVE_METRIC = "sharpe"


def parse_algorithms(value: str | None, fallback: str = "ppo") -> list[str]:
    if not value:
        return [fallback.lower()]
    algorithms = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(algorithms) - set(DEFAULT_ALGORITHMS))
    if invalid:
        raise ValueError(f"unsupported trial algorithm(s): {', '.join(invalid)}")
    return algorithms


def _trial_float(trial: Any, name: str, low: float, high: float, *, log: bool = False) -> float:
    return float(trial.suggest_float(name, low, high, log=log))


def _trial_seed(algo: str, trial_number: int, seed: int) -> int:
    return seed + trial_number * 1009 + sum(ord(char) for char in algo)


def _fallback_hyperparameters(algo: str, trial_number: int, seed: int) -> dict[str, Any]:
    rng = random.Random(_trial_seed(algo, trial_number, seed))

    def log_uniform(low: float, high: float) -> float:
        return 10 ** rng.uniform(math.log10(low), math.log10(high))

    n_steps = rng.choice([64, 128, 256, 512])
    params: dict[str, Any] = {
        "learning_rate": log_uniform(1e-5, 3e-3),
        "gamma": rng.choice([0.90, 0.95, 0.98, 0.99, 0.995]),
        "gae_lambda": rng.uniform(0.80, 0.99),
        "ent_coef": rng.choice([0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
        "n_steps": n_steps,
    }
    if algo == "ppo":
        params["clip_range"] = rng.uniform(0.10, 0.30)
        batch_size = rng.choice([32, 64, 128, 256])
        params["batch_size"] = min(batch_size, n_steps)
    return params


def suggest_hyperparameters(algo: str, trial_number: int, seed: int) -> dict[str, Any]:
    try:
        import optuna
    except ImportError:
        return _fallback_hyperparameters(algo, trial_number, seed)

    sampler = optuna.samplers.RandomSampler(seed=_trial_seed(algo, trial_number, seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial = study.ask()

    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512])
    params: dict[str, Any] = {
        "learning_rate": _trial_float(trial, "learning_rate", 1e-5, 3e-3, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.90, 0.95, 0.98, 0.99, 0.995]),
        "gae_lambda": _trial_float(trial, "gae_lambda", 0.80, 0.99),
        "ent_coef": trial.suggest_categorical("ent_coef", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
        "n_steps": n_steps,
    }
    if algo == "ppo":
        params["clip_range"] = _trial_float(trial, "clip_range", 0.10, 0.30)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        params["batch_size"] = min(batch_size, n_steps)
    return params


def select_trial_algorithm(trial_id: int, algorithms: list[str]) -> tuple[str, int]:
    if not algorithms:
        raise ValueError("at least one algorithm is required")
    algorithm_index = trial_id % len(algorithms)
    return algorithms[algorithm_index], trial_id // len(algorithms)


def apply_trial_config(
    config: dict[str, Any],
    *,
    algo: str,
    trial_id: int,
    params: dict[str, Any],
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    trial_config = copy.deepcopy(config)
    base_seed = int(trial_config.get("project", {}).get("seed", 42))
    seed = base_seed + trial_id
    trial_config.setdefault("project", {})["seed"] = seed
    training = trial_config.setdefault("training", {})
    training["algo"] = algo
    training["seed"] = seed
    training.update(params)
    if total_timesteps is not None:
        training["total_timesteps"] = int(total_timesteps)
    return trial_config


def read_objective(output_dir: Path, algo: str, objective_metric: str = OBJECTIVE_METRIC) -> tuple[float, dict[str, Any]]:
    import pandas as pd

    metrics_path = output_dir / f"{algo}_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        raise ValueError(f"metrics file is empty: {metrics_path}")
    row = metrics.iloc[0].to_dict()
    if objective_metric not in row:
        raise KeyError(f"objective metric {objective_metric!r} was not found in {metrics_path}")
    return float(row[objective_metric]), row


def write_trial_metadata(
    output_dir: Path,
    *,
    trial_id: int,
    trial_number: int,
    seed: int,
    config_path: str,
    algo: str,
    params: dict[str, Any],
    objective_metric: str,
    objective_value: float,
    metrics: dict[str, Any],
) -> None:
    metadata = {
        "trial_id": trial_id,
        "trial_number": trial_number,
        "seed": seed,
        "config_path": config_path,
        "algorithm": algo,
        "sampled_params": params,
        "objective_metric": objective_metric,
        "objective_value": objective_value,
        "metrics": metrics,
    }
    (output_dir / "trial_metadata.json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", type=int, required=True)
    parser.add_argument("--config", default="config/optuna_search.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--algo", choices=DEFAULT_ALGORITHMS, default=None)
    parser.add_argument("--algorithms", default=None, help="Comma-separated algorithms to alternate across array tasks.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--objective-metric", default=OBJECTIVE_METRIC)
    args = parser.parse_args()

    config = load_config(args.config)
    search_cfg = config.get("search", {})
    algorithm_value = args.algo or args.algorithms or search_cfg.get("algorithms")
    algorithms = [args.algo] if args.algo else parse_algorithms(
        algorithm_value,
        fallback=str(config.get("training", {}).get("algo", "ppo")),
    )
    algo, trial_number = select_trial_algorithm(args.trial_id, algorithms)
    base_seed = int(config.get("project", {}).get("seed", 42))
    params = suggest_hyperparameters(algo, trial_number, base_seed)
    total_timesteps = args.total_timesteps or search_cfg.get("total_timesteps")
    trial_config = apply_trial_config(
        config,
        algo=algo,
        trial_id=args.trial_id,
        params=params,
        total_timesteps=total_timesteps,
    )

    output_dir = Path(args.output_dir)
    ensure_dirs(output_dir, trial_config["paths"]["models_dir"])
    ensure_feature_inputs(trial_config)
    write_split_summary(trial_config, output_dir)

    if not train_with_stable_baselines(trial_config, output_dir):
        raise RuntimeError("stable-baselines3 is required for hyperparameter trials")

    objective_value, metrics = read_objective(output_dir, algo, args.objective_metric)
    write_trial_metadata(
        output_dir,
        trial_id=args.trial_id,
        trial_number=trial_number,
        seed=int(trial_config["training"]["seed"]),
        config_path=args.config,
        algo=algo,
        params=params,
        objective_metric=args.objective_metric,
        objective_value=objective_value,
        metrics=metrics,
    )
    print(json.dumps({"trial_id": args.trial_id, "algorithm": algo, "objective": objective_value}, indent=2))


if __name__ == "__main__":
    main()
