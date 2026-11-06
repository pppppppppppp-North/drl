from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.agents.baselines import run_baselines_on_features
from src.data.splits import generate_walk_forward_splits, split_summary, walk_forward_config_from_mapping
from src.envs.trading_env import ThaiStockTradingEnv, load_feature_table
from src.evaluation.metrics import equity_metrics
from src.train import (
    _algorithm_kwargs,
    _stable_baselines_algorithm,
    build_vectorized_training_env,
    ensure_feature_inputs,
    evaluate_model,
    make_trading_env_from_features,
)
from src.utils.checkpoints import prune_checkpoints
from src.utils.config import ensure_dirs, load_config


def make_env_from_features(config: dict, features: pd.DataFrame) -> ThaiStockTradingEnv:
    return make_trading_env_from_features(config, features)


def _metric_row(window_id: int, seed: int, split: str, policy: str, curve: pd.DataFrame) -> dict:
    row = {"window": window_id, "seed": seed, "split": split, "policy": policy}
    row.update(equity_metrics(curve))
    return row


def run_walk_forward_window(
    config: dict,
    window_id: int,
    output_dir: Path,
    algo: str | None = None,
    total_timesteps: int | None = None,
    seed: int | None = None,
    evaluation_splits: tuple[str, ...] = ("validation", "test"),
) -> pd.DataFrame:
    ensure_feature_inputs(config)
    features = load_feature_table(config["paths"]["features"], config["paths"].get("fallback_features_csv"))
    windows = generate_walk_forward_splits(features, walk_forward_config_from_mapping(config.get("walk_forward")))
    if window_id < 0 or window_id >= len(windows):
        raise ValueError(f"window_id must be between 0 and {len(windows) - 1}; got {window_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    window = windows[window_id]
    split_summary(window).assign(window=window_id).to_csv(output_dir / "window_split_summary.csv", index=False)

    training_config = dict(config.get("training", {}))
    algo = (algo or training_config.get("algo", "ppo")).lower()
    training_config["algo"] = algo
    if total_timesteps is not None:
        training_config["total_timesteps"] = total_timesteps
    total_timesteps = int(training_config.get("total_timesteps", 10000))
    seed = int(seed if seed is not None else training_config.get("seed", config["project"].get("seed", 42)))
    invalid_splits = sorted(set(evaluation_splits) - {"validation", "test"})
    if invalid_splits or not evaluation_splits:
        raise ValueError(f"evaluation_splits must contain validation and/or test; got {evaluation_splits}")

    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "window": window_id,
                "seed": seed,
                "algorithm": algo,
                "total_timesteps": total_timesteps,
                "evaluation_splits": list(evaluation_splits),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model_cls = _stable_baselines_algorithm(algo)
    if model_cls is None:
        raise ImportError(f"stable-baselines3 algorithm is unavailable: {algo}")

    from stable_baselines3.common.callbacks import CheckpointCallback
    task_config = dict(config)
    task_config["training"] = training_config
    train_env = build_vectorized_training_env(task_config, window["train"], seed)
    model = model_cls(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(output_dir / "tensorboard"),
        **_algorithm_kwargs(algo, total_timesteps, training_config),
    )
    checkpoint_dir = output_dir / "checkpoints" / algo
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = f"{algo}_window_{window_id}_checkpoint"
    checkpoint_callback = CheckpointCallback(
        save_freq=max(
            int(training_config.get("checkpoint_freq", max(total_timesteps // 5, 1)))
            // int(training_config.get("num_envs", 1)),
            1,
        ),
        save_path=str(checkpoint_dir),
        name_prefix=checkpoint_prefix,
        save_replay_buffer=bool(
            training_config.get("save_replay_buffer", algo in {"ddpg", "sac", "td3"})
        ),
        save_vecnormalize=True,
    )
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save(output_dir / f"{algo}_window_{window_id}_model")
    train_env.close()

    metric_rows = []
    for split in evaluation_splits:
        curve, actions = evaluate_model(
            model,
            make_env_from_features(config, window[split]),
            seed,
            policy_name=algo,
            return_actions=True,
        )
        curve.to_csv(output_dir / f"{algo}_{split}_equity_curve.csv", index=False)
        actions.to_csv(output_dir / f"{algo}_{split}_actions.csv", index=False)
        metric_rows.append(_metric_row(window_id, seed, split, algo, curve))

        baseline_curves, baseline_metrics = run_baselines_on_features(config, window[split], seed=seed)
        baseline_curves.to_csv(output_dir / f"baselines_{split}_equity_curves.csv", index=False)
        baseline_metrics.insert(0, "seed", seed)
        baseline_metrics.insert(0, "split", split)
        baseline_metrics.insert(0, "window", window_id)
        baseline_metrics.to_csv(output_dir / f"baselines_{split}_metrics.csv", index=False)
        metric_rows.extend(baseline_metrics.to_dict("records"))

    metrics = pd.DataFrame(metric_rows).sort_values(["split", "sharpe"], ascending=[True, False])
    metrics.to_csv(output_dir / "walk_forward_metrics.csv", index=False)
    prune_checkpoints(
        checkpoint_dir,
        name_prefix=checkpoint_prefix,
        keep_last=int(training_config.get("checkpoint_keep_last", 2)),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv.yaml")
    parser.add_argument("--window-id", type=int, required=True)
    parser.add_argument("--algo", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--evaluation-splits", default="validation,test")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["paths"]["results_dir"]) / f"walk_forward_window_{args.window_id}"
    ensure_dirs(output_dir)
    metrics = run_walk_forward_window(
        config,
        window_id=args.window_id,
        output_dir=output_dir,
        algo=args.algo,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        evaluation_splits=tuple(value.strip() for value in args.evaluation_splits.split(",") if value.strip()),
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
