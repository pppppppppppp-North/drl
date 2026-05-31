from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.agents.baselines import run_baselines_on_features
from src.data.splits import generate_walk_forward_splits, split_summary, walk_forward_config_from_mapping
from src.envs.trading_env import RewardConfig, ThaiStockTradingEnv, load_feature_table
from src.evaluation.metrics import equity_metrics
from src.train import _algorithm_kwargs, _stable_baselines_algorithm, ensure_feature_inputs, evaluate_model
from src.utils.config import ensure_dirs, load_config


def make_env_from_features(config: dict, features: pd.DataFrame) -> ThaiStockTradingEnv:
    env_cfg = config["environment"]
    return ThaiStockTradingEnv(
        features=features,
        feature_columns=config["features"]["columns"],
        lookback=config["features"]["lookback"],
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        reward_config=RewardConfig(**env_cfg.get("reward", {})),
    )


def _metric_row(window_id: int, split: str, policy: str, curve: pd.DataFrame) -> dict:
    row = {"window": window_id, "split": split, "policy": policy}
    row.update(equity_metrics(curve))
    return row


def run_walk_forward_window(
    config: dict,
    window_id: int,
    output_dir: Path,
    algo: str | None = None,
    total_timesteps: int | None = None,
    seed: int | None = None,
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

    model_cls = _stable_baselines_algorithm(algo)
    if model_cls is None:
        raise ImportError(f"stable-baselines3 algorithm is unavailable: {algo}")

    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    train_env = make_env_from_features(config, window["train"])
    model = model_cls(
        "MlpPolicy",
        Monitor(train_env),
        verbose=1,
        seed=seed,
        tensorboard_log=str(output_dir / "tensorboard"),
        **_algorithm_kwargs(algo, total_timesteps, training_config),
    )
    checkpoint_dir = output_dir / "checkpoints" / algo
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=int(training_config.get("checkpoint_freq", max(total_timesteps // 5, 1))),
        save_path=str(checkpoint_dir),
        name_prefix=f"{algo}_window_{window_id}_checkpoint",
        save_replay_buffer=algo in {"ddpg", "sac", "td3"},
        save_vecnormalize=True,
    )
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save(output_dir / f"{algo}_window_{window_id}_model")

    metric_rows = []
    for split in ("validation", "test"):
        curve, actions = evaluate_model(
            model,
            make_env_from_features(config, window[split]),
            seed,
            policy_name=algo,
            return_actions=True,
        )
        curve.to_csv(output_dir / f"{algo}_{split}_equity_curve.csv", index=False)
        actions.to_csv(output_dir / f"{algo}_{split}_actions.csv", index=False)
        metric_rows.append(_metric_row(window_id, split, algo, curve))

        baseline_curves, baseline_metrics = run_baselines_on_features(config, window[split], seed=seed)
        baseline_curves.to_csv(output_dir / f"baselines_{split}_equity_curves.csv", index=False)
        baseline_metrics.insert(0, "split", split)
        baseline_metrics.insert(0, "window", window_id)
        baseline_metrics.to_csv(output_dir / f"baselines_{split}_metrics.csv", index=False)
        metric_rows.extend(baseline_metrics.to_dict("records"))

    metrics = pd.DataFrame(metric_rows).sort_values(["split", "sharpe"], ascending=[True, False])
    metrics.to_csv(output_dir / "walk_forward_metrics.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv.yaml")
    parser.add_argument("--window-id", type=int, required=True)
    parser.add_argument("--algo", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
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
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
