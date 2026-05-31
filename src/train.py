from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.agents.baselines import run_baselines
from src.data.splits import (
    chronological_split,
    generate_walk_forward_splits,
    select_chronological_split,
    split_config_from_mapping,
    split_summary,
    walk_forward_config_from_mapping,
    walk_forward_summary,
)
from src.data.synthetic import make_synthetic_prices
from src.envs.trading_env import RewardConfig, ThaiStockTradingEnv, load_feature_table
from src.evaluation.metrics import equity_metrics
from src.features.build_features import write_features
from src.features.technical import add_technical_features
from src.utils.config import ensure_dirs, load_config


def _pilot_features_match_config(features: pd.DataFrame, config: dict) -> bool:
    expected_tickers = set(config["data"]["tickers"])
    actual_tickers = set(features["ticker"].dropna().astype(str).unique()) if "ticker" in features else set()
    if actual_tickers != expected_tickers:
        return False

    missing_columns = set(config["features"]["columns"] + ["date", "ticker", "close"]) - set(features.columns)
    if missing_columns:
        return False

    dates = pd.to_datetime(features["date"])
    expected_start = pd.Timestamp(config["data"]["start"])
    expected_end = pd.Timestamp(config["data"]["end"])
    return bool(dates.min() <= expected_start and dates.max() >= expected_end)


def ensure_pilot_features(config: dict) -> None:
    raw_path = Path(config["paths"]["raw_prices"])
    feature_path = Path(config["paths"]["features"])
    csv_path = Path(config["paths"].get("fallback_features_csv", feature_path.with_suffix(".csv")))
    if feature_path.exists() or csv_path.exists():
        existing = load_feature_table(feature_path, csv_path)
        if _pilot_features_match_config(existing, config):
            return
        print("existing pilot features do not match active config; regenerating synthetic pilot data")
        for path in (feature_path, csv_path):
            if path.exists():
                path.unlink()
    ensure_dirs(raw_path.parent)
    prices = make_synthetic_prices(config)
    prices.to_csv(raw_path, index=False)
    write_features(add_technical_features(prices), feature_path, csv_path)


def ensure_feature_inputs(config: dict) -> None:
    data_source = str(config.get("data", {}).get("source", "synthetic")).lower()
    if data_source == "synthetic":
        ensure_pilot_features(config)
        return

    feature_path = Path(config["paths"]["features"])
    csv_path = Path(config["paths"].get("fallback_features_csv", feature_path.with_suffix(".csv")))
    if feature_path.exists() or csv_path.exists():
        return
    raise FileNotFoundError(
        f"feature table for data.source={data_source!r} is missing; "
        f"build it first with `python -m src.features.build_features --config <config>`"
    )


def _features_for_split(config: dict, split: str = "all") -> pd.DataFrame:
    features = load_feature_table(config["paths"]["features"], config["paths"].get("fallback_features_csv"))
    if split == "all":
        return features
    return select_chronological_split(features, split, split_config_from_mapping(config.get("splits")))


def _validate_env_length(features: pd.DataFrame, lookback: int, split: str) -> None:
    unique_dates = pd.to_datetime(features["date"]).nunique()
    if unique_dates <= lookback + 1:
        raise ValueError(
            f"split {split!r} has {unique_dates} unique dates, but lookback={lookback} requires at least {lookback + 2}"
        )


def make_trading_env(config: dict, split: str = "all") -> ThaiStockTradingEnv:
    features = _features_for_split(config, split)
    _validate_env_length(features, int(config["features"]["lookback"]), split)
    env_cfg = config["environment"]
    return ThaiStockTradingEnv(
        features=features,
        feature_columns=config["features"]["columns"],
        lookback=config["features"]["lookback"],
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        reward_config=RewardConfig(**env_cfg.get("reward", {})),
    )


def evaluate_model(
    model,
    env: ThaiStockTradingEnv,
    seed: int,
    policy_name: str = "ppo",
    return_actions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    obs, info = env.reset(seed=seed)
    rows = [
        {
            "date": info["date"],
            "portfolio_value": info["portfolio_value"],
            "policy": policy_name,
            "net_return": info["net_return"],
            "turnover": info["turnover"],
            "drawdown": info["drawdown"],
            "cash_weight": info["cash_weight"],
        }
    ]
    action_rows = [
        {"date": info["date"], "policy": policy_name, "ticker": ticker, "weight": float(weight)}
        for ticker, weight in zip(env.tickers, info["weights"])
    ]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "date": info["date"],
                "portfolio_value": info["portfolio_value"],
                "policy": policy_name,
                "net_return": info["net_return"],
                "turnover": info["turnover"],
                "drawdown": info["drawdown"],
                "cash_weight": info["cash_weight"],
            }
        )
        action_rows.extend(
            {"date": info["date"], "policy": policy_name, "ticker": ticker, "weight": float(weight)}
            for ticker, weight in zip(env.tickers, info["weights"])
        )
        done = terminated or truncated

    curve = pd.DataFrame(rows)
    if return_actions:
        return curve, pd.DataFrame(action_rows)
    return curve


def _stable_baselines_algorithm(algo: str):
    try:
        from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    except ImportError:
        return None

    return {
        "a2c": A2C,
        "ddpg": DDPG,
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3,
    }.get(algo)


def _algorithm_kwargs(algo: str, total_timesteps: int, training_config: dict | None = None) -> dict:
    if algo == "ppo":
        kwargs = {"n_steps": 128, "batch_size": 64}
        allowed = {"learning_rate", "n_steps", "batch_size", "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef"}
    elif algo == "a2c":
        kwargs = {"n_steps": 64}
        allowed = {"learning_rate", "n_steps", "gamma", "gae_lambda", "ent_coef", "vf_coef"}
    else:
        kwargs = {
            "buffer_size": 50_000,
            "learning_starts": min(1_000, max(100, total_timesteps // 10)),
            "batch_size": 64,
        }
        allowed = {"learning_rate", "buffer_size", "learning_starts", "batch_size", "gamma", "tau"}

    for key, value in (training_config or {}).items():
        if key in allowed:
            kwargs[key] = value
    return kwargs


def _training_split(config: dict) -> str:
    if "splits" not in config:
        return "all"
    return str(config["training"].get("train_split", config["splits"].get("train_split", "train")))


def _evaluation_split(config: dict) -> str:
    if "splits" not in config:
        return "all"
    return str(config["training"].get("eval_split", config["splits"].get("eval_split", "validation")))


def write_split_summary(config: dict, output_dir: Path) -> None:
    if "splits" not in config:
        return
    features = load_feature_table(config["paths"]["features"], config["paths"].get("fallback_features_csv"))
    splits = chronological_split(features, split_config_from_mapping(config.get("splits")))
    split_summary(splits).to_csv(output_dir / "split_summary.csv", index=False)


def write_walk_forward_summary(config: dict, output_dir: Path) -> None:
    if "walk_forward" not in config:
        return
    features = load_feature_table(config["paths"]["features"], config["paths"].get("fallback_features_csv"))
    windows = generate_walk_forward_splits(features, walk_forward_config_from_mapping(config.get("walk_forward")))
    walk_forward_summary(windows).to_csv(output_dir / "walk_forward_summary.csv", index=False)


def train_with_stable_baselines(config: dict, output_dir: Path) -> bool:
    try:
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        return False

    algo = config["training"].get("algo", "ppo").lower()
    model_cls = _stable_baselines_algorithm(algo)
    if model_cls is None:
        return False

    train_split = _training_split(config)
    eval_split = _evaluation_split(config)
    train_env = make_trading_env(config, split=train_split)
    total_timesteps = int(config["training"].get("total_timesteps", 10000))
    model = model_cls(
        "MlpPolicy",
        Monitor(train_env),
        verbose=1,
        seed=config["training"].get("seed", 42),
        tensorboard_log=str(output_dir / "tensorboard"),
        **_algorithm_kwargs(algo, total_timesteps, config.get("training", {})),
    )
    checkpoint_dir = output_dir / "checkpoints" / algo
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_freq = int(config["training"].get("checkpoint_freq", max(total_timesteps // 5, 1)))
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=f"{algo}_checkpoint",
        save_replay_buffer=algo in {"ddpg", "sac", "td3"},
        save_vecnormalize=True,
    )
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save(output_dir / f"{algo}_model")

    seed = int(config["training"].get("seed", config["project"].get("seed", 42)))
    eval_curve, eval_actions = evaluate_model(
        model,
        make_trading_env(config, split=eval_split),
        seed,
        policy_name=algo,
        return_actions=True,
    )
    eval_curve.to_csv(output_dir / f"{algo}_equity_curve.csv", index=False)
    eval_actions.to_csv(output_dir / f"{algo}_actions.csv", index=False)
    metrics = {"policy": algo}
    metrics.update(equity_metrics(eval_curve))
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / f"{algo}_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--algo", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--train-split", choices=["all", "train", "validation", "test"], default=None)
    parser.add_argument("--eval-split", choices=["all", "train", "validation", "test"], default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.algo:
        config["training"]["algo"] = args.algo
    if args.num_envs:
        config["training"]["num_envs"] = args.num_envs
    if args.seed is not None:
        config["training"]["seed"] = args.seed
        config["project"]["seed"] = args.seed
    if args.total_timesteps is not None:
        config["training"]["total_timesteps"] = args.total_timesteps
    if args.train_split:
        config.setdefault("splits", {})
        config["training"]["train_split"] = args.train_split
    if args.eval_split:
        config.setdefault("splits", {})
        config["training"]["eval_split"] = args.eval_split

    output_dir = Path(args.output_dir or config["paths"]["results_dir"])
    ensure_dirs(output_dir, config["paths"]["models_dir"])
    ensure_feature_inputs(config)
    write_split_summary(config, output_dir)
    write_walk_forward_summary(config, output_dir)

    trained = train_with_stable_baselines(config, output_dir)

    if not trained:
        print("requested stable-baselines3 algorithm is unavailable; running baseline smoke instead")
        curves, metrics = run_baselines(config)
        curves.to_csv(output_dir / "baseline_equity_curves.csv", index=False)
        metrics.to_csv(output_dir / "baselines_metrics.csv", index=False)
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
