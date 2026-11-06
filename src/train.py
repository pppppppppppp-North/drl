from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
from src.data.ticker_universe import resolve_tickers_from_config
from src.envs.trading_env import RewardConfig, ThaiStockTradingEnv, load_feature_table
from src.evaluation.metrics import equity_metrics
from src.features.build_features import write_features
from src.features.technical import add_technical_features
from src.utils.checkpoints import prune_checkpoints
from src.utils.config import ensure_dirs, load_config


def _pilot_features_match_config(features: pd.DataFrame, config: dict) -> bool:
    expected_tickers = set(resolve_tickers_from_config(config))
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
    return make_trading_env_from_features(config, features)


def make_trading_env_from_features(config: dict, features: pd.DataFrame) -> ThaiStockTradingEnv:
    env_cfg = config["environment"]
    return ThaiStockTradingEnv(
        features=features,
        feature_columns=config["features"]["columns"],
        lookback=config["features"]["lookback"],
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        slippage=env_cfg.get("slippage", 0.0),
        max_turnover=env_cfg.get("max_turnover"),
        max_position_weight=env_cfg.get("max_position_weight", 1.0),
        min_cash_weight=env_cfg.get("min_cash_weight", 0.0),
        reward_config=RewardConfig(**env_cfg.get("reward", {})),
    )


def build_training_feature_slices(
    features: pd.DataFrame,
    *,
    num_envs: int,
    lookback: int,
    seed: int,
    window_dates: int | None = None,
) -> list[pd.DataFrame]:
    """Create distinct chronological slices for parallel training environments."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    dates = pd.Index(pd.to_datetime(features["date"]).drop_duplicates().sort_values())
    minimum_dates = lookback + 2
    if len(dates) < minimum_dates:
        raise ValueError(f"training features need at least {minimum_dates} dates; found {len(dates)}")
    if num_envs == 1:
        return [features.copy().reset_index(drop=True)]

    default_window = max(minimum_dates, int(np.ceil(len(dates) * 0.75)))
    window_dates = int(window_dates or default_window)
    if window_dates < minimum_dates or window_dates > len(dates):
        raise ValueError(f"training env window must be between {minimum_dates} and {len(dates)} dates")
    possible_starts = len(dates) - window_dates + 1
    if possible_starts < num_envs:
        raise ValueError(
            f"cannot build {num_envs} distinct training slices of {window_dates} dates from {len(dates)} dates"
        )

    rng = np.random.default_rng(seed)
    starts = sorted(int(value) for value in rng.choice(possible_starts, size=num_envs, replace=False))
    slices = []
    for start in starts:
        selected_dates = dates[start : start + window_dates]
        part = features[pd.to_datetime(features["date"]).isin(selected_dates)].copy()
        part["date"] = pd.to_datetime(part["date"])
        slices.append(part.sort_values(["date", "ticker"]).reset_index(drop=True))
    return slices


def build_vectorized_training_env(config: dict, features: pd.DataFrame, seed: int):
    """Build one monitored env or genuinely distinct parallel vector envs."""
    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:  # pragma: no cover - depends on optional training stack.
        raise ImportError("stable-baselines3 is required for vectorized training") from exc

    training = config.get("training", {})
    num_envs = int(training.get("num_envs", 1))
    lookback = int(config["features"]["lookback"])
    slices = build_training_feature_slices(
        features,
        num_envs=num_envs,
        lookback=lookback,
        seed=seed,
        window_dates=training.get("env_window_dates"),
    )

    def env_factory(part: pd.DataFrame, rank: int):
        def _make():
            env = make_trading_env_from_features(config, part)
            env.reset(seed=seed + rank)
            return Monitor(env)

        return _make

    factories = [env_factory(part, rank) for rank, part in enumerate(slices)]
    if num_envs == 1:
        return factories[0]()
    backend = str(training.get("vectorization_backend", "subproc")).lower()
    if backend == "dummy":
        return DummyVecEnv(factories)
    if backend != "subproc":
        raise ValueError("training.vectorization_backend must be 'dummy' or 'subproc'")
    return SubprocVecEnv(factories, start_method=str(training.get("subproc_start_method", "forkserver")))


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
            "transaction_cost_paid": info.get("transaction_cost_paid", 0.0),
            "slippage_paid": info.get("slippage_paid", 0.0),
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
                "transaction_cost_paid": info.get("transaction_cost_paid", 0.0),
                "slippage_paid": info.get("slippage_paid", 0.0),
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
        allowed = {
            "learning_rate",
            "buffer_size",
            "learning_starts",
            "batch_size",
            "gamma",
            "tau",
            "train_freq",
            "gradient_steps",
        }
        if algo == "sac":
            allowed.update({"ent_coef", "target_entropy"})
        if algo == "td3":
            allowed.update({"policy_delay", "target_policy_noise", "target_noise_clip"})

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
    except ImportError:
        return False

    algo = config["training"].get("algo", "ppo").lower()
    model_cls = _stable_baselines_algorithm(algo)
    if model_cls is None:
        return False

    train_split = _training_split(config)
    eval_split = _evaluation_split(config)
    training_features = _features_for_split(config, train_split)
    seed = int(config["training"].get("seed", config["project"].get("seed", 42)))
    train_env = build_vectorized_training_env(config, training_features, seed)
    total_timesteps = int(config["training"].get("total_timesteps", 10000))
    model = model_cls(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(output_dir / "tensorboard"),
        **_algorithm_kwargs(algo, total_timesteps, config.get("training", {})),
    )
    checkpoint_dir = output_dir / "checkpoints" / algo
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_freq = int(config["training"].get("checkpoint_freq", max(total_timesteps // 5, 1)))
    checkpoint_prefix = f"{algo}_checkpoint"
    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_freq // int(config["training"].get("num_envs", 1)), 1),
        save_path=str(checkpoint_dir),
        name_prefix=checkpoint_prefix,
        save_replay_buffer=bool(
            config["training"].get("save_replay_buffer", algo in {"ddpg", "sac", "td3"})
        ),
        save_vecnormalize=True,
    )
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save(output_dir / f"{algo}_model")
    train_env.close()

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
    prune_checkpoints(
        checkpoint_dir,
        name_prefix=checkpoint_prefix,
        keep_last=int(config["training"].get("checkpoint_keep_last", 2)),
    )
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
