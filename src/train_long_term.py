"""Training entrypoint for the long-term environment (milestones M1 and M5).

``src/train.py`` belongs to the manifest-locked V2 release deployed on BistKA and is left untouched.
This module reuses its unchanged helpers and adds:

- ``LongTermTradingEnv`` options from the ``environment`` config block (``accounting``,
  ``rebalance_every``, ``action_transform``, ``reward.type`` ...); ``train_random_start`` and
  ``train_episode_length`` apply to training environments only;
- ``training.policy``: SB3 ``policy_kwargs``, e.g. the ``asset_lstm`` fusion feature extractor;
- ``training.normalize_observations``: train-only ``VecNormalize`` statistics, frozen and reused at
  evaluation.

    python -m src.train_long_term --config <config.yaml> [--algo ppo] [--total-timesteps N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.envs.long_term_env import LongTermRewardConfig, LongTermTradingEnv
from src.evaluation.metrics import equity_metrics
from src.train import (
    _algorithm_kwargs,
    _evaluation_split,
    _features_for_split,
    _stable_baselines_algorithm,
    _training_split,
    _validate_env_length,
    build_training_feature_slices,
    ensure_feature_inputs,
    write_split_summary,
    write_walk_forward_summary,
)
from src.utils.checkpoints import prune_checkpoints
from src.utils.config import ensure_dirs, load_config


def make_long_term_env_from_features(
    config: dict, features: pd.DataFrame, *, training: bool = False
) -> LongTermTradingEnv:
    """Build the environment; ``training=True`` enables training-only episode sampling options."""

    env_cfg = config["environment"]
    episode_length = env_cfg.get("train_episode_length") if training else None
    return LongTermTradingEnv(
        features=features,
        feature_columns=config["features"]["columns"],
        lookback=config["features"]["lookback"],
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        slippage=env_cfg.get("slippage", 0.0),
        max_turnover=env_cfg.get("max_turnover"),
        max_position_weight=env_cfg.get("max_position_weight", 1.0),
        min_cash_weight=env_cfg.get("min_cash_weight", 0.0),
        reward_config=LongTermRewardConfig(**env_cfg.get("reward", {})),
        accounting=env_cfg.get("accounting", "legacy"),
        rebalance_every=int(env_cfg.get("rebalance_every", 1)),
        random_start=bool(env_cfg.get("train_random_start", False)) if training else False,
        episode_length=int(episode_length) if episode_length is not None else None,
        action_transform=env_cfg.get("action_transform", "clip"),
    )


def make_long_term_env(config: dict, split: str = "all") -> LongTermTradingEnv:
    features = _features_for_split(config, split)
    _validate_env_length(features, int(config["features"]["lookback"]), split)
    return make_long_term_env_from_features(config, features)


def build_long_term_training_env(config: dict, features: pd.DataFrame, seed: int):
    """One monitored env, or distinct parallel slices; optionally wrapped in train-only VecNormalize."""

    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    except ImportError as exc:  # pragma: no cover - depends on optional training stack.
        raise ImportError("stable-baselines3 is required for training") from exc

    training = config.get("training", {})
    num_envs = int(training.get("num_envs", 1))
    slices = build_training_feature_slices(
        features,
        num_envs=num_envs,
        lookback=int(config["features"]["lookback"]),
        seed=seed,
        window_dates=training.get("env_window_dates"),
    )

    def env_factory(part: pd.DataFrame, rank: int):
        def _make():
            env = make_long_term_env_from_features(config, part, training=True)
            env.reset(seed=seed + rank)
            return Monitor(env)

        return _make

    factories = [env_factory(part, rank) for rank, part in enumerate(slices)]
    normalize = bool(training.get("normalize_observations", False))
    if num_envs == 1 and not normalize:
        return factories[0]()
    backend = str(training.get("vectorization_backend", "subproc")).lower()
    if num_envs == 1 or backend == "dummy":
        vec_env = DummyVecEnv(factories)
    elif backend != "subproc":
        raise ValueError("training.vectorization_backend must be 'dummy' or 'subproc'")
    else:
        vec_env = SubprocVecEnv(factories, start_method=str(training.get("subproc_start_method", "forkserver")))
    if normalize:
        # Observation statistics are fitted on training rollouts only and frozen for evaluation.
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec_env


def policy_kwargs_from_config(config: dict, features: pd.DataFrame) -> dict:
    """SB3 ``policy_kwargs`` from ``training.policy`` (empty dict keeps the default MLP)."""

    policy_config = config.get("training", {}).get("policy")
    if not policy_config:
        return {}
    from src.models.feature_extractors import build_policy_kwargs

    return build_policy_kwargs(
        policy_config,
        lookback=int(config["features"]["lookback"]),
        num_assets=int(features["ticker"].nunique()),
        feature_columns=list(config["features"]["columns"]),
    )


def evaluate_long_term_model(
    model,
    env: LongTermTradingEnv,
    seed: int,
    policy_name: str = "ppo",
    return_actions: bool = False,
    obs_normalizer=None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic rollout; ``obs_normalizer`` (e.g. frozen ``VecNormalize.normalize_obs``) is applied first."""

    def _row(info: dict) -> dict:
        return {
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

    obs, info = env.reset(seed=seed)
    rows = [_row(info)]
    action_rows = [
        {"date": info["date"], "policy": policy_name, "ticker": t, "weight": float(w)}
        for t, w in zip(env.tickers, info["weights"])
    ]
    done = False
    while not done:
        prepared = obs_normalizer(obs) if obs_normalizer is not None else obs
        action, _ = model.predict(prepared, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        rows.append(_row(info))
        action_rows.extend(
            {"date": info["date"], "policy": policy_name, "ticker": t, "weight": float(w)}
            for t, w in zip(env.tickers, info["weights"])
        )
        done = terminated or truncated
    curve = pd.DataFrame(rows)
    if return_actions:
        return curve, pd.DataFrame(action_rows)
    return curve


def train_long_term(config: dict, output_dir: Path) -> None:
    from stable_baselines3.common.callbacks import CheckpointCallback

    algo = config["training"].get("algo", "ppo").lower()
    model_cls = _stable_baselines_algorithm(algo)
    if model_cls is None:
        raise ImportError(f"stable-baselines3 algorithm {algo!r} is unavailable")

    training_features = _features_for_split(config, _training_split(config))
    seed = int(config["training"].get("seed", config["project"].get("seed", 42)))
    train_env = build_long_term_training_env(config, training_features, seed)
    total_timesteps = int(config["training"].get("total_timesteps", 10000))
    model_kwargs = _algorithm_kwargs(algo, total_timesteps, config.get("training", {}))
    policy_kwargs = policy_kwargs_from_config(config, training_features)
    if policy_kwargs:
        model_kwargs["policy_kwargs"] = policy_kwargs
    model = model_cls(
        "MlpPolicy",
        train_env,
        verbose=int(config["training"].get("verbose", 1)),
        seed=seed,
        tensorboard_log=str(output_dir / "tensorboard"),
        **model_kwargs,
    )
    checkpoint_dir = output_dir / "checkpoints" / algo
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_freq = int(config["training"].get("checkpoint_freq", max(total_timesteps // 5, 1)))
    checkpoint_prefix = f"{algo}_checkpoint"
    callback = CheckpointCallback(
        save_freq=max(checkpoint_freq // int(config["training"].get("num_envs", 1)), 1),
        save_path=str(checkpoint_dir),
        name_prefix=checkpoint_prefix,
        save_replay_buffer=bool(config["training"].get("save_replay_buffer", algo in {"ddpg", "sac", "td3"})),
        save_vecnormalize=True,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(output_dir / f"{algo}_model")
    obs_normalizer = None
    if hasattr(train_env, "normalize_obs") and hasattr(train_env, "obs_rms"):
        train_env.training = False
        train_env.save(str(output_dir / f"{algo}_vecnormalize.pkl"))
        obs_normalizer = train_env.normalize_obs
    train_env.close()

    eval_curve, eval_actions = evaluate_long_term_model(
        model,
        make_long_term_env(config, split=_evaluation_split(config)),
        seed,
        policy_name=algo,
        return_actions=True,
        obs_normalizer=obs_normalizer,
    )
    eval_curve.to_csv(output_dir / f"{algo}_equity_curve.csv", index=False)
    eval_actions.to_csv(output_dir / f"{algo}_actions.csv", index=False)
    metrics = pd.DataFrame([{"policy": algo, **equity_metrics(eval_curve)}])
    metrics.to_csv(output_dir / f"{algo}_metrics.csv", index=False)
    prune_checkpoints(checkpoint_dir, name_prefix=checkpoint_prefix, keep_last=int(config["training"].get("checkpoint_keep_last", 2)))
    print(metrics.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--algo", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--train-split", choices=["all", "train", "validation", "test"], default=None)
    parser.add_argument("--eval-split", choices=["all", "train", "validation", "test"], default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.algo:
        config["training"]["algo"] = args.algo
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
    train_long_term(config, output_dir)


if __name__ == "__main__":
    main()
