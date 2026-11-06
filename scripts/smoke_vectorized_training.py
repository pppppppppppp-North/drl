from __future__ import annotations

import argparse

from src.data.synthetic import make_synthetic_prices
from src.features.technical import add_technical_features
from src.train import build_vectorized_training_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["ppo", "a2c", "ddpg", "td3", "sac"], default="ppo")
    parser.add_argument("--backend", choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=32)
    args = parser.parse_args()

    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

    config = {
        "project": {"seed": 1},
        "data": {
            "tickers": ["PTT.BK", "AOT.BK"],
            "start": "2023-01-01",
            "end": "2023-06-30",
            "initial_price": 100.0,
        },
        "features": {
            "lookback": 5,
            "columns": ["return_1d", "return_5d", "volatility_20d", "ma_ratio_10", "rsi_14"],
        },
        "environment": {"initial_cash": 1_000_000.0, "transaction_cost": 0.001, "reward": {}},
        "training": {
            "num_envs": args.num_envs,
            "env_window_dates": 60,
            "vectorization_backend": args.backend,
            "subproc_start_method": "forkserver",
        },
    }
    features = add_technical_features(make_synthetic_prices(config))
    env = build_vectorized_training_env(config, features, seed=1)
    algorithm_classes = {"ppo": PPO, "a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC}
    if args.algorithm == "ppo":
        algorithm_kwargs = {"n_steps": 8, "batch_size": 8}
    elif args.algorithm == "a2c":
        algorithm_kwargs = {"n_steps": 5}
    else:
        algorithm_kwargs = {
            "learning_starts": 0,
            "buffer_size": 1_000,
            "batch_size": 8,
            "train_freq": 1,
            "gradient_steps": 1,
        }
    model = algorithm_classes[args.algorithm](
        "MlpPolicy",
        env,
        verbose=0,
        seed=1,
        **algorithm_kwargs,
    )
    model.learn(total_timesteps=args.timesteps)
    env.close()
    print(
        f"algorithm={args.algorithm} backend={args.backend} envs={args.num_envs} "
        f"timesteps={model.num_timesteps} status=pass"
    )


if __name__ == "__main__":
    main()
