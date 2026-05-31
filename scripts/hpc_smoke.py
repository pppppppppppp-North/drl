from __future__ import annotations

import argparse

import numpy as np

from src.data.synthetic import make_synthetic_prices
from src.envs.trading_env import RewardConfig, ThaiStockTradingEnv
from src.features.technical import add_technical_features
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/debug.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    prices = make_synthetic_prices(config)
    features = add_technical_features(prices)
    env_cfg = config["environment"]
    env = ThaiStockTradingEnv(
        features=features,
        feature_columns=config["features"]["columns"],
        lookback=config["features"]["lookback"],
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        reward_config=RewardConfig(**env_cfg.get("reward", {})),
    )
    obs, info = env.reset(seed=config["project"]["seed"])
    for _ in range(5):
        action = np.full(env.num_assets, 1.0 / env.num_assets)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(
        "hpc smoke ok",
        f"obs_shape={obs.shape}",
        f"date={info['date']}",
        f"value={info['portfolio_value']:.2f}",
    )


if __name__ == "__main__":
    main()

