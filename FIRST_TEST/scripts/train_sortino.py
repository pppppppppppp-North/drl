#!/usr/bin/env python3
"""Train with Sortino Ratio + trading bonuses."""
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
sys.path.insert(0, '/lustrefs/project/25sfcs03/FIRST_TEST')

from src.env.trading_env_sortino import TradingEnvSortino

print("="*70)
print("Training with SORTINO RATIO + Trading Bonuses")
print("="*70)

df = pd.read_csv("/lustrefs/project/25sfcs03/FIRST_TEST/data/cached/PTT_BK_2020-01-01_2024-01-01.csv", 
                 index_col=0, parse_dates=True)
print(f"\nData: {len(df)} days")
print(f"Range: {df.index[0]} to {df.index[-1]}\n")

train_df = df.copy()

# Sortino environment with trading bonuses
env = DummyVecEnv([lambda: TradingEnvSortino(
    train_df, 
    use_feature_pipeline=True, 
    transaction_fee=0.001,  # Small fees
    reward_window=20
)])

# Higher exploration
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.1, learning_rate=0.0003)
model.learn(total_timesteps=100000)
model.save("/lustrefs/project/25sfcs03/FIRST_TEST/checkpoints/ptt_sortino_100k")

print("\n✓ Sortino training complete!")
print("✓ Saved to checkpoints/ptt_sortino_100k")
