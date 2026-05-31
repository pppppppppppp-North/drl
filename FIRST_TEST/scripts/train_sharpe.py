#!/usr/bin/env python3
"""Train agent with Sharpe Ratio rewards on real PTT.BK data."""
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
sys.path.insert(0, '/lustrefs/project/25sfcs03/FIRST_TEST')

from src.env.trading_env_sharpe import TradingEnvSharpe

print("="*70)
print("Training with SHARPE RATIO Rewards")
print("="*70)

# Load real data
df = pd.read_csv("/lustrefs/project/25sfcs03/FIRST_TEST/data/cached/PTT_BK_2020-01-01_2024-01-01.csv", 
                 index_col=0, parse_dates=True)
print(f"\nLoaded {len(df)} days of PTT.BK data")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Train/test split
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
print(f"Training on {len(train_df)} days\n")

# Create environment with SHARPE rewards, no fees initially
env = DummyVecEnv([lambda: TradingEnvSharpe(
    train_df, 
    use_feature_pipeline=True, 
    transaction_fee=0.0,  # No fees to encourage trading
    reward_window=20  # 20-day rolling Sharpe
)])

# Train with more exploration
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.05, learning_rate=0.0003)
model.learn(total_timesteps=50000)
model.save("/lustrefs/project/25sfcs03/FIRST_TEST/checkpoints/ptt_sharpe_agent")

print("\n✓ Training complete with SHARPE rewards!")
print("✓ Model saved to checkpoints/ptt_sharpe_agent")
