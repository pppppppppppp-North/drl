#!/usr/bin/env python3
"""Extended training with Sharpe Ratio rewards - 100k timesteps."""
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
sys.path.insert(0, '/lustrefs/project/25sfcs03/FIRST_TEST')
from src.env.trading_env_sharpe import TradingEnvSharpe
print("="*70)
print("EXTENDED Training with SHARPE RATIO Rewards (100k steps)")
print("="*70)
# Load real data
df = pd.read_csv("/lustrefs/project/25sfcs03/FIRST_TEST/data/cached/PTT_BK_2020-01-01_2024-01-01.csv", 
                 index_col=0, parse_dates=True)
print(f"\nLoaded {len(df)} days of PTT.BK data")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
# Use ALL data for training (more scenarios)
train_df = df.copy()
print(f"Training on ALL {len(train_df)} days to see more up/down trends\n")
# Create environment with SHARPE rewards, no fees
env = DummyVecEnv([lambda: TradingEnvSharpe(
    train_df, 
    use_feature_pipeline=True, 
    transaction_fee=0.0,
    reward_window=20
)])
# Checkpoint callback to save periodically
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="/lustrefs/project/25sfcs03/FIRST_TEST/checkpoints/",
    name_prefix="sharpe_long"
)
# Train with exploration
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.05, learning_rate=0.0003)
model.learn(total_timesteps=100000, callback=checkpoint_callback)
model.save("/lustrefs/project/25sfcs03/FIRST_TEST/checkpoints/ptt_sharpe_100k")
print("\n✓ Extended training complete (100k steps)!")
print("✓ Model saved to checkpoints/ptt_sharpe_100k")
