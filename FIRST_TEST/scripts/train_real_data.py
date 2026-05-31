import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.trading_env import TradingEnv
print("="*70)
print("Training on REAL PTT.BK Data")
print("="*70)
# Load cached real data
df = pd.read_csv("data/cached/PTT_BK_2020-01-01_2024-01-01.csv", index_col=0, parse_dates=True)
print(f"\nLoaded {len(df)} days of PTT.BK data")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
# Train/test split
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
print(f"\nTraining on {len(train_df)} days")
# Create environment with real data, low fees
env = DummyVecEnv([lambda: TradingEnv(train_df, use_feature_pipeline=True, transaction_fee=0.001)])
# Train
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01)
model.learn(total_timesteps=50000)
model.save("checkpoints/ptt_real_agent")
print("\n✓ Training complete on REAL data!")
print("✓ Model saved to checkpoints/ptt_real_agent")
