import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.trading_env import TradingEnv
from src.data.technicals import TechnicalIndicatorEngine
print("Starting no-fee training...")
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=100)
df = pd.DataFrame({
    'Close': 100 + np.cumsum(np.random.randn(100) * 2),
    'Open': 100 + np.cumsum(np.random.randn(100) * 2),
    'High': 105 + np.cumsum(np.random.randn(100) * 2),
    'Low': 95 + np.cumsum(np.random.randn(100) * 2),
    'Volume': np.random.randint(1000, 10000, 100)
}, index=dates)
ta = TechnicalIndicatorEngine()
df = ta.add_indicators(df)
df.dropna(inplace=True)
print(f"Data: {len(df)} days")
env = DummyVecEnv([lambda: TradingEnv(df, use_feature_pipeline=True, transaction_fee=0.0)])
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01)
model.learn(total_timesteps=20000)
model.save("checkpoints/demo_agent_no_fees")
print("✓ Model trained WITHOUT transaction fees")
