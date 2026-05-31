import numpy as np
from stable_baselines3 import PPO
from src.env.trading_env import TradingEnv
from src.data.technicals import TechnicalIndicatorEngine
import pandas as pd
# Same seed as TRAINING (42)
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=50)
df = pd.DataFrame({
    'Close': 100 + np.cumsum(np.random.randn(50) * 2),
    'Open': 100 + np.cumsum(np.random.randn(50) * 2),
    'High': 105 + np.cumsum(np.random.randn(50) * 2),
    'Low': 95 + np.cumsum(np.random.randn(50) * 2),
    'Volume': np.random.randint(1000, 10000, 50)
}, index=dates)
ta = TechnicalIndicatorEngine()
df = ta.add_indicators(df)
df.dropna(inplace=True)
model = PPO.load("checkpoints/demo_agent")
env = TradingEnv(df, use_feature_pipeline=True)
obs, _ = env.reset()
actions_taken = []
for i in range(min(20, len(df)-1)):
    action, _ = model.predict(obs, deterministic=True)
    actions_taken.append(int(action))
    obs, reward, done, trunc, info = env.step(action)
    if done or trunc:
        break
action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
print("Actions on TRAINING distribution:")
for i, a in enumerate(actions_taken):
    print(f"  Step {i}: {action_names[a]}")
buys = actions_taken.count(1)
sells = actions_taken.count(2)
holds = actions_taken.count(0)
print(f"\nSummary: {buys} buys, {sells} sells, {holds} holds")
