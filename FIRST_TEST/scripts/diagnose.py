import numpy as np
from stable_baselines3 import PPO
from src.env.trading_env import TradingEnv
from src.data.technicals import TechnicalIndicatorEngine
import pandas as pd
# Create test data
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
print("Data shape:", df.shape)
print("Price range:", df['Close'].min(), "-", df['Close'].max())
# Load model and test
model = PPO.load("checkpoints/demo_agent")
env = TradingEnv(df, use_feature_pipeline=True)
obs, _ = env.reset()
print("\nObservation shape:", obs.shape)
print("Observation sample:", obs[:10])
# Test predictions
for i in range(5):
    action, _ = model.predict(obs)
    print(f"Step {i}: Action={action}, Action probs: {model.policy.get_distribution(obs).distribution.probs}")
    obs, reward, done, trunc, info = env.step(action)
    if done or trunc:
        break
print("\nDiagnosis: Check if model is actually learning to predict different actions")
