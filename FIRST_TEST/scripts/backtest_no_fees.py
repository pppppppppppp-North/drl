import numpy as np
from stable_baselines3 import PPO
from src.env.trading_env import TradingEnv
from src.data.technicals import TechnicalIndicatorEngine
import pandas as pd
print("="*70)
print("BACKTEST: No-Fee Model")
print("="*70)
# Same seed as training
np.random.seed(42)
dates = pd.date_range(start="2023-02-01", periods=50)
df = pd.DataFrame({
    'Close': 110 + np.cumsum(np.random.randn(50) * 2),
    'Open': 110 + np.cumsum(np.random.randn(50) * 2),
    'High': 115 + np.cumsum(np.random.randn(50) * 2),
    'Low': 105 + np.cumsum(np.random.randn(50) * 2),
    'Volume': np.random.randint(1000, 10000, 50)
}, index=dates)
ta = TechnicalIndicatorEngine()
df = ta.add_indicators(df)
df.dropna(inplace=True)
print(f"\nTest data: {len(df)} days")
model = PPO.load("checkpoints/demo_agent_no_fees")
env = TradingEnv(df, use_feature_pipeline=True, transaction_fee=0.0)
obs, _ = env.reset()
actions = []
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    actions.append(action)
initial = 100000
final = info['portfolio_value']
returns = (final - initial) / initial * 100
action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
buys = actions.count(1)
sells = actions.count(2)
holds = actions.count(0)
print(f"\nResults:")
print(f"  • Initial: ${initial:,.2f}")
print(f"  • Final: ${final:,.2f}")
print(f"  • Returns: {returns:+.2f}%")
print(f"  • Actions: {buys} buys, {sells} sells, {holds} holds")
print("\n" + "="*70)
