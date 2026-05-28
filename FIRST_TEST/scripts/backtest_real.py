import pandas as pd
from stable_baselines3 import PPO
from src.env.trading_env import TradingEnv
print("="*70)
print("BACKTEST: Real PTT.BK Data")
print("="*70)
# Load real data
df = pd.read_csv("data/cached/PTT_BK_2020-01-01_2024-01-01.csv", index_col=0, parse_dates=True)
# Use test split (last 20%)
train_size = int(len(df) * 0.8)
test_df = df.iloc[train_size:]
print(f"\nTest period: {test_df.index[0]} to {test_df.index[-1]}")
print(f"Test days: {len(test_df)}")
print(f"Price range: {test_df['Close'].min():.2f} - {test_df['Close'].max():.2f}")
# Load trained model
model = PPO.load("checkpoints/ptt_real_agent")
env = TradingEnv(test_df, use_feature_pipeline=True, transaction_fee=0.001)
# Run backtest
obs, _ = env.reset()
actions = []
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    actions.append(int(action))
# Results
initial = 100000
final = info['portfolio_value']
returns = (final - initial) / initial * 100
buys = actions.count(1)
sells = actions.count(2)
holds = actions.count(0)
print(f"\nResults:")
print(f"  Initial Balance:  ${initial:,.2f}")
print(f"  Final Balance:    ${final:,.2f}")
print(f"  Returns:          {returns:+.2f}%")
print(f"  Actions: {buys} buys, {sells} sells, {holds} holds")
# Buy & Hold benchmark
buy_hold_return = (test_df['Close'].iloc[-1] / test_df['Close'].iloc[0] - 1) * 100
print(f"\nBuy & Hold Benchmark: {buy_hold_return:+.2f}%")
print("="*70)
