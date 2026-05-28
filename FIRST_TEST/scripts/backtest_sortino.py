import pandas as pd
from stable_baselines3 import PPO
import sys
sys.path.insert(0, '/lustrefs/project/25sfcs03/FIRST_TEST')
from src.env.trading_env_sortino import TradingEnvSortino
print("="*70)
print("BACKTEST: Sortino Model")
print("="*70)
df = pd.read_csv("/lustrefs/project/25sfcs03/FIRST_TEST/data/cached/PTT_BK_2020-01-01_2024-01-01.csv", 
                 index_col=0, parse_dates=True)
# Test on 2023
test_df = df[df.index.year == 2023]
print(f"\nTest: {test_df.index[0]} to {test_df.index[-1]}")
print(f"Days: {len(test_df)}")
model = PPO.load("/lustrefs/project/25sfcs03/FIRST_TEST/checkpoints/ptt_sortino_100k")
env = TradingEnvSortino(test_df, use_feature_pipeline=True, transaction_fee=0.001)
obs, _ = env.reset()
actions = []
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    actions.append(int(action))
initial = 100000
final = info['portfolio_value']
returns = (final - initial) / initial * 100
buys = actions.count(1)
sells = actions.count(2)
holds = actions.count(0)
print(f"\nResults:")
print(f"  Initial:  ${initial:,.2f}")
print(f"  Final:    ${final:,.2f}")
print(f"  Returns:  {returns:+.2f}%")
print(f"  Actions: {buys} buys, {sells} sells, {holds} holds")
buy_hold = (test_df['Close'].iloc[-1] / test_df['Close'].iloc[0] - 1) * 100
print(f"\nBuy & Hold: {buy_hold:+.2f}%")
if sells > 5:
    print("\n🎉 SUCCESS! Agent learned to SELL!")
elif buys > 5:
    print("\n⚠️ More buys, but still not many sells")
else:
    print("\n❌ Back to passive")
print("="*70)
