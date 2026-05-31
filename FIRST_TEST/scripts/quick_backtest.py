#!/usr/bin/env python3
"""
Quick backtest for the demo agent.
"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.env.trading_env import TradingEnv
from src.data.technicals import TechnicalIndicatorEngine

def create_test_data(n_days: int = 50) -> pd.DataFrame:
    """Generate synthetic test data."""
    np.random.seed(99)  # Different seed for test data
    dates = pd.date_range(start="2023-02-01", periods=n_days)
    
    price = 110
    prices = [price]
    for _ in range(n_days - 1):
        change = np.random.randn() * 2
        price = max(price + change, 50)
        prices.append(price)
    
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
        'Low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days)
    }, index=dates)
    
    return df

def quick_backtest():
    """Run quick backtest on demo agent."""
    print("=" * 70)
    print("QUICK BACKTEST: Demo Agent")
    print("=" * 70)
    
    # Generate test data
    print("\n[1/4] Generating test data...")
    df = create_test_data(50)
    ta_engine = TechnicalIndicatorEngine()
    df = ta_engine.add_indicators(df)
    df.dropna(inplace=True)
    print(f"   ✓ {len(df)} days of test data")
    
    # Load model
    print("\n[2/4] Loading trained model...")
    try:
        model = PPO.load("checkpoints/demo_agent")
        print("   ✓ Model loaded")
    except FileNotFoundError:
        print("   ✗ Model not found. Run quick_demo.py first!")
        return
    
    # Create environment and run
    print("\n[3/4] Running agent...")
    env = TradingEnv(df, use_feature_pipeline=True, initial_balance=100000.0)
    obs, _ = env.reset()
    
    portfolio_values = [100000.0]
    actions = []
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append(info['portfolio_value'])
        actions.append(action)
    
    # Calculate results
    print("\n[4/4] Results:")
    initial = portfolio_values[0]
    final = portfolio_values[-1]
    returns = (final - initial) / initial * 100
    
    buy_count = actions.count(1)
    sell_count = actions.count(2)
    hold_count = actions.count(0)
    
    print(f"   • Initial Balance:  ${initial:,.2f}")
    print(f"   • Final Balance:    ${final:,.2f}")
    print(f"   • Returns:          {returns:+.2f}%")
    print(f"   • Actions: {buy_count} buys, {sell_count} sells, {hold_count} holds")
    
    print("\n" + "=" * 70)
    print("✅ BACKTEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    quick_backtest()
