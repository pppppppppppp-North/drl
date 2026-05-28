#!/usr/bin/env python3
"""
Quick demo trainer using synthetic data to prove the pipeline works.
This bypasses network issues and runs locally.
"""
import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
from src.env.trading_env import TradingEnv
from src.data.technicals import TechnicalIndicatorEngine

def create_synthetic_data(n_days: int = 100) -> pd.DataFrame:
    """Generate synthetic market data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    # Generate realistic price movement
    price = 100
    prices = [price]
    for _ in range(n_days - 1):
        change = np.random.randn() * 2  # Daily volatility
        price = max(price + change, 50)  # Prevent negative prices
        prices.append(price)
    
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
        'Low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days)
    }, index=dates)
    
    return df

def quick_demo_train(n_days: int = 100, timesteps: int = 2000):
    """
    Quick demo: synthetic data → feature pipeline → train → save
    """
    print("=" * 70)
    print("QUICK DEMO: End-to-End Pipeline Test")
    print("=" * 70)
    
    # 1. Create synthetic data
    print("\n[1/5] Generating synthetic market data...")
    df = create_synthetic_data(n_days)
    print(f"   ✓ Created {len(df)} days of data")
    
    # 2. Add technical indicators
    print("\n[2/5] Adding technical indicators...")
    ta_engine = TechnicalIndicatorEngine()
    df = ta_engine.add_indicators(df)
    df.dropna(inplace=True)
    print(f"   ✓ Added indicators, {len(df)} days after cleanup")
    
    # 3. Create environment with feature pipeline
    print("\n[3/5] Creating trading environment...")
    env = TradingEnv(df, use_feature_pipeline=True, sequence_length=20)
    print(f"   ✓ Environment created")
    print(f"   ✓ Observation space: {env.observation_space.shape}")
    print(f"   ✓ Feature pipeline: ENABLED")
    
    # Test observation generation
    obs, _ = env.reset()
    print(f"   ✓ Sample observation shape: {obs.shape}")
    
    # 4. Train PPO agent
    print(f"\n[4/5] Training PPO agent ({timesteps} steps)...")
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=timesteps)
    print("   ✓ Training complete")
    
    # 5. Save model
    print("\n[5/5] Saving model...")
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/demo_agent")
    print("   ✓ Model saved to checkpoints/demo_agent")
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE - Pipeline is working!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Run: .venv/bin/python scripts/quick_backtest.py")
    print("  - Or train longer with real data once network improves")
    
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=100, help='Days of synthetic data')
    parser.add_argument('--timesteps', type=int, default=2000, help='Training timesteps')
    args = parser.parse_args()
    
    quick_demo_train(n_days=args.days, timesteps=args.timesteps)
