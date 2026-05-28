import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np
import os
import argparse
from src.env.trading_env import TradingEnv
from src.data.loader import SET50Loader
from src.data.technicals import TechnicalIndicatorEngine

def train_agent(
    tickers: list = ["PTT.BK"], 
    start_date: str = "2020-01-01", 
    end_date: str = "2023-01-01", 
    total_timesteps: int = 10000
):
    """
    Trains a PPO agent on the given dataframe.
    """
    # 1. Load Data
    loader = SET50Loader(tickers)
    # For now, yfinance might fail if no internet or symbols wrong, so we wrap somewhat safely or expect errors.
    # In a real run, we'd handle caching.
    try:
        df = loader.fetch_data(start_date, end_date)
        if df.empty:
            raise ValueError("No data fetched. Check internet connection or tickers.")
        
        # Handle MultiIndex if necessary, for single ticker simplified logic:
        # If multiple tickers, we might need to train on them sequentially or parallel envs.
        # For MVP, let's assume we train on the first ticker if multi-index, or just the df.
        if isinstance(df.columns, pd.MultiIndex):
            # Just take the first ticker for this MVP demo
            ticker = df.columns.levels[0][0]
            df = df[ticker]
            
    except Exception as e:
        print(f"Data fetch failed: {e}")
        print("Using dummy data for verification fallback.")
        dates = pd.date_range(start=start_date, periods=100)
        df = pd.DataFrame({'Close': np.random.uniform(100, 200, size=100), 
                           'High': np.random.uniform(105, 205, size=100),
                           'Low': np.random.uniform(95, 195, size=100),
                           'Open': np.random.uniform(100, 200, size=100),
                           'Volume': np.random.randint(1000, 10000, size=100)}, index=dates)

    # 2. Add Technical Indicators
    ta_engine = TechnicalIndicatorEngine()
    df = ta_engine.add_indicators(df)
    
    # Drops NaNs from indicators
    df.dropna(inplace=True) 

    # 3. Create Environment
    env_maker = lambda: TradingEnv(df)
    env = DummyVecEnv([env_maker])
    
    # 4. Initialize PPO Agent
    # MlpPolicy is used for 1D vector observations. 
    # If we had image data, we'd use CnnPolicy.
    model = PPO("MlpPolicy", env, verbose=1)
    
    # 5. Train
    print(f"Starting training on {len(df)} data points...")
    model.learn(total_timesteps=total_timesteps)
    
    # 6. Save
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/ppo_trading_agent")
    print("Model saved to checkpoints/ppo_trading_agent")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["PTT.BK"])
    parser.add_argument("--timesteps", type=int, default=5000)
    args = parser.parse_args()
    
    train_agent(tickers=args.tickers, total_timesteps=args.timesteps)
