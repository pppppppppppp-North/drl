import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np
import os
import argparse
import torch
from src.env.trading_env import TradingEnv
from src.data.loader import SET50Loader
from src.data.technicals import TechnicalIndicatorEngine
from src.models.combined import MultiModalExtractor
from src.models.llm_extractor import FinBERTExtractor, MockLLMExtractor

def train_agent(
    tickers: list = ["PTT.BK"], 
    start_date: str = "2020-01-01", 
    end_date: str = "2023-01-01", 
    total_timesteps: int = 10000,
    news_file: str = None
):
    """
    Trains a PPO agent using Bi-LSTM + FinBERT features.
    """
    # 1. Load Data
    loader = SET50Loader(tickers)
    try:
        df = loader.fetch_data(start_date, end_date)
        if df.empty: raise ValueError("Empty DF")
        if isinstance(df.columns, pd.MultiIndex):
            ticker = df.columns.levels[0][0]
            df = df[ticker]
    except Exception as e:
        print(f"Data fetch failed: {e}. using dummy.")
        dates = pd.date_range(start=start_date, periods=100)
        df = pd.DataFrame({'Close': np.random.uniform(100, 200, size=100), 'High':100,'Low':100,'Open':100,'Volume':100}, index=dates)

    # 2. Add Technical Indicators
    ta_engine = TechnicalIndicatorEngine()
    df = ta_engine.add_indicators(df)
    df.dropna(inplace=True) 

    # 3. Generate LLM Embeddings
    print("Generating LLM Embeddings (FinBERT)...")
    try:
        # Try loading real extractor
        llm = FinBERTExtractor(device="cpu") 
        
        # Prepare text data
        headlines = []
        if news_file and os.path.exists(news_file):
            print(f"Loading news from {news_file}...")
            news_df = pd.read_csv(news_file, parse_dates=['Date'])
            # We need to align news with market data dates.
            # Simplified strategy: Group news by date and join with market dataframe.
            # Convert market index to date (it's likely datetime already)
            
            # Create a Series of joined headlines per day
            # Format: "Title1. Title2."
            daily_news = news_df.groupby(news_df['Date'].dt.date)['Title'].apply(lambda x: ". ".join(x))
            
            # Reindex to match market df
            # Market df index is datetime, convert to date for mapping
            market_dates = df.index.date
            
            aligned_news = []
            for d in market_dates:
                if d in daily_news.index:
                    aligned_news.append(daily_news[d])
                else:
                    aligned_news.append("No news today.") # Placeholder
            headlines = aligned_news
        else:
            print("No news file provided or found. Using synthetic headlines.")
            headlines = [f"Market report for {d}" for d in df.index]

        # Batch Inference
        # For demo speed, we might trim or use mock if list is huge.
        print(f"Embedding {len(headlines)} items...")
        batch_size = 32
        embeddings = []
        # If too many, warn user? For now just run it.
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i+batch_size]
            emb = llm.get_embedding(batch)
            embeddings.append(emb)
        llm_embeddings = torch.cat(embeddings).numpy()
        
    except Exception as e:
        print(f"LLM extraction failed: {e}. Using Mock.")
        llm = MockLLMExtractor()
        llm_embeddings = llm.get_embedding([""] * len(df)).numpy()

    # 4. Create Environment with Dict Observation
    env_maker = lambda: TradingEnv(df, llm_embeddings=llm_embeddings)
    env = DummyVecEnv([env_maker])
    
    # 5. Initialize PPO Agent with MultiInputPolicy and Custom Extractor
    policy_kwargs = dict(
        features_extractor_class=MultiModalExtractor,
        features_extractor_kwargs=dict(features_dim=64),
    )
    
    model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    
    # 6. Train
    print(f"Starting training on {len(df)} data points...")
    model.learn(total_timesteps=total_timesteps)
    
    # 7. Save
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/ppo_multimodal_agent")
    print("Model saved to checkpoints/ppo_multimodal_agent")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["PTT.BK"])
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--news_file", type=str, default=None, help="Path to CSV with Date and Title columns")
    args = parser.parse_args()
    
    train_agent(tickers=args.tickers, total_timesteps=args.timesteps, news_file=args.news_file)
