import pandas as pd
import numpy as np
class TechnicalIndicatorEngine:
    def __init__(self):
        pass
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Simple Moving Averages
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        
        # EMA
        df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        
        # RSI (simplified)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        df.fillna(0, inplace=True)
        return df
