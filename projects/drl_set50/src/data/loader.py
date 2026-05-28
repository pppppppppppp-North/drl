import yfinance as yf
import pandas as pd
from typing import List, Optional

class SET50Loader:
    """
    Data loader for Thailand SET50 stocks.
    """
    def __init__(self, tickers: Optional[List[str]] = None):
        # Default top SET50 stocks if none provided
        self.tickers = tickers if tickers else [
            "PTT.BK", "AOT.BK", "CPALL.BK", "ADVANC.BK", "scb.bk",
            "kbuk.bk", "bdms.bk", "aot.bk", "delta.bk", "intuch.bk"
        ]
        # Normalize tickers to uppercase
        self.tickers = [t.upper() for t in self.tickers]

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches OHLCV data for the initialized tickers from Yahoo Finance.
        Returns a MultiIndex DataFrame (Ticker, Date).
        """
        print(f"Fetching data for {len(self.tickers)} tickers from {start_date} to {end_date}...")
        
        # Download data
        data = yf.download(
            self.tickers, 
            start=start_date, 
            end=end_date, 
            group_by='ticker',
            progress=False
        )
        
        # If single ticker, yfinance doesn't return MultiIndex columns in the same way
        # enforcing consistent structure if needed, but group_by='ticker' usually handles it.
        
        return data

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Loads data from a local CSV file.
        Expects a format compatible with the training pipeline.
        """
        return pd.read_csv(filepath, index_col=0, parse_dates=True)

if __name__ == "__main__":
    # Simple test
    loader = SET50Loader(["PTT.BK", "AOT.BK"])
    df = loader.fetch_data("2023-01-01", "2023-01-10")
    print(df.head())
