#!/usr/bin/env python3
"""
Data caching script to download and save SET50 data locally.
This avoids repeated API calls during training/testing.
"""
import pandas as pd
import os
from src.data.loader import SET50Loader
from src.data.technicals import TechnicalIndicatorEngine
import argparse

def cache_data(
    tickers: list,
    start_date: str,
    end_date: str,
    output_dir: str = "data/cached"
):
    """
    Download data and save to CSV for offline use.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    loader = SET50Loader(tickers)
    ta_engine = TechnicalIndicatorEngine()
    
    print(f"Fetching data for {len(tickers)} tickers: {tickers}")
    print(f"Date range: {start_date} to {end_date}")
    
    for ticker in tickers:
        print(f"\n[{ticker}] Downloading...")
        
        try:
            # Fetch single ticker
            single_loader = SET50Loader([ticker])
            df = single_loader.fetch_data(start_date, end_date)
            
            # Handle MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df = df[ticker]
            
            # Add technical indicators
            print(f"[{ticker}] Adding technical indicators...")
            df = ta_engine.add_indicators(df)
            df.dropna(inplace=True)
            
            # Save
            filename = f"{ticker.replace('.', '_')}_{start_date}_{end_date}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath)
            
            print(f"[{ticker}] ✓ Saved {len(df)} rows to {filepath}")
            
        except Exception as e:
            print(f"[{ticker}] ✗ Error: {e}")
            continue
    
    print(f"\n✓ Data cached in {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cache SET50 data locally')
    parser.add_argument('--tickers', nargs='+', default=['PTT.BK'],
                        help='Stock tickers to cache')
    parser.add_argument('--start', type=str, default='2023-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-02-01',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data/cached',
                        help='Output directory')
    
    args = parser.parse_args()
    
    cache_data(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output
    )
