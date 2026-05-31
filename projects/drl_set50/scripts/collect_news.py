import pandas as pd
import os
import argparse
from datetime import datetime
from src.data.news import GoogleNewsLoader

def collect_news(query: str, output_file: str):
    """
    Fetches latest news and appends to a CSV file.
    """
    loader = GoogleNewsLoader(query=query)
    new_df = loader.fetch_news(limit=100)
    
    if new_df.empty:
        print("No news found.")
        return

    # Check if file exists to append or create
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file, parse_dates=['Date'])
        # Concatenate
        combined_df = pd.concat([existing_df, new_df])
        # Drop duplicates based on Title and Date
        combined_df.drop_duplicates(subset=['Title', 'Date'], inplace=True)
        combined_df.sort_values(by='Date', inplace=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Appended {len(new_df)} articles. Total: {len(combined_df)}")
    else:
        new_df.sort_values(by='Date', inplace=True)
        new_df.to_csv(output_file, index=False)
        print(f"Created {output_file} with {len(new_df)} articles.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="SET50 Thailand Stock")
    parser.add_argument("--output", type=str, default="data/news_data.csv")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    collect_news(args.query, args.output)
