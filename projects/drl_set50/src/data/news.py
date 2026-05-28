import pandas as pd
import feedparser
import newspaper
from typing import List, Optional
from datetime import datetime
import time
import urllib.parse

class GoogleNewsLoader:
    """
    Fetches financial news from Google News RSS feed.
    """
    def __init__(self, query: str = "Tailand Stock Market", lang: str = "en"):
        encoded_query = urllib.parse.quote(query)
        self.base_url = f"https://news.google.com/rss/search?q={encoded_query}&hl={lang}-TH&gl=TH&ceid=TH:{lang}"
        
    def fetch_news(self, limit: int = 50) -> pd.DataFrame:
        """
        Fetches latest news from the RSS feed.
        """
        print(f"Fetching news from: {self.base_url}")
        feed = feedparser.parse(self.base_url)
        
        articles = []
        for entry in feed.entries[:limit]:
            title = entry.title
            link = entry.link
            pub_date = entry.published
            
            # Simple parsing using newspaper3k to get full text
            # Note: This is slow for many articles. For MVP we might just use title.
            # Using just title is often enough for Sentiment Analysis.
            # description = entry.description
            
            articles.append({
                "Date": pub_date,
                "Title": title,
                "Link": link,
                "Source": entry.source.title if 'source' in entry else "Google News"
            })
            
        df = pd.DataFrame(articles)
        # Convert Date to datetime if possible
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except:
            pass
            
        return df

if __name__ == "__main__":
    loader = GoogleNewsLoader(query="SET50 Thailand Economy")
    df = loader.fetch_news(limit=5)
    print(df.head())
