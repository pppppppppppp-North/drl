from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import ensure_dirs, load_config


NewsProvider = Callable[[str], list[dict[str, Any]]]
NEWS_COLUMNS = ["date", "published_at", "ticker", "title", "summary", "publisher", "url", "text", "source"]


def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only when optional dependency is absent.
        raise ImportError("install yfinance to download Yahoo Finance news data") from exc
    return yf


def _yahoo_news_provider(ticker: str) -> list[dict[str, Any]]:
    yf = _import_yfinance()
    return yf.Ticker(ticker).news or []


def _content(item: dict[str, Any]) -> dict[str, Any]:
    content = item.get("content")
    return content if isinstance(content, dict) else item


def _url_from_content(content: dict[str, Any]) -> str:
    for key in ("canonicalUrl", "clickThroughUrl", "link"):
        value = content.get(key)
        if isinstance(value, dict) and value.get("url"):
            return str(value["url"])
        if isinstance(value, str):
            return value
    return ""


def _published_at(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def _normalize_news_item(ticker: str, item: dict[str, Any]) -> dict[str, Any] | None:
    content = _content(item)
    published = _published_at(content.get("pubDate", content.get("providerPublishTime")))
    if pd.isna(published):
        return None

    title = str(content.get("title") or "").strip()
    summary = str(content.get("summary") or content.get("description") or "").strip()
    if not title and not summary:
        return None

    provider = content.get("provider") or content.get("publisher") or {}
    if isinstance(provider, dict):
        publisher = str(provider.get("displayName") or provider.get("name") or "").strip()
    else:
        publisher = str(provider or "").strip()

    text = ". ".join(part for part in [title, summary] if part)
    published = published.tz_convert(UTC)
    return {
        "date": published.date().isoformat(),
        "published_at": published.isoformat(),
        "ticker": ticker,
        "title": title,
        "summary": summary,
        "publisher": publisher,
        "url": _url_from_content(content),
        "text": text,
        "source": "Yahoo Finance news via yfinance",
    }


def download_yahoo_news(tickers: list[str], provider: NewsProvider | None = None) -> pd.DataFrame:
    if not tickers:
        raise ValueError("at least one ticker is required")
    provider = provider or _yahoo_news_provider
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        for item in provider(ticker):
            normalized = _normalize_news_item(ticker, item)
            if normalized is not None:
                rows.append(normalized)
    if not rows:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    news = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    news = news.drop_duplicates(["ticker", "published_at", "title"]).sort_values(["date", "ticker", "published_at"])
    return news.reset_index(drop=True)


def write_news_manifest_row(
    manifest_path: Path,
    *,
    news: pd.DataFrame,
    raw_file_path: Path,
    source_name: str = "yahoo_news_latest",
) -> None:
    ensure_dirs(manifest_path.parent)
    if news.empty:
        date_range = "empty"
        symbols = ""
        missing_value_rate = 0.0
    else:
        date_range = (
            f"{pd.to_datetime(news['date']).min().date().isoformat()} to "
            f"{pd.to_datetime(news['date']).max().date().isoformat()}"
        )
        symbols = ";".join(sorted(news["ticker"].dropna().astype(str).unique()))
        missing_value_rate = float(news.isna().mean().mean())

    row = {
        "source": source_name,
        "access_method": "Yahoo Finance news via yfinance",
        "frequency": "event",
        "date_range": date_range,
        "symbols": symbols,
        "columns": ";".join(NEWS_COLUMNS),
        "license_note": "Yahoo Finance news access through yfinance; verify usage rights before publication.",
        "raw_file_path": str(raw_file_path),
        "missing_value_rate": missing_value_rate,
    }
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["source"] != source_name]
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    else:
        manifest = pd.DataFrame([row])
    manifest.to_csv(manifest_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_sentiment.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default="data/data_manifest.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    sentiment_config = config.get("sentiment", {})
    output = Path(args.output or sentiment_config.get("raw_news_path", "data/raw/news_yahoo_latest.csv"))
    ensure_dirs(output.parent)
    news = download_yahoo_news(list(config["data"]["tickers"]))
    news.to_csv(output, index=False)
    write_news_manifest_row(Path(args.manifest), news=news, raw_file_path=output)
    print(f"wrote {output} rows={len(news)} tickers={news['ticker'].nunique() if not news.empty else 0}")


if __name__ == "__main__":
    main()
