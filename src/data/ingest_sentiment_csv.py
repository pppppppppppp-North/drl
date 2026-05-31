from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.ingest_news_yahoo import NEWS_COLUMNS
from src.sentiment.extract_embeddings import _hash_sentiment
from src.utils.config import ensure_dirs, load_config


OPTIONAL_COLUMNS = ["sentiment_score"]
OUTPUT_COLUMNS = NEWS_COLUMNS + OPTIONAL_COLUMNS
DAILY_COLUMNS = ["date", "ticker", "sentiment_score", "news_count", "sentiment_source"]


def _column_or_blank(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column and column in df.columns:
        return df[column].fillna("").astype(str).str.strip()
    return pd.Series([""] * len(df), index=df.index)


def _parse_required_dates(values: pd.Series, *, column_name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        bad = values[parsed.isna()].astype(str).head(5).tolist()
        raise ValueError(f"could not parse {column_name} values: {bad}")
    return parsed


def _clean_ticker(values: pd.Series) -> pd.Series:
    tickers = values.fillna("").astype(str).str.strip().str.upper()
    if tickers.eq("").any():
        raise ValueError("sentiment CSV contains blank ticker values")
    return tickers


def normalize_sentiment_csv(
    raw: pd.DataFrame,
    *,
    date_column: str,
    published_at_column: str | None,
    ticker_column: str,
    title_column: str | None,
    summary_column: str | None,
    text_column: str | None,
    sentiment_column: str | None,
    publisher_column: str | None,
    url_column: str | None,
    source_name: str,
) -> pd.DataFrame:
    required = {date_column, ticker_column}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"sentiment CSV is missing columns: {sorted(missing)}")

    if not text_column and not title_column and not summary_column and not sentiment_column:
        raise ValueError("sentiment CSV needs a text/title/summary column or a sentiment score column")

    df = raw.copy()
    dates = _parse_required_dates(df[date_column], column_name=date_column)
    if published_at_column and published_at_column in df.columns:
        published = _parse_required_dates(df[published_at_column], column_name=published_at_column)
    else:
        published = dates

    title = _column_or_blank(df, title_column)
    summary = _column_or_blank(df, summary_column)
    text = _column_or_blank(df, text_column)
    combined_text = text.mask(text.eq(""), (title + ". " + summary).str.strip(". "))

    normalized = pd.DataFrame(
        {
            "date": dates.dt.date.astype(str),
            "published_at": published.dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "ticker": _clean_ticker(df[ticker_column]),
            "title": title,
            "summary": summary,
            "publisher": _column_or_blank(df, publisher_column),
            "url": _column_or_blank(df, url_column),
            "text": combined_text.fillna("").astype(str).str.strip(),
            "source": source_name,
        }
    )

    if sentiment_column and sentiment_column in df.columns:
        normalized["sentiment_score"] = pd.to_numeric(df[sentiment_column], errors="coerce")
    else:
        normalized["sentiment_score"] = pd.NA

    no_text = normalized["text"].eq("")
    no_score = normalized["sentiment_score"].isna()
    if (no_text & no_score).any():
        bad = normalized.loc[no_text & no_score, ["date", "ticker"]].head(5).to_dict("records")
        raise ValueError(f"sentiment CSV rows need text or a numeric sentiment score: {bad}")

    normalized = normalized.drop_duplicates(["ticker", "published_at", "title", "text"])
    return normalized.sort_values(["date", "ticker", "published_at"]).reset_index(drop=True)


def build_daily_sentiment(news: pd.DataFrame, *, score_source: str) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    required = {"date", "ticker", "text"}
    missing = required - set(news.columns)
    if missing:
        raise ValueError(f"normalized sentiment table is missing columns: {sorted(missing)}")

    daily_input = news.copy()
    if "sentiment_score" in daily_input.columns and daily_input["sentiment_score"].notna().any():
        daily_input["sentiment_score"] = daily_input["sentiment_score"].fillna(
            daily_input["text"].map(_hash_sentiment)
        )
    else:
        daily_input["sentiment_score"] = daily_input["text"].map(_hash_sentiment)

    daily = (
        daily_input.groupby(["date", "ticker"], as_index=False)
        .agg(sentiment_score=("sentiment_score", "mean"), news_count=("text", "size"))
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    daily["sentiment_source"] = score_source
    return daily[DAILY_COLUMNS]


def _write_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dirs(path.parent)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def write_sentiment_csv_manifest(
    manifest_path: Path,
    news: pd.DataFrame,
    raw_file_path: Path,
    *,
    source_name: str,
    source_label: str,
) -> None:
    ensure_dirs(manifest_path.parent)
    if news.empty:
        date_range = "empty"
        symbols = ""
        missing_value_rate = 0.0
    else:
        parsed_dates = pd.to_datetime(news["date"])
        date_range = f"{parsed_dates.min().date().isoformat()} to {parsed_dates.max().date().isoformat()}"
        symbols = ";".join(sorted(news["ticker"].dropna().astype(str).unique()))
        missing_value_rate = float(news[OUTPUT_COLUMNS].isna().mean().mean())

    row: dict[str, Any] = {
        "source": source_name,
        "access_method": "User-provided historical news/sentiment CSV import",
        "frequency": "event",
        "date_range": date_range,
        "symbols": symbols,
        "columns": ";".join(OUTPUT_COLUMNS),
        "license_note": f"{source_label}; verify source terms, redistribution rights, and sentiment-score methodology before publication.",
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--daily-output", default=None)
    parser.add_argument("--manifest", default="data/data_manifest.csv")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--published-at-column", default="published_at")
    parser.add_argument("--ticker-column", default="ticker")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--summary-column", default="summary")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--sentiment-column", default="sentiment_score")
    parser.add_argument("--publisher-column", default="publisher")
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--source-name", default="user_historical_sentiment_csv")
    parser.add_argument("--manifest-source-name", default="historical_sentiment_csv_import")
    args = parser.parse_args()

    config = load_config(args.config)
    sentiment_config = config.get("sentiment", {})
    output = Path(args.output or sentiment_config.get("raw_news_path", "data/raw/news_historical.csv"))

    raw = pd.read_csv(args.input)
    news = normalize_sentiment_csv(
        raw,
        date_column=args.date_column,
        published_at_column=args.published_at_column,
        ticker_column=args.ticker_column,
        title_column=args.title_column,
        summary_column=args.summary_column,
        text_column=args.text_column,
        sentiment_column=args.sentiment_column,
        publisher_column=args.publisher_column,
        url_column=args.url_column,
        source_name=args.source_name,
    )
    _write_table(news[OUTPUT_COLUMNS], output)
    write_sentiment_csv_manifest(
        Path(args.manifest),
        news,
        output,
        source_name=args.manifest_source_name,
        source_label=args.source_name,
    )

    daily_message = ""
    if args.daily_output:
        daily = build_daily_sentiment(news, score_source=args.source_name)
        daily_output = Path(args.daily_output)
        _write_table(daily, daily_output)
        daily_message = f" daily_output={daily_output} daily_rows={len(daily)}"

    print(
        f"wrote {output} rows={len(news)} tickers={news['ticker'].nunique()} "
        f"date_range={news['date'].min()} to {news['date'].max()}{daily_message}"
    )


if __name__ == "__main__":
    main()
