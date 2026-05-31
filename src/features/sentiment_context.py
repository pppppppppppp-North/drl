from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.envs.trading_env import load_feature_table
from src.features.build_features import write_features
from src.utils.config import load_config


SENTIMENT_COLUMNS = ["sentiment_score", "news_count", "sentiment_news_age_days", "sentiment_release_date"]


def _empty_sentiment_features(features: pd.DataFrame) -> pd.DataFrame:
    merged = features.copy()
    if "sentiment_score" in merged.columns:
        merged = merged.drop(columns=["sentiment_score"])
    merged["sentiment_score"] = 0.0
    merged["news_count"] = 0.0
    merged["sentiment_news_age_days"] = 0.0
    merged["sentiment_release_date"] = pd.NaT
    return merged


def build_sentiment_context(daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    if daily_sentiment.empty:
        return pd.DataFrame(columns=["date", "ticker", "sentiment_score", "news_count"])
    missing = {"date", "ticker", "sentiment_score"} - set(daily_sentiment.columns)
    if missing:
        raise ValueError(f"sentiment table is missing columns: {sorted(missing)}")

    context = daily_sentiment.copy()
    context["date"] = pd.to_datetime(context["date"])
    context["sentiment_score"] = pd.to_numeric(context["sentiment_score"], errors="coerce").fillna(0.0)
    if "news_count" not in context.columns:
        context["news_count"] = 1.0
    context["news_count"] = pd.to_numeric(context["news_count"], errors="coerce").fillna(0.0)
    return (
        context.groupby(["date", "ticker"], as_index=False)
        .agg(sentiment_score=("sentiment_score", "mean"), news_count=("news_count", "sum"))
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def merge_sentiment_context(
    features: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    *,
    max_age_days: int = 7,
) -> pd.DataFrame:
    merged = features.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    if daily_sentiment.empty:
        return _empty_sentiment_features(merged).sort_values(["date", "ticker"]).reset_index(drop=True)

    context = build_sentiment_context(daily_sentiment)
    if "sentiment_score" in merged.columns:
        merged = merged.drop(columns=["sentiment_score"])
    for column in ["news_count", "sentiment_news_age_days", "sentiment_release_date"]:
        if column in merged.columns:
            merged = merged.drop(columns=[column])

    frames = []
    tolerance = pd.Timedelta(days=max_age_days)
    for ticker, ticker_features in merged.groupby("ticker", sort=True):
        left = ticker_features.sort_values("date").copy()
        right = context[context["ticker"].eq(ticker)].sort_values("date").copy()
        right = right.rename(columns={"date": "sentiment_release_date"})
        right = right[["sentiment_release_date", "sentiment_score", "news_count"]]
        if right.empty:
            left["sentiment_score"] = 0.0
            left["news_count"] = 0.0
            left["sentiment_news_age_days"] = 0.0
            left["sentiment_release_date"] = pd.NaT
        else:
            left = pd.merge_asof(
                left,
                right,
                left_on="date",
                right_on="sentiment_release_date",
                direction="backward",
                tolerance=tolerance,
            )
            age = left["date"] - left["sentiment_release_date"]
            left["sentiment_news_age_days"] = age.dt.days.fillna(0.0).astype(float)
            left["sentiment_score"] = left["sentiment_score"].fillna(0.0)
            left["news_count"] = left["news_count"].fillna(0.0)
        frames.append(left)

    result = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    return result.replace([float("inf"), float("-inf")], 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_sentiment.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--sentiment", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    path_config = config["paths"]
    sentiment_config = config.get("sentiment", {})
    feature_path = Path(args.features or path_config.get("base_features", path_config["features"]))
    fallback_csv = path_config.get("fallback_base_features_csv", path_config.get("fallback_features_csv"))
    output_path = Path(args.output or path_config["features"])
    output_csv = Path(path_config.get("fallback_features_csv", output_path.with_suffix(".csv")))
    sentiment_path = Path(args.sentiment or sentiment_config.get("daily_path", "data/processed/sentiment_daily.parquet"))

    features = load_feature_table(feature_path, fallback_csv)
    if sentiment_path.suffix == ".parquet":
        daily_sentiment = pd.read_parquet(sentiment_path)
    else:
        daily_sentiment = pd.read_csv(sentiment_path)
    merged = merge_sentiment_context(
        features,
        daily_sentiment,
        max_age_days=int(sentiment_config.get("max_age_days", 7)),
    )
    written = write_features(merged, output_path, output_csv)
    print(f"wrote {written} rows={len(merged)} columns={len(merged.columns)}")


if __name__ == "__main__":
    main()
