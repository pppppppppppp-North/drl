from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.envs.trading_env import load_feature_table
from src.features.build_features import write_features
from src.utils.config import load_config


def build_index_context(index_prices: pd.DataFrame, index_name: str = "set") -> pd.DataFrame:
    df = index_prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "index_name" in df.columns:
        df = df[df["index_name"].astype(str).eq(index_name)].copy()
    if df.empty:
        raise ValueError(f"no index rows available for {index_name!r}")

    df = df.sort_values("date")
    close = pd.to_numeric(df["close"], errors="coerce")
    context = pd.DataFrame({"date": df["date"]})
    context[f"{index_name}_return_1d"] = close.pct_change().fillna(0.0)
    context[f"{index_name}_return_5d"] = close.pct_change(5).fillna(0.0)
    context[f"{index_name}_return_20d"] = close.pct_change(20).fillna(0.0)
    context[f"{index_name}_volatility_20d"] = context[f"{index_name}_return_1d"].rolling(20, min_periods=2).std().fillna(0.0)
    context[f"{index_name}_ma_ratio_20"] = close / close.rolling(20, min_periods=1).mean() - 1.0
    return context.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)


def merge_market_context(
    features: pd.DataFrame,
    index_prices: pd.DataFrame,
    index_name: str = "set",
) -> pd.DataFrame:
    merged = features.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    context = build_index_context(index_prices, index_name)
    merged = merged.merge(context, on="date", how="left")
    context_columns = [column for column in context.columns if column != "date"]
    merged[context_columns] = merged[context_columns].fillna(0.0)
    primary_return = f"{index_name}_return_1d"
    if primary_return in merged.columns:
        merged["market_return_1d"] = merged[primary_return]
    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_market.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    feature_path = Path(args.features or config["paths"].get("base_features", config["paths"]["features"]))
    fallback_csv = config["paths"].get("fallback_base_features_csv", config["paths"].get("fallback_features_csv"))
    output_path = Path(args.output or config["paths"]["features"])
    output_csv = Path(config["paths"].get("fallback_features_csv", output_path.with_suffix(".csv")))
    index_config = config.get("market_indices", {})
    index_path = Path(args.indices or index_config.get("raw_prices", "data/raw/prices_market_indices.csv"))
    index_name = str(index_config.get("primary", "set"))

    features = load_feature_table(feature_path, fallback_csv)
    indices = pd.read_csv(index_path)
    merged = merge_market_context(features, indices, index_name)
    written = write_features(merged, output_path, output_csv)
    print(f"wrote {written} rows={len(merged)} columns={len(merged.columns)}")


if __name__ == "__main__":
    main()
