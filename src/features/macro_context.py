from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.envs.trading_env import load_feature_table
from src.features.build_features import write_features
from src.utils.config import load_config


def build_macro_context(macro_prices: pd.DataFrame) -> pd.DataFrame:
    df = macro_prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "macro_name" not in df.columns:
        raise ValueError("macro price table must include macro_name")
    if df.empty:
        raise ValueError("macro price table is empty")

    context_frames = []
    for name, part in df.groupby("macro_name", sort=True):
        safe_name = str(name).strip().lower().replace("-", "_")
        part = part.sort_values("date")
        close = pd.to_numeric(part["close"], errors="coerce")
        frame = pd.DataFrame({"date": part["date"]})
        frame[f"{safe_name}_level"] = close
        frame[f"{safe_name}_change_1d"] = close.diff().fillna(0.0)
        frame[f"{safe_name}_return_1d"] = close.pct_change().fillna(0.0)
        frame[f"{safe_name}_return_5d"] = close.pct_change(5).fillna(0.0)
        frame[f"{safe_name}_return_20d"] = close.pct_change(20).fillna(0.0)
        frame[f"{safe_name}_volatility_20d"] = frame[f"{safe_name}_return_1d"].rolling(20, min_periods=2).std().fillna(0.0)
        context_frames.append(frame)

    context = context_frames[0]
    for frame in context_frames[1:]:
        context = context.merge(frame, on="date", how="outer")
    return context.sort_values("date").replace([float("inf"), float("-inf")], 0.0).ffill().fillna(0.0)


def merge_macro_context(
    features: pd.DataFrame,
    macro_prices: pd.DataFrame,
    *,
    primary_rate_column: str = "us10y_change_1d",
) -> pd.DataFrame:
    merged = features.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    context = build_macro_context(macro_prices)

    feature_dates = pd.DataFrame({"date": sorted(merged["date"].drop_duplicates())})
    context = (
        context.set_index("date")
        .sort_index()
        .reindex(pd.to_datetime(feature_dates["date"]), method="ffill")
        .reset_index()
        .rename(columns={"index": "date"})
    )
    context_columns = [column for column in context.columns if column != "date"]
    context[context_columns] = context[context_columns].fillna(0.0)
    merged = merged.merge(context, on="date", how="left", validate="many_to_one")
    merged[context_columns] = merged[context_columns].fillna(0.0)
    if primary_rate_column in merged.columns:
        merged["macro_rate_change"] = merged[primary_rate_column]
    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_macro.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--macro", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    path_config = config["paths"]
    macro_config = config.get("macro_context", {})
    feature_path = Path(args.features or path_config.get("base_features", path_config["features"]))
    fallback_csv = path_config.get("fallback_base_features_csv", path_config.get("fallback_features_csv"))
    output_path = Path(args.output or path_config["features"])
    output_csv = Path(path_config.get("fallback_features_csv", output_path.with_suffix(".csv")))
    macro_path = Path(args.macro or macro_config.get("raw_prices", "data/raw/prices_macro_yahoo.csv"))
    primary_rate_column = str(macro_config.get("primary_rate_column", "us10y_change_1d"))

    features = load_feature_table(feature_path, fallback_csv)
    macro_prices = pd.read_csv(macro_path)
    merged = merge_macro_context(features, macro_prices, primary_rate_column=primary_rate_column)
    written = write_features(merged, output_path, output_csv)
    print(f"wrote {written} rows={len(merged)} columns={len(merged.columns)}")


if __name__ == "__main__":
    main()
