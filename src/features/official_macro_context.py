from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.envs.trading_env import load_feature_table
from src.features.build_features import write_features
from src.utils.config import load_config


DEFAULT_METRICS = {
    "leading_economic_index": "bot_leading_economic_index",
    "index_change": "bot_leading_index_change",
    "business_sentiment_index_3_months": "bot_bsi_3m",
    "export_volume_index_exclude_gold": "bot_export_volume_index_ex_gold",
    "broad_money_at_2000_prices_million_baht": "bot_real_broad_money",
}


def build_official_macro_context(
    macro: pd.DataFrame,
    *,
    metric_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    if macro.empty:
        raise ValueError("official macro table is empty")
    missing = {"period", "release_date", "metric", "value"} - set(macro.columns)
    if missing:
        raise ValueError(f"official macro table is missing columns: {sorted(missing)}")

    metric_map = metric_map or DEFAULT_METRICS
    df = macro.copy()
    df["period"] = pd.to_datetime(df["period"])
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["metric"].isin(metric_map)].dropna(subset=["value"]).copy()
    if df.empty:
        raise ValueError("official macro table contains none of the configured metrics")

    df["feature"] = df["metric"].map(metric_map)
    context = (
        df.pivot_table(index=["period", "release_date"], columns="feature", values="value", aggfunc="last")
        .reset_index()
        .sort_values("release_date")
    )
    context.columns.name = None
    context["bot_macro_release_date"] = context["release_date"]
    return context.replace([float("inf"), float("-inf")], 0.0).ffill().fillna(0.0)


def merge_official_macro_context(
    features: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    metric_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    merged = features.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    context = build_official_macro_context(macro, metric_map=metric_map)
    context = context.sort_values("release_date")

    feature_dates = pd.DataFrame({"date": sorted(merged["date"].drop_duplicates())})
    aligned = pd.merge_asof(
        feature_dates,
        context,
        left_on="date",
        right_on="release_date",
        direction="backward",
    )
    feature_columns = [
        column
        for column in aligned.columns
        if column not in {"date", "period", "release_date", "bot_macro_release_date"}
    ]
    aligned[feature_columns] = aligned[feature_columns].fillna(0.0)
    aligned["bot_macro_release_age_days"] = (aligned["date"] - aligned["release_date"]).dt.days.fillna(0.0)
    aligned = aligned.drop(columns=["period", "release_date"])

    for column in [*feature_columns, "bot_macro_release_age_days", "bot_macro_release_date"]:
        if column in merged.columns:
            merged = merged.drop(columns=[column])
    merged = merged.merge(aligned, on="date", how="left", validate="many_to_one")
    fill_columns = [*feature_columns, "bot_macro_release_age_days"]
    merged[fill_columns] = merged[fill_columns].fillna(0.0)
    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_official_macro.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--macro", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    path_config = config["paths"]
    macro_config = config.get("official_macro_context", {})
    feature_path = Path(args.features or path_config.get("base_features", path_config["features"]))
    fallback_csv = path_config.get("fallback_base_features_csv", path_config.get("fallback_features_csv"))
    output_path = Path(args.output or path_config["features"])
    output_csv = Path(path_config.get("fallback_features_csv", output_path.with_suffix(".csv")))
    macro_path = Path(args.macro or macro_config.get("raw_path", "data/raw/bot_official_macro.csv"))
    metric_map = dict(macro_config.get("metrics", DEFAULT_METRICS))

    features = load_feature_table(feature_path, fallback_csv)
    macro = pd.read_csv(macro_path)
    merged = merge_official_macro_context(features, macro, metric_map=metric_map)
    written = write_features(merged, output_path, output_csv)
    print(f"wrote {written} rows={len(merged)} columns={len(merged.columns)}")


if __name__ == "__main__":
    main()
