from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.envs.trading_env import load_feature_table
from src.features.build_features import write_features
from src.utils.config import load_config


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_")


def load_sector_mapping(path: str | Path) -> pd.DataFrame:
    mapping = pd.read_csv(path)
    required = {"ticker", "sector"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"sector mapping missing columns: {sorted(missing)}")
    mapping = mapping.copy()
    mapping["ticker"] = mapping["ticker"].astype(str).str.strip()
    mapping["sector"] = mapping["sector"].astype(str).str.strip()
    if mapping["ticker"].duplicated().any():
        duplicates = sorted(mapping.loc[mapping["ticker"].duplicated(), "ticker"].unique())
        raise ValueError(f"duplicate ticker sector mappings: {duplicates}")
    return mapping


def add_sector_context(
    features: pd.DataFrame,
    sector_mapping: pd.DataFrame,
    *,
    return_column: str = "return_1d",
) -> pd.DataFrame:
    required_features = {"date", "ticker", return_column}
    missing = required_features - set(features.columns)
    if missing:
        raise ValueError(f"feature table missing columns: {sorted(missing)}")

    mapping_columns = [col for col in ["ticker", "sector", "industry_group", "source"] if col in sector_mapping.columns]
    merged = features.merge(sector_mapping[mapping_columns], on="ticker", how="left", validate="many_to_one")
    missing_tickers = sorted(merged.loc[merged["sector"].isna(), "ticker"].dropna().unique())
    if missing_tickers:
        raise ValueError(f"missing sector mapping for tickers: {missing_tickers}")

    merged["date"] = pd.to_datetime(merged["date"])
    merged["sector_slug"] = merged["sector"].map(_slug)

    sector_returns = (
        merged.groupby(["date", "sector"], as_index=False)
        .agg(
            sector_equal_weight_return_1d=(return_column, "mean"),
            sector_peer_count=("ticker", "nunique"),
        )
    )
    merged = merged.merge(sector_returns, on=["date", "sector"], how="left", validate="many_to_one")
    merged["sector_relative_return_1d"] = merged[return_column] - merged["sector_equal_weight_return_1d"]

    for slug in sorted(merged["sector_slug"].unique()):
        merged[f"sector_{slug}"] = (merged["sector_slug"] == slug).astype(float)

    numeric_columns = ["sector_equal_weight_return_1d", "sector_relative_return_1d", "sector_peer_count"]
    sector_flag_columns = [
        col
        for col in merged.columns
        if col.startswith("sector_") and col not in {"sector_slug", *numeric_columns}
    ]
    merged[numeric_columns + sector_flag_columns] = merged[numeric_columns + sector_flag_columns].fillna(0.0)
    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_sector.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    path_config = config["paths"]
    sector_config = config.get("sector_context", {})
    input_path = args.input or path_config.get("base_features") or path_config["features"]
    fallback_input = path_config.get("fallback_base_features_csv")
    mapping_path = args.mapping or sector_config.get("mapping_path", "data/reference/sector_mapping_thai_pilot.csv")
    output_path = Path(args.output or path_config["features"])
    fallback_output = Path(path_config.get("fallback_features_csv", output_path.with_suffix(".csv")))

    features = load_feature_table(input_path, fallback_input)
    mapping = load_sector_mapping(mapping_path)
    sector_features = add_sector_context(features, mapping)
    written = write_features(sector_features, output_path, fallback_output)
    print(f"wrote {written} rows={len(sector_features)} columns={len(sector_features.columns)}")


if __name__ == "__main__":
    main()
