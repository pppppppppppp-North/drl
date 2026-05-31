from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.technical import add_technical_features
from src.utils.config import ensure_dirs, load_config


def write_features(features: pd.DataFrame, parquet_path: Path, csv_path: Path) -> Path:
    ensure_dirs(parquet_path.parent, csv_path.parent)
    try:
        features.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:
        features.to_csv(csv_path, index=False)
        print(f"parquet unavailable ({exc}); wrote CSV fallback {csv_path}")
        return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    raw_path = Path(args.input or config["paths"]["raw_prices"])
    parquet_path = Path(args.output or config["paths"]["features"])
    csv_path = Path(config["paths"].get("fallback_features_csv", parquet_path.with_suffix(".csv")))

    prices = pd.read_csv(raw_path)
    features = add_technical_features(prices)
    written = write_features(features, parquet_path, csv_path)
    print(f"wrote {written} rows={len(features)}")


if __name__ == "__main__":
    main()

