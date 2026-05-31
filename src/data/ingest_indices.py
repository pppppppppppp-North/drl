from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.ingest_ohlcv import _normalize_provider_frame, _yahoo_provider, write_manifest_row
from src.utils.config import ensure_dirs, load_config


def download_indices(indices: dict[str, str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for name, symbol in indices.items():
        frame = _normalize_provider_frame(_yahoo_provider(symbol, start, end), symbol)
        frame["index_name"] = name
        frames.append(frame)
    if not frames:
        raise ValueError("at least one index symbol is required")
    return pd.concat(frames, ignore_index=True).sort_values(["date", "index_name"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_market.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default="data/data_manifest.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    index_config = config.get("market_indices", {})
    indices = dict(index_config.get("symbols", {}))
    output = Path(args.output or index_config.get("raw_prices", "data/raw/prices_market_indices.csv"))
    ensure_dirs(output.parent)
    prices = download_indices(
        indices,
        start=str(config["data"]["start"]),
        end=str(config["data"]["end"]),
    )
    prices.to_csv(output, index=False)
    manifest_prices = prices.rename(columns={"index_name": "source_index"}).copy()
    manifest_prices["ticker"] = manifest_prices["source_index"] + ":" + manifest_prices["ticker"]
    write_manifest_row(
        Path(args.manifest),
        source_name="yahoo_thai_market_indices",
        access_method="Yahoo Finance via yfinance",
        prices=manifest_prices,
        raw_file_path=output,
        license_note="Yahoo Finance index data access through yfinance; verify usage rights before publication.",
    )
    print(
        f"wrote {output} rows={len(prices)} "
        f"indices={';'.join(sorted(prices['index_name'].dropna().unique()))}"
    )


if __name__ == "__main__":
    main()
