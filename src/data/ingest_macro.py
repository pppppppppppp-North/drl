from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.ingest_ohlcv import _normalize_provider_frame, _yahoo_provider, write_manifest_row
from src.utils.config import ensure_dirs, load_config


def download_macro_series(series: dict[str, str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for name, symbol in series.items():
        frame = _normalize_provider_frame(_yahoo_provider(symbol, start, end), symbol)
        frame["macro_name"] = name
        frames.append(frame)
    if not frames:
        raise ValueError("at least one macro symbol is required")
    return pd.concat(frames, ignore_index=True).sort_values(["date", "macro_name"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_macro.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default="data/data_manifest.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    macro_config = config.get("macro_context", {})
    series = dict(macro_config.get("symbols", {}))
    output = Path(args.output or macro_config.get("raw_prices", "data/raw/prices_macro_yahoo.csv"))
    ensure_dirs(output.parent)

    prices = download_macro_series(
        series,
        start=str(config["data"]["start"]),
        end=str(config["data"]["end"]),
    )
    prices.to_csv(output, index=False)

    manifest_prices = prices.rename(columns={"macro_name": "source_macro"}).copy()
    manifest_prices["ticker"] = manifest_prices["source_macro"] + ":" + manifest_prices["ticker"]
    write_manifest_row(
        Path(args.manifest),
        source_name="yahoo_macro_proxies",
        access_method="Yahoo Finance via yfinance",
        prices=manifest_prices,
        raw_file_path=output,
        license_note="Yahoo Finance macro proxy data access through yfinance; verify usage rights before publication.",
    )
    print(
        f"wrote {output} rows={len(prices)} "
        f"series={';'.join(sorted(prices['macro_name'].dropna().unique()))}"
    )


if __name__ == "__main__":
    main()
