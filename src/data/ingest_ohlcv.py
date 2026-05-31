from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.utils.config import ensure_dirs, load_config


Provider = Callable[[str, str, str], pd.DataFrame]
PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only when optional dependency is absent.
        raise ImportError("install yfinance to download Yahoo Finance OHLCV data") from exc
    return yf


def _yahoo_provider(ticker: str, start: str, end: str) -> pd.DataFrame:
    yf = _import_yfinance()
    return yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [str(next(part for part in column if part)) for column in frame.columns]
    return frame


def _normalize_provider_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"provider returned no rows for {ticker}")

    normalized = _flatten_columns(frame).reset_index()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]
    if "datetime" in normalized.columns and "date" not in normalized.columns:
        normalized = normalized.rename(columns={"datetime": "date"})
    if "adj_close" in normalized.columns and "close" not in normalized.columns:
        normalized = normalized.rename(columns={"adj_close": "close"})

    missing = set(["date", "open", "high", "low", "close", "volume"]) - set(normalized.columns)
    if missing:
        raise ValueError(f"{ticker} data missing required OHLCV columns: {sorted(missing)}")

    normalized = normalized[["date", "open", "high", "low", "close", "volume"]].copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.date
    normalized["ticker"] = ticker
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    if normalized.empty:
        raise ValueError(f"{ticker} data has no valid OHLCV rows after normalization")
    return normalized[PRICE_COLUMNS]


def add_placeholder_context_columns(prices: pd.DataFrame) -> pd.DataFrame:
    enriched = prices.copy()
    enriched["date"] = pd.to_datetime(enriched["date"])
    close = enriched.pivot(index="date", columns="ticker", values="close").sort_index()
    market_return = close.pct_change().mean(axis=1).fillna(0.0)
    enriched["market_return_1d"] = enriched["date"].map(market_return).astype(float)
    enriched["macro_rate_change"] = 0.0
    enriched["sentiment_score"] = 0.0
    enriched["date"] = enriched["date"].dt.date
    return enriched


def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    provider: Provider | None = None,
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("at least one ticker is required")
    provider = provider or _yahoo_provider
    frames = [_normalize_provider_frame(provider(ticker, start, end), ticker) for ticker in tickers]
    prices = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    duplicate_count = int(prices.duplicated(["date", "ticker"]).sum())
    if duplicate_count:
        raise ValueError(f"duplicate date/ticker rows after download: {duplicate_count}")
    return add_placeholder_context_columns(prices)


def write_manifest_row(
    manifest_path: Path,
    *,
    source_name: str,
    access_method: str,
    prices: pd.DataFrame,
    raw_file_path: Path,
    license_note: str,
) -> None:
    ensure_dirs(manifest_path.parent)
    date_range = (
        f"{pd.to_datetime(prices['date']).min().date().isoformat()} to "
        f"{pd.to_datetime(prices['date']).max().date().isoformat()}"
    )
    row = {
        "source": source_name,
        "access_method": access_method,
        "frequency": "daily",
        "date_range": date_range,
        "symbols": ";".join(sorted(prices["ticker"].dropna().astype(str).unique())),
        "columns": ";".join(prices.columns),
        "license_note": license_note,
        "raw_file_path": str(raw_file_path),
        "missing_value_rate": float(prices.isna().mean().mean()),
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
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default="data/data_manifest.csv")
    parser.add_argument("--source-name", default="yahoo_thai_ohlcv")
    args = parser.parse_args()

    config = load_config(args.config)
    output = Path(args.output or config["paths"]["raw_prices"])
    ensure_dirs(output.parent)
    prices = download_ohlcv(
        tickers=list(config["data"]["tickers"]),
        start=str(config["data"]["start"]),
        end=str(config["data"]["end"]),
    )
    prices.to_csv(output, index=False)
    write_manifest_row(
        Path(args.manifest),
        source_name=args.source_name,
        access_method="Yahoo Finance via yfinance",
        prices=prices,
        raw_file_path=output,
        license_note="Yahoo Finance data access through yfinance; verify usage rights before publication.",
    )
    print(f"wrote {output} rows={len(prices)} tickers={prices['ticker'].nunique()}")


if __name__ == "__main__":
    main()
