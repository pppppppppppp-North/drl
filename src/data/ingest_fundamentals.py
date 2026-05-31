from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.ingest_ohlcv import _import_yfinance
from src.data.ticker_universe import resolve_tickers_from_config
from src.utils.config import ensure_dirs, load_config


STATEMENT_ATTRS = {
    "income_annual": "income_stmt",
    "balance_annual": "balance_sheet",
    "cashflow_annual": "cashflow",
    "income_quarterly": "quarterly_financials",
    "balance_quarterly": "quarterly_balance_sheet",
    "cashflow_quarterly": "quarterly_cashflow",
}


def _statement_to_long(ticker: str, statement_type: str, statement: pd.DataFrame) -> pd.DataFrame:
    if statement.empty:
        return pd.DataFrame(columns=["ticker", "statement_type", "metric", "period_end", "value"])
    long = statement.copy().rename_axis(index="metric", columns="period_end").stack().reset_index(name="value")
    long["ticker"] = ticker
    long["statement_type"] = statement_type
    long["period_end"] = pd.to_datetime(long["period_end"]).dt.date
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["period_end", "metric", "value"])
    return long[["ticker", "statement_type", "metric", "period_end", "value"]]


def download_fundamentals(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        raise ValueError("at least one ticker is required")
    yf = _import_yfinance()
    frames = []
    for ticker in tickers:
        yf_ticker = yf.Ticker(ticker)
        for statement_type, attr in STATEMENT_ATTRS.items():
            statement = getattr(yf_ticker, attr)
            frames.append(_statement_to_long(ticker, statement_type, statement))
    fundamentals = pd.concat(frames, ignore_index=True)
    if fundamentals.empty:
        raise ValueError("provider returned no fundamentals rows")
    return fundamentals.sort_values(["ticker", "period_end", "statement_type", "metric"]).reset_index(drop=True)


def write_fundamentals_manifest_row(
    manifest_path: Path,
    *,
    fundamentals: pd.DataFrame,
    raw_file_path: Path,
) -> None:
    ensure_dirs(manifest_path.parent)
    date_range = (
        f"{pd.to_datetime(fundamentals['period_end']).min().date().isoformat()} to "
        f"{pd.to_datetime(fundamentals['period_end']).max().date().isoformat()}"
    )
    row = {
        "source": "yahoo_financial_statements",
        "access_method": "Yahoo Finance via yfinance",
        "frequency": "annual_and_quarterly",
        "date_range": date_range,
        "symbols": ";".join(sorted(fundamentals["ticker"].dropna().astype(str).unique())),
        "columns": ";".join(fundamentals.columns),
        "license_note": "Yahoo Finance fundamentals data access through yfinance; verify usage rights before publication.",
        "raw_file_path": str(raw_file_path),
        "missing_value_rate": float(fundamentals.isna().mean().mean()),
    }
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["source"] != row["source"]]
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    else:
        manifest = pd.DataFrame([row])
    manifest.to_csv(manifest_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_fundamentals.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default="data/data_manifest.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    output = Path(args.output or config.get("fundamentals", {}).get("raw_path", "data/raw/fundamentals_yahoo_quarterly.csv"))
    ensure_dirs(output.parent)
    fundamentals = download_fundamentals(resolve_tickers_from_config(config))
    fundamentals.to_csv(output, index=False)
    write_fundamentals_manifest_row(Path(args.manifest), fundamentals=fundamentals, raw_file_path=output)
    print(
        f"wrote {output} rows={len(fundamentals)} "
        f"tickers={fundamentals['ticker'].nunique()} "
        f"periods={fundamentals['period_end'].nunique()}"
    )


if __name__ == "__main__":
    main()
