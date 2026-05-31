from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _clean_ticker(value: object) -> str:
    ticker = str(value).strip().upper()
    if not ticker:
        raise ValueError("ticker universe contains a blank ticker")
    return ticker


def _dedupe_preserve_order(tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for ticker in tickers:
        if ticker in seen:
            continue
        seen.add(ticker)
        unique.append(ticker)
    return unique


def load_ticker_universe_file(
    path: str | Path,
    *,
    column: str = "ticker",
    suffix: str | None = None,
) -> list[str]:
    universe_path = Path(path)
    if not universe_path.exists():
        raise FileNotFoundError(f"ticker universe file does not exist: {universe_path}")

    if universe_path.suffix.lower() in {".txt", ".lst"}:
        raw_tickers = [line.strip() for line in universe_path.read_text(encoding="utf-8").splitlines()]
        tickers = [_clean_ticker(ticker) for ticker in raw_tickers if ticker and not ticker.startswith("#")]
    else:
        universe = pd.read_csv(universe_path)
        if column not in universe.columns:
            raise ValueError(f"ticker universe file is missing column: {column}")
        tickers = [_clean_ticker(value) for value in universe[column].dropna().tolist()]

    if suffix:
        suffix = suffix.upper()
        tickers = [ticker if ticker.endswith(suffix) else f"{ticker}{suffix}" for ticker in tickers]
    tickers = _dedupe_preserve_order(tickers)
    if not tickers:
        raise ValueError("ticker universe is empty")
    return tickers


def resolve_tickers_from_config(config: dict[str, Any]) -> list[str]:
    data_config = config.get("data", {})
    universe_config = data_config.get("ticker_universe")
    if universe_config:
        if isinstance(universe_config, (str, Path)):
            return load_ticker_universe_file(universe_config)
        return load_ticker_universe_file(
            universe_config["path"],
            column=str(universe_config.get("column", "ticker")),
            suffix=universe_config.get("suffix"),
        )

    tickers = [_clean_ticker(ticker) for ticker in data_config.get("tickers", [])]
    tickers = _dedupe_preserve_order(tickers)
    if not tickers:
        raise ValueError("config data section must provide tickers or ticker_universe.path")
    return tickers
