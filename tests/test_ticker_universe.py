from __future__ import annotations

import pandas as pd
import pytest

from src.data.ticker_universe import load_ticker_universe_file, resolve_tickers_from_config


def test_load_ticker_universe_csv_deduplicates_and_normalizes(tmp_path) -> None:
    path = tmp_path / "universe.csv"
    pd.DataFrame({"ticker": ["ptt.bk", "AOT.BK", "PTT.BK"]}).to_csv(path, index=False)

    assert load_ticker_universe_file(path) == ["PTT.BK", "AOT.BK"]


def test_load_ticker_universe_can_append_suffix(tmp_path) -> None:
    path = tmp_path / "set50.csv"
    pd.DataFrame({"symbol": ["ptt", "AOT.BK", "cpall"]}).to_csv(path, index=False)

    assert load_ticker_universe_file(path, column="symbol", suffix=".BK") == ["PTT.BK", "AOT.BK", "CPALL.BK"]


def test_resolve_tickers_prefers_universe_file(tmp_path) -> None:
    path = tmp_path / "universe.csv"
    pd.DataFrame({"ticker": ["KBANK.BK"]}).to_csv(path, index=False)
    config = {
        "data": {
            "tickers": ["PTT.BK"],
            "ticker_universe": {"path": str(path), "column": "ticker"},
        }
    }

    assert resolve_tickers_from_config(config) == ["KBANK.BK"]


def test_load_ticker_universe_rejects_missing_column(tmp_path) -> None:
    path = tmp_path / "universe.csv"
    pd.DataFrame({"symbol": ["PTT.BK"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing column"):
        load_ticker_universe_file(path, column="ticker")
