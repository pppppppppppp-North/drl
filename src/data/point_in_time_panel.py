"""Point-in-time SET50/SET100 session panel, 2012-2024 (milestone M3).

Joins three inputs:

* the official membership archive (``data/reference/set50_set100_membership_2012_2024.csv``);
* the per-list price map, which names the Yahoo series that prices each member in each list
  (``member_price_map_2012_2024.csv``; renamed members use their current symbol's series);
* the Yahoo price files.

From them it builds:

* a session calendar: dates on which most quoted series traded;
* a session membership mask: one row per session and index member, with the price series to use;
* a feature panel: technical features per price series on its own past prices, the SET50 ETF
  market context, the number of sessions of price history so far, and the trailing traded value
  used to order the environment's slots (as of the previous session).

A list applies from its effective date and never before its publication date, the archive's
as-of rule. Every feature at a session uses only prices up to that session, and the trading
environment adds one more session of lag before a decision.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.market_context import build_index_context
from src.features.technical import add_technical_features

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
STOCK_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "ma_ratio_10",
    "ma_ratio_30",
    "rsi_14",
    "macd",
    "bollinger_z",
    "atr_14",
    "volume_ratio_20",
]
COVERED = ("own_symbol", "current_symbol")


def load_prices(paths: Sequence[Path]) -> pd.DataFrame:
    """Concatenate price files; a series may come from only one file."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=PRICE_COLUMNS, parse_dates=["date"])
        frame["source_file"] = Path(path).name
        frames.append(frame)
    prices = pd.concat(frames, ignore_index=True)
    owners = prices.groupby("ticker")["source_file"].nunique()
    shared = sorted(owners[owners > 1].index)
    if shared:
        raise ValueError(f"series appear in more than one price file: {shared}")
    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("price files contain duplicate date/ticker rows")
    return prices.drop(columns="source_file").sort_values(["date", "ticker"]).reset_index(drop=True)


def session_calendar(prices: pd.DataFrame, *, min_active_share: float = 0.5) -> pd.DatetimeIndex:
    """Dates on which at least ``min_active_share`` of the series quoted that day traded.

    Yahoo sometimes carries Thai holidays as zero-volume rows with repeated prices; those dates
    are not sessions.
    """
    active = (prices["volume"] > 0).groupby(prices["date"]).mean()
    return pd.DatetimeIndex(sorted(active[active >= min_active_share].index), name="date")


def session_membership_mask(
    membership: pd.DataFrame,
    price_map: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    """One row per session and SET100 member: its list, SET50 flag and price series."""
    intervals = membership[
        ["record_id", "effective_from", "effective_to_exclusive", "published_at"]
    ].drop_duplicates()
    session_frame = pd.DataFrame({"date": sessions})
    pieces = []
    for interval in intervals.itertuples(index=False):
        start = max(pd.Timestamp(interval.effective_from), pd.Timestamp(interval.published_at))
        end = pd.Timestamp(interval.effective_to_exclusive)
        dates = session_frame[(session_frame["date"] >= start) & (session_frame["date"] < end)]
        if dates.empty:
            continue
        members = membership.loc[
            membership["record_id"] == interval.record_id, ["record_id", "ticker", "in_set50"]
        ]
        pieces.append(dates.merge(members, how="cross"))
    mask = pd.concat(pieces, ignore_index=True)
    if mask.duplicated(["date", "ticker"]).any():
        raise ValueError("a session falls inside two membership lists")
    covered_dates = set(mask["date"])
    missing = [d for d in sessions if d not in covered_dates]
    if missing:
        raise ValueError(f"{len(missing)} sessions have no membership list, first {missing[0].date()}")
    mapped = price_map[["record_id", "ticker", "price_ticker", "coverage"]].fillna({"price_ticker": ""})
    mask = mask.merge(mapped, on=["record_id", "ticker"], how="left", validate="many_to_one")
    if mask["coverage"].isna().any():
        raise ValueError("the price map does not cover every list member")
    mask["in_set50"] = mask["in_set50"].astype(str).eq("True")
    mask["price_ticker"] = mask["price_ticker"].where(mask["coverage"].isin(COVERED), "")
    counts = mask.groupby("date").agg(members=("ticker", "size"), set50=("in_set50", "sum"))
    if not (counts["members"].eq(100).all() and counts["set50"].eq(50).all()):
        raise ValueError("every session must have 100 SET100 and 50 SET50 members")
    return mask.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_feature_panel(
    prices: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    series: Sequence[str],
    index_prices: pd.DataFrame,
    *,
    index_name: str = "set50_etf",
    rank_window: int = 60,
) -> pd.DataFrame:
    """Technical features per price series (session rows only) plus the market context."""
    wanted = set(series)
    rows = prices[prices["ticker"].isin(wanted) & prices["date"].isin(sessions)]
    rows = rows[(rows["close"] > 0) & np.isfinite(rows["close"])]
    features = add_technical_features(rows[PRICE_COLUMNS])
    features = features.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = features.groupby("ticker", sort=False)
    features["history_sessions"] = grouped.cumcount() + 1
    traded_value = features["close"] * features["volume"].clip(lower=0)
    # Ordering key for the environment's slots: mean traded value up to the previous session.
    features["rank_value"] = (
        traded_value.groupby(features["ticker"], sort=False)
        .transform(lambda s: s.rolling(rank_window, min_periods=1).mean().shift(1))
        .fillna(0.0)
    )
    index = index_prices.copy()
    index["date"] = pd.to_datetime(index["date"])
    index = index.assign(index_name=index_name)
    context = build_index_context(index, index_name, target_dates=pd.Series(sessions))
    benchmark = index.drop_duplicates("date").set_index("date")["close"].reindex(sessions).ffill()
    context["benchmark_close"] = benchmark.to_numpy()
    panel = features.merge(context, on="date", how="left")
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def market_feature_columns(index_name: str = "set50_etf") -> list[str]:
    return [
        f"{index_name}_return_1d",
        f"{index_name}_return_5d",
        f"{index_name}_return_20d",
        f"{index_name}_volatility_20d",
        f"{index_name}_ma_ratio_20",
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_point_in_time_inputs(
    *,
    root: Path,
    membership_path: str,
    price_map_path: str,
    price_paths: Sequence[str],
    index_path: str,
    index_name: str = "set50_etf",
    end: str = "2024-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Session mask and feature panel for every series that prices a member, with input hashes."""
    root = Path(root)
    membership = pd.read_csv(root / membership_path, dtype=str, keep_default_na=False)
    price_map = pd.read_csv(root / price_map_path, dtype=str, keep_default_na=False)
    prices = load_prices([root / path for path in price_paths])
    prices = prices[prices["date"] <= pd.Timestamp(end)]
    sessions = session_calendar(prices)
    mask = session_membership_mask(membership, price_map, sessions)
    series = sorted(set(mask["price_ticker"]) - {""})
    index_prices = pd.read_csv(root / index_path, parse_dates=["date"])
    panel = build_feature_panel(prices, sessions, series, index_prices, index_name=index_name)
    inputs = [membership_path, price_map_path, *price_paths, index_path]
    receipt = {
        "inputs": {path: _sha256(root / path) for path in inputs},
        "sessions": [str(sessions.min().date()), str(sessions.max().date()), int(len(sessions))],
        "series": len(series),
        "mask_rows": int(len(mask)),
        "set50_member_sessions_with_prices": float(
            (mask.loc[mask["in_set50"], "price_ticker"] != "").mean()
        ),
        "set100_member_sessions_with_prices": float((mask["price_ticker"] != "").mean()),
    }
    return mask, panel, receipt


def write_point_in_time_inputs(mask: pd.DataFrame, panel: pd.DataFrame, receipt: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "set_membership_session_mask_2012_2024.csv.gz"
    panel_path = out_dir / "pit_feature_panel_2012_2024.csv.gz"
    mask.assign(date=mask["date"].dt.date).to_csv(mask_path, index=False)
    panel.assign(date=panel["date"].dt.date).to_csv(panel_path, index=False)
    receipt = {
        **receipt,
        "outputs": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in (mask_path, panel_path)
        },
    }
    (out_dir / "pit_inputs_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
