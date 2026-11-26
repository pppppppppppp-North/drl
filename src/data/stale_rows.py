"""Stale price rows: Yahoo's repeated quotes on days a series did not trade (milestone M6).

Yahoo's 2024 prices for the Thai market carry rows with zero volume whose open, high, low and close
all equal the previous close, on real trading days (``docs/final_test_results_20261125.md``). A stale
row says nothing about that day's price. The point-in-time session calendar drops a date when fewer
than half of the quoted series traded, and the five-stock pipeline drops a date when none did, so a
day on which most series are stale leaves the calendar even if the market was open.

:func:`stale_summary` counts them in a window (for the holdout, 2025 onward) so the holdout stage can
report them before any model is evaluated. It reports the data; it does not change it.
"""
from __future__ import annotations

import pandas as pd


def stale_mask(prices: pd.DataFrame) -> pd.Series:
    """True for rows with no volume whose open, high, low and close all equal the previous close."""

    frame = prices.sort_values(["ticker", "date"])
    previous = frame.groupby("ticker")["close"].shift(1)
    flat = frame["open"].eq(frame["close"]) & frame["high"].eq(frame["close"]) & frame["low"].eq(frame["close"])
    stale = frame["volume"].fillna(0).le(0) & flat & frame["close"].eq(previous)
    return stale.reindex(prices.index)


def stale_summary(prices: pd.DataFrame, *, since: str | pd.Timestamp | None = None,
                  min_active_share: float = 0.5) -> dict:
    """Stale rows from ``since`` on: totals, the series and dates affected, and the dates a calendar
    that needs ``min_active_share`` of series trading would drop."""

    frame = prices.assign(date=pd.to_datetime(prices["date"]))
    frame = frame.assign(stale=stale_mask(frame))
    if since is not None:
        frame = frame[frame["date"] >= pd.Timestamp(since)]
    if frame.empty:
        return {"rows": 0, "stale_rows": 0, "stale_share": 0.0, "series": 0, "series_with_stale_rows": 0,
                "dates": 0, "dates_with_stale_rows": 0, "weekdays_below_active_share": 0, "by_year": {}}
    traded = frame["volume"].fillna(0).gt(0).groupby(frame["date"]).mean()
    low = traded[traded < min_active_share]
    stale_by_series = frame.groupby("ticker")["stale"].sum()
    by_year = frame.groupby(frame["date"].dt.year).agg(rows=("stale", "size"), stale_rows=("stale", "sum"))
    return {
        "rows": int(len(frame)),
        "stale_rows": int(frame["stale"].sum()),
        "stale_share": float(frame["stale"].mean()),
        "series": int(frame["ticker"].nunique()),
        "series_with_stale_rows": int((stale_by_series > 0).sum()),
        "median_stale_rows_per_affected_series": float(stale_by_series[stale_by_series > 0].median())
        if (stale_by_series > 0).any() else 0.0,
        "dates": int(frame["date"].nunique()),
        "dates_with_stale_rows": int(frame.loc[frame["stale"], "date"].nunique()),
        "weekdays_below_active_share": int((pd.DatetimeIndex(low.index).dayofweek < 5).sum()),
        "by_year": {int(year): {"rows": int(row.rows), "stale_rows": int(row.stale_rows)}
                    for year, row in by_year.iterrows()},
    }
