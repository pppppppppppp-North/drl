"""Stale price rows: zero volume with open, high, low and close equal to the previous close."""
from __future__ import annotations

import pandas as pd

from src.data.stale_rows import stale_mask, stale_summary


def _frame():
    dates = pd.bdate_range("2024-12-30", periods=6)
    rows = []
    for ticker, base in (("A", 10.0), ("B", 20.0), ("C", 30.0)):
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": ticker, "open": base + i, "high": base + i + 1, "low": base + i - 1,
                         "close": base + i, "volume": 100})
    frame = pd.DataFrame(rows)
    # A is stale on the 4th session: no volume, flat at the previous close.
    stale = (frame["ticker"] == "A") & (frame["date"] == dates[3])
    frame.loc[stale, ["open", "high", "low", "close", "volume"]] = [12.0, 12.0, 12.0, 12.0, 0]
    # On the 5th session every series is flat with no volume (a holiday, or a day most quotes went stale).
    holiday = frame["date"] == dates[4]
    previous = frame.loc[frame["date"] == dates[3]].set_index("ticker")["close"]
    for ticker in "ABC":
        row = holiday & (frame["ticker"] == ticker)
        frame.loc[row, ["open", "high", "low", "close", "volume"]] = [previous[ticker]] * 4 + [0]
    # B has no volume on the 6th session but moved: not stale.
    moved = (frame["ticker"] == "B") & (frame["date"] == dates[5])
    frame.loc[moved, "volume"] = 0
    return frame, dates


def test_stale_mask_needs_no_volume_a_flat_bar_and_the_previous_close():
    frame, dates = _frame()
    flagged = frame[stale_mask(frame)]
    assert set(zip(flagged["ticker"], flagged["date"])) == {("A", dates[3]), ("A", dates[4]), ("B", dates[4]),
                                                            ("C", dates[4])}


def test_stale_summary_counts_from_a_start_date():
    frame, dates = _frame()
    summary = stale_summary(frame, since="2025-01-01")
    assert summary["rows"] == 12 and summary["stale_rows"] == 4
    assert summary["series_with_stale_rows"] == 3 and summary["dates_with_stale_rows"] == 2
    assert summary["weekdays_below_active_share"] == 1
    assert summary["by_year"] == {2025: {"rows": 12, "stale_rows": 4}}
    assert stale_summary(frame, since="2030-01-01")["rows"] == 0
