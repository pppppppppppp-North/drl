from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.source_coverage_table import carried_forward, member_sessions


def test_carried_forward_counts_same_day_carried_and_missing_sessions() -> None:
    sessions = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]))
    observed = pd.DatetimeIndex(pd.to_datetime(["2020-01-03", "2020-01-07"]))
    out = carried_forward(sessions, observed)
    assert out["units"] == 5
    assert out["covered_share"] == pytest.approx(2 / 5)
    assert out["carried_or_stale_share"] == pytest.approx(2 / 5)  # 01-06 and 01-08
    assert out["missing_share"] == pytest.approx(1 / 5)  # 01-02, before the first observation
    assert out["longest_carry_days"] == 3.0  # Friday's close on Monday


def test_member_sessions_flags_unmapped_stale_and_warm_up_rows() -> None:
    date = pd.Timestamp("2020-01-02")
    mask = pd.DataFrame({"date": [date] * 4, "ticker": ["A", "B", "C", "D"],
                         "price_ticker": ["A.BK", "B.BK", "", "D.BK"]})
    panel = pd.DataFrame({"date": [date] * 3, "ticker": ["A.BK", "B.BK", "D.BK"], "close": [1.0, 2.0, 3.0],
                          "volume": [100, 0, 50], "history_sessions": [200, 200, 5], "return_1d": [0.0, 0.1, np.nan],
                          "sector_peers_available": [1.0, 0.0, 1.0]})
    rows = member_sessions(mask, panel, ["return_1d"]).set_index("member")
    assert rows["mapped"].tolist() == [True, True, False, True]
    assert rows["priced"].tolist() == [True, True, False, True]
    assert rows["stale"].tolist() == [False, True, False, False]
    assert rows["technical"].tolist() == [True, True, False, False]
    assert rows["warm_up"].tolist() == [False, False, False, True]
    assert rows["sector_peer"].tolist() == [True, False, False, True]
