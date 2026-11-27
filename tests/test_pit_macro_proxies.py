from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_pit_walk_forward import add_macro_proxies


def test_macro_proxies_are_market_level_carried_forward_and_stop_at_the_validation_end() -> None:
    dates = pd.bdate_range("2022-12-26", periods=8)  # Thai sessions through 2023-01-04
    panel = pd.DataFrame([{"date": d, "ticker": t, "close": 1.0} for d in dates for t in ("A.BK", "B.BK")])
    macro_dates = pd.to_datetime(["2022-12-23", "2022-12-27", "2022-12-28", "2023-01-03"])
    macro = pd.DataFrame({"date": macro_dates, "ticker": "BZ=F", "close": [100.0, 110.0, 99.0, 500.0]})
    out = add_macro_proxies(panel, macro, {"BZ=F": "brent"}, "2022-12-30")
    assert len(out) == len(panel)
    level = out.drop_duplicates("date").set_index("date")["brent_level"]
    assert level[pd.Timestamp("2022-12-26")] == 100.0  # the last close on or before the session
    assert level[pd.Timestamp("2022-12-27")] == 110.0
    assert level[pd.Timestamp("2022-12-29")] == 99.0  # no new close: carried forward
    assert level[pd.Timestamp("2023-01-04")] == 99.0  # the 2023 close is never read
    one_day = out.drop_duplicates("date").set_index("date")["brent_return_1d"]
    assert one_day[pd.Timestamp("2022-12-27")] == pytest.approx(0.10)
    assert (out.groupby("date")["brent_return_5d"].nunique() == 1).all()  # the same value for every stock
