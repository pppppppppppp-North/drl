from __future__ import annotations

import pandas as pd

from scripts.build_member_price_map import build_map, to_csv_bytes

LISTS = [("l1", "2012-01-01", "2012-07-01"), ("l2", "2012-07-01", "2013-01-01"), ("l3", "2013-01-01", "2013-07-01")]


def _membership(tickers: dict[str, list[str]]) -> pd.DataFrame:
    rows = [{"record_id": rid, "effective_from": start, "effective_to_exclusive": end, "ticker": t, "in_set50": True}
            for rid, start, end in LISTS for t, lists in tickers.items() if rid in lists]
    return pd.DataFrame(rows)


def _coverage(spans: dict[str, tuple[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame([{"ticker": t, "first_date": pd.Timestamp(a), "last_date": pd.Timestamp(b),
                           "price_file": "prices.csv.gz"} for t, (a, b) in spans.items()])
    return frame.set_index("ticker")


def _kinds(frame: pd.DataFrame, ticker: str) -> list[tuple[str, str]]:
    part = frame[frame["ticker"] == ticker]
    return list(zip(part["coverage"], part["price_ticker"]))


def test_own_current_partial_and_none() -> None:
    membership = _membership({"OWN.BK": ["l1", "l2", "l3"], "OLD.BK": ["l1", "l2"], "LATE.BK": ["l1", "l2", "l3"],
                              "GONE.BK": ["l1"]})
    coverage = _coverage({"OWN.BK": ("2012-01-04", "2013-06-28"), "NEW.BK": ("2012-01-04", "2013-06-28"),
                          "LATE.BK": ("2012-09-03", "2013-06-28")})
    frame, reused = build_map(membership, coverage, pd.Series({"OLD.BK": "NEW.BK"}))
    assert _kinds(frame, "OWN.BK") == [("own_symbol", "OWN.BK")] * 3
    assert _kinds(frame, "OLD.BK") == [("current_symbol", "NEW.BK")] * 2  # renamed: the current symbol's series
    assert _kinds(frame, "LATE.BK") == [("none", ""), ("partial_own_symbol", "LATE.BK"), ("own_symbol", "LATE.BK")]
    assert _kinds(frame, "GONE.BK") == [("none", "")]
    assert frame.loc[frame["ticker"] == "GONE.BK", "price_file"].tolist() == [""]
    assert reused == []


def test_a_series_that_starts_after_the_last_list_is_another_security() -> None:
    membership = _membership({"KK.BK": ["l1", "l2"]})
    coverage = _coverage({"KK.BK": ("2020-10-07", "2024-12-30"), "KKP.BK": ("2012-01-04", "2013-06-28")})
    frame, reused = build_map(membership, coverage, pd.Series({"KK.BK": "KKP.BK"}))
    assert reused == ["KK.BK"]
    assert _kinds(frame, "KK.BK") == [("current_symbol", "KKP.BK")] * 2


def test_the_tolerance_is_ten_days_at_each_end() -> None:
    membership = _membership({"A.BK": ["l1"], "B.BK": ["l1"], "C.BK": ["l1"]})
    coverage = _coverage({"A.BK": ("2012-01-11", "2012-06-21"),   # 10 days late, 10 days early: covered
                          "B.BK": ("2012-01-12", "2012-06-30"),   # 11 days late: partial
                          "C.BK": ("2012-01-04", "2012-06-20")})  # stops 11 days early: partial
    frame, _ = build_map(membership, coverage, pd.Series(dtype=str))
    assert _kinds(frame, "A.BK") == [("own_symbol", "A.BK")]
    assert _kinds(frame, "B.BK") == [("partial_own_symbol", "B.BK")]
    assert _kinds(frame, "C.BK") == [("partial_own_symbol", "C.BK")]


def test_csv_writes_booleans_and_empty_strings_as_the_2026_11_07_map() -> None:
    membership = _membership({"GONE.BK": ["l1"]})
    frame, _ = build_map(membership, _coverage({"X.BK": ("2012-01-04", "2012-06-29")}), pd.Series(dtype=str))
    assert to_csv_bytes(frame).decode().splitlines()[1] == "l1,2012-01-01,2012-07-01,GONE.BK,True,,none,"
