from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.pit_sector_context import SECTOR_COLUMNS, add_sector_context, session_sectors

DATES = pd.bdate_range("2019-01-01", periods=10)
FIRST, SECOND = DATES[:5], DATES[5:]
# List r1 for the first five sessions, r2 for the rest. C changes sector, D leaves, E has no prices, G joins.
LISTS = {
    "r1": {"A": "Banking", "B": "Banking", "C": "Banking", "D": "Energy"},
    "r2": {"A": "Banking", "B": "Banking", "C": "Energy", "E": "Banking", "G": "Banking"},
}


def _membership() -> pd.DataFrame:
    return pd.DataFrame([{"record_id": r, "ticker": t, "sector": s} for r, m in LISTS.items() for t, s in m.items()])


def _mask() -> pd.DataFrame:
    rows = []
    for record, dates in (("r1", FIRST), ("r2", SECOND)):
        for date in dates:
            for ticker in LISTS[record]:
                rows.append({"date": date, "record_id": record, "ticker": ticker,
                             "price_ticker": "" if ticker == "E" else f"{ticker}.BK", "in_set50": True})
    return pd.DataFrame(rows)


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [{"date": d, "ticker": t, "close": 1.0, "return_5d": rng.normal(0, 0.02), "return_20d": rng.normal(0, 0.05)}
            for d in DATES for t in ("A.BK", "B.BK", "C.BK", "D.BK", "F.BK", "G.BK")]
    panel = pd.DataFrame(rows).assign(history_sessions=100)
    panel.loc[(panel["ticker"] == "B.BK") & (panel["date"] == DATES[6]), "return_20d"] = np.nan
    panel.loc[(panel["ticker"] == "G.BK") & (panel["date"] == DATES[8]), "history_sessions"] = 10  # a new series
    return panel


def _cell(out: pd.DataFrame, date, ticker: str, column: str) -> float:
    return float(out.loc[(out["date"] == date) & (out["ticker"] == ticker), column].iloc[0])


def test_peers_are_the_other_listed_priced_members_of_the_dated_sector() -> None:
    panel = _panel()
    out = add_sector_context(panel, _mask(), _membership())
    pd.testing.assert_frame_equal(out[panel.columns], panel.sort_values(["date", "ticker"]).reset_index(drop=True))
    first, second = FIRST[2], SECOND[2]
    # First list: A's Banking peers are B and C; G is not listed yet and has no sector.
    expected = _cell(out, first, "A.BK", "return_5d") - np.mean([_cell(out, first, t, "return_5d") for t in ("B.BK", "C.BK")])
    assert _cell(out, first, "A.BK", "sector_relative_return_5d") == pytest.approx(expected)
    assert [_cell(out, first, "G.BK", c) for c in SECTOR_COLUMNS] == [0.0, 0.0, 0.0, 0.0]
    # Second list: A's Banking peers are B and G (E has no prices; C moved to Energy).
    expected = np.mean([_cell(out, second, t, "return_20d") for t in ("B.BK", "G.BK")])
    assert _cell(out, second, "A.BK", "sector_peer_return_20d") == pytest.approx(expected)
    # C is Energy's only listed member, so it has no peer; D has left but keeps its earlier sector, Energy,
    # and its peer is C.
    assert [_cell(out, second, "C.BK", c) for c in SECTOR_COLUMNS] == [0.0, 0.0, 0.0, 0.0]
    assert _cell(out, second, "D.BK", "sector_peer_return_20d") == pytest.approx(_cell(out, second, "C.BK", "return_20d"))
    assert _cell(out, second, "D.BK", "sector_peers_available") == 1.0
    # F is never listed: no sector on any session.
    assert (out.loc[out["ticker"] == "F.BK", SECTOR_COLUMNS] == 0.0).all().all()


def test_a_peer_without_the_horizon_return_is_left_out_and_a_missing_own_return_stays_missing() -> None:
    out = add_sector_context(_panel(), _mask(), _membership())
    date = DATES[6]
    assert _cell(out, date, "A.BK", "sector_peer_return_20d") == pytest.approx(_cell(out, date, "G.BK", "return_20d"))
    assert np.isnan(_cell(out, date, "B.BK", "sector_relative_return_20d"))
    assert np.isfinite(_cell(out, date, "B.BK", "sector_peer_return_20d"))


def test_a_peer_in_its_zero_filled_warm_up_counts_only_for_the_shorter_horizon() -> None:
    out = add_sector_context(_panel(), _mask(), _membership())
    date = DATES[8]  # G has 10 sessions: a real 5-session return, a zero-filled 20-session one
    assert _cell(out, date, "A.BK", "sector_peer_return_20d") == pytest.approx(_cell(out, date, "B.BK", "return_20d"))
    expected = _cell(out, date, "A.BK", "return_5d") - np.mean([_cell(out, date, t, "return_5d") for t in ("B.BK", "G.BK")])
    assert _cell(out, date, "A.BK", "sector_relative_return_5d") == pytest.approx(expected)


def test_the_inputs_are_finite_wherever_the_own_returns_are() -> None:
    out = add_sector_context(_panel(), _mask(), _membership())
    own = np.isfinite(out[["return_5d", "return_20d"]]).all(axis=1)
    assert np.isfinite(out.loc[own, SECTOR_COLUMNS]).all().all()


def test_a_series_with_two_sectors_on_one_session_is_rejected() -> None:
    mask = _mask()
    extra = mask[(mask["ticker"] == "D") & (mask["record_id"] == "r1")].assign(ticker="H")
    membership = pd.concat([_membership(), pd.DataFrame([{"record_id": "r1", "ticker": "H", "sector": "Banking"}])])
    with pytest.raises(ValueError, match="two sectors"):
        session_sectors(pd.concat([mask, extra]), membership)
