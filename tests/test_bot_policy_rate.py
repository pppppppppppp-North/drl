from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ingest_bot_policy_rate import (
    normalize_policy_rate_history,
    parse_announcement_date,
    parse_decision,
    parse_votes,
    read_xlsx_rows,
)
from src.features.policy_rate_context import merge_policy_rate_context

SOURCE = Path(__file__).resolve().parents[1] / "data/external/original/bot_policy_rate_mpc_table_asof_20261028.xlsx"


def test_parsers() -> None:
    assert parse_decision("Hold") == 0
    assert parse_decision("Raise 25 bps.") == 25
    assert parse_decision("Cut\xa0100 bps") == -100
    with pytest.raises(ValueError):
        parse_decision("Pause")
    assert parse_votes("65/") == 6.0 and parse_votes("7") == 7.0 and np.isnan(parse_votes(None))
    assert str(parse_announcement_date("28-29 May 2013")) == "2013-05-29"
    assert str(parse_announcement_date("8-Feb-2017", "42774")) == "2017-02-08"
    assert str(parse_announcement_date("43770")) == "2019-11-01"


@pytest.mark.skipif(not SOURCE.exists(), reason="preserved BOT source file not present")
def test_official_history_matches_known_decisions() -> None:
    history = normalize_policy_rate_history(read_xlsx_rows(SOURCE), source_name="bot")
    assert len(history) == 207
    by_date = history.set_index(history["announcement_date"].dt.strftime("%Y-%m-%d"))
    assert by_date.loc["2013-05-29", "change_bp"] == -25  # two-day meeting stored as text
    assert by_date.loc["2014-03-12", ["votes_hold", "votes_cut_25"]].tolist() == [3.0, 4.0]
    assert by_date.loc["2019-11-06", "policy_rate"] == 1.25
    assert by_date.loc["2020-02-05", "policy_rate"] == 1.0
    assert history["policy_rate"].iloc[-1] == 1.0


def test_level_mismatch_fails_closed() -> None:
    rows = [
        {"C": "43770", "E": "MPC", "G": "Hold", "H": "1.5"},
        {"C": "43775", "E": "MPC", "G": "Hold", "H": "1.25"},  # level moved without a decision
    ]
    with pytest.raises(ValueError, match="disagree"):
        normalize_policy_rate_history(rows, source_name="x")


def test_policy_rate_available_only_from_next_session() -> None:
    history = pd.DataFrame(
        {
            "announcement_date": pd.to_datetime(["2019-08-07", "2019-11-06"]),
            "policy_rate": [1.5, 1.25],
            "change_bp": [-25, -25],
            "dissent_share": [2 / 7, 2 / 7],
        }
    )
    dates = pd.bdate_range("2019-08-01", "2020-12-31")
    stocks = pd.DataFrame([{"date": d, "ticker": t, "close": 1.0} for d in dates for t in ("AAA", "BBB")])
    merged = merge_policy_rate_context(stocks, history).set_index(["date", "ticker"])
    assert merged.loc[(pd.Timestamp("2019-08-07"), "AAA"), "bot_policy_available"] == 0.0
    assert merged.loc[(pd.Timestamp("2019-11-06"), "AAA"), "bot_policy_rate"] == 1.5  # announced that day
    assert merged.loc[(pd.Timestamp("2019-11-07"), "AAA"), "bot_policy_rate"] == 1.25
    assert merged.loc[(pd.Timestamp("2019-11-07"), "BBB"), "bot_last_change_bp"] == -25
    # 365 days before 2020-11-05 is 2019-11-06, when the November cut was not yet available.
    assert merged.loc[(pd.Timestamp("2020-11-05"), "AAA"), "bot_policy_rate_change_1y"] == pytest.approx(-0.25)
    assert merged.loc[(pd.Timestamp("2020-11-09"), "AAA"), "bot_policy_rate_change_1y"] == pytest.approx(0.0)
    used = merged["bot_source_announcement"].notna()
    assert (merged.loc[used, "bot_source_announcement"] < merged.loc[used].index.get_level_values("date")).all()
