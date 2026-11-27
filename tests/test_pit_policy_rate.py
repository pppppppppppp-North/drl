from __future__ import annotations

import pandas as pd

from scripts.run_pit_walk_forward import add_policy_rate


def test_policy_rate_features_lag_one_session_and_ignore_decisions_after_the_validation_end() -> None:
    dates = pd.bdate_range("2022-12-26", periods=8)  # through 2023-01-04
    panel = pd.DataFrame([{"date": d, "ticker": t, "close": 1.0} for d in dates for t in ("A.BK", "B.BK")])
    history = pd.DataFrame({"announcement_date": ["2022-12-27", "2023-01-02"], "policy_rate": [1.25, 1.5],
                            "change_bp": [25, 25]})
    out = add_policy_rate(panel, history, "2022-12-30")
    assert len(out) == len(panel)
    assert "bot_source_announcement" not in out.columns
    rate = out.drop_duplicates("date").set_index("date")["bot_policy_rate"]
    assert rate[pd.Timestamp("2022-12-27")] == 0.0  # announced that day: available from the next session
    assert rate[pd.Timestamp("2022-12-28")] == 1.25
    assert rate[pd.Timestamp("2023-01-04")] == 1.25  # the 2023 decision is never read
    per_date = out.groupby("date")["bot_policy_rate"].nunique()
    assert (per_date == 1).all()  # market-level: the same value for every stock
