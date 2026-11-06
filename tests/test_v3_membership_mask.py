from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.v3.membership_mask import (
    MASK_KIND,
    REVISION_AVAILABILITY_RULE,
    attach_membership_context,
    build_point_in_time_membership_panel,
    validate_membership_panel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = (
    PROJECT_ROOT / "data/reference/set50_set100_membership_2021_2024.csv"
)
UNIVERSE_PATH = (
    PROJECT_ROOT / "data/reference/set50_or_set100_universe.csv"
)
SET_CALENDAR_PATH = PROJECT_ROOT / "data/raw/prices_market_indices.csv"


def _universe() -> tuple[str, ...]:
    return tuple(pd.read_csv(UNIVERSE_PATH)["ticker"].astype(str))


def _panel(dates: list[str], *, index_name: str = "SET100") -> pd.DataFrame:
    return build_point_in_time_membership_panel(
        HISTORY_PATH,
        dates,
        _universe(),
        index_name=index_name,
    )


def _active(panel: pd.DataFrame, date: str) -> pd.DataFrame:
    return panel[
        panel["date"].eq(pd.Timestamp(date)) & panel["is_index_member"]
    ]


def test_2021_revision_is_delayed_until_next_observed_session() -> None:
    panel = _panel(["2021-02-10", "2021-02-11", "2021-02-15"])

    assert _active(panel, "2021-02-10")["membership_record_id"].unique().tolist() == [
        "set_2021_h1_original"
    ]
    assert _active(panel, "2021-02-11")["membership_record_id"].unique().tolist() == [
        "set_2021_h1_original"
    ]
    assert _active(panel, "2021-02-15")["membership_record_id"].unique().tolist() == [
        "set_2021_h1_revised"
    ]
    assert "TKN.BK" in set(_active(panel, "2021-02-11")["ticker"])
    assert "OR.BK" not in set(_active(panel, "2021-02-11")["ticker"])
    assert "OR.BK" in set(_active(panel, "2021-02-15")["ticker"])
    assert "TKN.BK" not in set(_active(panel, "2021-02-15")["ticker"])


def test_2023_revision_is_delayed_until_next_observed_session() -> None:
    panel = _panel(["2023-04-05", "2023-04-07", "2023-04-10"])

    assert _active(panel, "2023-04-07")["membership_record_id"].unique().tolist() == [
        "set_2023_h1_original"
    ]
    assert _active(panel, "2023-04-10")["membership_record_id"].unique().tolist() == [
        "set_2023_h1_revised"
    ]
    before = set(_active(panel, "2023-04-07")["ticker"])
    after = set(_active(panel, "2023-04-10")["ticker"])
    assert {"SIRI.BK", "SJWD.BK"}.issubset(after)
    assert {"DTAC.BK", "STARK.BK"}.issubset(before)
    assert {"SIRI.BK", "SJWD.BK"}.isdisjoint(before)
    assert {"DTAC.BK", "STARK.BK"}.isdisjoint(after)


def test_scheduled_half_year_boundaries_remain_half_open() -> None:
    panel = _panel(
        ["2024-01-01", "2024-01-02", "2024-06-28", "2024-07-01"],
        index_name="SET50",
    )

    assert _active(panel, "2024-01-01")["membership_record_id"].unique().tolist() == [
        "set_2023_h2"
    ]
    assert _active(panel, "2024-01-02")["membership_record_id"].unique().tolist() == [
        "set_2024_h1"
    ]
    assert _active(panel, "2024-06-28")["membership_record_id"].unique().tolist() == [
        "set_2024_h1"
    ]
    assert _active(panel, "2024-07-01")["membership_record_id"].unique().tolist() == [
        "set_2024_h2"
    ]
    assert panel.groupby("date")["is_index_member"].sum().eq(50).all()


def test_full_observed_set_session_grid_is_dense_and_stable() -> None:
    calendar = pd.read_csv(SET_CALENDAR_PATH)
    dates = pd.to_datetime(calendar["date"]).drop_duplicates().sort_values()
    panel = build_point_in_time_membership_panel(
        HISTORY_PATH,
        dates.tolist(),
        _universe(),
        index_name="SET100",
    )

    assert len(panel) == len(dates) * 138
    assert panel["date"].nunique() == len(dates)
    assert panel.groupby("date")["is_index_member"].sum().eq(100).all()
    assert panel.groupby("date")["membership_record_id"].nunique().eq(1).all()
    assert panel["mask_kind"].eq(MASK_KIND).all()
    assert panel["revision_availability_rule"].eq(
        REVISION_AVAILABILITY_RULE
    ).all()


def test_historical_sector_changes_are_joined_as_of_session() -> None:
    panel = _panel(["2023-06-30", "2023-07-03"])

    sectors = (
        panel[panel["ticker"].isin(["MINT.BK", "NEX.BK"])]
        .set_index(["date", "ticker"])["sector_as_of"]
        .to_dict()
    )
    assert sectors[(pd.Timestamp("2023-06-30"), "MINT.BK")] == "Food & Beverage"
    assert sectors[(pd.Timestamp("2023-07-03"), "MINT.BK")] == "Tourism & Leisure"
    assert (
        sectors[(pd.Timestamp("2023-06-30"), "NEX.BK")]
        == "Electronic Components"
    )
    assert sectors[(pd.Timestamp("2023-07-03"), "NEX.BK")] == "Automotive"


def test_attach_context_does_not_relabel_inactive_rows_as_members() -> None:
    dates = ["2021-02-11", "2021-02-15"]
    features = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "close": 100.0}
            for date in dates
            for ticker in ("OR.BK", "TKN.BK")
        ]
    )
    attached = attach_membership_context(
        features,
        HISTORY_PATH,
        asset_universe=_universe(),
        index_name="SET100",
    )

    status = attached.set_index(["date", "ticker"])["is_index_member"].to_dict()
    assert status[(pd.Timestamp("2021-02-11"), "TKN.BK")]
    assert not status[(pd.Timestamp("2021-02-11"), "OR.BK")]
    assert status[(pd.Timestamp("2021-02-15"), "OR.BK")]
    assert not status[(pd.Timestamp("2021-02-15"), "TKN.BK")]


def test_membership_builder_fails_closed_on_unknown_ticker_and_date() -> None:
    with pytest.raises(ValueError, match="absent from the archive"):
        build_point_in_time_membership_panel(
            HISTORY_PATH,
            ["2024-07-01"],
            [*_universe(), "UNKNOWN.BK"],
        )
    with pytest.raises(ValueError, match="outside membership coverage"):
        _panel(["2025-01-01"])


def test_membership_validator_rejects_same_day_revision_use() -> None:
    panel = _panel(["2021-02-15"])
    tampered = panel.copy()
    tampered["date"] = pd.Timestamp("2021-02-11")

    with pytest.raises(ValueError, match="before it is safely available"):
        validate_membership_panel(tampered)

