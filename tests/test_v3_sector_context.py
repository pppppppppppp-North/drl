from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v3.membership_mask import (
    MASK_KIND,
    REVISION_AVAILABILITY_RULE,
    build_point_in_time_membership_panel,
    validate_membership_panel,
)
from src.v3.sector_context import (
    SECTOR_CONTEXT_KIND,
    build_active_member_sector_context,
    validate_active_member_sector_context,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = (
    PROJECT_ROOT / "data/reference/set50_set100_membership_2021_2024.csv"
)
UNIVERSE_PATH = (
    PROJECT_ROOT / "data/reference/set50_or_set100_universe.csv"
)


def _membership(
    *,
    sectors: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    sectors = sectors or {
        "AAA.BK": "Energy",
        "BBB.BK": "Energy",
        "CCC.BK": None,
    }
    rows = []
    for slot, ticker in enumerate(("AAA.BK", "BBB.BK", "CCC.BK")):
        active = ticker != "CCC.BK"
        rows.append(
            {
                "date": pd.Timestamp("2024-01-02"),
                "asset_slot": slot,
                "ticker": ticker,
                "index_name": "SET100",
                "is_index_member": active,
                "membership_record_id": "synthetic_sector_record",
                "membership_published_at": pd.Timestamp("2023-12-01"),
                "revision_status": "original",
                "sector_as_of": sectors[ticker] if active else None,
                "symbol": ticker.removesuffix(".BK") if active else None,
                "company_name": ticker if active else None,
                "source_url": "https://example.invalid/sector-test",
                "source_file_sha256": "a" * 64,
                "revision_availability_rule": REVISION_AVAILABILITY_RULE,
                "mask_kind": MASK_KIND,
            }
        )
    return validate_membership_panel(pd.DataFrame(rows))


def _returns() -> pd.DataFrame:
    date = pd.Timestamp("2024-01-02")
    return pd.DataFrame(
        [
            {
                "date": date,
                "ticker": "AAA.BK",
                "return_1d": 0.10,
                "return_1d_is_observed": True,
                "return_information_available_at": date,
            },
            {
                "date": date,
                "ticker": "BBB.BK",
                "return_1d": 0.02,
                "return_1d_is_observed": True,
                "return_information_available_at": date,
            },
            {
                "date": date,
                "ticker": "CCC.BK",
                "return_1d": 0.90,
                "return_1d_is_observed": True,
                "return_information_available_at": date,
            },
        ]
    )


def test_context_uses_active_members_and_excludes_inactive_extreme_return() -> None:
    context = build_active_member_sector_context(_returns(), _membership())
    by_ticker = context.set_index("ticker")

    assert context["sector_context_kind"].eq(SECTOR_CONTEXT_KIND).all()
    assert by_ticker.loc["AAA.BK", "sector_active_equal_weight_return_1d"] == pytest.approx(
        0.06
    )
    assert by_ticker.loc["BBB.BK", "sector_active_equal_weight_return_1d"] == pytest.approx(
        0.06
    )
    assert by_ticker.loc["AAA.BK", "sector_return_contributor_count"] == 2
    assert by_ticker.loc[
        "AAA.BK",
        "sector_peer_ex_self_equal_weight_return_1d",
    ] == pytest.approx(0.02)
    assert by_ticker.loc[
        "BBB.BK",
        "sector_peer_ex_self_equal_weight_return_1d",
    ] == pytest.approx(0.10)
    assert not by_ticker.loc["CCC.BK", "sector_benchmark_available"]
    assert by_ticker.loc["CCC.BK", "sector_active_equal_weight_return_1d"] == 0.0


def test_context_fails_closed_when_an_active_status_row_is_missing() -> None:
    returns = _returns()
    returns = returns[returns["ticker"].ne("BBB.BK")]

    with pytest.raises(ValueError, match="missing active member/status rows"):
        build_active_member_sector_context(returns, _membership())


def test_context_rejects_duplicate_keys_and_fabricated_unobserved_zero() -> None:
    duplicate = pd.concat([_returns(), _returns().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate date/ticker"):
        build_active_member_sector_context(duplicate, _membership())

    fabricated = _returns()
    fabricated.loc[
        fabricated["ticker"].eq("BBB.BK"),
        "return_1d_is_observed",
    ] = False
    with pytest.raises(ValueError, match="fabricated value"):
        build_active_member_sector_context(fabricated, _membership())


def test_missing_active_return_changes_contributors_not_member_denominator() -> None:
    returns = _returns()
    missing = returns["ticker"].eq("BBB.BK")
    returns.loc[missing, "return_1d"] = np.nan
    returns.loc[missing, "return_1d_is_observed"] = False
    returns.loc[missing, "return_information_available_at"] = pd.NaT

    context = build_active_member_sector_context(returns, _membership())
    by_ticker = context.set_index("ticker")

    assert by_ticker.loc["AAA.BK", "sector_active_member_count"] == 2
    assert by_ticker.loc["AAA.BK", "sector_return_contributor_count"] == 1
    assert by_ticker.loc["AAA.BK", "sector_return_coverage"] == pytest.approx(0.5)
    assert by_ticker.loc["AAA.BK", "sector_active_equal_weight_return_1d"] == pytest.approx(
        0.10
    )
    assert not by_ticker.loc["AAA.BK", "sector_peer_ex_self_available"]
    assert by_ticker.loc["BBB.BK", "sector_benchmark_available"]
    assert not by_ticker.loc["BBB.BK", "sector_relative_to_active_available"]


def test_singleton_sector_has_inclusive_context_but_no_ex_self_peer() -> None:
    panel = _membership(
        sectors={
            "AAA.BK": "Energy",
            "BBB.BK": "Banking",
            "CCC.BK": None,
        }
    )
    context = build_active_member_sector_context(_returns(), panel).set_index("ticker")

    assert context.loc["AAA.BK", "sector_active_member_count"] == 1
    assert context.loc["AAA.BK", "sector_active_equal_weight_return_1d"] == pytest.approx(
        0.10
    )
    assert context.loc["AAA.BK", "sector_relative_to_active_return_1d"] == pytest.approx(
        0.0
    )
    assert not context.loc["AAA.BK", "sector_peer_ex_self_available"]


def test_context_rejects_future_return_availability() -> None:
    returns = _returns()
    returns.loc[
        returns["ticker"].eq("AAA.BK"),
        "return_information_available_at",
    ] = pd.Timestamp("2024-01-03")

    with pytest.raises(ValueError, match="own session date"):
        build_active_member_sector_context(returns, _membership())


def test_real_archive_sector_changes_are_preserved_in_context() -> None:
    universe = tuple(
        pd.read_csv(UNIVERSE_PATH)["ticker"].astype(str)
    )
    dates = [pd.Timestamp("2023-06-30"), pd.Timestamp("2023-07-03")]
    membership = build_point_in_time_membership_panel(
        HISTORY_PATH,
        dates,
        universe,
        index_name="SET100",
    )
    active = membership[membership["is_index_member"]][["date", "ticker"]].copy()
    active["return_1d"] = 0.0
    active["return_1d_is_observed"] = True
    active["return_information_available_at"] = active["date"]

    context = build_active_member_sector_context(active, membership)
    sectors = (
        context[context["ticker"].isin(["MINT.BK", "NEX.BK"])]
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
    validate_active_member_sector_context(
        context,
        expected_dates=dates,
        expected_tickers=universe,
    )
