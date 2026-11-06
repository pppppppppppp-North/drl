from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.v3.investability_policy import (
    assess_membership_exit_paths,
    membership_exit_transitions,
    validate_investability_policy,
)
from src.v3.market_data_readiness import CORPORATE_ACTION_COLUMNS
from src.v3.membership_mask import (
    MASK_KIND,
    REVISION_AVAILABILITY_RULE,
    validate_membership_panel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config/v3/investability_policy_draft.yaml"


def _dates() -> pd.DatetimeIndex:
    return pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )


def _membership() -> pd.DataFrame:
    rows = []
    for day, date in enumerate(_dates()):
        for slot, ticker in enumerate(("AAA.BK", "BBB.BK")):
            is_member = ticker == "BBB.BK" or day < 2
            rows.append(
                {
                    "date": date,
                    "asset_slot": slot,
                    "ticker": ticker,
                    "index_name": "SET100",
                    "is_index_member": is_member,
                    "membership_record_id": f"membership-{day}",
                    "membership_published_at": pd.Timestamp("2023-12-01"),
                    "revision_status": "original",
                    "sector_as_of": "Energy" if is_member else "",
                    "symbol": ticker.removesuffix(".BK"),
                    "company_name": ticker,
                    "source_url": "https://example.invalid/investability-test",
                    "source_file_sha256": "a" * 64,
                    "revision_availability_rule": REVISION_AVAILABILITY_RULE,
                    "mask_kind": MASK_KIND,
                }
            )
    return validate_membership_panel(pd.DataFrame(rows))


def _sessions(*, include_buffer: bool = False) -> pd.DataFrame:
    dates = list(_dates())
    if include_buffer:
        dates.append(pd.Timestamp("2024-01-08"))
    return pd.DataFrame(
        {
            "date": dates,
            "session_available_at": ["2023-12-01T00:00:00+07:00"] * len(dates),
            "source_record_id": [f"session-{date.date()}" for date in dates],
        }
    )


def _status(*, include_buffer: bool = False) -> pd.DataFrame:
    dates = list(_dates())
    if include_buffer:
        dates.append(pd.Timestamp("2024-01-08"))
    return pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "trading_status": "trading",
                "status_available_at": f"{date.date()}T17:00:00+07:00",
                "source_record_id": f"status-{date.date()}-{ticker}",
            }
            for date in dates
            for ticker in ("AAA.BK", "BBB.BK")
        ]
    )


def _bars(*, include_buffer: bool = False) -> pd.DataFrame:
    dates = list(_dates())
    if include_buffer:
        dates.append(pd.Timestamp("2024-01-08"))
    return pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "adjusted_close": 100.0,
                "volume": 1_000.0,
                "bar_available_at": f"{date.date()}T17:00:00+07:00",
                "source_record_id": f"bar-{date.date()}-{ticker}",
            }
            for date in dates
            for ticker in ("AAA.BK", "BBB.BK")
        ]
    )


def _symbols() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instrument_id": f"instrument-{ticker}",
                "ticker": ticker,
                "effective_from": "2024-01-01",
                "effective_to_exclusive": "2025-01-01",
                "source_record_id": f"symbol-{ticker}",
            }
            for ticker in ("AAA.BK", "BBB.BK")
        ]
    )


def _actions() -> pd.DataFrame:
    return pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)


def _assess(**overrides) -> pd.DataFrame:
    arguments = {
        "membership_panel": _membership(),
        "bars": _bars(),
        "trading_status": _status(),
        "sessions": _sessions(),
        "corporate_actions": _actions(),
        "symbol_history": _symbols(),
    }
    arguments.update(overrides)
    return assess_membership_exit_paths(**arguments)


def test_draft_policy_is_structurally_valid_but_not_lockable() -> None:
    issues, policy = validate_investability_policy(POLICY_PATH)
    assert issues == []
    assert policy["registration"]["status"] == "draft_not_approved"

    locked_issues, _ = validate_investability_policy(
        POLICY_PATH,
        require_locked=True,
    )
    assert "V3 investability policy is not locked" in locked_issues
    assert (
        "V3 investability policy is not approved by both authors"
        in locked_issues
    )
    assert "locked policy cannot retain unresolved author decisions" in locked_issues


def test_membership_exit_is_detected_on_first_nonmember_session() -> None:
    exits = membership_exit_transitions(_membership())

    assert len(exits) == 1
    assert exits.iloc[0]["ticker"] == "AAA.BK"
    assert exits.iloc[0]["last_active_date"] == pd.Timestamp("2024-01-03")
    assert exits.iloc[0]["exit_effective_date"] == pd.Timestamp("2024-01-04")


def test_immediate_trading_bar_covers_membership_exit() -> None:
    ledger = _assess()

    assert len(ledger) == 1
    assert bool(ledger.iloc[0]["path_covered"])
    assert ledger.iloc[0]["path_status"] == "tradable_liquidation"
    assert ledger.iloc[0]["liquidation_date"] == pd.Timestamp("2024-01-04")
    assert ledger.iloc[0]["liquidation_lag_sessions"] == 0


def test_suspended_exit_waits_for_first_buffer_trading_bar() -> None:
    status = _status(include_buffer=True)
    status.loc[
        status["ticker"].eq("AAA.BK")
        & status["date"].isin(
            [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")]
        ),
        "trading_status",
    ] = "suspended"
    ledger = _assess(
        bars=_bars(include_buffer=True),
        trading_status=status,
        sessions=_sessions(include_buffer=True),
    )

    assert bool(ledger.iloc[0]["path_covered"])
    assert ledger.iloc[0]["liquidation_date"] == pd.Timestamp("2024-01-08")
    assert ledger.iloc[0]["liquidation_lag_sessions"] == 2
    assert ledger.iloc[0]["observed_status_sessions"] == 3


def test_inactive_exit_status_cannot_be_missing_or_inferred_from_a_bar() -> None:
    status = _status()
    status = status[
        ~(
            status["ticker"].eq("AAA.BK")
            & status["date"].eq(pd.Timestamp("2024-01-04"))
        )
    ]
    ledger = _assess(trading_status=status)

    assert not bool(ledger.iloc[0]["path_covered"])
    assert ledger.iloc[0]["blocking_reason"] == "missing_contiguous_exit_status"


def test_delisting_requires_explicit_terminal_resolution() -> None:
    status = _status()
    status.loc[
        status["ticker"].eq("AAA.BK")
        & status["date"].eq(pd.Timestamp("2024-01-04")),
        "trading_status",
    ] = "delisted"
    unresolved = _assess(trading_status=status)
    assert (
        unresolved.iloc[0]["blocking_reason"]
        == "delisting_without_explicit_resolution"
    )

    action = {column: None for column in CORPORATE_ACTION_COLUMNS}
    action.update(
        {
            "action_id": "delisting-aaa",
            "ticker": "AAA.BK",
            "action_type": "delisting",
            "announcement_at": "2024-01-03T12:00:00+07:00",
            "effective_date": "2024-01-04",
            "cash_amount": 0.0,
            "currency": "THB",
            "source_record_id": "action-source",
        }
    )
    resolved = _assess(
        trading_status=status,
        corporate_actions=pd.DataFrame([action]),
    )
    assert bool(resolved.iloc[0]["path_covered"])
    assert resolved.iloc[0]["path_status"] == "terminal_settlement"
    assert resolved.iloc[0]["terminal_action_id"] == "delisting-aaa"


def test_resolved_exit_is_invariant_to_later_market_suffix() -> None:
    original = _assess(
        bars=_bars(include_buffer=True),
        trading_status=_status(include_buffer=True),
        sessions=_sessions(include_buffer=True),
    )
    changed_status = _status(include_buffer=True)
    changed_status.loc[
        changed_status["date"].eq(pd.Timestamp("2024-01-08")),
        "trading_status",
    ] = "halted"
    changed_bars = _bars(include_buffer=True)
    changed_bars = changed_bars[
        changed_bars["date"].ne(pd.Timestamp("2024-01-08"))
    ]
    changed = _assess(
        bars=changed_bars,
        trading_status=changed_status,
        sessions=_sessions(include_buffer=True),
    )

    pd.testing.assert_frame_equal(original, changed)


def test_missing_symbol_identity_blocks_an_otherwise_tradable_exit() -> None:
    ledger = _assess(symbol_history=None)

    assert not bool(ledger.iloc[0]["path_covered"])
    assert (
        ledger.iloc[0]["blocking_reason"]
        == "missing_stable_instrument_identity"
    )
