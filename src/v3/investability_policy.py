"""Draft V3 investability policy and observable membership-exit paths.

This module keeps membership, order permission, execution, and valuation
separate.  In particular, leaving an index never implies that an existing
position was sold: the package must show a contiguous status path to a valid
tradable bar or an explicit terminal settlement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.v3.membership_mask import validate_membership_panel


POLICY_SCHEMA_VERSION = 1
POLICY_ID = "thai_drl_v3_investability_lifecycle"
POLICY_STATUSES = {"draft_not_approved", "locked"}
NONTRADING_STATUSES = {"suspended", "halted", "no_trade"}
EXIT_LEDGER_COLUMNS = (
    "exit_transition_id",
    "ticker",
    "asset_slot",
    "instrument_id",
    "last_active_date",
    "exit_effective_date",
    "liquidation_date",
    "liquidation_lag_sessions",
    "observed_status_sessions",
    "terminal_action_id",
    "path_status",
    "path_covered",
    "blocking_reason",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str, issues: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        issues.append(f"{label} must be a mapping")
        return {}
    return value


def _require_exact(
    mapping: Mapping[str, Any],
    key: str,
    expected: Any,
    *,
    label: str,
    issues: list[str],
) -> None:
    if mapping.get(key) != expected:
        issues.append(f"{label}.{key} must be {expected!r}")


def validate_investability_policy(
    path: Path,
    *,
    require_locked: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Validate the draft policy without converting it into approval."""

    policy_path = Path(path)
    if not policy_path.is_file():
        return [f"V3 investability policy is missing: {policy_path}"], {}
    try:
        loaded = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot read V3 investability policy: {exc}"], {}
    issues: list[str] = []
    policy = dict(_mapping(loaded, "policy", issues))
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        issues.append(f"policy schema_version must be {POLICY_SCHEMA_VERSION}")

    registration = _mapping(policy.get("registration"), "registration", issues)
    _require_exact(
        registration,
        "id",
        POLICY_ID,
        label="registration",
        issues=issues,
    )
    status = registration.get("status")
    if status not in POLICY_STATUSES:
        issues.append(
            "registration.status must be draft_not_approved or locked"
        )
    _require_exact(
        registration,
        "replaces_v2",
        False,
        label="registration",
        issues=issues,
    )

    boundary = _mapping(
        policy.get("scientific_boundary"),
        "scientific_boundary",
        issues,
    )
    _require_exact(
        boundary,
        "development_only",
        True,
        label="scientific_boundary",
        issues=issues,
    )
    if status == "draft_not_approved":
        for key in (
            "final_investability_ready",
            "training_enabled",
            "publication_claim_enabled",
        ):
            _require_exact(
                boundary,
                key,
                False,
                label="scientific_boundary",
                issues=issues,
            )

    timing = _mapping(policy.get("timing"), "timing", issues)
    required_timing = {
        "timezone": "Asia/Bangkok",
        "decision_horizon": "close_t_to_close_t_plus_1",
        "model_information_rule": "strictly_available_before_decision_cutoff",
        "same_session_after_close_bar_role": (
            "execution_evidence_only_not_model_input"
        ),
        "later_correction_rule": "never_rewrite_an_earlier_decision",
    }
    for key, expected in required_timing.items():
        _require_exact(
            timing,
            key,
            expected,
            label="timing",
            issues=issues,
        )

    rules = _mapping(policy.get("rules"), "rules", issues)
    expected_rule_values = {
        ("entry", "membership_required"): True,
        ("entry", "complete_prior_lookback_required"): True,
        ("entry", "observable_trading_permission_required"): True,
        ("entry", "valid_execution_bar_required_for_fill"): True,
        ("entry", "failed_fill_destination"): "remain_in_cash",
        ("membership_exit", "request_on_first_nonmember_session"): True,
        ("membership_exit", "execute_on_first_subsequent_trading_bar"): True,
        ("membership_exit", "while_untradeable"): "freeze_existing_shares",
        ("membership_exit", "never_convert_unfilled_holding_to_cash"): True,
        ("delisting", "assumed_recovery_value"): "forbidden",
        ("delisting", "required_resolution"): (
            "explicit_cash_settlement_or_successor_instrument"
        ),
        ("delisting", "unresolved_resolution"): "block_experiment",
        ("stale_valuation", "method"): (
            "carry_last_observed_raw_close_with_age_flag"
        ),
        ("stale_valuation", "fabricated_observed_return"): "forbidden",
        ("corporate_actions", "wealth_basis"): (
            "raw_execution_prices_plus_explicit_actions"
        ),
        ("corporate_actions", "adjusted_close_role"): (
            "model_features_and_return_reconciliation_only"
        ),
        ("corporate_actions", "double_count_adjusted_return_and_cash_action"): (
            "forbidden"
        ),
        ("symbol_change", "identity_key"): "stable_instrument_id",
        ("symbol_change", "ticker_string_as_identity"): "forbidden",
    }
    for (section_name, key), expected in expected_rule_values.items():
        section = _mapping(
            rules.get(section_name),
            f"rules.{section_name}",
            issues,
        )
        _require_exact(
            section,
            key,
            expected,
            label=f"rules.{section_name}",
            issues=issues,
        )
    for section_name in ("suspension", "halt", "no_trade"):
        section = _mapping(
            rules.get(section_name),
            f"rules.{section_name}",
            issues,
        )
        for key, expected in {
            "new_entry_allowed": False,
            "sale_allowed": False,
            "holding_rule": "freeze_existing_shares",
            "return_observation_rule": (
                "stale_valuation_is_not_an_observed_zero_return"
            ),
        }.items():
            _require_exact(
                section,
                key,
                expected,
                label=f"rules.{section_name}",
                issues=issues,
            )

    accounting = _mapping(policy.get("accounting"), "accounting", issues)
    expected_accounting = {
        "ledger": "self_financing_security_and_cash",
        "turnover": (
            "sum_absolute_security_trades_excluding_cash_settlement_leg"
        ),
        "transaction_cost_basis": "gross_traded_security_notional",
        "slippage_basis": "gross_traded_security_notional",
        "post_return_state": "carry_drifted_security_and_cash_weights",
        "cost_funding": "reduce_purchases_before_allowing_negative_cash",
        "borrowing_allowed": False,
        "forced_exit_overrides_discretionary_turnover_cap": True,
        "untradeable_forced_exit": "remain_pending_without_trade_cost",
    }
    for key, expected in expected_accounting.items():
        _require_exact(
            accounting,
            key,
            expected,
            label="accounting",
            issues=issues,
        )

    evidence = policy.get("required_provider_evidence")
    expected_evidence = [
        "adjusted_ohlcv_with_availability",
        "daily_trading_status_with_availability",
        "authoritative_set_sessions",
        "corporate_actions_with_announcement_and_effective_dates",
        "stable_instrument_symbol_history",
    ]
    if evidence != expected_evidence:
        issues.append(
            "required_provider_evidence must contain the five ordered inputs"
        )
    decisions = policy.get("author_decisions_required_before_lock")
    if not isinstance(decisions, list) or not all(
        isinstance(value, str) and value.strip() for value in decisions
    ):
        issues.append(
            "author_decisions_required_before_lock must be a string list"
        )
        decisions = []
    if status == "draft_not_approved" and not decisions:
        issues.append("draft policy must disclose unresolved author decisions")

    if require_locked:
        if status != "locked":
            issues.append("V3 investability policy is not locked")
        if registration.get("approved_by_authors") is not True:
            issues.append("V3 investability policy is not approved by both authors")
        if boundary.get("final_investability_ready") is not True:
            issues.append("locked policy must mark final investability ready")
        if boundary.get("training_enabled") is not True:
            issues.append("locked policy must explicitly enable training")
        if decisions:
            issues.append("locked policy cannot retain unresolved author decisions")
        sidecar = Path(f"{policy_path}.sha256")
        if not sidecar.is_file():
            issues.append("locked V3 investability policy sidecar is missing")
        else:
            tokens = sidecar.read_text(encoding="utf-8").split()
            if not tokens or tokens[0].lower() != sha256_file(policy_path):
                issues.append("locked V3 investability policy sidecar does not match")

    return list(dict.fromkeys(issues)), policy


def membership_exit_transitions(
    membership_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Return first non-member sessions following observed active membership."""

    panel = validate_membership_panel(membership_panel)
    ordered = panel.sort_values(["asset_slot", "date"], kind="mergesort").copy()
    ordered["_previous_member"] = ordered.groupby(
        "ticker",
        sort=False,
    )["is_index_member"].shift(1)
    ordered["_previous_date"] = ordered.groupby(
        "ticker",
        sort=False,
    )["date"].shift(1)
    exits = ordered[
        ordered["_previous_member"].eq(True)
        & ordered["is_index_member"].eq(False)
    ].copy()
    if exits.empty:
        return pd.DataFrame(
            columns=[
                "exit_transition_id",
                "ticker",
                "asset_slot",
                "last_active_date",
                "exit_effective_date",
            ]
        )
    exits["exit_transition_id"] = (
        exits["ticker"].astype(str)
        + "|"
        + exits["date"].dt.strftime("%Y-%m-%d")
    )
    exits = exits.rename(
        columns={
            "_previous_date": "last_active_date",
            "date": "exit_effective_date",
        }
    )
    return exits[
        [
            "exit_transition_id",
            "ticker",
            "asset_slot",
            "last_active_date",
            "exit_effective_date",
        ]
    ].sort_values(
        ["exit_effective_date", "asset_slot"],
        kind="mergesort",
    ).reset_index(drop=True)


def _normalize_sessions(sessions: pd.DataFrame | None) -> pd.DatetimeIndex:
    if sessions is None or "date" not in sessions:
        return pd.DatetimeIndex([])
    values = pd.to_datetime(
        sessions["date"],
        errors="coerce",
    ).dt.normalize().dropna()
    return pd.DatetimeIndex(values.drop_duplicates().sort_values())


def _normalize_status(status: pd.DataFrame | None) -> pd.DataFrame:
    if status is None or not {"date", "ticker", "trading_status"}.issubset(status):
        return pd.DataFrame(columns=["date", "ticker", "trading_status"])
    normalized = status[["date", "ticker", "trading_status"]].copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"],
        errors="coerce",
    ).dt.normalize()
    normalized["ticker"] = (
        normalized["ticker"].astype(str).str.strip().str.upper()
    )
    normalized["trading_status"] = (
        normalized["trading_status"].astype(str).str.strip().str.lower()
    )
    return normalized.dropna(subset=["date"])


def _valid_bar_key_set(bars: pd.DataFrame | None) -> set[tuple[pd.Timestamp, str]]:
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    if bars is None or not required.issubset(bars):
        return set()
    normalized = bars[list(required)].copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"],
        errors="coerce",
    ).dt.normalize()
    normalized["ticker"] = (
        normalized["ticker"].astype(str).str.strip().str.upper()
    )
    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    prices = normalized[["open", "high", "low", "close"]].to_numpy(
        dtype=np.float64
    )
    valid = (
        normalized["date"].notna().to_numpy()
        & np.isfinite(prices).all(axis=1)
        & (prices > 0.0).all(axis=1)
        & np.isfinite(normalized["volume"].to_numpy(dtype=np.float64))
        & normalized["volume"].ge(0.0).to_numpy()
    )
    return set(
        normalized.loc[valid, ["date", "ticker"]].itertuples(
            index=False,
            name=None,
        )
    )


def _instrument_at_exit(
    symbol_history: pd.DataFrame | None,
    *,
    ticker: str,
    date: pd.Timestamp,
) -> str:
    required = {
        "instrument_id",
        "ticker",
        "effective_from",
        "effective_to_exclusive",
    }
    if symbol_history is None or not required.issubset(symbol_history):
        return ""
    symbols = symbol_history[list(required)].copy()
    symbols["ticker"] = symbols["ticker"].astype(str).str.strip().str.upper()
    symbols["effective_from"] = pd.to_datetime(
        symbols["effective_from"],
        errors="coerce",
    ).dt.normalize()
    symbols["effective_to_exclusive"] = pd.to_datetime(
        symbols["effective_to_exclusive"],
        errors="coerce",
    ).dt.normalize()
    match = symbols[
        symbols["ticker"].eq(ticker)
        & symbols["effective_from"].le(date)
        & symbols["effective_to_exclusive"].gt(date)
    ]
    if len(match) != 1:
        return ""
    return str(match.iloc[0]["instrument_id"]).strip()


def _terminal_action(
    corporate_actions: pd.DataFrame | None,
    *,
    ticker: str,
    date: pd.Timestamp,
) -> str:
    required = {
        "action_id",
        "ticker",
        "action_type",
        "effective_date",
        "cash_amount",
        "successor_ticker",
    }
    if corporate_actions is None or not required.issubset(corporate_actions):
        return ""
    actions = corporate_actions[list(required)].copy()
    actions["ticker"] = actions["ticker"].astype(str).str.strip().str.upper()
    actions["action_type"] = (
        actions["action_type"].astype(str).str.strip().str.lower()
    )
    actions["effective_date"] = pd.to_datetime(
        actions["effective_date"],
        errors="coerce",
    ).dt.normalize()
    actions["cash_amount"] = pd.to_numeric(
        actions["cash_amount"],
        errors="coerce",
    )
    candidates = actions[
        actions["ticker"].eq(ticker)
        & actions["effective_date"].le(date)
        & actions["action_type"].isin({"delisting", "merger"})
    ]
    for row in candidates.sort_values(
        ["effective_date", "action_id"],
        kind="mergesort",
    ).itertuples(index=False):
        successor = (
            str(row.successor_ticker).strip()
            if pd.notna(row.successor_ticker)
            else ""
        )
        explicit_cash = pd.notna(row.cash_amount) and float(row.cash_amount) >= 0.0
        if explicit_cash or successor:
            return str(row.action_id).strip()
    return ""


def assess_membership_exit_paths(
    membership_panel: pd.DataFrame,
    *,
    bars: pd.DataFrame | None,
    trading_status: pd.DataFrame | None,
    sessions: pd.DataFrame | None,
    corporate_actions: pd.DataFrame | None,
    symbol_history: pd.DataFrame | None,
) -> pd.DataFrame:
    """Assess whether every observed membership exit has an executable path."""

    transitions = membership_exit_transitions(membership_panel)
    if transitions.empty:
        return pd.DataFrame(columns=EXIT_LEDGER_COLUMNS)
    session_dates = _normalize_sessions(sessions)
    status = _normalize_status(trading_status)
    if status.duplicated(["date", "ticker"]).any():
        status = pd.DataFrame(columns=status.columns)
    status_by_key = {
        (row.date, row.ticker): row.trading_status
        for row in status.itertuples(index=False)
    }
    valid_bars = _valid_bar_key_set(bars)

    rows: list[dict[str, Any]] = []
    for exit_row in transitions.itertuples(index=False):
        exit_date = pd.Timestamp(exit_row.exit_effective_date).normalize()
        ticker = str(exit_row.ticker)
        instrument_id = _instrument_at_exit(
            symbol_history,
            ticker=ticker,
            date=exit_date,
        )
        base = {
            "exit_transition_id": exit_row.exit_transition_id,
            "ticker": ticker,
            "asset_slot": int(exit_row.asset_slot),
            "instrument_id": instrument_id,
            "last_active_date": pd.Timestamp(exit_row.last_active_date),
            "exit_effective_date": exit_date,
            "liquidation_date": pd.NaT,
            "liquidation_lag_sessions": pd.NA,
            "observed_status_sessions": 0,
            "terminal_action_id": "",
            "path_status": "unresolved",
            "path_covered": False,
            "blocking_reason": "",
        }
        if not instrument_id:
            base["blocking_reason"] = "missing_stable_instrument_identity"
            rows.append(base)
            continue
        candidates = session_dates[session_dates >= exit_date]
        if candidates.empty:
            base["blocking_reason"] = "missing_authoritative_exit_sessions"
            rows.append(base)
            continue

        observed = 0
        resolved = False
        for lag, date in enumerate(candidates):
            state = status_by_key.get((pd.Timestamp(date), ticker))
            if state is None:
                base["blocking_reason"] = "missing_contiguous_exit_status"
                break
            observed += 1
            if state in NONTRADING_STATUSES:
                continue
            if state == "trading":
                if (pd.Timestamp(date), ticker) not in valid_bars:
                    base["blocking_reason"] = "trading_exit_without_valid_bar"
                    break
                base.update(
                    {
                        "liquidation_date": pd.Timestamp(date),
                        "liquidation_lag_sessions": int(lag),
                        "path_status": "tradable_liquidation",
                        "path_covered": True,
                        "blocking_reason": "",
                    }
                )
                resolved = True
                break
            if state == "delisted":
                action_id = _terminal_action(
                    corporate_actions,
                    ticker=ticker,
                    date=pd.Timestamp(date),
                )
                if not action_id:
                    base["blocking_reason"] = "delisting_without_explicit_resolution"
                    break
                base.update(
                    {
                        "liquidation_date": pd.Timestamp(date),
                        "liquidation_lag_sessions": int(lag),
                        "terminal_action_id": action_id,
                        "path_status": "terminal_settlement",
                        "path_covered": True,
                        "blocking_reason": "",
                    }
                )
                resolved = True
                break
            base["blocking_reason"] = f"inconsistent_exit_status:{state}"
            break
        base["observed_status_sessions"] = int(observed)
        if not resolved and not base["blocking_reason"]:
            base["blocking_reason"] = "exit_path_ends_before_resolution"
        rows.append(base)

    ledger = pd.DataFrame(rows, columns=EXIT_LEDGER_COLUMNS)
    return ledger.sort_values(
        ["exit_effective_date", "asset_slot"],
        kind="mergesort",
    ).reset_index(drop=True)
