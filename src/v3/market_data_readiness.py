"""Fail-closed licensed-market package readiness for the isolated V3 path.

This module does not reuse the five-ticker V2 readiness receipt.  It evaluates
the broader acquisition union against point-in-time active membership and
requires separate, checksum-bound bars, trading status, authoritative sessions,
corporate actions, and symbol history.  A passing package is still not a final
investability or training approval; those require a locked execution policy and
V3 protocol.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.v3.investability_policy import assess_membership_exit_paths
from src.v3.membership_mask import validate_membership_panel


MANIFEST_SCHEMA_VERSION = 1
REQUIRED_BINDINGS = (
    "bars",
    "trading_status",
    "sessions",
    "corporate_actions",
    "symbol_history",
)
BAR_COLUMNS = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "bar_available_at",
    "source_record_id",
)
STATUS_COLUMNS = (
    "date",
    "ticker",
    "trading_status",
    "status_available_at",
    "source_record_id",
)
SESSION_COLUMNS = (
    "date",
    "session_available_at",
    "source_record_id",
)
CORPORATE_ACTION_COLUMNS = (
    "action_id",
    "ticker",
    "action_type",
    "announcement_at",
    "effective_date",
    "adjustment_factor",
    "cash_amount",
    "currency",
    "successor_ticker",
    "source_record_id",
)
SYMBOL_HISTORY_COLUMNS = (
    "instrument_id",
    "ticker",
    "effective_from",
    "effective_to_exclusive",
    "source_record_id",
)
ALLOWED_TRADING_STATUSES = {
    "trading",
    "suspended",
    "halted",
    "no_trade",
    "delisted",
    "not_yet_listed",
}
ALLOWED_ACTION_TYPES = {
    "dividend",
    "split",
    "reverse_split",
    "rights",
    "spin_off",
    "merger",
    "symbol_change",
    "delisting",
}
REQUIRED_ATTESTATIONS = (
    "authoritative_session_calendar_complete",
    "trading_status_complete",
    "membership_exit_path_status_complete",
    "corporate_actions_complete",
    "symbol_history_complete",
    "adjustment_method_documented",
    "point_in_time_feature_safe",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(
    checks: list[dict[str, Any]],
    check: str,
    passed: bool,
    *,
    severity: str,
    observed: Any,
    requirement: str,
    details: str,
) -> None:
    checks.append(
        {
            "check": check,
            "status": "pass" if bool(passed) else "fail",
            "severity": severity,
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def _missing_columns(frame: pd.DataFrame | None, required: Sequence[str]) -> list[str]:
    if frame is None:
        return list(required)
    return sorted(set(required) - set(frame.columns))


def _normalize_key_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or not {"date", "ticker"}.issubset(frame.columns):
        return pd.DataFrame(columns=["date", "ticker"])
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"],
        errors="coerce",
    ).dt.normalize()
    normalized["ticker"] = (
        normalized["ticker"].astype(str).str.strip().str.upper()
    )
    return normalized


def _bangkok_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)


def _nonblank(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _coverage_row(
    metric: str,
    numerator: int,
    denominator: int,
    *,
    denominator_definition: str,
) -> dict[str, Any]:
    ratio = numerator / denominator if denominator else 0.0
    return {
        "metric": metric,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "coverage_ratio": float(ratio),
        "coverage_percent": float(ratio * 100.0),
        "denominator_definition": denominator_definition,
    }


def _manifest_checks(
    manifest: Mapping[str, Any] | None,
    checks: list[dict[str, Any]],
    *,
    binding_results: pd.DataFrame | None,
) -> None:
    value: Mapping[str, Any] = manifest or {}
    _check(
        checks,
        "manifest_schema",
        value.get("schema_version") == MANIFEST_SCHEMA_VERSION,
        severity="critical",
        observed=value.get("schema_version"),
        requirement=f"schema_version={MANIFEST_SCHEMA_VERSION}",
        details="The candidate package must declare the V3 licensed-market schema.",
    )
    identity_fields = ("package_id", "provider", "retrieved_at_utc")
    missing_identity = [
        field for field in identity_fields if not str(value.get(field, "")).strip()
    ]
    _check(
        checks,
        "manifest_identity",
        not missing_identity,
        severity="critical",
        observed={"missing": missing_identity},
        requirement=f"nonblank {list(identity_fields)}",
        details="Provider identity and retrieval time are required for provenance.",
    )

    license_value = value.get("license")
    license_mapping = license_value if isinstance(license_value, Mapping) else {}
    license_ok = (
        bool(str(license_mapping.get("reference", "")).strip())
        and license_mapping.get("research_use_allowed") is True
        and license_mapping.get("derived_publication_allowed") is True
    )
    _check(
        checks,
        "license_allows_research_and_derived_publication",
        license_ok,
        severity="critical",
        observed={
            "reference_present": bool(
                str(license_mapping.get("reference", "")).strip()
            ),
            "research_use_allowed": license_mapping.get(
                "research_use_allowed"
            ),
            "derived_publication_allowed": license_mapping.get(
                "derived_publication_allowed"
            ),
            "raw_redistribution_allowed": license_mapping.get(
                "raw_redistribution_allowed"
            ),
        },
        requirement=(
            "license reference plus research_use_allowed=true and "
            "derived_publication_allowed=true"
        ),
        details=(
            "Raw redistribution may remain forbidden, but the project must be "
            "allowed to analyze the data and publish derived findings."
        ),
    )

    attestations_value = value.get("attestations")
    attestations = (
        attestations_value
        if isinstance(attestations_value, Mapping)
        else {}
    )
    failed_attestations = [
        name for name in REQUIRED_ATTESTATIONS if attestations.get(name) is not True
    ]
    _check(
        checks,
        "manifest_completeness_attestations",
        not failed_attestations,
        severity="critical",
        observed={"not_true": failed_attestations},
        requirement=f"all declarations true: {list(REQUIRED_ATTESTATIONS)}",
        details=(
            "A missing file cannot be reinterpreted as no event, no suspension, "
            "or no symbol change."
        ),
    )

    binding_pass = False
    observed_bindings: dict[str, Any] = {"binding_results_present": False}
    if binding_results is not None and not binding_results.empty:
        observed_bindings = {
            "binding_results_present": True,
            "checks": int(len(binding_results)),
            "failed": binding_results.loc[
                binding_results["status"].ne("pass"),
                "check",
            ].astype(str).tolist(),
        }
        expected_checks = {
            "manifest_sidecar",
            *{f"binding_{label}" for label in REQUIRED_BINDINGS},
        }
        observed_checks = set(binding_results["check"].astype(str))
        binding_pass = (
            expected_checks.issubset(observed_checks)
            and binding_results["status"].eq("pass").all()
        )
    _check(
        checks,
        "manifest_bindings_verified",
        binding_pass,
        severity="critical",
        observed=observed_bindings,
        requirement=(
            "manifest sidecar and all five path/hash/byte/row bindings pass"
        ),
        details="Every input must be immutable and remain inside the project.",
    )


def verify_market_package_bindings(
    manifest_path: Path,
    *,
    project_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, pd.DataFrame],
    pd.DataFrame,
    list[Path],
]:
    """Load a candidate manifest and verify every bound CSV before assessment."""

    root = Path(project_root).resolve()
    path = Path(manifest_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("market package manifest is outside the project")
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("market package manifest must be a mapping")
    manifest = dict(loaded)
    rows: list[dict[str, Any]] = []
    referenced = [path]
    frames: dict[str, pd.DataFrame] = {}

    sidecar = Path(f"{path}.sha256")
    sidecar_ok = False
    sidecar_observed: Any = {"path": str(sidecar)}
    if sidecar.is_file():
        referenced.append(sidecar)
        tokens = sidecar.read_text(encoding="utf-8").split()
        actual = sha256_file(path)
        sidecar_ok = bool(tokens and tokens[0].lower() == actual)
        sidecar_observed = {
            "path": str(sidecar),
            "expected": tokens[0].lower() if tokens else None,
            "actual": actual,
        }
    _check(
        rows,
        "manifest_sidecar",
        sidecar_ok,
        severity="critical",
        observed=sidecar_observed,
        requirement="matching SHA-256 sidecar",
        details="The package manifest itself must be immutable.",
    )

    bindings_value = manifest.get("bindings")
    bindings = bindings_value if isinstance(bindings_value, Mapping) else {}
    for label in REQUIRED_BINDINGS:
        binding_value = bindings.get(label)
        binding = binding_value if isinstance(binding_value, Mapping) else {}
        raw_path = binding.get("path")
        bound_path: Path | None = None
        issue: str | None = None
        if not isinstance(raw_path, str) or not raw_path.strip():
            issue = "missing project-relative path"
        else:
            relative = Path(raw_path)
            candidate = (root / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not candidate.is_relative_to(root)
            ):
                issue = "path escapes the project"
            else:
                bound_path = candidate
        actual_hash = None
        actual_bytes = None
        actual_rows = None
        if issue is None and bound_path is not None:
            if not bound_path.is_file():
                issue = "bound file is missing"
            else:
                referenced.append(bound_path)
                try:
                    frame = pd.read_csv(bound_path)
                except Exception as exc:  # pragma: no cover - parser detail is provider-specific
                    issue = f"cannot parse CSV: {exc}"
                else:
                    frames[label] = frame
                    actual_hash = sha256_file(bound_path)
                    actual_bytes = bound_path.stat().st_size
                    actual_rows = len(frame)
        expected_hash = str(binding.get("sha256", "")).lower()
        expected_bytes = binding.get("bytes")
        expected_rows = binding.get("rows")
        passed = (
            issue is None
            and len(expected_hash) == 64
            and actual_hash == expected_hash
            and actual_bytes == expected_bytes
            and actual_rows == expected_rows
        )
        _check(
            rows,
            f"binding_{label}",
            passed,
            severity="critical",
            observed={
                "path": raw_path,
                "issue": issue,
                "sha256": actual_hash,
                "bytes": actual_bytes,
                "rows": actual_rows,
            },
            requirement={
                "sha256": expected_hash,
                "bytes": expected_bytes,
                "rows": expected_rows,
            },
            details=f"The {label} file must match its manifest binding exactly.",
        )
    return manifest, frames, pd.DataFrame(rows), sorted(set(referenced))


def assess_v3_market_package(
    bars: pd.DataFrame,
    membership_panel: pd.DataFrame,
    acquisition_universe: pd.DataFrame,
    *,
    trading_status: pd.DataFrame | None = None,
    sessions: pd.DataFrame | None = None,
    corporate_actions: pd.DataFrame | None = None,
    symbol_history: pd.DataFrame | None = None,
    manifest: Mapping[str, Any] | None = None,
    binding_results: pd.DataFrame | None = None,
    generated_at_utc: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Assess a candidate package without converting missing evidence into errors."""

    membership = validate_membership_panel(membership_panel)
    if "ticker" not in acquisition_universe.columns:
        raise ValueError("acquisition universe is missing ticker")
    universe = acquisition_universe.copy()
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    if universe["ticker"].eq("").any() or universe["ticker"].duplicated().any():
        raise ValueError("acquisition universe contains blank or duplicate tickers")
    expected_tickers = set(universe["ticker"])
    membership_tickers = set(membership["ticker"].astype(str))
    if membership_tickers != expected_tickers:
        raise ValueError("membership panel and acquisition universe disagree")

    checks: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    active = membership[membership["is_index_member"]][
        ["date", "ticker", "sector_as_of"]
    ].copy()
    active_keys = active[["date", "ticker"]]
    active_key_count = len(active_keys)
    expected_dates = set(membership["date"].drop_duplicates())

    missing_bar_columns = _missing_columns(bars, BAR_COLUMNS)
    _check(
        checks,
        "bar_required_columns",
        not missing_bar_columns,
        severity="critical",
        observed={"missing": missing_bar_columns},
        requirement=f"columns include {list(BAR_COLUMNS)}",
        details=(
            "Adjusted close, availability, and source identity are required in "
            "addition to canonical OHLCV."
        ),
    )
    normalized_bars = _normalize_key_frame(bars)
    valid_bar_keys = normalized_bars[
        normalized_bars["date"].notna() & normalized_bars["ticker"].ne("")
    ][["date", "ticker"]].drop_duplicates()
    bar_duplicates = (
        int(normalized_bars.duplicated(["date", "ticker"]).sum())
        if {"date", "ticker"}.issubset(normalized_bars.columns)
        else 0
    )
    _check(
        checks,
        "bar_unique_daily_grain",
        bar_duplicates == 0,
        severity="critical",
        observed=bar_duplicates,
        requirement="0 duplicate date/ticker rows",
        details="Silent deduplication can choose a later revision and bias returns.",
    )
    observed_bar_tickers = set(valid_bar_keys["ticker"])
    missing_tickers = sorted(expected_tickers - observed_bar_tickers)
    unknown_tickers = sorted(observed_bar_tickers - expected_tickers)
    _check(
        checks,
        "bar_acquisition_ticker_contract",
        not missing_tickers and not unknown_tickers,
        severity="critical",
        observed={
            "observed": len(observed_bar_tickers),
            "expected": len(expected_tickers),
            "missing_count": len(missing_tickers),
            "missing_sample": missing_tickers[:15],
            "unknown": unknown_tickers[:15],
        },
        requirement="observed ticker set equals the checksum-bound acquisition union",
        details=(
            "The acquisition union is a coverage contract, not a static "
            "membership mask."
        ),
    )
    coverage.append(
        _coverage_row(
            "acquisition_ticker_coverage",
            len(expected_tickers & observed_bar_tickers),
            len(expected_tickers),
            denominator_definition="checksum-bound acquisition tickers",
        )
    )

    active_bar_join = active.merge(
        valid_bar_keys.assign(_bar_present=True),
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    active_bar_count = int(active_bar_join["_bar_present"].eq(True).sum())
    coverage.append(
        _coverage_row(
            "active_member_bar_coverage",
            active_bar_count,
            active_key_count,
            denominator_definition="point-in-time active SET100 session keys",
        )
    )

    covered_active = active_bar_join[
        active_bar_join["_bar_present"].eq(True)
    ].copy()
    if covered_active.empty:
        peer_covered_count = 0
    else:
        sector_counts = covered_active.groupby(
            ["date", "sector_as_of"],
            dropna=False,
        )["ticker"].transform("nunique")
        peer_covered_count = int(sector_counts.ge(2).sum())
    coverage.append(
        _coverage_row(
            "active_bar_sector_peer_coverage",
            peer_covered_count,
            active_bar_count,
            denominator_definition=(
                "active member/session keys with a bar; peer means at least one "
                "other covered active member in the dated sector"
            ),
        )
    )

    bar_dates = set(valid_bar_keys["date"].dropna())
    provisional_extra_bar_dates = sorted(bar_dates - expected_dates)

    if not missing_bar_columns:
        for column in ("open", "high", "low", "close", "adjusted_close", "volume"):
            normalized_bars[column] = pd.to_numeric(
                normalized_bars[column],
                errors="coerce",
            )
        price_columns = ["open", "high", "low", "close", "adjusted_close"]
        numeric = normalized_bars[[*price_columns, "volume"]].to_numpy(
            dtype=np.float64
        )
        invalid_numeric = int((~np.isfinite(numeric)).sum())
        nonpositive_prices = int(
            normalized_bars[price_columns].le(0.0).any(axis=1).sum()
        )
        negative_volume = int(normalized_bars["volume"].lt(0.0).sum())
        _check(
            checks,
            "bar_numeric_domain",
            (
                invalid_numeric == 0
                and nonpositive_prices == 0
                and negative_volume == 0
            ),
            severity="critical",
            observed={
                "nonfinite_values": invalid_numeric,
                "nonpositive_price_rows": nonpositive_prices,
                "negative_volume_rows": negative_volume,
            },
            requirement="finite positive prices and nonnegative volume",
            details="Zero volume is permitted; invalid price levels are not.",
        )
        tolerance = 1e-8
        envelope_violations = int(
            (
                (
                    normalized_bars["high"] + tolerance
                    < normalized_bars[["open", "close"]].max(axis=1)
                )
                | (
                    normalized_bars["low"] - tolerance
                    > normalized_bars[["open", "close"]].min(axis=1)
                )
                | (
                    normalized_bars["low"]
                    > normalized_bars["high"] + tolerance
                )
            ).sum()
        )
        _check(
            checks,
            "bar_ohlc_envelope",
            envelope_violations == 0,
            severity="critical",
            observed=envelope_violations,
            requirement="low <= open/close <= high for every bar",
            details="Impossible OHLC envelopes corrupt execution and return logic.",
        )
        bar_information = _bangkok_timestamp(
            normalized_bars["bar_available_at"]
        )
        availability_invalid = int(bar_information.isna().sum())
        availability_wrong_date = int(
            (
                bar_information.dt.normalize()
                != normalized_bars["date"]
            ).fillna(True).sum()
        )
        _check(
            checks,
            "bar_point_in_time_availability",
            availability_invalid == 0 and availability_wrong_date == 0,
            severity="critical",
            observed={
                "invalid": availability_invalid,
                "not_same_session": availability_wrong_date,
            },
            requirement="every daily bar is first available on its own SET session",
            details=(
                "Date-level availability supports after-close/later-decision use; "
                "it does not authorize same-close execution."
            ),
        )
        blank_source_ids = int(
            (~_nonblank(normalized_bars["source_record_id"])).sum()
        )
        _check(
            checks,
            "bar_source_identity",
            blank_source_ids == 0,
            severity="high",
            observed=blank_source_ids,
            requirement="0 blank source_record_id values",
            details="Every bar must be traceable to a provider record.",
        )
    else:
        for check_id, requirement, details in (
            (
                "bar_numeric_domain",
                "finite positive prices and nonnegative volume",
                "Not evaluated because the bar schema is incomplete.",
            ),
            (
                "bar_ohlc_envelope",
                "low <= open/close <= high for every bar",
                "Not evaluated because the bar schema is incomplete.",
            ),
            (
                "bar_point_in_time_availability",
                "every daily bar is first available on its own SET session",
                "Not evaluated because the availability field is missing.",
            ),
            (
                "bar_source_identity",
                "0 blank source_record_id values",
                "Not evaluated because the source identity field is missing.",
            ),
        ):
            _check(
                checks,
                check_id,
                False,
                severity="critical" if check_id != "bar_source_identity" else "high",
                observed={"blocked_by": "bar_required_columns"},
                requirement=requirement,
                details=details,
            )

    missing_status_columns = _missing_columns(trading_status, STATUS_COLUMNS)
    _check(
        checks,
        "status_required_columns",
        not missing_status_columns,
        severity="critical",
        observed={"missing": missing_status_columns},
        requirement=f"columns include {list(STATUS_COLUMNS)}",
        details="Trading state must be explicit for every active session.",
    )
    normalized_status = _normalize_key_frame(trading_status)
    status_duplicates = (
        int(normalized_status.duplicated(["date", "ticker"]).sum())
        if {"date", "ticker"}.issubset(normalized_status.columns)
        else 0
    )
    _check(
        checks,
        "status_unique_daily_grain",
        not missing_status_columns and status_duplicates == 0,
        severity="critical",
        observed=(
            status_duplicates
            if not missing_status_columns
            else {"blocked_by": "status_required_columns"}
        ),
        requirement="0 duplicate date/ticker status rows",
        details="Conflicting daily states cannot be resolved silently.",
    )
    status_keys = normalized_status[
        normalized_status["date"].notna() & normalized_status["ticker"].ne("")
    ][["date", "ticker"]].drop_duplicates()
    active_status_join = active_keys.merge(
        status_keys.assign(_status_present=True),
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    active_status_count = int(
        active_status_join["_status_present"].eq(True).sum()
    )
    _check(
        checks,
        "active_member_status_coverage",
        active_status_count == active_key_count,
        severity="critical",
        observed={
            "covered": active_status_count,
            "required": active_key_count,
            "coverage_ratio": (
                active_status_count / active_key_count
                if active_key_count
                else 0.0
            ),
        },
        requirement="100% of active member/session keys have explicit status",
        details="Missing status cannot be inferred from a missing bar.",
    )
    coverage.append(
        _coverage_row(
            "active_member_status_coverage",
            active_status_count,
            active_key_count,
            denominator_definition="point-in-time active SET100 session keys",
        )
    )

    if not missing_status_columns:
        normalized_status["trading_status"] = (
            normalized_status["trading_status"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        unknown_status_tickers = sorted(
            set(normalized_status["ticker"]) - expected_tickers
        )
        _check(
            checks,
            "status_ticker_and_session_contract",
            not unknown_status_tickers,
            severity="critical",
            observed={
                "unknown_tickers": unknown_status_tickers[:15],
            },
            requirement="status rows use only acquisition-universe tickers",
            details=(
                "Session membership is checked separately against the licensed "
                "calendar so provider warm-up rows can be retained."
            ),
        )
        invalid_status = sorted(
            set(normalized_status["trading_status"])
            - ALLOWED_TRADING_STATUSES
        )
        _check(
            checks,
            "status_allowed_values",
            not invalid_status,
            severity="critical",
            observed=invalid_status,
            requirement=f"statuses in {sorted(ALLOWED_TRADING_STATUSES)}",
            details="Status categories have locked, explicit meanings.",
        )
        status_information = _bangkok_timestamp(
            normalized_status["status_available_at"]
        )
        wrong_status_date = int(
            (
                status_information.dt.normalize()
                != normalized_status["date"]
            ).fillna(True).sum()
        )
        blank_status_sources = int(
            (~_nonblank(normalized_status["source_record_id"])).sum()
        )
        _check(
            checks,
            "status_point_in_time_availability",
            wrong_status_date == 0,
            severity="critical",
            observed=wrong_status_date,
            requirement="every daily status is first available on its own session",
            details="Later corrections cannot define an earlier trading decision.",
        )
        _check(
            checks,
            "status_source_identity",
            blank_status_sources == 0,
            severity="high",
            observed=blank_status_sources,
            requirement="0 blank source_record_id values",
            details="Every status must be traceable to a provider record.",
        )

        active_states = active_keys.merge(
            normalized_status[
                ["date", "ticker", "trading_status"]
            ],
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
        )
        impossible_active = int(
            active_states["trading_status"]
            .isin({"not_yet_listed", "delisted"})
            .sum()
        )
        _check(
            checks,
            "active_member_status_consistency",
            impossible_active == 0,
            severity="critical",
            observed=impossible_active,
            requirement="active members are not not_yet_listed or delisted",
            details="Membership and instrument state must agree on each session.",
        )
        trading_active = active_states[
            active_states["trading_status"].eq("trading")
        ][["date", "ticker"]]
        trading_bar_join = trading_active.merge(
            valid_bar_keys.assign(_bar_present=True),
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
        )
        trading_bar_count = int(
            trading_bar_join["_bar_present"].eq(True).sum()
        )
        _check(
            checks,
            "trading_status_requires_bar",
            trading_bar_count == len(trading_active),
            severity="critical",
            observed={
                "covered": trading_bar_count,
                "trading_keys": len(trading_active),
            },
            requirement="100% of active trading statuses have a valid bar row",
            details="A security marked trading cannot lack its market bar.",
        )
        active_market = active_states.merge(
            valid_bar_keys.assign(_bar_present=True),
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
        )
        explicit_nontrading = active_market["trading_status"].isin(
            {"suspended", "halted", "no_trade"}
        )
        market_observation_covered = (
            (
                active_market["trading_status"].eq("trading")
                & active_market["_bar_present"].eq(True)
            )
            | explicit_nontrading
        )
        market_observation_count = int(market_observation_covered.sum())
        _check(
            checks,
            "active_member_market_observation_coverage",
            market_observation_count == active_key_count,
            severity="critical",
            observed={
                "covered": market_observation_count,
                "required": active_key_count,
            },
            requirement=(
                "100% of active keys have either a trading bar or explicit "
                "suspended/halted/no_trade status"
            ),
            details=(
                "A missing bar is acceptable only when an authoritative "
                "non-trading state explains it."
            ),
        )
        coverage.append(
            _coverage_row(
                "active_trading_status_bar_coverage",
                trading_bar_count,
                len(trading_active),
                denominator_definition="active keys explicitly marked trading",
            )
        )
    else:
        for check_id in (
            "status_allowed_values",
            "status_ticker_and_session_contract",
            "status_point_in_time_availability",
            "status_source_identity",
            "active_member_status_consistency",
            "trading_status_requires_bar",
            "active_member_market_observation_coverage",
        ):
            _check(
                checks,
                check_id,
                False,
                severity="critical",
                observed={"blocked_by": "status_required_columns"},
                requirement="complete, valid, point-in-time trading status",
                details="Not evaluated because the status package is absent or incomplete.",
            )
        coverage.append(
            _coverage_row(
                "active_trading_status_bar_coverage",
                0,
                active_key_count,
                denominator_definition=(
                    "active keys; exact trading denominator unavailable because "
                    "status data is missing"
                ),
            )
        )

    missing_session_columns = _missing_columns(sessions, SESSION_COLUMNS)
    _check(
        checks,
        "session_required_columns",
        not missing_session_columns,
        severity="critical",
        observed={"missing": missing_session_columns},
        requirement=f"columns include {list(SESSION_COLUMNS)}",
        details="Observed provider dates cannot substitute for an authoritative calendar.",
    )
    session_dates: set[pd.Timestamp] = set()
    if not missing_session_columns and sessions is not None:
        normalized_sessions = sessions.copy()
        normalized_sessions["date"] = pd.to_datetime(
            normalized_sessions["date"],
            errors="coerce",
        ).dt.normalize()
        session_duplicates = int(normalized_sessions["date"].duplicated().sum())
        invalid_session_dates = int(normalized_sessions["date"].isna().sum())
        session_dates = set(normalized_sessions["date"].dropna())
        calendar_match = (
            session_duplicates == 0
            and invalid_session_dates == 0
            and expected_dates.issubset(session_dates)
        )
        _check(
            checks,
            "authoritative_session_calendar_match",
            calendar_match,
            severity="critical",
            observed={
                "session_count": len(session_dates),
                "membership_session_count": len(expected_dates),
                "duplicates": session_duplicates,
                "invalid_dates": invalid_session_dates,
                "missing_from_package": len(expected_dates - session_dates),
                "extra_buffer_sessions": len(session_dates - expected_dates),
            },
            requirement="licensed sessions cover every membership date",
            details=(
                "Extra authoritative buffer sessions are allowed. Missing "
                "membership sessions require rebuilding, not dropping rows."
            ),
        )
        session_information = _bangkok_timestamp(
            normalized_sessions["session_available_at"]
        )
        future_calendar = int(
            (
                session_information.dt.normalize()
                > normalized_sessions["date"]
            ).fillna(True).sum()
        )
        blank_session_sources = int(
            (~_nonblank(normalized_sessions["source_record_id"])).sum()
        )
        _check(
            checks,
            "session_provenance_and_availability",
            future_calendar == 0 and blank_session_sources == 0,
            severity="critical",
            observed={
                "future_availability": future_calendar,
                "blank_source_ids": blank_session_sources,
            },
            requirement="calendar is known by each session and source IDs are complete",
            details="Calendar provenance must not be reconstructed from later bars.",
        )
        bars_outside_sessions = len(bar_dates - session_dates)
        _check(
            checks,
            "bars_on_authoritative_sessions",
            bars_outside_sessions == 0,
            severity="critical",
            observed=bars_outside_sessions,
            requirement="0 bar dates outside the licensed session calendar",
            details="Off-calendar rows need provider resolution before use.",
        )
        status_dates = set(normalized_status["date"].dropna())
        statuses_outside_sessions = len(status_dates - session_dates)
        _check(
            checks,
            "statuses_on_authoritative_sessions",
            not missing_status_columns and statuses_outside_sessions == 0,
            severity="critical",
            observed=(
                statuses_outside_sessions
                if not missing_status_columns
                else {"blocked_by": "status_required_columns"}
            ),
            requirement="0 status dates outside the licensed session calendar",
            details="Status rows may include buffer sessions but not off-calendar dates.",
        )
        coverage.append(
            _coverage_row(
                "membership_session_calendar_coverage",
                len(expected_dates & session_dates),
                len(expected_dates),
                denominator_definition="membership panel session dates",
            )
        )
    else:
        for check_id in (
            "authoritative_session_calendar_match",
            "session_provenance_and_availability",
            "bars_on_authoritative_sessions",
            "statuses_on_authoritative_sessions",
        ):
            _check(
                checks,
                check_id,
                False,
                severity="critical",
                observed={"blocked_by": "session_required_columns"},
                requirement="complete authoritative SET session package",
                details="Not evaluated because the session calendar is absent.",
            )
        coverage.append(
            _coverage_row(
                "membership_session_calendar_coverage",
                0,
                len(expected_dates),
                denominator_definition="membership panel session dates",
            )
        )

    missing_action_columns = _missing_columns(
        corporate_actions,
        CORPORATE_ACTION_COLUMNS,
    )
    _check(
        checks,
        "corporate_action_required_columns",
        not missing_action_columns,
        severity="critical",
        observed={"missing": missing_action_columns},
        requirement=f"columns include {list(CORPORATE_ACTION_COLUMNS)}",
        details="An empty but complete action table must still carry the full schema.",
    )
    if not missing_action_columns and corporate_actions is not None:
        actions = corporate_actions.copy()
        actions["ticker"] = actions["ticker"].astype(str).str.strip().str.upper()
        actions["action_type"] = (
            actions["action_type"].astype(str).str.strip().str.lower()
        )
        actions["announcement_at"] = _bangkok_timestamp(
            actions["announcement_at"]
        )
        actions["effective_date"] = pd.to_datetime(
            actions["effective_date"],
            errors="coerce",
        ).dt.normalize()
        actions["adjustment_factor"] = pd.to_numeric(
            actions["adjustment_factor"],
            errors="coerce",
        )
        actions["cash_amount"] = pd.to_numeric(
            actions["cash_amount"],
            errors="coerce",
        )
        duplicate_actions = int(actions["action_id"].duplicated().sum())
        invalid_action_types = sorted(
            set(actions["action_type"]) - ALLOWED_ACTION_TYPES
        )
        unknown_action_tickers = sorted(
            set(actions["ticker"]) - expected_tickers
        )
        invalid_dates = int(
            (
                actions["announcement_at"].isna()
                | actions["effective_date"].isna()
                | (
                    actions["announcement_at"].dt.normalize()
                    > actions["effective_date"]
                )
            ).sum()
        )
        factor_required = actions["action_type"].isin(
            {"split", "reverse_split", "rights", "spin_off"}
        )
        invalid_factors = int(
            (
                factor_required
                & (
                    actions["adjustment_factor"].isna()
                    | actions["adjustment_factor"].le(0.0)
                )
            ).sum()
        )
        invalid_cash = int(actions["cash_amount"].dropna().lt(0.0).sum())
        successor_required = actions["action_type"].isin(
            {"merger", "symbol_change"}
        )
        missing_successor = int(
            (successor_required & ~_nonblank(actions["successor_ticker"])).sum()
        )
        blank_action_sources = int(
            (~_nonblank(actions["source_record_id"])).sum()
        )
        action_ok = (
            duplicate_actions == 0
            and not invalid_action_types
            and not unknown_action_tickers
            and invalid_dates == 0
            and invalid_factors == 0
            and invalid_cash == 0
            and missing_successor == 0
            and blank_action_sources == 0
        )
        _check(
            checks,
            "corporate_action_domain_and_lineage",
            action_ok,
            severity="critical",
            observed={
                "rows": len(actions),
                "duplicate_action_ids": duplicate_actions,
                "invalid_types": invalid_action_types,
                "unknown_tickers": unknown_action_tickers[:15],
                "invalid_dates": invalid_dates,
                "invalid_factors": invalid_factors,
                "negative_cash_amounts": invalid_cash,
                "missing_successors": missing_successor,
                "blank_source_ids": blank_action_sources,
            },
            requirement="unique, valid, point-in-time corporate actions with lineage",
            details="Action-free history is accepted only as an explicit empty table plus attestation.",
        )
    else:
        _check(
            checks,
            "corporate_action_domain_and_lineage",
            False,
            severity="critical",
            observed={"blocked_by": "corporate_action_required_columns"},
            requirement="unique, valid, point-in-time corporate actions with lineage",
            details="Not evaluated because the corporate-action package is absent.",
        )

    missing_symbol_columns = _missing_columns(
        symbol_history,
        SYMBOL_HISTORY_COLUMNS,
    )
    _check(
        checks,
        "symbol_history_required_columns",
        not missing_symbol_columns,
        severity="critical",
        observed={"missing": missing_symbol_columns},
        requirement=f"columns include {list(SYMBOL_HISTORY_COLUMNS)}",
        details="Stable instrument IDs prevent ticker changes from breaking continuity.",
    )
    if not missing_symbol_columns and symbol_history is not None:
        symbols = symbol_history.copy()
        symbols["ticker"] = symbols["ticker"].astype(str).str.strip().str.upper()
        symbols["effective_from"] = pd.to_datetime(
            symbols["effective_from"],
            errors="coerce",
        ).dt.normalize()
        symbols["effective_to_exclusive"] = pd.to_datetime(
            symbols["effective_to_exclusive"],
            errors="coerce",
        ).dt.normalize()
        invalid_intervals = int(
            (
                symbols["effective_from"].isna()
                | symbols["effective_to_exclusive"].isna()
                | (
                    symbols["effective_from"]
                    >= symbols["effective_to_exclusive"]
                )
            ).sum()
        )
        blank_instruments = int((~_nonblank(symbols["instrument_id"])).sum())
        blank_symbol_sources = int(
            (~_nonblank(symbols["source_record_id"])).sum()
        )
        symbol_tickers = set(symbols["ticker"])
        ticker_contract = symbol_tickers == expected_tickers
        overlaps = 0
        for _, group in symbols.sort_values(
            ["instrument_id", "effective_from"]
        ).groupby("instrument_id", sort=False):
            if len(group) > 1:
                previous_end = group["effective_to_exclusive"].shift(1)
                overlaps += int(
                    (group["effective_from"] < previous_end).fillna(False).sum()
                )
        active_symbol = active_keys.merge(
            symbols[
                [
                    "ticker",
                    "effective_from",
                    "effective_to_exclusive",
                ]
            ],
            on="ticker",
            how="left",
        )
        active_symbol["_covered"] = (
            active_symbol["effective_from"].notna()
            & active_symbol["effective_to_exclusive"].notna()
            & (active_symbol["date"] >= active_symbol["effective_from"])
            & (
                active_symbol["date"]
                < active_symbol["effective_to_exclusive"]
            )
        )
        symbol_covered_keys = int(
            active_symbol.groupby(["date", "ticker"])["_covered"].any().sum()
        )
        symbol_ok = (
            invalid_intervals == 0
            and blank_instruments == 0
            and blank_symbol_sources == 0
            and ticker_contract
            and overlaps == 0
            and symbol_covered_keys == active_key_count
        )
        _check(
            checks,
            "symbol_history_continuity",
            symbol_ok,
            severity="critical",
            observed={
                "rows": len(symbols),
                "ticker_count": len(symbol_tickers),
                "expected_tickers": len(expected_tickers),
                "invalid_intervals": invalid_intervals,
                "overlaps": overlaps,
                "blank_instrument_ids": blank_instruments,
                "blank_source_ids": blank_symbol_sources,
                "active_keys_covered": symbol_covered_keys,
                "active_keys_required": active_key_count,
            },
            requirement=(
                "exact ticker union, non-overlapping stable-ID intervals, and "
                "100% active-key coverage"
            ),
            details="Symbol history must cover entries, exits, and successor symbols.",
        )
    else:
        _check(
            checks,
            "symbol_history_continuity",
            False,
            severity="critical",
            observed={"blocked_by": "symbol_history_required_columns"},
            requirement="complete, non-overlapping stable instrument history",
            details="Not evaluated because symbol history is absent.",
        )

    exit_ledger = assess_membership_exit_paths(
        membership,
        bars=bars,
        trading_status=trading_status,
        sessions=sessions,
        corporate_actions=corporate_actions,
        symbol_history=symbol_history,
    )
    exit_transition_count = len(exit_ledger)
    covered_exit_count = (
        int(exit_ledger["path_covered"].astype(bool).sum())
        if exit_transition_count
        else 0
    )
    exit_blockers = (
        exit_ledger.loc[
            ~exit_ledger["path_covered"].astype(bool),
            ["ticker", "exit_effective_date", "blocking_reason"],
        ]
        .head(15)
        .assign(
            exit_effective_date=lambda frame: frame[
                "exit_effective_date"
            ].astype(str)
        )
        .to_dict(orient="records")
        if exit_transition_count
        else []
    )
    _check(
        checks,
        "membership_exit_path_coverage",
        covered_exit_count == exit_transition_count,
        severity="critical",
        observed={
            "covered": covered_exit_count,
            "required": exit_transition_count,
            "blocking_sample": exit_blockers,
        },
        requirement=(
            "every observed membership exit has contiguous status through a "
            "valid liquidation bar or explicit terminal settlement"
        ),
        details=(
            "Leaving SET100 does not itself execute a sale. Suspended or halted "
            "holdings must remain pending until observable execution is possible."
        ),
    )
    coverage.append(
        _coverage_row(
            "membership_exit_path_coverage",
            covered_exit_count,
            exit_transition_count,
            denominator_definition=(
                "first non-member sessions following observed active membership"
            ),
        )
    )

    _manifest_checks(
        manifest,
        checks,
        binding_results=binding_results,
    )

    checks_frame = pd.DataFrame(checks)
    coverage_frame = pd.DataFrame(coverage)
    failed = checks_frame.loc[
        checks_frame["status"].ne("pass"),
        ["check", "severity"],
    ]
    package_eligible = failed.empty
    generated = generated_at_utc or datetime.now(timezone.utc).isoformat()
    receipt = {
        "schema_version": 1,
        "generated_at_utc": generated,
        "status": "pass" if package_eligible else "fail",
        "evidence_class": (
            "licensed_market_package"
            if package_eligible
            else "v3_market_package_gap_assessment"
        ),
        "metrics": {
            "acquisition_tickers_expected": len(expected_tickers),
            "bar_tickers_observed": len(observed_bar_tickers),
            "membership_sessions": len(expected_dates),
            "active_membership_keys": active_key_count,
            "active_bar_keys": active_bar_count,
            "active_status_keys": active_status_count,
            "bar_dates_outside_development_membership_grid": len(
                provisional_extra_bar_dates
            ),
            "active_bar_keys_with_sector_peer": peer_covered_count,
            "membership_exit_transitions": exit_transition_count,
            "membership_exit_paths_covered": covered_exit_count,
            "checks_total": len(checks_frame),
            "checks_passed": int(checks_frame["status"].eq("pass").sum()),
            "checks_failed": int(checks_frame["status"].ne("pass").sum()),
        },
        "coverage": {
            row["metric"]: {
                "numerator": int(row["numerator"]),
                "denominator": int(row["denominator"]),
                "coverage_ratio": float(row["coverage_ratio"]),
            }
            for row in coverage
        },
        "blocking_checks": failed.to_dict(orient="records"),
        "licensed_market_package_eligible": package_eligible,
        "final_investability_eligible": False,
        "model_input_eligible": False,
        "training_enabled": False,
        "publication_claim_eligible": False,
        "hpc_jobs_submitted": 0,
        "scientific_boundary": (
            "A passing intake certifies the licensed market package only. "
            "Final investability, model input, training, holdout access, and "
            "publication claims remain blocked until the execution policy and "
            "V3 protocol are author-approved, checksum-locked, and separately released."
        ),
    }
    return receipt, checks_frame, coverage_frame


def checks_for_csv(checks: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic, flat check table suitable for CSV evidence."""

    flat = checks.copy()
    flat["observed"] = flat["observed"].map(
        lambda value: json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if isinstance(value, (dict, list, tuple))
        else str(value)
    )
    flat["requirement"] = flat["requirement"].map(
        lambda value: json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if isinstance(value, (dict, list, tuple))
        else str(value)
    )
    return flat
