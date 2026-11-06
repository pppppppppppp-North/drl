"""Leakage-safe development eligibility derived from membership and past data.

The output is suitable for exercising the isolated V3 environment, but it is
not a final investability assertion.  Observable suspension/trading-status and
corporate-action rules still have to be supplied by a licensed market source.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.v3.membership_mask import validate_membership_panel


DEVELOPMENT_ELIGIBILITY_KIND = (
    "v3_development_eligibility_not_final_investability"
)
REQUIRED_COLUMNS = (
    "date",
    "asset_slot",
    "ticker",
    "is_index_member",
    "has_observed_bar",
    "has_minimum_history",
    "is_eligible",
    "information_available_at",
    "mask_kind",
)


def _coerce_boolean(series: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized) - {"true", "false"})
    if unexpected:
        raise ValueError(
            f"eligibility column {column} contains non-Boolean values: "
            + ", ".join(unexpected)
        )
    return normalized.eq("true")


def validate_development_eligibility_mask(
    mask: pd.DataFrame,
    *,
    expected_dates: Sequence[str | pd.Timestamp] | None = None,
    expected_tickers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate an exact, past-only development eligibility grid."""

    missing = [column for column in REQUIRED_COLUMNS if column not in mask.columns]
    if missing:
        raise ValueError("eligibility mask is missing columns: " + ", ".join(missing))
    checked = mask.copy()
    checked["date"] = pd.to_datetime(checked["date"], errors="coerce").dt.normalize()
    checked["information_available_at"] = pd.to_datetime(
        checked["information_available_at"],
        errors="coerce",
    ).dt.normalize()
    if checked[["date", "information_available_at"]].isna().any().any():
        raise ValueError("eligibility mask contains invalid dates")
    for column in (
        "is_index_member",
        "has_observed_bar",
        "has_minimum_history",
        "is_eligible",
    ):
        checked[column] = _coerce_boolean(checked[column], column=column)

    if checked.duplicated(["date", "ticker"]).any():
        raise ValueError("eligibility mask has duplicate date/ticker rows")
    if checked["mask_kind"].nunique() != 1 or (
        checked["mask_kind"].iloc[0] != DEVELOPMENT_ELIGIBILITY_KIND
    ):
        raise ValueError(
            "eligibility mask must declare the isolated V3 development kind"
        )
    if (checked["information_available_at"] > checked["date"]).any():
        raise ValueError("eligibility mask contains future information")

    expected = (
        checked["is_index_member"]
        & checked["has_observed_bar"]
        & checked["has_minimum_history"]
    )
    if not checked["is_eligible"].equals(expected):
        raise ValueError(
            "is_eligible must equal membership AND current bar AND past history"
        )

    dates = pd.DatetimeIndex(checked["date"].drop_duplicates().sort_values())
    if expected_dates is not None:
        requested_dates = pd.DatetimeIndex(
            pd.to_datetime(list(expected_dates), errors="coerce")
        ).normalize().drop_duplicates().sort_values()
        if requested_dates.isna().any() or not dates.equals(requested_dates):
            raise ValueError("eligibility mask date grid does not match the request")

    slot_table = (
        checked[["asset_slot", "ticker"]]
        .drop_duplicates()
        .sort_values("asset_slot")
    )
    if slot_table["asset_slot"].duplicated().any() or slot_table["ticker"].duplicated().any():
        raise ValueError("eligibility mask has an unstable asset-slot mapping")
    tickers = tuple(slot_table["ticker"].astype(str))
    if expected_tickers is not None:
        requested_tickers = tuple(str(value).strip().upper() for value in expected_tickers)
        if tickers != requested_tickers:
            raise ValueError("eligibility mask ticker slots do not match the request")

    expected_rows = len(dates) * len(tickers)
    if len(checked) != expected_rows:
        raise ValueError(
            f"eligibility mask has {len(checked)} rows, expected {expected_rows}"
        )
    expected_slots = list(range(len(tickers)))
    for _, group in checked.groupby("date", sort=False):
        if group.sort_values("asset_slot")["asset_slot"].tolist() != expected_slots:
            raise ValueError("eligibility mask has an incomplete date/ticker grid")

    return checked.sort_values(["date", "asset_slot"], kind="mergesort").reset_index(
        drop=True
    )


def build_development_eligibility_mask(
    membership_panel: pd.DataFrame,
    market_rows: pd.DataFrame,
    *,
    lookback: int,
    price_columns: Sequence[str] = ("open", "high", "low", "close"),
    volume_column: str = "volume",
) -> pd.DataFrame:
    """Combine membership with current and strictly historical market coverage.

    The history condition uses only the ``lookback`` sessions *before* each
    decision date.  Next-session price availability never determines today's
    eligibility.
    """

    if lookback <= 0:
        raise ValueError("lookback must be positive")
    membership = validate_membership_panel(membership_panel)
    required_market = {"date", "ticker", *price_columns, volume_column}
    missing = sorted(required_market - set(market_rows.columns))
    if missing:
        raise ValueError(f"market rows are missing columns: {missing}")

    market = market_rows.copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["ticker"] = market["ticker"].astype(str).str.strip().str.upper()
    if market["date"].isna().any():
        raise ValueError("market rows contain invalid dates")
    if market.duplicated(["date", "ticker"]).any():
        raise ValueError("market rows have duplicate date/ticker rows")

    numeric_columns = [*price_columns, volume_column]
    for column in numeric_columns:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    merged = membership.merge(
        market[["date", "ticker", *numeric_columns]],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    price_values = merged[list(price_columns)].to_numpy(dtype=np.float64)
    volume_values = merged[volume_column].to_numpy(dtype=np.float64)
    merged["has_observed_bar"] = (
        np.isfinite(price_values).all(axis=1)
        & (price_values > 0.0).all(axis=1)
        & np.isfinite(volume_values)
        & (volume_values >= 0.0)
    )

    merged = merged.sort_values(["asset_slot", "date"], kind="mergesort")
    prior_history = (
        merged.groupby("ticker", sort=False)["has_observed_bar"]
        .transform(
            lambda values: values.shift(1)
            .rolling(window=lookback, min_periods=lookback)
            .sum()
        )
    )
    merged["has_minimum_history"] = prior_history.eq(float(lookback))
    merged["is_eligible"] = (
        merged["is_index_member"]
        & merged["has_observed_bar"]
        & merged["has_minimum_history"]
    )
    merged["information_available_at"] = merged["date"]
    merged["mask_kind"] = DEVELOPMENT_ELIGIBILITY_KIND

    return validate_development_eligibility_mask(
        merged,
        expected_dates=membership["date"].drop_duplicates().tolist(),
        expected_tickers=tuple(
            membership.sort_values("asset_slot")
            .drop_duplicates("asset_slot")["ticker"]
            .astype(str)
        ),
    )

