"""Point-in-time sector context for the isolated V3 development path.

The shared V2 sector helper intentionally remains unchanged.  This module uses
the dated ``sector_as_of`` labels in the validated membership panel, excludes
inactive acquisition-universe rows from every aggregate, and requires an
explicit observed-return flag.  Its close-based values are available only
after the corresponding session and are intended for a later decision.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.v3.membership_mask import validate_membership_panel


SECTOR_CONTEXT_KIND = "v3_active_member_point_in_time_sector_context"
SECTOR_CONTEXT_TIMING_RULE = "after_close_same_session_for_later_decision"

OUTPUT_COLUMNS = (
    "date",
    "asset_slot",
    "ticker",
    "is_index_member",
    "sector_as_of",
    "asset_return_1d",
    "return_1d_is_observed",
    "return_information_available_at",
    "sector_active_member_count",
    "sector_return_contributor_count",
    "sector_return_coverage",
    "sector_active_equal_weight_return_1d",
    "sector_benchmark_available",
    "sector_relative_to_active_return_1d",
    "sector_relative_to_active_available",
    "sector_peer_ex_self_count",
    "sector_peer_ex_self_equal_weight_return_1d",
    "sector_peer_ex_self_available",
    "sector_relative_to_peers_return_1d",
    "sector_relative_to_peers_available",
    "sector_context_information_available_at",
    "sector_context_kind",
    "sector_context_timing_rule",
)


def _coerce_boolean(series: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized) - {"true", "false"})
    if unexpected:
        raise ValueError(
            f"sector-context column {column} contains non-Boolean values: "
            + ", ".join(unexpected)
        )
    return normalized.eq("true")


def _normalize_return_rows(
    features: pd.DataFrame,
    *,
    return_column: str,
    observed_column: str,
    information_column: str,
) -> pd.DataFrame:
    required = {"date", "ticker", return_column, observed_column, information_column}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"sector return rows are missing columns: {missing}")

    rows = features[
        ["date", "ticker", return_column, observed_column, information_column]
    ].copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.normalize()
    rows["ticker"] = rows["ticker"].astype(str).str.strip().str.upper()
    rows[return_column] = pd.to_numeric(rows[return_column], errors="coerce")
    rows[observed_column] = _coerce_boolean(
        rows[observed_column],
        column=observed_column,
    )
    rows[information_column] = pd.to_datetime(
        rows[information_column],
        errors="coerce",
    )
    if rows["date"].isna().any() or rows["ticker"].eq("").any():
        raise ValueError("sector return rows contain invalid date/ticker keys")
    if rows.duplicated(["date", "ticker"]).any():
        raise ValueError("sector return rows have duplicate date/ticker rows")

    observed = rows[observed_column]
    numeric = rows[return_column].to_numpy(dtype=np.float64)
    invalid_observed = observed & ~np.isfinite(numeric)
    if invalid_observed.any():
        raise ValueError("observed sector returns must be finite")
    fabricated_unobserved = ~observed & rows[return_column].notna()
    if fabricated_unobserved.any():
        raise ValueError(
            "unobserved sector returns must remain missing, not a fabricated value"
        )
    if rows.loc[observed, information_column].isna().any():
        raise ValueError("observed sector returns need an availability timestamp")
    observed_information_dates = rows.loc[
        observed,
        information_column,
    ].dt.normalize()
    if not observed_information_dates.equals(rows.loc[observed, "date"]):
        raise ValueError(
            "observed daily returns must become available on their own session date"
        )
    return rows


def _expected_metrics(frame: pd.DataFrame, *, neutral_value: float) -> pd.DataFrame:
    expected = frame.copy()
    active = expected["is_index_member"]
    observed = expected["return_1d_is_observed"]
    contributor = active & observed
    group_columns = ["date", "sector_as_of"]

    expected["_active_value"] = active.astype(np.int64)
    expected["_contributor_value"] = contributor.astype(np.int64)
    expected["_contributor_return"] = np.where(
        contributor,
        expected["asset_return_1d"],
        0.0,
    )
    expected["_contributor_information"] = expected[
        "return_information_available_at"
    ].where(contributor)

    grouped = expected.groupby(group_columns, dropna=False, sort=False)
    member_count = grouped["_active_value"].transform("sum").astype(np.int64)
    contributor_count = grouped["_contributor_value"].transform("sum").astype(
        np.int64
    )
    contributor_sum = grouped["_contributor_return"].transform("sum")
    context_information = grouped["_contributor_information"].transform("max")

    expected["sector_active_member_count"] = np.where(
        active,
        member_count,
        0,
    ).astype(np.int64)
    expected["sector_return_contributor_count"] = np.where(
        active,
        contributor_count,
        0,
    ).astype(np.int64)
    expected["sector_return_coverage"] = np.where(
        active & member_count.gt(0),
        contributor_count / member_count,
        0.0,
    )

    benchmark_available = active & contributor_count.gt(0)
    inclusive_return = np.divide(
        contributor_sum,
        contributor_count,
        out=np.full(len(expected), neutral_value, dtype=np.float64),
        where=contributor_count.to_numpy() > 0,
    )
    expected["sector_benchmark_available"] = benchmark_available
    expected["sector_active_equal_weight_return_1d"] = np.where(
        benchmark_available,
        inclusive_return,
        neutral_value,
    )
    relative_active_available = benchmark_available & observed
    expected["sector_relative_to_active_available"] = relative_active_available
    expected["sector_relative_to_active_return_1d"] = np.where(
        relative_active_available,
        expected["asset_return_1d"] - inclusive_return,
        neutral_value,
    )

    self_contribution = contributor.astype(np.int64)
    peer_count = contributor_count - self_contribution
    peer_sum = contributor_sum - np.where(
        contributor,
        expected["asset_return_1d"],
        0.0,
    )
    peer_available = active & peer_count.gt(0)
    peer_return = np.divide(
        peer_sum,
        peer_count,
        out=np.full(len(expected), neutral_value, dtype=np.float64),
        where=peer_count.to_numpy() > 0,
    )
    expected["sector_peer_ex_self_count"] = np.where(
        active,
        peer_count,
        0,
    ).astype(np.int64)
    expected["sector_peer_ex_self_available"] = peer_available
    expected["sector_peer_ex_self_equal_weight_return_1d"] = np.where(
        peer_available,
        peer_return,
        neutral_value,
    )
    relative_peer_available = peer_available & observed
    expected["sector_relative_to_peers_available"] = relative_peer_available
    expected["sector_relative_to_peers_return_1d"] = np.where(
        relative_peer_available,
        expected["asset_return_1d"] - peer_return,
        neutral_value,
    )
    expected["sector_context_information_available_at"] = context_information.where(
        benchmark_available
    )
    return expected


def validate_active_member_sector_context(
    context: pd.DataFrame,
    *,
    expected_dates: Sequence[str | pd.Timestamp] | None = None,
    expected_tickers: Sequence[str] | None = None,
    neutral_value: float = 0.0,
) -> pd.DataFrame:
    """Validate the exact V3 sector-context grid and recompute its aggregates."""

    if not np.isfinite(neutral_value):
        raise ValueError("neutral_value must be finite")
    missing = [column for column in OUTPUT_COLUMNS if column not in context.columns]
    if missing:
        raise ValueError("sector context is missing columns: " + ", ".join(missing))

    checked = context[list(OUTPUT_COLUMNS)].copy()
    checked["date"] = pd.to_datetime(checked["date"], errors="coerce").dt.normalize()
    checked["ticker"] = checked["ticker"].astype(str).str.strip().str.upper()
    for column in (
        "return_information_available_at",
        "sector_context_information_available_at",
    ):
        checked[column] = pd.to_datetime(checked[column], errors="coerce")
    for column in (
        "is_index_member",
        "return_1d_is_observed",
        "sector_benchmark_available",
        "sector_relative_to_active_available",
        "sector_peer_ex_self_available",
        "sector_relative_to_peers_available",
    ):
        checked[column] = _coerce_boolean(checked[column], column=column)
    if checked["date"].isna().any() or checked["ticker"].eq("").any():
        raise ValueError("sector context contains invalid date/ticker keys")
    if checked.duplicated(["date", "ticker"]).any():
        raise ValueError("sector context has duplicate date/ticker rows")
    if (
        checked["sector_context_kind"].nunique() != 1
        or checked["sector_context_kind"].iloc[0] != SECTOR_CONTEXT_KIND
    ):
        raise ValueError("sector context kind is invalid")
    if (
        checked["sector_context_timing_rule"].nunique() != 1
        or checked["sector_context_timing_rule"].iloc[0]
        != SECTOR_CONTEXT_TIMING_RULE
    ):
        raise ValueError("sector context timing rule is invalid")

    active = checked["is_index_member"]
    if checked.loc[active, "sector_as_of"].isna().any() or (
        checked.loc[active, "sector_as_of"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("active sector-context rows have blank dated sectors")
    observed = checked["return_1d_is_observed"]
    returns = pd.to_numeric(checked["asset_return_1d"], errors="coerce")
    checked["asset_return_1d"] = returns
    if (observed & ~np.isfinite(returns.to_numpy(dtype=np.float64))).any():
        raise ValueError("observed sector-context returns must be finite")
    if (~observed & returns.notna()).any():
        raise ValueError(
            "unobserved sector-context returns must remain missing"
        )
    if checked.loc[observed, "return_information_available_at"].isna().any():
        raise ValueError("observed sector-context returns lack availability")
    if (
        checked.loc[observed, "return_information_available_at"].dt.normalize()
        != checked.loc[observed, "date"]
    ).any():
        raise ValueError("sector-context return availability is not same-session")
    context_available = checked["sector_benchmark_available"]
    if checked.loc[
        context_available,
        "sector_context_information_available_at",
    ].isna().any():
        raise ValueError("available sector context lacks availability metadata")
    if (
        checked.loc[
            context_available,
            "sector_context_information_available_at",
        ].dt.normalize()
        > checked.loc[context_available, "date"]
    ).any():
        raise ValueError("sector context contains future information")

    slot_table = (
        checked[["asset_slot", "ticker"]]
        .drop_duplicates()
        .sort_values("asset_slot")
    )
    if (
        slot_table["asset_slot"].duplicated().any()
        or slot_table["ticker"].duplicated().any()
    ):
        raise ValueError("sector context has an unstable asset-slot mapping")
    tickers = tuple(slot_table["ticker"].astype(str))
    dates = pd.DatetimeIndex(checked["date"].drop_duplicates().sort_values())
    if expected_dates is not None:
        requested_dates = pd.DatetimeIndex(
            pd.to_datetime(list(expected_dates), errors="coerce")
        ).normalize().drop_duplicates().sort_values()
        if requested_dates.isna().any() or not dates.equals(requested_dates):
            raise ValueError("sector-context date grid does not match the request")
    if expected_tickers is not None:
        requested_tickers = tuple(
            str(value).strip().upper() for value in expected_tickers
        )
        if tickers != requested_tickers:
            raise ValueError("sector-context ticker slots do not match the request")
    if len(checked) != len(dates) * len(tickers):
        raise ValueError("sector context is not a dense date-by-ticker grid")
    expected_slots = list(range(len(tickers)))
    for _, group in checked.groupby("date", sort=False):
        if group.sort_values("asset_slot")["asset_slot"].tolist() != expected_slots:
            raise ValueError("sector context has an incomplete asset-slot grid")

    numeric_columns = (
        "sector_active_member_count",
        "sector_return_contributor_count",
        "sector_return_coverage",
        "sector_active_equal_weight_return_1d",
        "sector_relative_to_active_return_1d",
        "sector_peer_ex_self_count",
        "sector_peer_ex_self_equal_weight_return_1d",
        "sector_relative_to_peers_return_1d",
    )
    for column in numeric_columns:
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
    if checked[list(numeric_columns)].isna().any().any():
        raise ValueError("sector context contains non-numeric aggregate values")

    recomputed = _expected_metrics(checked, neutral_value=neutral_value)
    exact_columns = (
        "sector_active_member_count",
        "sector_return_contributor_count",
        "sector_benchmark_available",
        "sector_relative_to_active_available",
        "sector_peer_ex_self_count",
        "sector_peer_ex_self_available",
        "sector_relative_to_peers_available",
    )
    for column in exact_columns:
        if not checked[column].equals(recomputed[column]):
            raise ValueError(f"sector context has inconsistent {column}")
    float_columns = (
        "sector_return_coverage",
        "sector_active_equal_weight_return_1d",
        "sector_relative_to_active_return_1d",
        "sector_peer_ex_self_equal_weight_return_1d",
        "sector_relative_to_peers_return_1d",
    )
    for column in float_columns:
        if not np.allclose(
            checked[column].to_numpy(dtype=np.float64),
            recomputed[column].to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(f"sector context has inconsistent {column}")

    return checked.sort_values(["date", "asset_slot"], kind="mergesort").reset_index(
        drop=True
    )


def build_active_member_sector_context(
    features: pd.DataFrame,
    membership_panel: pd.DataFrame,
    *,
    return_column: str = "return_1d",
    observed_column: str = "return_1d_is_observed",
    information_column: str = "return_information_available_at",
    neutral_value: float = 0.0,
) -> pd.DataFrame:
    """Build dated sector features from active members and observed returns only.

    A feature/status row is required for every active member/session key.
    Inactive acquisition rows may contain returns, but they never contribute.
    Unavailable numeric context uses ``neutral_value`` only beside explicit
    availability flags.
    """

    if not np.isfinite(neutral_value):
        raise ValueError("neutral_value must be finite")
    membership = validate_membership_panel(membership_panel)
    returns = _normalize_return_rows(
        features,
        return_column=return_column,
        observed_column=observed_column,
        information_column=information_column,
    )
    membership_keys = membership[["date", "ticker"]]
    orphans = returns.merge(
        membership_keys,
        on=["date", "ticker"],
        how="left",
        indicator=True,
    )
    if orphans["_merge"].eq("left_only").any():
        sample = orphans.loc[
            orphans["_merge"].eq("left_only"),
            ["date", "ticker"],
        ].head(10)
        raise ValueError(
            "sector return rows are outside the membership grid: "
            + repr(sample.to_dict(orient="records"))
        )

    merged = membership[
        ["date", "asset_slot", "ticker", "is_index_member", "sector_as_of"]
    ].merge(
        returns,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
        indicator="_return_row",
    )
    missing_active = merged["is_index_member"] & merged["_return_row"].ne("both")
    if missing_active.any():
        sample = merged.loc[
            missing_active,
            ["date", "ticker"],
        ].head(10)
        raise ValueError(
            "sector context is missing active member/status rows: "
            f"count={int(missing_active.sum())}, "
            f"sample={sample.to_dict(orient='records')!r}"
        )

    merged = merged.rename(
        columns={
            return_column: "asset_return_1d",
            observed_column: "return_1d_is_observed",
            information_column: "return_information_available_at",
        }
    )
    merged["return_1d_is_observed"] = merged[
        "return_1d_is_observed"
    ].eq(True)
    merged["sector_context_kind"] = SECTOR_CONTEXT_KIND
    merged["sector_context_timing_rule"] = SECTOR_CONTEXT_TIMING_RULE
    built = _expected_metrics(merged, neutral_value=neutral_value)
    return validate_active_member_sector_context(
        built[list(OUTPUT_COLUMNS)],
        expected_dates=membership["date"].drop_duplicates().tolist(),
        expected_tickers=tuple(
            membership.sort_values("asset_slot")
            .drop_duplicates("asset_slot")["ticker"]
            .astype(str)
        ),
        neutral_value=neutral_value,
    )
