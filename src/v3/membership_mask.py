"""Point-in-time SET membership panels for the isolated V3 development path.

This module deliberately produces a *membership* mask, not a final
investability mask.  Membership is only one input to investability; licensed
prices, observable trading status, corporate actions, and a locked handling
policy are still required before model training.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from src.data.constituent_history import (
    IndexName,
    load_constituent_history,
    validate_constituent_history,
)


REVISION_AVAILABILITY_RULE = "next_observed_session_after_publication"
MASK_KIND = "membership_only_not_investability"

PANEL_COLUMNS = (
    "date",
    "asset_slot",
    "ticker",
    "index_name",
    "is_index_member",
    "membership_record_id",
    "membership_published_at",
    "revision_status",
    "sector_as_of",
    "symbol",
    "company_name",
    "source_url",
    "source_file_sha256",
    "revision_availability_rule",
    "mask_kind",
)


def _history_frame(history: pd.DataFrame | Path) -> pd.DataFrame:
    if isinstance(history, Path):
        return load_constituent_history(history)
    return validate_constituent_history(history)


def _normalize_sessions(values: Iterable[str | pd.Timestamp]) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(pd.Series(list(values), dtype="object"), errors="coerce")
    if parsed.empty:
        raise ValueError("at least one session date is required")
    if parsed.isna().any():
        raise ValueError("session dates contain invalid values")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert(None)
    normalized = pd.DatetimeIndex(parsed.dt.normalize().drop_duplicates().sort_values())
    if normalized.empty:
        raise ValueError("at least one session date is required")
    return normalized


def _normalize_tickers(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip().upper() for value in values)
    if not normalized:
        raise ValueError("at least one ticker is required")
    if any(not value for value in normalized):
        raise ValueError("ticker universe contains a blank value")
    if len(normalized) != len(set(normalized)):
        raise ValueError("ticker universe contains duplicates")
    malformed = sorted(
        value
        for value in normalized
        if not pd.Series([value]).str.fullmatch(r"[A-Z0-9&.-]+\.BK").iloc[0]
    )
    if malformed:
        raise ValueError(f"ticker universe contains malformed .BK tickers: {malformed}")
    return normalized


def _record_table(history: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "record_id",
        "effective_from",
        "published_at",
        "revision_status",
        "source_url",
        "source_file_sha256",
    ]
    for field in fields:
        inconsistent = history.groupby("record_id")[field].nunique(dropna=False)
        if (inconsistent != 1).any():
            record_ids = inconsistent[inconsistent != 1].index.astype(str).tolist()
            raise ValueError(
                f"membership records disagree on {field}: {record_ids}"
            )
    return (
        history[fields]
        .drop_duplicates("record_id")
        .sort_values(["effective_from", "published_at", "record_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _select_record_for_session(
    records: pd.DataFrame,
    session: pd.Timestamp,
) -> pd.Series:
    available = records[
        (records["effective_from"] <= session)
        & (records["published_at"] <= session)
        & (
            records["revision_status"].ne("revised")
            | (records["published_at"] < session)
        )
    ]
    if available.empty:
        raise ValueError(
            "no point-in-time constituent record is safely available for "
            f"session {session.date()}"
        )
    return available.iloc[-1]


def validate_membership_panel(
    panel: pd.DataFrame,
    *,
    expected_dates: Sequence[str | pd.Timestamp] | None = None,
    expected_tickers: Sequence[str] | None = None,
    expected_active_count: int | None = None,
) -> pd.DataFrame:
    """Validate a dense, publication-gated date-by-ticker membership panel."""

    missing = [column for column in PANEL_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError("membership panel is missing columns: " + ", ".join(missing))

    checked = panel.copy()
    checked["date"] = pd.to_datetime(checked["date"], errors="coerce")
    checked["membership_published_at"] = pd.to_datetime(
        checked["membership_published_at"],
        errors="coerce",
    )
    if checked[["date", "membership_published_at"]].isna().any().any():
        raise ValueError("membership panel contains invalid dates")

    if not pd.api.types.is_bool_dtype(checked["is_index_member"]):
        normalized = checked["is_index_member"].astype(str).str.strip().str.lower()
        unexpected = sorted(set(normalized) - {"true", "false"})
        if unexpected:
            raise ValueError(
                "membership panel contains non-Boolean membership values: "
                + ", ".join(unexpected)
            )
        checked["is_index_member"] = normalized.eq("true")

    if checked.duplicated(["date", "ticker"]).any():
        raise ValueError("membership panel has duplicate date/ticker rows")
    if checked["mask_kind"].nunique() != 1 or checked["mask_kind"].iloc[0] != MASK_KIND:
        raise ValueError(f"membership panel mask_kind must be {MASK_KIND!r}")
    if (
        checked["revision_availability_rule"].nunique() != 1
        or checked["revision_availability_rule"].iloc[0] != REVISION_AVAILABILITY_RULE
    ):
        raise ValueError(
            "membership panel does not use the locked next-session revision rule"
        )
    if checked["index_name"].nunique() != 1:
        raise ValueError("membership panel mixes index names")
    if checked["membership_record_id"].astype(str).str.strip().eq("").any():
        raise ValueError("membership panel has blank record IDs")
    if checked.groupby("date")["membership_record_id"].nunique().ne(1).any():
        raise ValueError("membership panel maps a session to multiple records")

    future_original = (
        checked["revision_status"].ne("revised")
        & (checked["membership_published_at"] > checked["date"])
    )
    future_revision = (
        checked["revision_status"].eq("revised")
        & (checked["membership_published_at"] >= checked["date"])
    )
    if future_original.any() or future_revision.any():
        raise ValueError("membership panel uses a record before it is safely available")

    member_rows = checked["is_index_member"]
    if checked.loc[member_rows, "sector_as_of"].astype(str).str.strip().eq("").any():
        raise ValueError("active membership rows have blank sectors")
    if checked.loc[member_rows, "symbol"].astype(str).str.strip().eq("").any():
        raise ValueError("active membership rows have blank symbols")

    if expected_dates is not None:
        dates = _normalize_sessions(expected_dates)
        observed = pd.DatetimeIndex(checked["date"].drop_duplicates().sort_values())
        if not observed.equals(dates):
            raise ValueError("membership panel date grid does not match the request")
    else:
        dates = pd.DatetimeIndex(checked["date"].drop_duplicates().sort_values())

    if expected_tickers is not None:
        tickers = _normalize_tickers(expected_tickers)
        observed_tickers = tuple(
            checked.sort_values("asset_slot")
            .drop_duplicates("asset_slot")["ticker"]
            .astype(str)
        )
        if observed_tickers != tickers:
            raise ValueError("membership panel ticker slots do not match the request")
    else:
        slot_table = (
            checked[["asset_slot", "ticker"]]
            .drop_duplicates()
            .sort_values("asset_slot")
        )
        tickers = tuple(slot_table["ticker"].astype(str))

    expected_rows = len(dates) * len(tickers)
    if len(checked) != expected_rows:
        raise ValueError(
            f"membership panel has {len(checked)} rows, expected {expected_rows}"
        )
    expected_slots = list(range(len(tickers)))
    for _, group in checked.groupby("date", sort=False):
        if group.sort_values("asset_slot")["asset_slot"].tolist() != expected_slots:
            raise ValueError("membership panel has an unstable asset-slot layout")

    if expected_active_count is not None:
        counts = checked.groupby("date")["is_index_member"].sum()
        if not counts.eq(expected_active_count).all():
            bad = counts[counts.ne(expected_active_count)]
            raise ValueError(
                "membership panel active count drift: "
                + ", ".join(
                    f"{date.date()}={int(count)}" for date, count in bad.head(10).items()
                )
            )

    return checked.sort_values(["date", "asset_slot"], kind="mergesort").reset_index(
        drop=True
    )


def build_point_in_time_membership_panel(
    history: pd.DataFrame | Path,
    dates: Sequence[str | pd.Timestamp],
    tickers: Sequence[str],
    *,
    index_name: IndexName = "SET100",
    require_complete_index: bool = True,
) -> pd.DataFrame:
    """Return an exact date-by-ticker point-in-time membership panel.

    Same-day revisions are not used until the next supplied session.  This is
    conservative for the close-to-close simulator because the source PDFs only
    provide publication dates, not intraday publication timestamps.
    """

    if index_name not in {"SET50", "SET100"}:
        raise ValueError("index_name must be SET50 or SET100")
    frame = _history_frame(history)
    sessions = _normalize_sessions(dates)
    ticker_slots = _normalize_tickers(tickers)

    coverage_start = frame["effective_from"].min()
    coverage_end = frame["effective_to_exclusive"].max()
    outside = sessions[(sessions < coverage_start) | (sessions >= coverage_end)]
    if len(outside):
        raise ValueError(
            "requested sessions fall outside membership coverage "
            f"[{coverage_start.date()}, {coverage_end.date()}): "
            + ", ".join(value.date().isoformat() for value in outside[:5])
        )

    archive_tickers = set(frame["ticker"].astype(str))
    unknown = sorted(set(ticker_slots) - archive_tickers)
    if unknown:
        raise ValueError(f"ticker universe contains symbols absent from the archive: {unknown}")

    records = _record_table(frame)
    selected_records = []
    for session in sessions:
        record = _select_record_for_session(records, session)
        selected_records.append(
            {
                "date": session,
                "membership_record_id": record["record_id"],
                "membership_published_at": record["published_at"],
                "revision_status": record["revision_status"],
                "source_url": record["source_url"],
                "source_file_sha256": record["source_file_sha256"],
            }
        )
    selected = pd.DataFrame(selected_records)
    slots = pd.DataFrame(
        {
            "asset_slot": range(len(ticker_slots)),
            "ticker": ticker_slots,
        }
    )
    panel = selected.merge(slots, how="cross")

    flag = "in_set50" if index_name == "SET50" else "in_set100"
    members = frame.loc[
        frame[flag],
        ["record_id", "ticker", "sector", "symbol", "company_name"],
    ].rename(
        columns={
            "record_id": "membership_record_id",
            "sector": "sector_as_of",
        }
    )
    members["is_index_member"] = True
    panel = panel.merge(
        members,
        on=["membership_record_id", "ticker"],
        how="left",
        validate="many_to_one",
    )
    panel["is_index_member"] = panel["is_index_member"].eq(True)
    panel["index_name"] = index_name
    panel["revision_availability_rule"] = REVISION_AVAILABILITY_RULE
    panel["mask_kind"] = MASK_KIND

    expected_active_count = 50 if index_name == "SET50" else 100
    checked = validate_membership_panel(
        panel[list(PANEL_COLUMNS)],
        expected_dates=sessions,
        expected_tickers=ticker_slots,
        expected_active_count=expected_active_count if require_complete_index else None,
    )

    if require_complete_index:
        active_tickers = set(
            frame.loc[
                frame["record_id"].isin(checked["membership_record_id"].unique())
                & frame[flag],
                "ticker",
            ].astype(str)
        )
        omitted = sorted(active_tickers - set(ticker_slots))
        if omitted:
            raise ValueError(
                f"ticker universe omits active {index_name} members: {omitted}"
            )
    return checked


def attach_membership_context(
    features: pd.DataFrame,
    history: pd.DataFrame | Path,
    *,
    asset_universe: Sequence[str],
    index_name: Literal["SET50", "SET100"] = "SET100",
) -> pd.DataFrame:
    """Attach dated membership and sector context to an existing feature table."""

    required = {"date", "ticker"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")
    if features.duplicated(["date", "ticker"]).any():
        raise ValueError("feature table has duplicate date/ticker rows")

    checked_features = features.copy()
    checked_features["date"] = pd.to_datetime(
        checked_features["date"],
        errors="coerce",
    ).dt.normalize()
    if checked_features["date"].isna().any():
        raise ValueError("feature table contains invalid dates")
    checked_features["ticker"] = checked_features["ticker"].astype(str).str.strip().str.upper()

    panel = build_point_in_time_membership_panel(
        history,
        dates=checked_features["date"].drop_duplicates().tolist(),
        tickers=asset_universe,
        index_name=index_name,
    )
    context_columns = [
        "date",
        "ticker",
        "asset_slot",
        "is_index_member",
        "membership_record_id",
        "membership_published_at",
        "revision_status",
        "sector_as_of",
        "source_file_sha256",
        "revision_availability_rule",
        "mask_kind",
    ]
    merged = checked_features.merge(
        panel[context_columns],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    if merged["mask_kind"].isna().any():
        missing_rows = merged.loc[
            merged["mask_kind"].isna(),
            ["date", "ticker"],
        ].head(10)
        raise ValueError(
            "feature rows are absent from the membership panel: "
            + missing_rows.to_dict(orient="records").__repr__()
        )
    return merged.sort_values(["date", "asset_slot"], kind="mergesort").reset_index(
        drop=True
    )
