from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


REQUIRED_COLUMNS = (
    "record_id",
    "effective_from",
    "effective_to_exclusive",
    "published_at",
    "symbol",
    "ticker",
    "sector",
    "in_set50",
    "in_set100",
)

IndexName = Literal["SET50", "SET100"]


def _coerce_boolean(series: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false"}
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        raise ValueError(
            f"membership column {column} contains non-Boolean values: "
            + ", ".join(unexpected)
        )
    return normalized.eq("true")


def validate_constituent_history(
    frame: pd.DataFrame,
    *,
    expected_start: str = "2021-01-01",
    expected_end_exclusive: str = "2025-01-01",
) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "membership archive is missing columns: " + ", ".join(missing)
        )
    normalized = frame.copy()
    for column in (
        "effective_from",
        "effective_to_exclusive",
        "published_at",
    ):
        normalized[column] = pd.to_datetime(
            normalized[column],
            errors="coerce",
        )
        if normalized[column].isna().any():
            raise ValueError(f"membership column {column} contains invalid dates")
    for column in ("in_set50", "in_set100"):
        normalized[column] = _coerce_boolean(
            normalized[column],
            column=column,
        )
    if normalized.duplicated(["record_id", "symbol"]).any():
        raise ValueError("membership archive has duplicate interval/symbol rows")
    if not normalized["ticker"].astype(str).str.fullmatch(
        r"[A-Z0-9&.-]+\.BK"
    ).all():
        raise ValueError("membership archive has malformed .BK ticker mappings")
    if normalized["sector"].astype(str).str.strip().eq("").any():
        raise ValueError("membership archive has blank sector values")
    if (~normalized["in_set100"]).any():
        raise ValueError("membership archive contains rows outside SET100")
    if (normalized["in_set50"] & ~normalized["in_set100"]).any():
        raise ValueError("membership archive has SET50 rows outside SET100")
    if (normalized["published_at"] > normalized["effective_from"]).any():
        raise ValueError(
            "membership archive applies a record before its publication date"
        )

    interval_rows = (
        normalized[
            [
                "record_id",
                "effective_from",
                "effective_to_exclusive",
                "published_at",
            ]
        ]
        .drop_duplicates()
        .sort_values(["effective_from", "record_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if interval_rows["record_id"].duplicated().any():
        raise ValueError("membership record ID maps to multiple date intervals")
    if interval_rows.iloc[0]["effective_from"] != pd.Timestamp(expected_start):
        raise ValueError(
            f"membership archive does not start at {expected_start}"
        )
    if interval_rows.iloc[-1]["effective_to_exclusive"] != pd.Timestamp(
        expected_end_exclusive
    ):
        raise ValueError(
            "membership archive does not end at "
            f"{expected_end_exclusive} exclusive"
        )
    if (
        interval_rows["effective_from"]
        >= interval_rows["effective_to_exclusive"]
    ).any():
        raise ValueError("membership archive contains non-positive intervals")
    for index in range(1, len(interval_rows)):
        prior_end = interval_rows.iloc[index - 1][
            "effective_to_exclusive"
        ]
        current_start = interval_rows.iloc[index]["effective_from"]
        if prior_end != current_start:
            relationship = "gap" if prior_end < current_start else "overlap"
            raise ValueError(
                f"membership archive interval {relationship}: "
                f"{prior_end.date()} -> {current_start.date()}"
            )

    for record_id, group in normalized.groupby("record_id", sort=False):
        if len(group) != 100:
            raise ValueError(
                f"membership interval {record_id} has {len(group)} rows, "
                "expected 100"
            )
        if group["symbol"].nunique() != 100:
            raise ValueError(
                f"membership interval {record_id} has duplicate symbols"
            )
        if int(group["in_set50"].sum()) != 50:
            raise ValueError(
                f"membership interval {record_id} does not contain 50 SET50 rows"
            )
    return normalized.sort_values(
        ["effective_from", "symbol"],
        kind="mergesort",
    ).reset_index(drop=True)


def load_constituent_history(path: Path) -> pd.DataFrame:
    return validate_constituent_history(pd.read_csv(path))


def membership_as_of(
    history: pd.DataFrame | Path,
    as_of: str | pd.Timestamp,
    *,
    index_name: IndexName = "SET100",
) -> pd.DataFrame:
    frame = (
        load_constituent_history(history)
        if isinstance(history, Path)
        else validate_constituent_history(history)
    )
    timestamp = pd.Timestamp(as_of)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    active = frame[
        (frame["effective_from"] <= timestamp)
        & (timestamp < frame["effective_to_exclusive"])
        & (frame["published_at"] <= timestamp)
    ].copy()
    if active.empty:
        raise ValueError(
            f"no point-in-time constituent record covers {timestamp.date()}"
        )
    flag = "in_set50" if index_name == "SET50" else "in_set100"
    active = active[active[flag]].copy()
    expected_count = 50 if index_name == "SET50" else 100
    if len(active) != expected_count:
        raise ValueError(
            f"{index_name} membership at {timestamp.date()} has "
            f"{len(active)} rows, expected {expected_count}"
        )
    return active.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def acquisition_universe(history: pd.DataFrame | Path) -> tuple[str, ...]:
    frame = (
        load_constituent_history(history)
        if isinstance(history, Path)
        else validate_constituent_history(history)
    )
    return tuple(sorted(frame["ticker"].astype(str).unique()))
