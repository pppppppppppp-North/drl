"""Canonical fold-scoped datasets for isolated V3 development.

The evaluation policy assigns every usable close-to-close transition to a
temporal fold and split.  This module turns that schedule into exact feature
and eligibility slices.  It is deliberately independent of the shared V2
splitter and does not authorize training, validation scoring, or holdout use.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.v3.eligibility import validate_development_eligibility_mask
from src.v3.evaluation_policy import (
    FOLD_IDS,
    SCHEDULE_COLUMNS,
    normalize_session_calendar,
    validate_fold_schedule,
)


FOLD_SPLITS = ("train", "validation")
TRANSITION_COLUMNS = (
    "fold_id",
    "split",
    "transition_index",
    "decision_date",
    "outcome_date",
    "global_session_index",
    "schedule_row_sha256",
)
BOOLEAN_SCHEDULE_COLUMNS = (
    "train_context",
    "train_decision",
    "train_outcome",
    "validation_context",
    "validation_decision",
    "validation_outcome",
    "preprocessing_fit_allowed",
    "model_update_allowed",
    "validation_scoring_allowed",
    "unused_boundary_transition_start",
    "holdout_access_allowed",
)


@dataclass(frozen=True)
class FoldDatasetSlice:
    """One exact environment input and its owned transition ledger."""

    fold_id: str
    split: str
    lookback: int
    dates: tuple[str, ...]
    features: pd.DataFrame
    eligibility_mask: pd.DataFrame
    transitions: pd.DataFrame
    features_sha256: str
    eligibility_sha256: str
    transitions_sha256: str


@dataclass(frozen=True)
class FoldDatasetBundle:
    """Train and validation slices bound to one canonical fold schedule."""

    fold_id: str
    lookback: int
    feature_columns: tuple[str, ...]
    asset_universe: tuple[str, ...]
    schedule_sha256: str
    source_features_sha256: str
    source_eligibility_sha256: str
    train: FoldDatasetSlice
    validation: FoldDatasetSlice


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash one already-canonical frame with stable null and float encoding."""

    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        date_format="%Y-%m-%d",
        float_format="%.17g",
        na_rep="<NA>",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_asset_universe(values: Sequence[str]) -> tuple[str, ...]:
    tickers = tuple(str(value).strip().upper() for value in values)
    if not tickers or any(not value for value in tickers):
        raise ValueError("asset_universe must contain nonblank tickers")
    if len(tickers) != len(set(tickers)):
        raise ValueError("asset_universe must contain unique tickers")
    return tickers


def _normalize_feature_columns(values: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(str(value).strip() for value in values)
    if not columns or any(not value for value in columns):
        raise ValueError("feature_columns must contain nonblank names")
    if len(columns) != len(set(columns)):
        raise ValueError("feature_columns must be unique")
    if any(value in {"date", "ticker"} for value in columns):
        raise ValueError("date and ticker cannot be model feature columns")
    return columns


def _strict_dates(values: pd.Series, *, label: str) -> pd.DatetimeIndex:
    try:
        parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains invalid dates") from exc
    if parsed.isna().any():
        raise ValueError(f"{label} contains invalid dates")
    if parsed.tz is not None:
        raise ValueError(f"{label} must use timezone-free exchange dates")
    if not parsed.equals(parsed.normalize()):
        raise ValueError(f"{label} contains intraday timestamps")
    return parsed.normalize()


def _coerce_boolean(series: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized) - {"true", "false"})
    if unexpected:
        raise ValueError(
            f"schedule column {column} contains non-Boolean values: {unexpected}"
        )
    return normalized.eq("true")


def _canonical_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    if list(schedule.columns) != list(SCHEDULE_COLUMNS):
        raise ValueError("fold schedule columns do not match the canonical schema")
    checked = schedule.copy()
    checked["date"] = _strict_dates(checked["date"], label="fold schedule")
    for column in BOOLEAN_SCHEDULE_COLUMNS:
        checked[column] = _coerce_boolean(checked[column], column=column)
    checked["global_session_index"] = pd.to_numeric(
        checked["global_session_index"], errors="coerce"
    )
    if checked["global_session_index"].isna().any():
        raise ValueError("fold schedule contains invalid global session indices")
    checked["global_session_index"] = checked["global_session_index"].astype(int)
    checked["fold_id"] = checked["fold_id"].astype(str)
    checked["transition_owner"] = checked["transition_owner"].fillna("").astype(str)
    checked["transition_outcome_date"] = (
        checked["transition_outcome_date"].fillna("").astype(str)
    )
    if checked.duplicated(["fold_id", "date"]).any():
        raise ValueError("fold schedule has duplicate fold/date rows")
    return checked.sort_values(
        ["fold_id", "global_session_index"], kind="mergesort"
    ).reset_index(drop=True)


def _canonical_features(
    features: pd.DataFrame,
    *,
    sessions: pd.DatetimeIndex,
    tickers: tuple[str, ...],
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    required = {"date", "ticker", "close", *feature_columns}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")
    checked = features.copy()
    checked["date"] = _strict_dates(checked["date"], label="feature table")
    checked["ticker"] = checked["ticker"].astype(str).str.strip().str.upper()
    if checked["ticker"].eq("").any():
        raise ValueError("feature table contains blank tickers")
    if checked.duplicated(["date", "ticker"]).any():
        raise ValueError("feature table has duplicate date/ticker rows")

    allowed_dates = set(sessions)
    orphan_dates = sorted(set(checked["date"]) - allowed_dates)
    if orphan_dates:
        sample = [value.date().isoformat() for value in orphan_dates[:5]]
        raise ValueError(f"feature table contains unauthorized session dates: {sample}")
    orphan_tickers = sorted(set(checked["ticker"]) - set(tickers))
    if orphan_tickers:
        raise ValueError(
            f"feature table contains tickers outside asset_universe: {orphan_tickers}"
        )

    expected_index = pd.MultiIndex.from_product(
        [sessions, tickers], names=["date", "ticker"]
    )
    observed_index = pd.MultiIndex.from_frame(checked[["date", "ticker"]])
    missing_keys = expected_index.difference(observed_index)
    if len(missing_keys):
        sample = [
            {"date": date.date().isoformat(), "ticker": ticker}
            for date, ticker in missing_keys[:5]
        ]
        raise ValueError(
            "feature table is missing required session/ticker rows: " + repr(sample)
        )
    if len(checked) != len(expected_index):
        raise ValueError("feature table does not match the exact session/ticker grid")

    numeric_columns = list(dict.fromkeys(("close", *feature_columns)))
    numeric = checked[numeric_columns].apply(pd.to_numeric, errors="coerce")
    nonfinite_features = ~np.isfinite(
        numeric[list(feature_columns)].to_numpy(dtype=np.float64)
    )
    if nonfinite_features.any():
        raise ValueError("feature table contains nonfinite model feature values")
    checked[numeric_columns] = numeric

    ticker_slot = {ticker: index for index, ticker in enumerate(tickers)}
    checked["__asset_slot"] = checked["ticker"].map(ticker_slot)
    ordered_columns = [
        "date",
        "ticker",
        *[column for column in checked.columns if column not in {"date", "ticker", "__asset_slot"}],
    ]
    return checked.sort_values(
        ["date", "__asset_slot"], kind="mergesort"
    )[ordered_columns].reset_index(drop=True)


def _canonical_eligibility(
    mask: pd.DataFrame,
    *,
    sessions: pd.DatetimeIndex,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    checked = validate_development_eligibility_mask(
        mask,
        expected_dates=sessions,
        expected_tickers=tickers,
    )
    if set(checked["date"]) != set(sessions):
        raise ValueError("eligibility mask contains unauthorized session dates")
    return checked


def _transition_ledger(fold: pd.DataFrame, *, split: str) -> pd.DataFrame:
    decision_column = f"{split}_decision"
    rows = fold.loc[fold[decision_column]].copy()
    if rows.empty:
        raise ValueError(f"fold has no {split} decisions")
    expected_owner = rows["transition_owner"].eq(split)
    if not expected_owner.all():
        raise ValueError(f"{split} decisions have incorrect transition ownership")
    if rows["unused_boundary_transition_start"].any():
        raise ValueError("an unused boundary transition was assigned to a split")
    if rows["transition_outcome_date"].eq("").any():
        raise ValueError(f"{split} decisions are missing terminal outcomes")
    if split == "train":
        if not rows["preprocessing_fit_allowed"].all():
            raise ValueError("train decisions are not preprocessing-fit authorized")
        if not rows["model_update_allowed"].all():
            raise ValueError("train decisions are not model-update authorized")
        if rows["validation_scoring_allowed"].any():
            raise ValueError("train decisions cannot be validation-scored")
    else:
        if rows["preprocessing_fit_allowed"].any() or rows[
            "model_update_allowed"
        ].any():
            raise ValueError("validation decisions cannot fit or update a model")
        if not rows["validation_scoring_allowed"].all():
            raise ValueError("validation decisions are not scoring-authorized")

    records: list[dict[str, Any]] = []
    for transition_index, row in enumerate(rows.itertuples(index=False)):
        decision = pd.Timestamp(row.date).date().isoformat()
        outcome = pd.Timestamp(row.transition_outcome_date).date().isoformat()
        row_payload = {
            column: (
                pd.Timestamp(getattr(row, column)).date().isoformat()
                if column == "date"
                else bool(getattr(row, column))
                if column in BOOLEAN_SCHEDULE_COLUMNS
                else int(getattr(row, column))
                if column == "global_session_index"
                else getattr(row, column)
            )
            for column in SCHEDULE_COLUMNS
        }
        records.append(
            {
                "fold_id": str(row.fold_id),
                "split": split,
                "transition_index": transition_index,
                "decision_date": decision,
                "outcome_date": outcome,
                "global_session_index": int(row.global_session_index),
                "schedule_row_sha256": _json_sha256(row_payload),
            }
        )
    ledger = pd.DataFrame(records, columns=TRANSITION_COLUMNS)
    if ledger.duplicated(["decision_date"]).any():
        raise ValueError(f"{split} transition ledger has duplicate decision dates")
    if ledger.duplicated(["outcome_date"]).any():
        raise ValueError(f"{split} transition ledger has duplicate outcome dates")
    return ledger


def _split_dates(fold: pd.DataFrame, *, split: str) -> tuple[pd.Timestamp, ...]:
    selected = fold[
        fold[f"{split}_context"]
        | fold[f"{split}_decision"]
        | fold[f"{split}_outcome"]
    ]
    return tuple(pd.Timestamp(value) for value in selected["date"])


def _validate_slice_geometry(
    *,
    split: str,
    dates: tuple[pd.Timestamp, ...],
    transitions: pd.DataFrame,
    lookback: int,
) -> None:
    if len(dates) != lookback + len(transitions) + 1:
        raise ValueError(
            f"{split} slice does not contain context + decisions + terminal outcome"
        )
    expected_decisions = [value.date().isoformat() for value in dates[lookback:-1]]
    expected_outcomes = [value.date().isoformat() for value in dates[lookback + 1 :]]
    if transitions["decision_date"].tolist() != expected_decisions:
        raise ValueError(f"{split} environment decisions differ from owned transitions")
    if transitions["outcome_date"].tolist() != expected_outcomes:
        raise ValueError(f"{split} environment outcomes differ from owned transitions")


def _validate_environment_coverage(
    features: pd.DataFrame,
    eligibility: pd.DataFrame,
    transitions: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    tickers: tuple[str, ...],
    lookback: int,
) -> None:
    dates = tuple(pd.Timestamp(value) for value in features["date"].drop_duplicates())
    feature_panel = features.set_index(["date", "ticker"])
    eligible_panel = eligibility.pivot(
        index="date", columns="ticker", values="is_eligible"
    ).reindex(index=dates, columns=tickers)
    if eligible_panel.isna().any().any():
        raise ValueError("slice eligibility grid is incomplete")

    for transition in transitions.itertuples(index=False):
        decision = pd.Timestamp(transition.decision_date)
        outcome = pd.Timestamp(transition.outcome_date)
        decision_index = dates.index(decision)
        if decision_index < lookback or dates[decision_index + 1] != outcome:
            raise ValueError("transition is not consecutive inside its fold slice")
        active = eligible_panel.loc[decision].to_numpy(dtype=bool)
        if not active.any():
            continue
        active_tickers = [ticker for ticker, value in zip(tickers, active, strict=True) if value]
        history_dates = dates[decision_index - lookback : decision_index]
        history = feature_panel.loc[
            pd.IndexSlice[list(history_dates), active_tickers], list(feature_columns)
        ].to_numpy(dtype=np.float64)
        if not np.isfinite(history).all():
            raise ValueError(
                "eligible assets have nonfinite features in an owned lookback window"
            )
        current = feature_panel.loc[
            pd.IndexSlice[decision, active_tickers], "close"
        ].to_numpy(dtype=np.float64)
        following = feature_panel.loc[
            pd.IndexSlice[outcome, active_tickers], "close"
        ].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(current).all()
            or not np.isfinite(following).all()
            or (current <= 0.0).any()
            or (following <= 0.0).any()
        ):
            raise ValueError(
                "eligible assets lack finite positive valuation prices on an owned transition"
            )


def _materialize_slice(
    *,
    fold_id: str,
    split: str,
    fold: pd.DataFrame,
    features: pd.DataFrame,
    eligibility: pd.DataFrame,
    tickers: tuple[str, ...],
    feature_columns: tuple[str, ...],
    lookback: int,
) -> FoldDatasetSlice:
    dates = _split_dates(fold, split=split)
    transitions = _transition_ledger(fold, split=split)
    _validate_slice_geometry(
        split=split,
        dates=dates,
        transitions=transitions,
        lookback=lookback,
    )
    date_set = set(dates)
    sliced_features = features[features["date"].isin(date_set)].copy()
    sliced_eligibility = eligibility[eligibility["date"].isin(date_set)].copy()
    sliced_eligibility = validate_development_eligibility_mask(
        sliced_eligibility,
        expected_dates=dates,
        expected_tickers=tickers,
    )
    _validate_environment_coverage(
        sliced_features,
        sliced_eligibility,
        transitions,
        feature_columns=feature_columns,
        tickers=tickers,
        lookback=lookback,
    )
    return FoldDatasetSlice(
        fold_id=fold_id,
        split=split,
        lookback=lookback,
        dates=tuple(value.date().isoformat() for value in dates),
        features=sliced_features.reset_index(drop=True),
        eligibility_mask=sliced_eligibility,
        transitions=transitions,
        features_sha256=canonical_frame_sha256(sliced_features),
        eligibility_sha256=canonical_frame_sha256(sliced_eligibility),
        transitions_sha256=canonical_frame_sha256(transitions),
    )


def materialize_fold_dataset(
    features: pd.DataFrame,
    eligibility_mask: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    sessions: Sequence[str | pd.Timestamp],
    policy: Mapping[str, Any],
    fold_id: str,
    feature_columns: Sequence[str],
    asset_universe: Sequence[str],
) -> FoldDatasetBundle:
    """Build exact train/validation environment inputs for one canonical fold."""

    if fold_id not in FOLD_IDS:
        raise ValueError(f"unsupported fold_id: {fold_id!r}")
    normalized_sessions = normalize_session_calendar(sessions)
    schedule_issues = validate_fold_schedule(schedule, normalized_sessions, policy)
    if schedule_issues:
        raise ValueError("; ".join(schedule_issues))
    canonical_schedule = _canonical_schedule(schedule)
    tickers = _normalize_asset_universe(asset_universe)
    columns = _normalize_feature_columns(feature_columns)

    fold_geometry = policy.get("fold_geometry")
    if not isinstance(fold_geometry, Mapping):
        raise ValueError("policy fold_geometry must be a mapping")
    train_lookback = int(fold_geometry.get("initial_train_context_sessions", -1))
    validation_lookback = int(
        fold_geometry.get("validation_context_sessions", -1)
    )
    if train_lookback <= 0 or train_lookback != validation_lookback:
        raise ValueError("train and validation context must define one positive lookback")

    canonical_features = _canonical_features(
        features,
        sessions=normalized_sessions,
        tickers=tickers,
        feature_columns=columns,
    )
    canonical_eligibility = _canonical_eligibility(
        eligibility_mask,
        sessions=normalized_sessions,
        tickers=tickers,
    )
    feature_keys = canonical_features[["date", "ticker"]]
    eligibility_keys = canonical_eligibility[["date", "ticker"]]
    if not feature_keys.equals(eligibility_keys):
        raise ValueError("feature and eligibility session/ticker grids differ")

    fold = canonical_schedule[canonical_schedule["fold_id"].eq(fold_id)].copy()
    if fold.empty:
        raise ValueError(f"canonical schedule is missing {fold_id}")
    train = _materialize_slice(
        fold_id=fold_id,
        split="train",
        fold=fold,
        features=canonical_features,
        eligibility=canonical_eligibility,
        tickers=tickers,
        feature_columns=columns,
        lookback=train_lookback,
    )
    validation = _materialize_slice(
        fold_id=fold_id,
        split="validation",
        fold=fold,
        features=canonical_features,
        eligibility=canonical_eligibility,
        tickers=tickers,
        feature_columns=columns,
        lookback=train_lookback,
    )
    return FoldDatasetBundle(
        fold_id=fold_id,
        lookback=train_lookback,
        feature_columns=columns,
        asset_universe=tickers,
        schedule_sha256=canonical_frame_sha256(canonical_schedule),
        source_features_sha256=canonical_frame_sha256(canonical_features),
        source_eligibility_sha256=canonical_frame_sha256(canonical_eligibility),
        train=train,
        validation=validation,
    )
