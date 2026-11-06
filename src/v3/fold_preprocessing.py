"""Train-only deterministic preprocessing for canonical V3 fold datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.v3.fold_dataset import (
    FoldDatasetBundle,
    FoldDatasetSlice,
    canonical_frame_sha256,
)


PREPROCESSING_SCHEMA_VERSION = 1
PREPROCESSING_METHOD = "eligible_train_decision_population_zscore_v1"
PROTECTED_RAW_COLUMNS = frozenset(
    {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "portfolio_value",
        "cash_weight",
        "weight",
        "weights",
        "gross_return",
        "net_return",
        "turnover",
        "transaction_cost_paid",
        "slippage_paid",
        "drawdown",
        "forced_exit_turnover",
        "self_financing_reconciliation_error",
    }
)


@dataclass(frozen=True)
class FeatureScale:
    """Population z-score parameters for one model feature."""

    feature: str
    count: int
    mean: float
    scale: float
    constant: bool


@dataclass(frozen=True)
class FoldPreprocessingState:
    """Checksum-bound scaler state fit on one fold's authorized rows."""

    schema_version: int
    fold_id: str
    method: str
    feature_columns: tuple[str, ...]
    fit_scope: str
    fit_row_count: int
    fit_rows_sha256: str
    schedule_sha256: str
    train_transitions_sha256: str
    parameters: tuple[FeatureScale, ...]
    state_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **_state_payload(self),
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True)
class PreprocessedFoldDataset:
    """Transformed fold frames paired with the exact state that produced them."""

    fold_id: str
    state_sha256: str
    train_features: pd.DataFrame
    validation_features: pd.DataFrame
    train_features_sha256: str
    validation_features_sha256: str


def _state_payload(state: FoldPreprocessingState) -> dict[str, Any]:
    return {
        "schema_version": state.schema_version,
        "fold_id": state.fold_id,
        "method": state.method,
        "feature_columns": list(state.feature_columns),
        "fit_scope": state.fit_scope,
        "fit_row_count": state.fit_row_count,
        "fit_rows_sha256": state.fit_rows_sha256,
        "schedule_sha256": state.schedule_sha256,
        "train_transitions_sha256": state.train_transitions_sha256,
        "parameters": [
            {
                "feature": value.feature,
                "count": value.count,
                "mean": value.mean,
                "scale": value.scale,
                "constant": value.constant,
            }
            for value in state.parameters
        ],
    }


def _payload_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def preprocessing_state_sha256(state: FoldPreprocessingState) -> str:
    """Recompute a state's canonical digest, excluding its stored digest."""

    return _payload_sha256(_state_payload(state))


def validate_preprocessing_state(state: FoldPreprocessingState) -> list[str]:
    issues: list[str] = []
    if state.schema_version != PREPROCESSING_SCHEMA_VERSION:
        issues.append("preprocessing state schema version is unsupported")
    if state.method != PREPROCESSING_METHOD:
        issues.append("preprocessing state method is unsupported")
    if state.fit_scope != "eligible_rows_on_train_decision_dates_only":
        issues.append("preprocessing state has an unauthorized fit scope")
    if not state.feature_columns or len(state.feature_columns) != len(
        set(state.feature_columns)
    ):
        issues.append("preprocessing state feature order is invalid")
    if set(state.feature_columns) & PROTECTED_RAW_COLUMNS:
        issues.append("preprocessing state attempts to scale protected raw columns")
    if state.fit_row_count <= 0:
        issues.append("preprocessing state has no fit rows")
    if len(state.parameters) != len(state.feature_columns):
        issues.append("preprocessing state parameter count does not match features")
    if tuple(value.feature for value in state.parameters) != state.feature_columns:
        issues.append("preprocessing state parameter order does not match features")
    for value in state.parameters:
        if value.count != state.fit_row_count:
            issues.append(f"preprocessing count differs for {value.feature}")
        if not np.isfinite(value.mean) or not np.isfinite(value.scale):
            issues.append(f"preprocessing parameters are nonfinite for {value.feature}")
        if value.scale <= 0.0:
            issues.append(f"preprocessing scale is not positive for {value.feature}")
        if value.constant != (value.scale == 1.0 and value.constant):
            # A constant feature is always represented with unit scale.  A
            # nonconstant feature may coincidentally have population scale 1.
            if value.constant and value.scale != 1.0:
                issues.append(f"constant feature {value.feature} must use unit scale")
    if state.state_sha256 != preprocessing_state_sha256(state):
        issues.append("preprocessing state SHA-256 does not match its payload")
    return list(dict.fromkeys(issues))


def _normalize_requested_dates(values: Sequence[str | pd.Timestamp]) -> tuple[str, ...]:
    parsed = pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce"))
    if parsed.isna().any():
        raise ValueError("preprocessing fit dates contain invalid values")
    if parsed.tz is not None or not parsed.equals(parsed.normalize()):
        raise ValueError("preprocessing fit dates must be timezone-free sessions")
    if parsed.duplicated().any():
        raise ValueError("preprocessing fit dates contain duplicates")
    return tuple(value.date().isoformat() for value in parsed)


def _authorized_fit_frame(
    bundle: FoldDatasetBundle,
    *,
    split: str,
    fit_dates: Sequence[str | pd.Timestamp] | None,
) -> pd.DataFrame:
    if split != "train":
        raise ValueError("preprocessing may be fit only on the train split")
    authorized_dates = tuple(bundle.train.transitions["decision_date"].astype(str))
    requested_dates = (
        authorized_dates
        if fit_dates is None
        else _normalize_requested_dates(fit_dates)
    )
    if requested_dates != authorized_dates:
        raise ValueError(
            "preprocessing fit dates must exactly equal authorized train decisions"
        )

    feature_keys = bundle.train.features[["date", "ticker", *bundle.feature_columns]].copy()
    feature_keys["date"] = pd.to_datetime(feature_keys["date"]).dt.normalize()
    mask = bundle.train.eligibility_mask[
        ["date", "asset_slot", "ticker", "is_eligible"]
    ].copy()
    mask["date"] = pd.to_datetime(mask["date"]).dt.normalize()
    joined = feature_keys.merge(
        mask,
        on=["date", "ticker"],
        how="inner",
        validate="one_to_one",
    )
    requested = pd.DatetimeIndex(pd.to_datetime(list(requested_dates)))
    fit = joined[joined["date"].isin(requested) & joined["is_eligible"]].copy()
    fit = fit.sort_values(["date", "asset_slot"], kind="mergesort").reset_index(
        drop=True
    )
    if fit.empty:
        raise ValueError("preprocessing has no eligible authorized train rows")
    values = fit[list(bundle.feature_columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError("preprocessing fit rows contain nonfinite feature values")
    fit[list(bundle.feature_columns)] = values
    return fit[["date", "asset_slot", "ticker", *bundle.feature_columns]]


def fit_fold_preprocessor(
    bundle: FoldDatasetBundle,
    *,
    split: str = "train",
    fit_dates: Sequence[str | pd.Timestamp] | None = None,
) -> FoldPreprocessingState:
    """Fit population z-scores on eligible training-decision rows only."""

    protected = set(bundle.feature_columns) & PROTECTED_RAW_COLUMNS
    if protected:
        raise ValueError(
            "protected price, valuation, or accounting columns cannot be scaled: "
            + ", ".join(sorted(protected))
        )
    if bundle.train.fold_id != bundle.fold_id or bundle.validation.fold_id != bundle.fold_id:
        raise ValueError("fold dataset slices do not match their bundle")
    fit = _authorized_fit_frame(bundle, split=split, fit_dates=fit_dates)
    parameters: list[FeatureScale] = []
    for feature in bundle.feature_columns:
        values = fit[feature].to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        population_scale = float(np.std(values, ddof=0))
        constant = population_scale == 0.0
        scale = 1.0 if constant else population_scale
        parameters.append(
            FeatureScale(
                feature=feature,
                count=len(values),
                mean=mean,
                scale=scale,
                constant=constant,
            )
        )

    provisional = FoldPreprocessingState(
        schema_version=PREPROCESSING_SCHEMA_VERSION,
        fold_id=bundle.fold_id,
        method=PREPROCESSING_METHOD,
        feature_columns=bundle.feature_columns,
        fit_scope="eligible_rows_on_train_decision_dates_only",
        fit_row_count=len(fit),
        fit_rows_sha256=canonical_frame_sha256(fit),
        schedule_sha256=bundle.schedule_sha256,
        train_transitions_sha256=bundle.train.transitions_sha256,
        parameters=tuple(parameters),
        state_sha256="",
    )
    state = FoldPreprocessingState(
        **{
            **provisional.__dict__,
            "state_sha256": preprocessing_state_sha256(provisional),
        }
    )
    issues = validate_preprocessing_state(state)
    if issues:
        raise ValueError("; ".join(issues))
    return state


def _verify_state_binding(
    bundle: FoldDatasetBundle,
    state: FoldPreprocessingState,
) -> None:
    issues = validate_preprocessing_state(state)
    if issues:
        raise ValueError("; ".join(issues))
    if state.fold_id != bundle.fold_id:
        raise ValueError("preprocessing state cannot be reused across folds")
    if state.schedule_sha256 != bundle.schedule_sha256:
        raise ValueError("preprocessing state is bound to a different schedule")
    if state.train_transitions_sha256 != bundle.train.transitions_sha256:
        raise ValueError("preprocessing state is bound to different train transitions")
    if state.feature_columns != bundle.feature_columns:
        raise ValueError("preprocessing state feature order differs from the fold bundle")
    fit = _authorized_fit_frame(bundle, split="train", fit_dates=None)
    if canonical_frame_sha256(fit) != state.fit_rows_sha256:
        raise ValueError("preprocessing state is bound to different fit rows")


def _transform_slice(
    dataset: FoldDatasetSlice,
    state: FoldPreprocessingState,
) -> pd.DataFrame:
    if dataset.fold_id != state.fold_id:
        raise ValueError("preprocessing state cannot transform another fold")
    transformed = dataset.features.copy()
    before_protected = transformed[
        [column for column in transformed.columns if column in PROTECTED_RAW_COLUMNS]
    ].copy()
    for parameter in state.parameters:
        values = pd.to_numeric(transformed[parameter.feature], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(
                f"fold slice contains nonfinite values for {parameter.feature}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = (values.astype(float) - parameter.mean) / parameter.scale
        if not np.isfinite(scaled.to_numpy(dtype=np.float64)).all():
            raise ValueError(
                f"preprocessing produced nonfinite values for {parameter.feature}"
            )
        transformed[parameter.feature] = scaled
    if not before_protected.equals(transformed[list(before_protected.columns)]):
        raise ValueError("preprocessing changed protected raw columns")
    return transformed


def transform_fold_dataset(
    bundle: FoldDatasetBundle,
    state: FoldPreprocessingState,
) -> PreprocessedFoldDataset:
    """Apply one verified training state to both slices without refitting."""

    _verify_state_binding(bundle, state)
    train = _transform_slice(bundle.train, state)
    validation = _transform_slice(bundle.validation, state)
    return PreprocessedFoldDataset(
        fold_id=bundle.fold_id,
        state_sha256=state.state_sha256,
        train_features=train,
        validation_features=validation,
        train_features_sha256=canonical_frame_sha256(train),
        validation_features_sha256=canonical_frame_sha256(validation),
    )
