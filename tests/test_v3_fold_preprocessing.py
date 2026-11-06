from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v3.eligibility import DEVELOPMENT_ELIGIBILITY_KIND
from src.v3.evaluation_policy import (
    build_evaluation_schedule,
    validate_evaluation_policy,
)
from src.v3.fold_dataset import materialize_fold_dataset
from src.v3.fold_preprocessing import (
    fit_fold_preprocessor,
    preprocessing_state_sha256,
    transform_fold_dataset,
    validate_preprocessing_state,
)


POLICY_PATH = Path("config/v3/evaluation_policy_draft.yaml")


def _inputs() -> tuple[
    pd.DatetimeIndex,
    dict,
    pd.DataFrame,
    tuple[str, ...],
    pd.DataFrame,
    pd.DataFrame,
]:
    issues, policy = validate_evaluation_policy(POLICY_PATH)
    assert issues == []
    sessions = pd.bdate_range("2021-01-01", "2024-12-31")
    schedule, _ = build_evaluation_schedule(sessions, policy)
    tickers = ("BBB.BK", "AAA.BK")
    features = []
    eligibility = []
    for day, date in enumerate(sessions):
        for slot, ticker in enumerate(tickers):
            close = 90.0 + day * 0.1 + slot
            features.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close - 0.1,
                    "high": close + 0.3,
                    "low": close - 0.4,
                    "close": close,
                    "volume": 2_000.0 + day,
                    "signal": float(day * 3 + slot),
                    "constant_signal": 4.0,
                    "net_return": float(day) / 1_000_000.0,
                    "turnover": float(slot) / 10.0,
                }
            )
            eligibility.append(
                {
                    "date": date,
                    "asset_slot": slot,
                    "ticker": ticker,
                    "is_index_member": True,
                    "has_observed_bar": True,
                    "has_minimum_history": True,
                    "is_eligible": True,
                    "information_available_at": date,
                    "mask_kind": DEVELOPMENT_ELIGIBILITY_KIND,
                }
            )
    return (
        sessions,
        policy,
        schedule,
        tickers,
        pd.DataFrame(features),
        pd.DataFrame(eligibility),
    )


def _bundle(
    fold_id: str = "fold_03",
    *,
    features: pd.DataFrame | None = None,
    eligibility: pd.DataFrame | None = None,
):
    sessions, policy, schedule, tickers, base_features, base_eligibility = _inputs()
    return materialize_fold_dataset(
        base_features if features is None else features,
        base_eligibility if eligibility is None else eligibility,
        schedule,
        sessions=sessions,
        policy=policy,
        fold_id=fold_id,
        feature_columns=("signal", "constant_signal"),
        asset_universe=tickers,
    )


def test_population_zscore_uses_only_eligible_training_decision_rows() -> None:
    sessions, policy, schedule, tickers, features, eligibility = _inputs()
    fold_id = "fold_03"
    decision_date = pd.Timestamp(
        schedule.loc[
            schedule["fold_id"].eq(fold_id) & schedule["train_decision"],
            "date",
        ].iloc[10]
    )
    inactive = eligibility["date"].eq(decision_date) & eligibility["ticker"].eq(
        tickers[0]
    )
    eligibility.loc[inactive, "is_index_member"] = False
    eligibility.loc[inactive, "is_eligible"] = False
    features.loc[
        features["date"].eq(decision_date) & features["ticker"].eq(tickers[0]),
        "signal",
    ] = 9_999_999.0
    bundle = materialize_fold_dataset(
        features,
        eligibility,
        schedule,
        sessions=sessions,
        policy=policy,
        fold_id=fold_id,
        feature_columns=("signal", "constant_signal"),
        asset_universe=tickers,
    )
    state = fit_fold_preprocessor(bundle)

    train = bundle.train.features.merge(
        bundle.train.eligibility_mask[["date", "ticker", "is_eligible"]],
        on=["date", "ticker"],
        validate="one_to_one",
    )
    decision_dates = pd.to_datetime(bundle.train.transitions["decision_date"])
    expected = train[train["date"].isin(decision_dates) & train["is_eligible"]][
        "signal"
    ].to_numpy(dtype=np.float64)
    signal = state.parameters[0]
    constant = state.parameters[1]
    assert state.fit_row_count == len(expected)
    assert signal.mean == pytest.approx(float(np.mean(expected)))
    assert signal.scale == pytest.approx(float(np.std(expected, ddof=0)))
    assert signal.constant is False
    assert constant.mean == pytest.approx(4.0)
    assert constant.scale == pytest.approx(1.0)
    assert constant.constant is True
    assert validate_preprocessing_state(state) == []
    assert preprocessing_state_sha256(state) == state.state_sha256


def test_context_and_validation_values_cannot_affect_fitted_state() -> None:
    sessions, policy, schedule, tickers, features, eligibility = _inputs()
    base = materialize_fold_dataset(
        features,
        eligibility,
        schedule,
        sessions=sessions,
        policy=policy,
        fold_id="fold_04",
        feature_columns=("signal", "constant_signal"),
        asset_universe=tickers,
    )
    base_state = fit_fold_preprocessor(base)

    changed = features.copy()
    train_context_date = pd.Timestamp(base.train.dates[0])
    validation_date = pd.Timestamp(base.validation.transitions["decision_date"].iloc[0])
    changed.loc[changed["date"].eq(train_context_date), "signal"] += 10_000.0
    changed.loc[changed["date"].eq(validation_date), "signal"] -= 20_000.0
    changed_bundle = materialize_fold_dataset(
        changed,
        eligibility,
        schedule,
        sessions=sessions,
        policy=policy,
        fold_id="fold_04",
        feature_columns=("signal", "constant_signal"),
        asset_universe=tickers,
    )
    changed_state = fit_fold_preprocessor(changed_bundle)

    assert changed_state == base_state


def test_eligible_training_decision_change_updates_state_deterministically() -> None:
    sessions, policy, schedule, tickers, features, eligibility = _inputs()
    first = materialize_fold_dataset(
        features,
        eligibility,
        schedule,
        sessions=sessions,
        policy=policy,
        fold_id="fold_02",
        feature_columns=("signal", "constant_signal"),
        asset_universe=tickers,
    )
    first_state = fit_fold_preprocessor(first)
    decision_date = pd.Timestamp(first.train.transitions["decision_date"].iloc[0])
    changed = features.copy()
    changed.loc[
        changed["date"].eq(decision_date) & changed["ticker"].eq(tickers[0]),
        "signal",
    ] += 100.0
    second = materialize_fold_dataset(
        changed,
        eligibility,
        schedule,
        sessions=sessions,
        policy=policy,
        fold_id="fold_02",
        feature_columns=("signal", "constant_signal"),
        asset_universe=tickers,
    )
    second_state = fit_fold_preprocessor(second)

    assert first_state.fit_rows_sha256 != second_state.fit_rows_sha256
    assert first_state.state_sha256 != second_state.state_sha256
    assert fit_fold_preprocessor(second) == second_state


def test_context_validation_or_partial_fit_requests_are_rejected() -> None:
    bundle = _bundle("fold_01")
    with pytest.raises(ValueError, match="only on the train split"):
        fit_fold_preprocessor(bundle, split="validation")
    with pytest.raises(ValueError, match="exactly equal authorized train decisions"):
        fit_fold_preprocessor(
            bundle,
            fit_dates=[bundle.validation.transitions["decision_date"].iloc[0]],
        )
    with pytest.raises(ValueError, match="exactly equal authorized train decisions"):
        fit_fold_preprocessor(
            bundle,
            fit_dates=bundle.train.transitions["decision_date"].iloc[:-1],
        )
    with pytest.raises(ValueError, match="exactly equal authorized train decisions"):
        fit_fold_preprocessor(bundle, fit_dates=[bundle.train.dates[0]])


def test_transform_uses_one_state_and_preserves_protected_raw_columns() -> None:
    bundle = _bundle("fold_05")
    state = fit_fold_preprocessor(bundle)
    transformed = transform_fold_dataset(bundle, state)

    protected = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "net_return",
        "turnover",
    ]
    pd.testing.assert_frame_equal(
        bundle.train.features[protected],
        transformed.train_features[protected],
    )
    pd.testing.assert_frame_equal(
        bundle.validation.features[protected],
        transformed.validation_features[protected],
    )
    assert transformed.train_features["constant_signal"].eq(0.0).all()
    assert transformed.validation_features["constant_signal"].eq(0.0).all()
    fit_dates = pd.to_datetime(bundle.train.transitions["decision_date"])
    fit_values = transformed.train_features[
        transformed.train_features["date"].isin(fit_dates)
    ]["signal"].to_numpy(dtype=np.float64)
    assert float(np.mean(fit_values)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.std(fit_values, ddof=0)) == pytest.approx(1.0, abs=1e-12)
    assert transformed.state_sha256 == state.state_sha256


def test_state_reuse_across_folds_and_state_tampering_fail_closed() -> None:
    first = _bundle("fold_01")
    second = _bundle("fold_02")
    state = fit_fold_preprocessor(first)
    with pytest.raises(ValueError, match="cannot be reused across folds"):
        transform_fold_dataset(second, state)

    tampered = replace(state, fit_row_count=state.fit_row_count + 1)
    with pytest.raises(ValueError, match="SHA-256"):
        transform_fold_dataset(first, tampered)


@pytest.mark.parametrize("operation", ["division", "subtraction"])
def test_finite_inputs_that_overflow_scaling_are_rejected_before_output_hashing(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.v3 import fold_preprocessing

    sessions, policy, schedule, tickers, features, eligibility = _inputs()
    fold = schedule[schedule["fold_id"].eq("fold_01")]
    validation_start = pd.Timestamp(
        fold.loc[fold["validation_decision"], "date"].iloc[0]
    )
    if operation == "division":
        features["signal"] *= 1e-6
    else:
        train_dates = pd.to_datetime(fold.loc[fold["train_decision"], "date"])
        train_rows = eligibility["date"].isin(train_dates)
        eligibility.loc[train_rows, ["is_index_member", "is_eligible"]] = False
        fit_row = eligibility["date"].eq(train_dates.iloc[0]) & eligibility[
            "ticker"
        ].eq(tickers[0])
        eligibility.loc[fit_row, ["is_index_member", "is_eligible"]] = True
        features.loc[fit_row, "signal"] = -1e308
    features.loc[features["date"].eq(validation_start), "signal"] = 1e308
    bundle = materialize_fold_dataset(
        features,
        eligibility,
        schedule,
        sessions=sessions,
        policy=policy,
        fold_id="fold_01",
        feature_columns=("signal", "constant_signal"),
        asset_universe=tickers,
    )
    state = fit_fold_preprocessor(bundle)
    assert validate_preprocessing_state(state) == []
    assert np.isfinite(features[list(bundle.feature_columns)].to_numpy()).all()
    if operation == "division":
        assert 0.0 < state.parameters[0].scale < 1.0
    else:
        assert state.fit_row_count == 1
        assert state.parameters[0].mean == -1e308
        assert state.parameters[0].scale == 1.0

    original_hash = fold_preprocessing.canonical_frame_sha256

    def reject_output_hashing(frame: pd.DataFrame) -> str:
        if "close" in frame.columns:
            pytest.fail("overflowing transformation reached output hashing")
        return original_hash(frame)

    monkeypatch.setattr(
        fold_preprocessing, "canonical_frame_sha256", reject_output_hashing
    )
    with pytest.raises(ValueError, match="preprocessing produced nonfinite values for signal"):
        transform_fold_dataset(bundle, state)


@pytest.mark.parametrize("protected", ["close", "open", "volume", "net_return", "turnover"])
def test_protected_raw_columns_cannot_be_selected_for_scaling(protected: str) -> None:
    sessions, policy, schedule, tickers, features, eligibility = _inputs()
    bundle = materialize_fold_dataset(
        features,
        eligibility,
        schedule,
        sessions=sessions,
        policy=policy,
        fold_id="fold_01",
        feature_columns=(protected,),
        asset_universe=tickers,
    )
    with pytest.raises(ValueError, match="cannot be scaled"):
        fit_fold_preprocessor(bundle)


def test_source_row_order_does_not_change_state_or_transformed_hashes() -> None:
    sessions, policy, schedule, tickers, features, eligibility = _inputs()
    first = materialize_fold_dataset(
        features,
        eligibility,
        schedule,
        sessions=sessions,
        policy=policy,
        fold_id="fold_06",
        feature_columns=("signal", "constant_signal"),
        asset_universe=tickers,
    )
    second = materialize_fold_dataset(
        features.sample(frac=1.0, random_state=31),
        eligibility.sample(frac=1.0, random_state=32),
        schedule,
        sessions=list(reversed(sessions)),
        policy=policy,
        fold_id="fold_06",
        feature_columns=("signal", "constant_signal"),
        asset_universe=tickers,
    )
    first_state = fit_fold_preprocessor(first)
    second_state = fit_fold_preprocessor(second)
    first_transformed = transform_fold_dataset(first, first_state)
    second_transformed = transform_fold_dataset(second, second_state)

    assert first_state == second_state
    assert (
        first_transformed.train_features_sha256
        == second_transformed.train_features_sha256
    )
    assert (
        first_transformed.validation_features_sha256
        == second_transformed.validation_features_sha256
    )
