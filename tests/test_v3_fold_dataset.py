from __future__ import annotations

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
from src.v3.trading_env import PointInTimeThaiStockTradingEnv


POLICY_PATH = Path("config/v3/evaluation_policy_draft.yaml")


def _fixture() -> tuple[
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
    feature_rows = []
    mask_rows = []
    for day, date in enumerate(sessions):
        for slot, ticker in enumerate(tickers):
            close = 100.0 + day * 0.05 + slot
            feature_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close - 0.2,
                    "high": close + 0.4,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1_000.0 + day + slot,
                    "signal": float(day * 2 + slot),
                    "constant_signal": 7.0,
                    "net_return": float(day) / 100_000.0,
                }
            )
            mask_rows.append(
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
        pd.DataFrame(feature_rows),
        pd.DataFrame(mask_rows),
    )


def _materialize(
    *,
    fold_id: str = "fold_03",
    features: pd.DataFrame | None = None,
    mask: pd.DataFrame | None = None,
    schedule: pd.DataFrame | None = None,
):
    sessions, policy, canonical_schedule, tickers, base_features, base_mask = _fixture()
    return materialize_fold_dataset(
        base_features if features is None else features,
        base_mask if mask is None else mask,
        canonical_schedule if schedule is None else schedule,
        sessions=sessions,
        policy=policy,
        fold_id=fold_id,
        feature_columns=("signal", "constant_signal"),
        asset_universe=tickers,
    )


def test_all_fold_slices_match_canonical_owned_transitions() -> None:
    sessions, policy, schedule, tickers, features, mask = _fixture()
    for fold_id in sorted(schedule["fold_id"].unique()):
        bundle = materialize_fold_dataset(
            features,
            mask,
            schedule,
            sessions=sessions,
            policy=policy,
            fold_id=fold_id,
            feature_columns=("signal", "constant_signal"),
            asset_universe=tickers,
        )
        fold = schedule[schedule["fold_id"].eq(fold_id)]
        assert len(bundle.train.transitions) == int(fold["train_decision"].sum())
        assert len(bundle.validation.transitions) == int(
            fold["validation_decision"].sum()
        )
        for split in (bundle.train, bundle.validation):
            assert len(split.dates) == bundle.lookback + len(split.transitions) + 1
            assert len(split.features) == len(split.dates) * len(tickers)
            assert split.transitions["split"].eq(split.split).all()
            assert split.transitions["decision_date"].tolist() == list(
                split.dates[bundle.lookback : -1]
            )
            assert split.transitions["outcome_date"].tolist() == list(
                split.dates[bundle.lookback + 1 :]
            )
            assert split.transitions["schedule_row_sha256"].str.fullmatch(
                r"[0-9a-f]{64}"
            ).all()


def test_materialized_validation_slice_drives_only_owned_environment_steps() -> None:
    bundle = _materialize(fold_id="fold_04")
    validation = bundle.validation
    env = PointInTimeThaiStockTradingEnv(
        features=validation.features,
        feature_columns=list(bundle.feature_columns),
        eligibility_mask=validation.eligibility_mask,
        asset_universe=bundle.asset_universe,
        lookback=bundle.lookback,
        transaction_cost=0.0,
    )
    env.reset(seed=21)
    observed = []
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(
            np.zeros(len(bundle.asset_universe), dtype=np.float64)
        )
        observed.append((info["decision_date"], info["date"]))
        done = terminated or truncated

    expected = list(
        validation.transitions[["decision_date", "outcome_date"]].itertuples(
            index=False, name=None
        )
    )
    assert observed == expected


def test_feature_and_mask_row_order_do_not_change_fold_artifacts() -> None:
    sessions, policy, schedule, tickers, features, mask = _fixture()
    first = materialize_fold_dataset(
        features,
        mask,
        schedule,
        sessions=sessions,
        policy=policy,
        fold_id="fold_05",
        feature_columns=("signal", "constant_signal"),
        asset_universe=tickers,
    )
    second = materialize_fold_dataset(
        features.sample(frac=1.0, random_state=11),
        mask.sample(frac=1.0, random_state=12),
        schedule,
        sessions=list(reversed(sessions)),
        policy=policy,
        fold_id="fold_05",
        feature_columns=("signal", "constant_signal"),
        asset_universe=tickers,
    )

    assert first.schedule_sha256 == second.schedule_sha256
    assert first.source_features_sha256 == second.source_features_sha256
    assert first.source_eligibility_sha256 == second.source_eligibility_sha256
    assert first.train.features_sha256 == second.train.features_sha256
    assert first.validation.transitions_sha256 == second.validation.transitions_sha256
    pd.testing.assert_frame_equal(first.train.features, second.train.features)
    pd.testing.assert_frame_equal(
        first.validation.eligibility_mask,
        second.validation.eligibility_mask,
    )


@pytest.mark.parametrize("kind", ["missing", "duplicate", "ticker", "date"])
def test_feature_grid_fails_closed_on_missing_duplicate_or_orphan_rows(
    kind: str,
) -> None:
    sessions, policy, schedule, tickers, features, mask = _fixture()
    if kind == "missing":
        changed = features.iloc[1:].copy()
        match = "missing required session/ticker"
    elif kind == "duplicate":
        changed = pd.concat([features, features.iloc[[0]]], ignore_index=True)
        match = "duplicate date/ticker"
    elif kind == "ticker":
        orphan = features.iloc[[0]].copy()
        orphan["ticker"] = "ORPHAN.BK"
        changed = pd.concat([features, orphan], ignore_index=True)
        match = "outside asset_universe"
    else:
        orphan = features.iloc[[0]].copy()
        orphan["date"] = pd.Timestamp("2025-01-02")
        changed = pd.concat([features, orphan], ignore_index=True)
        match = "unauthorized session dates"

    with pytest.raises(ValueError, match=match):
        materialize_fold_dataset(
            changed,
            mask,
            schedule,
            sessions=sessions,
            policy=policy,
            fold_id="fold_01",
            feature_columns=("signal", "constant_signal"),
            asset_universe=tickers,
        )


def test_mask_gaps_and_schedule_tampering_fail_closed() -> None:
    sessions, policy, schedule, tickers, features, mask = _fixture()
    with pytest.raises(ValueError, match="expected"):
        materialize_fold_dataset(
            features,
            mask.iloc[1:].copy(),
            schedule,
            sessions=sessions,
            policy=policy,
            fold_id="fold_02",
            feature_columns=("signal", "constant_signal"),
            asset_universe=tickers,
        )

    changed_schedule = schedule.copy()
    decision = changed_schedule["validation_decision"].idxmax()
    changed_schedule.loc[decision, "transition_owner"] = "train"
    with pytest.raises(ValueError, match="canonical policy materialization"):
        materialize_fold_dataset(
            features,
            mask,
            changed_schedule,
            sessions=sessions,
            policy=policy,
            fold_id="fold_02",
            feature_columns=("signal", "constant_signal"),
            asset_universe=tickers,
        )


def test_nonfinite_model_features_and_bad_active_valuations_are_rejected() -> None:
    sessions, policy, schedule, tickers, features, mask = _fixture()
    nonfinite = features.copy()
    nonfinite.loc[0, "signal"] = np.nan
    with pytest.raises(ValueError, match="nonfinite model feature"):
        materialize_fold_dataset(
            nonfinite,
            mask,
            schedule,
            sessions=sessions,
            policy=policy,
            fold_id="fold_01",
            feature_columns=("signal", "constant_signal"),
            asset_universe=tickers,
        )

    bad_price = features.copy()
    first_decision = schedule.loc[
        schedule["fold_id"].eq("fold_01") & schedule["train_decision"], "date"
    ].iloc[0]
    bad_price.loc[
        bad_price["date"].eq(pd.Timestamp(first_decision))
        & bad_price["ticker"].eq(tickers[0]),
        "close",
    ] = 0.0
    with pytest.raises(ValueError, match="finite positive valuation"):
        materialize_fold_dataset(
            bad_price,
            mask,
            schedule,
            sessions=sessions,
            policy=policy,
            fold_id="fold_01",
            feature_columns=("signal", "constant_signal"),
            asset_universe=tickers,
        )
