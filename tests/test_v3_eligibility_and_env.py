from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.envs.trading_env import RewardConfig
from src.v3.baselines import active_equal_weight_action
from src.v3.eligibility import (
    DEVELOPMENT_ELIGIBILITY_KIND,
    build_development_eligibility_mask,
    validate_development_eligibility_mask,
)
from src.v3.membership_mask import (
    MASK_KIND,
    REVISION_AVAILABILITY_RULE,
    validate_membership_panel,
)
from src.v3.trading_env import PointInTimeThaiStockTradingEnv


def _dates(count: int = 8) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=count)


def _membership_panel(
    *,
    count: int = 8,
    tickers: tuple[str, ...] = ("AAA.BK", "BBB.BK"),
) -> pd.DataFrame:
    rows = []
    for date in _dates(count):
        for slot, ticker in enumerate(tickers):
            rows.append(
                {
                    "date": date,
                    "asset_slot": slot,
                    "ticker": ticker,
                    "index_name": "SET100",
                    "is_index_member": True,
                    "membership_record_id": "synthetic_structural_record",
                    "membership_published_at": pd.Timestamp("2023-12-01"),
                    "revision_status": "original",
                    "sector_as_of": "Synthetic Sector",
                    "symbol": ticker.removesuffix(".BK"),
                    "company_name": ticker,
                    "source_url": "https://example.invalid/structural-test",
                    "source_file_sha256": "a" * 64,
                    "revision_availability_rule": REVISION_AVAILABILITY_RULE,
                    "mask_kind": MASK_KIND,
                }
            )
    return validate_membership_panel(pd.DataFrame(rows))


def _market_rows(
    *,
    count: int = 8,
    tickers: tuple[str, ...] = ("AAA.BK", "BBB.BK"),
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "open": 100.0 + day,
                "high": 101.0 + day,
                "low": 99.0 + day,
                "close": 100.0 + day,
                "volume": 1_000.0,
            }
            for day, date in enumerate(_dates(count))
            for ticker in tickers
        ]
    )


def _eligibility_mask(
    pattern: list[tuple[bool, bool]],
    *,
    count: int = 8,
    tickers: tuple[str, str] = ("AAA.BK", "BBB.BK"),
) -> pd.DataFrame:
    rows = []
    for date, active in zip(_dates(count), pattern, strict=True):
        for slot, (ticker, eligible) in enumerate(
            zip(tickers, active, strict=True)
        ):
            rows.append(
                {
                    "date": date,
                    "asset_slot": slot,
                    "ticker": ticker,
                    "is_index_member": eligible,
                    "has_observed_bar": True,
                    "has_minimum_history": True,
                    "is_eligible": eligible,
                    "information_available_at": date,
                    "mask_kind": DEVELOPMENT_ELIGIBILITY_KIND,
                }
            )
    return validate_development_eligibility_mask(pd.DataFrame(rows))


def _features(*, count: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "close": 100.0 + day,
                "signal": float(day + slot),
            }
            for day, date in enumerate(_dates(count))
            for slot, ticker in enumerate(("AAA.BK", "BBB.BK"))
        ]
    )


def _env(mask: pd.DataFrame, **kwargs) -> PointInTimeThaiStockTradingEnv:
    return PointInTimeThaiStockTradingEnv(
        features=_features(),
        feature_columns=["signal"],
        eligibility_mask=mask,
        asset_universe=["AAA.BK", "BBB.BK"],
        lookback=2,
        transaction_cost=kwargs.pop("transaction_cost", 0.001),
        reward_config=RewardConfig(
            return_weight=1.0,
            turnover_penalty=0.0,
            drawdown_penalty=0.0,
        ),
        **kwargs,
    )


def test_development_eligibility_uses_only_current_and_prior_bars() -> None:
    membership = _membership_panel()
    market = _market_rows()
    mask = build_development_eligibility_mask(
        membership,
        market,
        lookback=2,
    )

    counts = mask.groupby("date")["is_eligible"].sum()
    assert counts.iloc[:2].eq(0).all()
    assert counts.iloc[2:].eq(2).all()
    assert (mask["information_available_at"] <= mask["date"]).all()

    missing_next = market[
        ~(
            market["date"].eq(_dates()[4])
            & market["ticker"].eq("AAA.BK")
        )
    ]
    changed = build_development_eligibility_mask(
        membership,
        missing_next,
        lookback=2,
    )
    today = changed[
        changed["date"].eq(_dates()[3]) & changed["ticker"].eq("AAA.BK")
    ].iloc[0]
    next_day = changed[
        changed["date"].eq(_dates()[4]) & changed["ticker"].eq("AAA.BK")
    ].iloc[0]
    assert today["is_eligible"]
    assert not next_day["is_eligible"]


def test_eligibility_validator_rejects_future_information() -> None:
    mask = build_development_eligibility_mask(
        _membership_panel(),
        _market_rows(),
        lookback=2,
    )
    tampered = mask.copy()
    tampered.loc[tampered.index[0], "information_available_at"] = (
        tampered.loc[tampered.index[0], "date"] + pd.Timedelta(days=1)
    )

    with pytest.raises(ValueError, match="future information"):
        validate_development_eligibility_mask(tampered)


def test_eligibility_history_is_invariant_to_future_market_suffix() -> None:
    membership = _membership_panel()
    market = _market_rows()
    original = build_development_eligibility_mask(
        membership,
        market,
        lookback=2,
    )
    cutoff = _dates()[3]
    perturbed = market.copy()
    future = perturbed["date"].gt(cutoff)
    perturbed.loc[future, ["open", "high", "low", "close"]] *= 100.0
    perturbed = perturbed[
        ~(
            perturbed["date"].eq(_dates()[6])
            & perturbed["ticker"].eq("AAA.BK")
        )
    ]
    changed = build_development_eligibility_mask(
        membership,
        perturbed,
        lookback=2,
    )

    columns = [
        "date",
        "asset_slot",
        "ticker",
        "is_index_member",
        "has_observed_bar",
        "has_minimum_history",
        "is_eligible",
        "information_available_at",
        "mask_kind",
    ]
    pd.testing.assert_frame_equal(
        original.loc[original["date"].le(cutoff), columns].reset_index(drop=True),
        changed.loc[changed["date"].le(cutoff), columns].reset_index(drop=True),
    )


def test_v3_observation_exposes_mask_and_inactive_action_stays_cash() -> None:
    pattern = [(True, True), (True, True), (False, True)] + [(False, True)] * 5
    env = _env(_eligibility_mask(pattern))
    observation, info = env.reset(seed=7)

    assert observation.shape == env.observation_space.shape == (9,)
    assert info["action_mask"].tolist() == [False, True]
    _, _, _, _, info = env.step(np.array([1.0, 0.0]))
    assert np.allclose(info["weights"], [0.0, 0.0])
    assert info["cash_weight"] == pytest.approx(1.0)


def test_forced_exit_overrides_turnover_cap_and_charges_cost_once() -> None:
    pattern = [(True, True), (True, True)] + [(False, True)] * 6
    env = _env(
        _eligibility_mask(pattern),
        max_turnover=0.1,
        transaction_cost=0.01,
    )
    env.reset(seed=3)
    env.weights = np.array([0.5, 0.0], dtype=np.float64)
    env.cash_weight = 0.5

    _, _, _, _, info = env.step(np.array([1.0, 0.0]))

    assert info["weights"][0] == pytest.approx(0.0)
    assert info["forced_exit_turnover"] == pytest.approx(0.5)
    assert info["turnover"] == pytest.approx(0.5)
    assert info["transaction_cost_paid"] == pytest.approx(0.005)
    assert info["turnover_limit_overridden"] is True


def test_environment_rejects_missing_next_valuation_for_eligible_asset() -> None:
    mask = _eligibility_mask([(True, True)] * 8)
    features = _features()
    features.loc[
        features["date"].eq(_dates()[3])
        & features["ticker"].eq("AAA.BK"),
        "close",
    ] = np.nan

    with pytest.raises(ValueError, match="current/next valuation prices"):
        PointInTimeThaiStockTradingEnv(
            features=features,
            feature_columns=["signal"],
            eligibility_mask=mask,
            asset_universe=["AAA.BK", "BBB.BK"],
            lookback=2,
        )


def test_membership_only_panel_cannot_be_used_as_eligibility() -> None:
    with pytest.raises(ValueError, match="eligibility mask is missing columns"):
        PointInTimeThaiStockTradingEnv(
            features=_features(),
            feature_columns=["signal"],
            eligibility_mask=_membership_panel(),
            asset_universe=["AAA.BK", "BBB.BK"],
            lookback=2,
        )


def test_fixed_asset_slots_survive_missing_rows_for_inactive_asset() -> None:
    pattern = [(True, False)] * 8
    mask = _eligibility_mask(pattern)
    features = _features()
    features = features[features["ticker"].ne("BBB.BK")].reset_index(drop=True)

    env = PointInTimeThaiStockTradingEnv(
        features=features,
        feature_columns=["signal"],
        eligibility_mask=mask,
        asset_universe=["AAA.BK", "BBB.BK"],
        lookback=2,
    )

    assert env.action_space.shape == (2,)
    _, _, _, _, info = env.step(np.array([0.5, 0.5]))
    assert info["weights"][1] == pytest.approx(0.0)


def test_all_inactive_session_forces_cash_only() -> None:
    pattern = [(True, True), (True, True)] + [(False, False)] * 6
    env = _env(_eligibility_mask(pattern))
    env.reset(seed=9)

    _, _, _, _, info = env.step(np.array([1.0, 1.0]))

    assert np.allclose(info["weights"], [0.0, 0.0])
    assert info["cash_weight"] == pytest.approx(1.0)
    assert info["eligible_count"] == 0


def test_active_equal_weight_uses_only_current_mask() -> None:
    action = active_equal_weight_action(np.array([True, False, True, False]))
    assert np.allclose(action, [0.5, 0.0, 0.5, 0.0])
    assert np.allclose(
        active_equal_weight_action(np.array([False, False])),
        [0.0, 0.0],
    )


def test_non_alphabetical_asset_slots_stay_aligned_with_prices() -> None:
    tickers = ("BBB.BK", "AAA.BK")
    dates = _dates()
    features = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "close": (
                    100.0
                    if ticker == "AAA.BK"
                    else (100.0 if day <= 2 else 200.0)
                ),
                "signal": float(slot),
            }
            for day, date in enumerate(dates)
            for slot, ticker in enumerate(tickers)
        ]
    )
    mask = _eligibility_mask(
        [(True, True)] * len(dates),
        tickers=tickers,
    )
    env = PointInTimeThaiStockTradingEnv(
        features=features,
        feature_columns=["signal"],
        eligibility_mask=mask,
        asset_universe=tickers,
        lookback=2,
        transaction_cost=0.0,
    )

    env.reset(seed=11)
    _, _, _, _, info = env.step(np.array([1.0, 0.0]))

    assert info["net_return"] == pytest.approx(1.0)
    assert info["weights"].tolist() == pytest.approx([1.0, 0.0])


def test_step_info_separates_decision_mask_from_next_observation_mask() -> None:
    pattern = [
        (True, True),
        (True, True),
        (True, False),
        (False, True),
        (False, True),
        (False, True),
        (False, True),
        (False, True),
    ]
    env = _env(_eligibility_mask(pattern))
    _, reset_info = env.reset(seed=13)

    assert reset_info["action_mask"].tolist() == [True, False]
    assert reset_info["next_action_mask"].tolist() == [True, False]

    _, _, _, _, info = env.step(np.array([1.0, 0.0]))

    assert info["decision_date"] == _dates()[2].date().isoformat()
    assert info["date"] == _dates()[3].date().isoformat()
    assert info["action_mask"].tolist() == [True, False]
    assert info["eligible_count"] == 1
    assert info["next_action_mask"].tolist() == [False, True]
    assert info["next_eligible_count"] == 1


def test_turnover_and_cost_use_one_way_security_notional() -> None:
    env = _env(
        _eligibility_mask([(True, True)] * 8),
        transaction_cost=0.01,
    )
    env.reset(seed=17)

    _, _, _, _, info = env.step(np.array([0.5, 0.0]))

    assert info["turnover"] == pytest.approx(0.5)
    assert info["transaction_cost_paid"] == pytest.approx(0.005)


def test_weights_drift_with_asset_returns_before_next_rebalance() -> None:
    dates = _dates()
    features = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "close": (
                    200.0
                    if ticker == "AAA.BK" and day >= 3
                    else 100.0
                ),
                "signal": float(day + slot),
            }
            for day, date in enumerate(dates)
            for slot, ticker in enumerate(("AAA.BK", "BBB.BK"))
        ]
    )
    env = PointInTimeThaiStockTradingEnv(
        features=features,
        feature_columns=["signal"],
        eligibility_mask=_eligibility_mask([(True, True)] * 8),
        asset_universe=["AAA.BK", "BBB.BK"],
        lookback=2,
        transaction_cost=0.0,
    )
    env.reset(seed=19)

    _, _, _, _, info = env.step(np.array([0.5, 0.5]))

    assert info["net_return"] == pytest.approx(0.5)
    assert info["weights"].tolist() == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert info["cash_weight"] == pytest.approx(0.0)
