from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.point_in_time_panel import session_membership_mask
from src.envs.long_term_env import LongTermRewardConfig
from src.envs.point_in_time_env import PointInTimeSlotEnv

STOCK = ["return_1d", "volatility_20d"]
MARKET = ["mkt_return_1d"]


def _panel(n_dates: int = 80, tickers=("A.BK", "B.BK", "C.BK"), value=None) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for i, ticker in enumerate(tickers):
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n_dates))
        for t, date in enumerate(dates):
            rows.append({
                "date": date, "ticker": ticker, "close": close[t], "history_sessions": t + 1,
                "rank_value": (value or {}).get(ticker, 10.0 - i), "return_1d": rng.normal(),
                "volatility_20d": abs(rng.normal()), "mkt_return_1d": float(t) / 100,
            })
    return pd.DataFrame(rows)


def _membership(panel: pd.DataFrame, spans: dict[str, tuple[int, int]]) -> pd.DataFrame:
    dates = sorted(panel["date"].unique())
    rows = []
    for ticker, (first, last) in spans.items():
        rows += [{"date": d, "ticker": ticker, "price_ticker": ticker} for d in dates[first:last]]
    return pd.DataFrame(rows)


def _env(panel, membership, **kwargs) -> PointInTimeSlotEnv:
    options = dict(num_slots=3, lookback=5, min_history_sessions=1, transaction_cost=0.001, slippage=0.0,
                   reward_config=LongTermRewardConfig(type="log_return", turnover_penalty=0.0, drawdown_penalty=0.0))
    options.update(kwargs)
    return PointInTimeSlotEnv(panel, membership, STOCK, MARKET, **options)


def test_membership_applies_from_the_session_after_it_starts() -> None:
    panel = _panel()
    env = _env(panel, _membership(panel, {"A.BK": (0, 80), "B.BK": (0, 80), "C.BK": (50, 80)}))
    c = env.tickers.index("C.BK")
    assert not env.tradable[50, c]  # the list in force at session 49 does not include C
    assert env.tradable[51, c]


def test_non_members_get_no_weight_and_leavers_are_sold_on_any_session() -> None:
    panel = _panel()
    env = _env(panel, _membership(panel, {"A.BK": (0, 80), "B.BK": (0, 60), "C.BK": (50, 80)}), rebalance_every=1000)
    env.reset(seed=0)
    b, c = env.tickers.index("B.BK"), env.tickers.index("C.BK")
    _, _, _, _, info = env.step(np.zeros(4))  # the only rebalance: equal logits over A, B and cash
    assert set(info["holdings"]) == {"A.BK", "B.BK"}
    forced = []
    done = False
    while not done:
        _, _, done, _, info = env.step(np.zeros(4))
        forced.append((info["date"], info["forced_exit_turnover"]))
        assert env.weights[c] == 0.0  # C joins later but is never bought without a rebalance
        if env.current_step > 61:
            assert env.weights[b] == 0.0
    sold = [date for date, amount in forced if amount > 0]
    assert len(sold) == 1  # B is sold once, at the session after its last listed session


def test_masked_softmax_puts_nothing_in_empty_slots() -> None:
    panel = _panel(tickers=("A.BK", "B.BK"))
    env = _env(panel, _membership(panel, {"A.BK": (0, 80), "B.BK": (0, 80)}))
    env.reset(seed=0)
    action = np.array([0.0, 0.0, 5.0, -5.0])  # slot 3 is empty but has the largest logit
    weights = env._slot_weights(action, env.slots[env.current_step])
    assert weights[2] == 0.0 and weights.sum() == pytest.approx(1.0, abs=1e-2)


def test_slots_are_ordered_by_trailing_traded_value() -> None:
    panel = _panel(value={"A.BK": 1.0, "B.BK": 3.0, "C.BK": 2.0})
    env = _env(panel, _membership(panel, {t: (0, 80) for t in ("A.BK", "B.BK", "C.BK")}))
    assert env.slot_tickers(40) == ["B.BK", "C.BK", "A.BK"]


def test_accounting_reconciles_and_weights_stay_a_budget() -> None:
    panel = _panel()
    env = _env(panel, _membership(panel, {"A.BK": (0, 70), "B.BK": (0, 80), "C.BK": (30, 80)}), rebalance_every=5)
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    value, done = env.initial_cash, False
    while not done:
        _, _, done, _, info = env.step(rng.normal(size=4))
        value *= 1 + info["net_return"]
        assert sum(info["holdings"].values()) + info["cash_weight"] == pytest.approx(1.0)
        assert info["portfolio_value"] == pytest.approx(value)


def test_observation_uses_only_past_sessions_and_fits_the_lstm_extractor() -> None:
    torch = pytest.importorskip("torch")
    from src.models.feature_extractors import AssetTemporalExtractor

    panel = _panel()
    env = _env(panel, _membership(panel, {t: (0, 80) for t in ("A.BK", "B.BK", "C.BK")}))
    obs, _ = env.reset(seed=0)
    width = len(env.feature_columns)
    window = obs[: env.lookback * env.num_slots * width].reshape(env.lookback, env.num_slots, width)
    # The market column is t/100 for session t; the newest row is the session before the decision.
    assert window[-1, 0, len(STOCK)] == pytest.approx((env.current_step - 1) / 100)
    extractor = AssetTemporalExtractor(env.observation_space, lookback=env.lookback, num_assets=env.num_slots,
                                       num_features=width, market_feature_indices=[len(STOCK)])
    assert extractor(torch.as_tensor(obs[None])).shape[0] == 1


def test_session_mask_applies_a_revision_from_its_publication_date() -> None:
    membership = pd.DataFrame(
        [{"record_id": rid, "effective_from": start, "effective_to_exclusive": end, "published_at": pub,
          "ticker": f"S{i}.BK", "in_set50": str(i < 50)}
         for rid, start, end, pub in (("a", "2020-01-01", "2020-01-08", "2019-12-15"),
                                      ("b", "2020-01-08", "2020-02-01", "2020-01-08"))
         for i in range(100)]
    )
    membership.loc[(membership.record_id == "b") & (membership.ticker == "S0.BK"), "ticker"] = "NEW.BK"
    price_map = membership[["record_id", "ticker"]].assign(price_ticker=membership["ticker"], coverage="own_symbol")
    sessions = pd.DatetimeIndex(pd.bdate_range("2020-01-02", "2020-01-31"))
    mask = session_membership_mask(membership, price_map, sessions)

    on = lambda day: set(mask.loc[mask["date"] == pd.Timestamp(day), "ticker"])  # noqa: E731
    assert "S0.BK" in on("2020-01-07") and "NEW.BK" not in on("2020-01-07")
    assert "NEW.BK" in on("2020-01-08") and "S0.BK" not in on("2020-01-08")


def test_validation_window_keeps_lookback_context_and_never_passes_its_end() -> None:
    from scripts.run_pit_walk_forward import window

    panel = _panel(n_dates=60)
    dates = sorted(panel["date"].unique())
    val = window(panel, str(pd.Timestamp(dates[40]).date()), str(pd.Timestamp(dates[50]).date()), lookback=5)
    kept = sorted(val["date"].unique())
    assert kept[0] == dates[35] and kept[-1] == dates[50]
    with pytest.raises(ValueError, match="lookback"):
        window(panel, str(pd.Timestamp(dates[2]).date()), str(pd.Timestamp(dates[50]).date()), lookback=5)


def test_ridge_baseline_uses_the_completed_row_and_only_tradable_slots() -> None:
    from scripts.pit_universe_baselines import fixed_universe_mask, ridge_policy
    from src.agents.predictive_baselines import RidgeForecaster

    panel = _panel()
    env = _env(panel, _membership(panel, {"A.BK": (0, 80), "B.BK": (0, 80)}))
    env.reset(seed=0)
    forecaster = RidgeForecaster(STOCK + MARKET)
    forecaster.mean_, forecaster.scale_ = np.zeros(3), np.ones(3)
    forecaster.coef_, forecaster.intercept_ = np.array([1.0, 0.0, 0.0]), 0.0  # forecast = yesterday's return_1d
    weights = ridge_policy(forecaster, k=1)(None, env, 0)
    row = env.current_step - 1
    best = int(np.argmax(env.stock_tensor[row, env.slots[env.current_step][:2], 0]))
    expected = np.zeros(env.num_slots)
    if env.stock_tensor[row, env.slots[env.current_step][best], 0] > 0:
        expected[best] = 1.0
    assert np.array_equal(weights, expected) and weights[2] == 0.0

    fixed = fixed_universe_mask(panel, {"C.BK"})
    assert set(fixed["price_ticker"]) == {"C.BK"} and fixed["date"].nunique() == 80


def test_sessions_drop_zero_volume_holidays_and_series_come_from_one_file(tmp_path) -> None:
    from src.data.point_in_time_panel import load_prices, session_calendar

    base = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}
    first = pd.DataFrame([{"date": d, "ticker": t, "volume": v, **base}
                          for d, v in (("2020-01-02", 10), ("2020-01-03", 0)) for t in ("A.BK", "B.BK")])
    first.to_csv(tmp_path / "a.csv", index=False)
    first[first.ticker == "A.BK"].to_csv(tmp_path / "b.csv", index=False)
    prices = load_prices([tmp_path / "a.csv"])
    assert list(session_calendar(prices)) == [pd.Timestamp("2020-01-02")]
    with pytest.raises(ValueError, match="more than one price file"):
        load_prices([tmp_path / "a.csv", tmp_path / "b.csv"])


def _seats(env: PointInTimeSlotEnv, step: int) -> list[str | None]:
    return [env.tickers[u] if u >= 0 else None for u in env.slots[step]]


def _listed(panel: pd.DataFrame, ranges: dict[str, list[tuple[int, int]]]) -> pd.DataFrame:
    dates = sorted(panel["date"].unique())
    return pd.DataFrame([{"date": d, "ticker": t, "price_ticker": t}
                         for t, spans in ranges.items() for first, last in spans for d in dates[first:last]])


def test_sticky_slots_do_not_move_members_when_their_ranks_change() -> None:
    panel = _panel(value={"A.BK": 3.0, "B.BK": 2.0, "C.BK": 1.0})
    later = panel["date"] >= sorted(panel["date"].unique())[40]
    panel.loc[later & (panel["ticker"] == "C.BK"), "rank_value"] = 9.0  # C becomes the most traded
    members = _membership(panel, {t: (0, 80) for t in ("A.BK", "B.BK", "C.BK")})
    by_value, sticky = _env(panel, members), _env(panel, members, slot_order="sticky")
    assert _seats(by_value, 30) == ["A.BK", "B.BK", "C.BK"] and _seats(by_value, 50) == ["C.BK", "A.BK", "B.BK"]
    assert _seats(sticky, 30) == _seats(sticky, 50) == ["A.BK", "B.BK", "C.BK"]
    with pytest.raises(ValueError, match="slot_order"):
        _env(panel, members, slot_order="alphabetical")


def test_sticky_slots_seat_rejoiners_in_their_old_slot_and_joiners_in_the_lowest_free_one() -> None:
    tickers = ("A.BK", "B.BK", "C.BK", "D.BK", "E.BK")
    panel = _panel(tickers=tickers, value={"A.BK": 10.0, "D.BK": 9.0, "C.BK": 8.0, "B.BK": 7.0, "E.BK": 6.0})
    members = _listed(panel, {"A.BK": [(0, 80)], "C.BK": [(0, 80)], "D.BK": [(0, 20)],
                              "B.BK": [(0, 30), (50, 80)], "E.BK": [(60, 80)]})
    env = _env(panel, members, num_slots=4, slot_order="sticky")
    assert _seats(env, 5) == ["A.BK", "D.BK", "C.BK", "B.BK"]
    assert _seats(env, 25) == ["A.BK", None, "C.BK", "B.BK"]  # D left; nobody moves up
    assert _seats(env, 40) == ["A.BK", None, "C.BK", None]
    assert _seats(env, 55) == ["A.BK", None, "C.BK", "B.BK"]  # B returns to its old slot, not the lowest free one
    assert _seats(env, 65) == ["A.BK", "E.BK", "C.BK", "B.BK"]  # a new joiner takes the lowest free slot
    assert _seats(_env(panel, members, num_slots=4), 65) == ["A.BK", "C.BK", "B.BK", "E.BK"]


def test_accounting_reconciles_with_sticky_slots_and_gaps() -> None:
    panel = _panel(tickers=("A.BK", "B.BK", "C.BK", "D.BK"))
    members = _listed(panel, {"A.BK": [(0, 80)], "B.BK": [(0, 35)], "C.BK": [(0, 80)], "D.BK": [(45, 80)]})
    env = _env(panel, members, num_slots=4, slot_order="sticky", rebalance_every=3)
    env.reset(seed=0)
    rng = np.random.default_rng(2)
    value, done = env.initial_cash, False
    while not done:
        slots = env.slots[env.current_step]
        _, _, done, _, info = env.step(rng.normal(size=5))
        value *= 1 + info["net_return"]
        assert sum(info["holdings"].values()) + info["cash_weight"] == pytest.approx(1.0)
        assert info["portfolio_value"] == pytest.approx(value)
        assert set(info["holdings"]) <= {env.tickers[u] for u in slots if u >= 0}
