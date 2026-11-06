from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.envs.long_term_env import (
    LongTermRewardConfig,
    LongTermTradingEnv,
    differential_downside_ratio,
    differential_sharpe,
)


def _panel(prices: dict[str, list[float]]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(next(iter(prices.values()))))
    rows = []
    for ticker, series in prices.items():
        for date, close in zip(dates, series):
            rows.append({"date": date, "ticker": ticker, "close": close, "f1": close / 100.0})
    return pd.DataFrame(rows)


def _env(prices=None, **kwargs) -> LongTermTradingEnv:
    prices = prices or {"AAA": [100, 110, 121, 110, 121, 133.1], "BBB": [100, 100, 100, 100, 100, 100]}
    return LongTermTradingEnv(_panel(prices), ["f1"], lookback=1, transaction_cost=0.01, **kwargs)


def test_defaults_reproduce_legacy_cash_leg_turnover() -> None:
    env = _env()
    _, reward, _, _, info = env.step(np.array([0.5, 0.0]))
    # Legacy turnover adds the cash leg: |0.5| + |0.5 - 1.0| = 1.0.
    assert info["turnover"] == pytest.approx(1.0)
    expected_net = 0.5 * 0.10 - 0.01 * 1.0
    assert info["net_return"] == pytest.approx(expected_net)
    assert reward == pytest.approx(expected_net - 0.05 * 1.0 - 0.1 * 0.0)
    # Legacy keeps the target weight even though prices moved.
    assert info["weights"][0] == pytest.approx(0.5)


def test_self_financing_counts_security_turnover_once_and_drifts_weights() -> None:
    env = _env(accounting="self_financing")
    _, _, _, _, info = env.step(np.array([0.5, 0.0]))
    assert info["turnover"] == pytest.approx(0.5)
    gross = 0.5 * 0.10
    assert info["net_return"] == pytest.approx((1 - 0.01 * 0.5) * (1 + gross) - 1)
    drifted = 0.5 * 1.10 / (1 + gross)
    assert info["weights"][0] == pytest.approx(drifted)
    assert info["cash_weight"] == pytest.approx(1 - drifted)
    # Holding the drifted weights costs nothing on the next decision.
    _, _, _, _, info2 = env.step(info["weights"])
    assert info2["turnover"] == pytest.approx(0.0, abs=1e-12)
    assert info2["transaction_cost_paid"] == pytest.approx(0.0, abs=1e-12)


def test_log_return_reward_and_incremental_drawdown() -> None:
    env = _env(
        accounting="self_financing",
        reward_config=LongTermRewardConfig(type="log_return", turnover_penalty=0.0, drawdown_penalty=1.0, drawdown_mode="increment"),
    )
    env.step(np.array([1.0, 0.0]))  # 110 -> 121: +10% less the entry cost
    _, reward_down, _, _, info_down = env.step(np.array([1.0, 0.0]))  # 121 -> 110: new drawdown
    dd = info_down["drawdown"]
    assert dd > 0
    assert reward_down == pytest.approx(math.log1p(info_down["net_return"]) - dd)
    _, reward_up, _, _, info_up = env.step(np.array([1.0, 0.0]))  # 110 -> 121: drawdown shrinks
    assert reward_up == pytest.approx(math.log1p(info_up["net_return"]))


def test_level_drawdown_mode_matches_legacy_definition() -> None:
    cfg = LongTermRewardConfig(type="simple", turnover_penalty=0.0, drawdown_penalty=1.0)
    env = _env(accounting="self_financing", reward_config=cfg)
    for _ in range(3):
        _, reward, _, _, info = env.step(np.array([1.0, 0.0]))
    assert reward == pytest.approx(info["net_return"] - info["drawdown"])


def test_plan_main_adds_clipped_differential_sharpe() -> None:
    cfg = LongTermRewardConfig(type="plan_main", turnover_penalty=0.0, drawdown_penalty=0.0, sharpe_weight=0.5, sharpe_eta=0.1)
    env = _env(accounting="self_financing", reward_config=cfg)
    _, first, _, _, info = env.step(np.array([1.0, 0.0]))
    # No variance history on the first step: the differential Sharpe term is zero.
    assert info["differential_ratio"] == 0.0
    assert first == pytest.approx(math.log1p(info["net_return"]))
    _, second, _, _, info2 = env.step(np.array([1.0, 0.0]))
    assert abs(info2["differential_ratio"]) <= cfg.sharpe_clip
    assert second == pytest.approx(math.log1p(info2["net_return"]) + 0.5 * info2["differential_ratio"])


def test_differential_ratios_are_defined_and_signed() -> None:
    assert differential_sharpe(0.01, 0.0, 0.0) == 0.0
    assert differential_sharpe(0.03, 0.01, 0.0002) > 0
    assert differential_sharpe(-0.03, 0.01, 0.0002) < 0
    assert differential_downside_ratio(0.01, 0.0, 0.0) == 0.0
    assert differential_downside_ratio(0.02, 0.001, 0.0001) > 0
    assert differential_downside_ratio(-0.02, 0.001, 0.0001) < 0


def test_rebalance_every_holds_positions_between_decisions() -> None:
    env = _env(accounting="self_financing", rebalance_every=2)
    _, _, _, _, info1 = env.step(np.array([1.0, 0.0]))
    assert info1["rebalanced"] is True
    _, _, _, _, info2 = env.step(np.array([0.0, 1.0]))  # ignored: not a rebalance step
    assert info2["rebalanced"] is False
    assert info2["turnover"] == pytest.approx(0.0, abs=1e-12)
    assert info2["target_weights"][0] == pytest.approx(info1["weights"][0])
    _, _, _, _, info3 = env.step(np.array([0.0, 1.0]))
    assert info3["rebalanced"] is True
    assert info3["target_weights"][1] == pytest.approx(1.0)


def test_random_start_episode_is_seeded_and_bounded() -> None:
    prices = {"AAA": list(np.linspace(100, 140, 40)), "BBB": list(np.linspace(100, 90, 40))}
    starts = set()
    for seed in range(8):
        env = _env(prices, random_start=True, episode_length=10)
        env.reset(seed=seed)
        starts.add(env.start_step)
        assert env.lookback <= env.start_step <= len(env.dates) - 1 - 10
        steps, done = 0, False
        while not done:
            _, _, done, _, _ = env.step(np.array([0.5, 0.5]))
            steps += 1
        assert steps == 10
    assert len(starts) > 1
    a, b = _env(prices, random_start=True, episode_length=10), _env(prices, random_start=True, episode_length=10)
    a.reset(seed=3)
    b.reset(seed=3)
    assert a.start_step == b.start_step


def test_softmax_actions_include_cash_and_accept_direct_weights() -> None:
    env = _env(action_transform="softmax", accounting="self_financing")
    assert env.action_space.shape == (3,)
    _, _, _, _, info = env.step(np.array([5.0, -5.0, -5.0]))
    assert info["target_weights"].sum() <= 1.0 + 1e-12
    assert info["target_weights"][0] > 0.99
    _, _, _, _, info2 = env.step(np.array([0.0, 0.0, 5.0]))  # mostly cash
    assert info2["target_weights"].sum() < 0.02
    _, _, _, _, info3 = env.step(np.array([0.25, 0.25]))  # direct N-length weights (baselines)
    assert info3["target_weights"] == pytest.approx([0.25, 0.25])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "unknown"},
        {"drawdown_mode": "average"},
        {"sharpe_eta": 0.0},
        {"sharpe_clip": 0.0},
        {"scale": -1.0},
    ],
)
def test_reward_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        LongTermRewardConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"accounting": "fifo"}, {"rebalance_every": 0}, {"episode_length": 0}, {"action_transform": "tanh"}],
)
def test_env_rejects_invalid_options(kwargs) -> None:
    with pytest.raises(ValueError):
        _env(**kwargs)


def test_train_helper_enables_random_start_only_for_training() -> None:
    from src.train_long_term import make_long_term_env_from_features

    prices = {"AAA": list(np.linspace(100, 140, 40)), "BBB": list(np.linspace(100, 90, 40))}
    config = {
        "features": {"columns": ["f1"], "lookback": 1},
        "environment": {
            "initial_cash": 1_000_000.0,
            "transaction_cost": 0.001,
            "accounting": "self_financing",
            "rebalance_every": 5,
            "action_transform": "softmax",
            "train_random_start": True,
            "train_episode_length": 10,
            "reward": {"type": "log_return", "turnover_penalty": 0.0, "drawdown_penalty": 0.0},
        },
    }
    train_env = make_long_term_env_from_features(config, _panel(prices), training=True)
    eval_env = make_long_term_env_from_features(config, _panel(prices))
    assert train_env.random_start and train_env.episode_length == 10
    assert not eval_env.random_start and eval_env.episode_length is None
    assert eval_env.start_step == eval_env.lookback
    assert eval_env.rebalance_every == 5 and eval_env.accounting == "self_financing"
    assert eval_env.reward_config.type == "log_return"


def test_defaults_are_bit_identical_to_locked_v2_environment() -> None:
    """With default options the long-term env must reproduce the locked V2 env exactly."""

    from src.envs.trading_env import RewardConfig as V2RewardConfig
    from src.envs.trading_env import ThaiStockTradingEnv as V2Env

    rng = np.random.default_rng(2026)
    checked = 0
    for trial in range(20):
        n_assets, n_dates, n_feats = int(rng.integers(1, 5)), int(rng.integers(12, 40)), int(rng.integers(1, 3))
        lookback = int(rng.integers(1, 6))
        dates = pd.bdate_range("2020-01-01", periods=n_dates)
        rows = []
        for a in range(n_assets):
            close = 100 * np.cumprod(1 + rng.normal(0, 0.02, n_dates))
            for i, d in enumerate(dates):
                row = {"date": d, "ticker": f"T{a}", "close": close[i]}
                row.update({f"f{f}": rng.normal() if rng.random() > 0.05 else np.nan for f in range(n_feats)})
                rows.append(row)
        frame = pd.DataFrame(rows)
        kwargs = dict(
            feature_columns=[f"f{f}" for f in range(n_feats)],
            lookback=lookback,
            transaction_cost=float(rng.uniform(0, 0.01)),
            slippage=float(rng.uniform(0, 0.005)),
            max_turnover=float(rng.uniform(0.1, 2.0)) if rng.random() < 0.5 else None,
            max_position_weight=float(rng.uniform(0.2, 1.0)),
            min_cash_weight=float(rng.uniform(0, 0.3)),
        )
        reward = dict(
            return_weight=float(rng.uniform(0.5, 2)),
            turnover_penalty=float(rng.uniform(0, 0.1)),
            drawdown_penalty=float(rng.uniform(0, 0.5)),
        )
        old = V2Env(frame, reward_config=V2RewardConfig(**reward), **kwargs)
        new = LongTermTradingEnv(frame, reward_config=LongTermRewardConfig(**reward), **kwargs)
        assert old.observation_space.shape == new.observation_space.shape
        assert old.action_space.shape == new.action_space.shape
        assert np.array_equal(old.reset(seed=trial)[0], new.reset(seed=trial)[0])
        done = False
        while not done:
            action = rng.uniform(-0.5, 1.5, n_assets)
            o1, r1, d1, _, i1 = old.step(action)
            o2, r2, d2, _, i2 = new.step(action)
            assert np.array_equal(o1, o2) and r1 == r2 and d1 == d2
            for key, value in i1.items():
                if isinstance(value, np.ndarray):
                    assert np.array_equal(value, i2[key]), key
                else:
                    assert value == i2[key], key
            checked += 1
            done = d1
    assert checked > 100
