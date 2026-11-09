from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

import gymnasium as gym  # noqa: E402
from stable_baselines3.common.policies import ActorCriticPolicy  # noqa: E402

from src.envs.long_term_env import LongTermRewardConfig  # noqa: E402
from src.envs.point_in_time_env import PointInTimeSlotEnv  # noqa: E402
from src.models.feature_extractors import AssetTemporalExtractor, build_policy_kwargs  # noqa: E402
from src.models.shared_asset_policy import (  # noqa: E402
    SharedAssetPolicy,
    SlotSetExtractor,
    feature_statistics,
    policy_from_config,
)

L, N, F = 4, 5, 4  # feature columns: two stock features, one market feature, slot_valid
MARKET, VALID = 2, 3
MEAN = np.array([0.5, -1.0, 0.1, 0.0], dtype=np.float32)
STD = np.array([2.0, 0.5, 3.0, 1.0], dtype=np.float32)
EXTRACTOR = dict(lookback=L, num_assets=N, num_features=F, market_feature_indices=[MARKET], hidden_size=6, fusion_size=5)


def _spaces():
    observation = gym.spaces.Box(-np.inf, np.inf, shape=(L * N * F + N + 1,), dtype=np.float32)
    action = gym.spaces.Box(-5.0, 5.0, shape=(N + 1,), dtype=np.float32)
    return observation, action


def _policy(seed: int = 0, large_outputs: bool = True) -> SharedAssetPolicy:
    torch.manual_seed(seed)
    policy = SharedAssetPolicy(*_spaces(), lambda _: 3e-4, net_arch=[8, 8], features_extractor_class=SlotSetExtractor,
                               features_extractor_kwargs=dict(EXTRACTOR, valid_feature_index=VALID,
                                                              feature_mean=MEAN.tolist(), feature_std=STD.tolist()))
    if large_outputs:  # make the logits O(1) so that the comparisons below are not trivially small
        for layer in policy.mlp_extractor.output_layers():
            torch.nn.init.normal_(layer.weight)
    return policy


def _observation(valid=(1, 1, 0, 1, 1), seed: int = 1):
    rng = np.random.default_rng(seed)
    window = rng.normal(size=(L, N, F)).astype(np.float32)
    window[:, :, MARKET] = rng.normal(size=(L, 1))  # market features repeat in every slot, as in the environment
    valid = np.asarray(valid, dtype=np.float32)
    window[:, :, VALID] = valid
    window[:, valid == 0, :MARKET] = 0.0  # empty slots carry no stock features
    weights = (rng.dirichlet(np.ones(N + 1))[:N] * valid).astype(np.float32)
    cash = np.array([1.0 - weights.sum()], dtype=np.float32)
    return window, weights, cash


def _flat(window, weights, cash) -> torch.Tensor:
    return torch.as_tensor(np.concatenate([window.reshape(-1), weights, cash]))[None]


def _outputs(policy, obs):
    with torch.no_grad():
        return policy.get_distribution(obs).distribution.mean[0], policy.predict_values(obs)[0, 0]


def test_actor_and_critic_do_not_depend_on_the_slot_order() -> None:
    policy = _policy()
    window, weights, cash = _observation()
    perm = np.array([3, 0, 4, 2, 1])  # new slot j holds the stock of old slot perm[j]
    obs, obs_perm = _flat(window, weights, cash), _flat(window[:, perm], weights[perm], cash)
    mean, value = _outputs(policy, obs)
    mean_perm, value_perm = _outputs(policy, obs_perm)
    assert mean[:N].abs().max() > 0.1
    assert torch.allclose(mean_perm[:N], mean[:N][perm], atol=1e-5)
    assert torch.allclose(mean_perm[N], mean[N], atol=1e-5)
    assert torch.allclose(value_perm, value, atol=1e-5)

    action = torch.randn(1, N + 1)
    action_perm = torch.cat([action[:, :N][:, perm], action[:, N:]], dim=1)
    with torch.no_grad():
        log_prob = policy.get_distribution(obs).log_prob(action)
        log_prob_perm = policy.get_distribution(obs_perm).log_prob(action_perm)
    assert torch.allclose(log_prob, log_prob_perm, atol=1e-5)
    log_std = policy.log_std.detach()
    assert log_std.shape == (N + 1,) and torch.all(log_std[:N] == log_std[0])


def test_the_dense_head_does_depend_on_the_slot_order() -> None:
    torch.manual_seed(0)
    dense = ActorCriticPolicy(*_spaces(), lambda _: 3e-4, net_arch=[8, 8], features_extractor_class=AssetTemporalExtractor,
                              features_extractor_kwargs=EXTRACTOR)
    torch.nn.init.normal_(dense.action_net.weight)
    window, weights, cash = _observation()
    perm = np.array([3, 0, 4, 2, 1])
    mean, _ = _outputs(dense, _flat(window, weights, cash))
    mean_perm, _ = _outputs(dense, _flat(window[:, perm], weights[perm], cash))
    assert not torch.allclose(mean_perm[:N], mean[:N][perm], atol=1e-3)


def test_empty_slots_change_neither_the_policy_nor_its_log_probability() -> None:
    policy = _policy()
    window, weights, cash = _observation(valid=(1, 1, 0, 1, 0))
    noisy = window.copy()
    noisy[:, [2, 4], :MARKET] = 50.0 * np.random.default_rng(2).normal(size=(L, 2, MARKET))
    obs = _flat(window, weights, cash)
    mean, value = _outputs(policy, obs)
    mean_noisy, value_noisy = _outputs(policy, _flat(noisy, weights, cash))
    kept = [0, 1, 3, N]
    assert torch.allclose(mean_noisy[kept], mean[kept], atol=1e-6)
    assert torch.allclose(value_noisy, value, atol=1e-6)

    action = torch.randn(1, N + 1)
    other = action.clone()
    other[0, [2, 4]] += 3.0
    with torch.no_grad():
        distribution = policy.get_distribution(obs)
        assert torch.allclose(distribution.log_prob(action), distribution.log_prob(other))
        # three stocks in slots plus cash, each a unit Gaussian at initialization
        assert distribution.entropy().item() == pytest.approx(4 * 0.5 * np.log(2 * np.pi * np.e), rel=1e-5)


def test_features_are_standardized_with_one_set_of_statistics_for_all_slots() -> None:
    torch.manual_seed(0)
    space, _ = _spaces()
    with_stats = SlotSetExtractor(space, valid_feature_index=VALID, feature_mean=MEAN, feature_std=STD, **EXTRACTOR)
    without = SlotSetExtractor(space, valid_feature_index=VALID, **EXTRACTOR)
    without.load_state_dict(with_stats.state_dict())
    window, weights, cash = _observation()
    scaled = np.clip((window - MEAN) / STD, -10.0, 10.0).astype(np.float32)
    out = with_stats(_flat(window, weights, cash))
    assert torch.allclose(out, without(_flat(scaled, weights, cash)), atol=1e-6)
    assert out.shape == (1, N * 5 + N + 1) == (1, with_stats.features_dim)
    assert out[0, N * 5 : N * 5 + N].tolist() == [1.0, 1.0, 0.0, 1.0, 1.0]
    assert out[0, -1].item() == pytest.approx(float(cash[0]))
    with pytest.raises(ValueError):
        SlotSetExtractor(space, feature_mean=MEAN, **EXTRACTOR)
    with pytest.raises(ValueError):
        SlotSetExtractor(space, feature_mean=MEAN, feature_std=np.zeros(F), **EXTRACTOR)


def test_untrained_policy_starts_near_an_even_allocation() -> None:
    policy = _policy(large_outputs=False)
    mean, _ = _outputs(policy, _flat(*_observation()))
    assert mean.abs().max() < 0.1


def test_policy_rejects_a_mismatched_action_space_and_state_dependent_noise() -> None:
    observation, _ = _spaces()
    kwargs = dict(net_arch=[8], features_extractor_class=SlotSetExtractor, features_extractor_kwargs=EXTRACTOR)
    with pytest.raises(ValueError):
        SharedAssetPolicy(observation, gym.spaces.Box(-5.0, 5.0, shape=(N,), dtype=np.float32), lambda _: 3e-4, **kwargs)
    with pytest.raises(ValueError):
        SharedAssetPolicy(*_spaces(), lambda _: 3e-4, use_sde=True, **kwargs)


def test_feature_statistics_pool_all_stocks_and_count_each_session_once_for_market_features() -> None:
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-02"] * 2 + ["2020-01-03"] * 2),
        "ticker": ["A", "B"] * 3,
        "ret": [1.0, 3.0, np.nan, 5.0, 7.0, 9.0],
        "flat": [2.0] * 6,
        "mkt": [0.0, 0.0, 1.0, 1.0, 5.0, 5.0],
    })
    stats = feature_statistics(panel, ["ret", "flat"], ["mkt"])
    assert stats["ret"] == pytest.approx((5.0, np.std([1.0, 3.0, 5.0, 7.0, 9.0])))
    assert stats["flat"] == (2.0, 1.0)
    assert stats["mkt"] == pytest.approx((2.0, np.std([0.0, 1.0, 5.0])))


def test_policy_from_config_keeps_the_dense_default_and_builds_the_shared_actor() -> None:
    columns = ["ret", "vol", "mkt", "slot_valid"]
    base = {"features_extractor": "asset_lstm", "market_feature_columns": ["mkt"], "hidden_size": 8, "fusion_size": 4,
            "net_arch": [16]}
    common = dict(lookback=5, num_assets=3, feature_columns=columns)
    assert policy_from_config(base, **common) == ("MlpPolicy", build_policy_kwargs(base, **common))
    assert policy_from_config(None, **common) == ("MlpPolicy", {})

    stats = {"ret": (0.1, 2.0), "vol": (1.0, 0.5), "mkt": (0.0, 3.0)}
    shared = {**base, "actor": "shared_asset"}
    policy, kwargs = policy_from_config(shared, feature_stats=stats, **common)
    assert policy is SharedAssetPolicy
    assert kwargs["features_extractor_class"] is SlotSetExtractor
    assert kwargs["net_arch"] == [16]
    extractor = kwargs["features_extractor_kwargs"]
    assert extractor["valid_feature_index"] == 3
    assert extractor["feature_mean"] == [0.1, 1.0, 0.0, 0.0]
    assert extractor["feature_std"] == [2.0, 0.5, 3.0, 1.0]
    assert "feature_mean" not in policy_from_config({**shared, "standardize_features": False}, **common)[1]["features_extractor_kwargs"]

    with pytest.raises(ValueError, match="normalize_observations"):
        policy_from_config(shared, feature_stats=stats, normalize_observations=True, **common)
    with pytest.raises(ValueError, match="feature_stats"):
        policy_from_config(shared, **common)
    with pytest.raises(ValueError, match="asset_lstm"):
        policy_from_config({"actor": "shared_asset", "standardize_features": False}, **common)
    with pytest.raises(ValueError, match="actor"):
        policy_from_config({**base, "actor": "transformer"}, **common)
    with pytest.raises(ValueError, match="shared_asset"):
        policy_from_config({**base, "standardize_features": True}, **common)


def _pit_env(slot_order: str = "traded_value") -> tuple[PointInTimeSlotEnv, pd.DataFrame]:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=80)
    rows = []
    for ticker in ("A.BK", "B.BK", "C.BK", "D.BK"):
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, len(dates)))
        for t, date in enumerate(dates):
            rows.append({"date": date, "ticker": ticker, "close": close[t], "history_sessions": t + 1,
                         "rank_value": float(rng.normal()), "return_1d": rng.normal(), "volatility_20d": abs(rng.normal()),
                         "mkt_return_1d": float(t) / 100})
    panel = pd.DataFrame(rows)
    spans = {"A.BK": (0, 80), "B.BK": (0, 60), "C.BK": (30, 80), "D.BK": (0, 80)}
    membership = pd.DataFrame([{"date": d, "ticker": t, "price_ticker": t} for t, (a, b) in spans.items() for d in dates[a:b]])
    env = PointInTimeSlotEnv(panel, membership, ["return_1d", "volatility_20d"], ["mkt_return_1d"], num_slots=5, lookback=5,
                             min_history_sessions=1, transaction_cost=0.001, slippage=0.0, rebalance_every=2,
                             reward_config=LongTermRewardConfig(type="log_return", turnover_penalty=0.0, drawdown_penalty=0.0),
                             slot_order=slot_order)
    return env, panel


def test_deterministic_rollouts_are_the_same_under_either_slot_order() -> None:
    torch.manual_seed(0)
    envs = {order: _pit_env(order) for order in ("sticky", "traded_value")}
    sticky, panel = envs["sticky"]
    stats = feature_statistics(panel, ["return_1d", "volatility_20d"], ["mkt_return_1d"])
    policy, kwargs = policy_from_config(
        {"actor": "shared_asset", "features_extractor": "asset_lstm", "market_feature_columns": ["mkt_return_1d"],
         "hidden_size": 8, "fusion_size": 4, "net_arch": [16]},
        lookback=5, num_assets=5, feature_columns=sticky.feature_columns, feature_stats=stats)
    model = policy(sticky.observation_space, sticky.action_space, lambda _: 3e-4, **kwargs)
    for layer in model.mlp_extractor.output_layers():
        torch.nn.init.normal_(layer.weight)
    paths = {}
    for order, (env, _) in envs.items():
        obs, _ = env.reset(seed=0)
        done, values = False, []
        while not done:
            with torch.no_grad():
                action = model.get_distribution(torch.as_tensor(obs)[None]).distribution.mean[0].numpy()
            obs, _, done, _, info = env.step(action)
            values.append(info["portfolio_value"])
        paths[order] = np.array(values)
    traded = envs["traded_value"][0]
    assert (sticky.slots != traded.slots).any(axis=1).sum() > 10  # the two orders really seat members differently
    assert np.abs(paths["sticky"] / paths["sticky"][0] - 1).max() > 1e-3  # and the portfolio does move
    np.testing.assert_allclose(paths["sticky"], paths["traded_value"], rtol=1e-6)


def test_ppo_trains_saves_and_reloads_with_the_shared_actor(tmp_path) -> None:
    from stable_baselines3 import PPO

    env, panel = _pit_env()
    stats = feature_statistics(panel, ["return_1d", "volatility_20d"], ["mkt_return_1d"])
    policy, kwargs = policy_from_config(
        {"actor": "shared_asset", "features_extractor": "asset_lstm", "market_feature_columns": ["mkt_return_1d"],
         "hidden_size": 8, "fusion_size": 4, "net_arch": [16]},
        lookback=5, num_assets=5, feature_columns=env.feature_columns, feature_stats=stats)
    model = PPO(policy, env, n_steps=32, batch_size=16, n_epochs=2, seed=0, policy_kwargs=kwargs, verbose=0, device="cpu")
    model.learn(total_timesteps=64)
    obs = env.reset(seed=0)[0]
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (6,) and np.isfinite(action).all()
    model.save(tmp_path / "shared")
    loaded = PPO.load(tmp_path / "shared", device="cpu")
    assert isinstance(loaded.policy, SharedAssetPolicy)
    np.testing.assert_allclose(loaded.predict(obs, deterministic=True)[0], action, atol=1e-6)
