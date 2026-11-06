from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from src.envs.long_term_env import LongTermTradingEnv
from src.models.feature_extractors import AssetTemporalExtractor, build_policy_kwargs


def _features(n_dates: int = 60, tickers=("AAA", "BBB", "CCC")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=n_dates)
    market = rng.normal(0, 0.01, n_dates)
    rows = []
    for t in tickers:
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_dates))
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t, "close": close[i], "ret": close[i] / close[max(i - 1, 0)] - 1, "mkt": market[i]})
    return pd.DataFrame(rows)


def test_extractor_output_shape_and_market_branch() -> None:
    env = LongTermTradingEnv(_features(), ["ret", "mkt"], lookback=5, action_transform="softmax")
    extractor = AssetTemporalExtractor(
        env.observation_space, lookback=5, num_assets=3, num_features=2, market_feature_indices=[1], hidden_size=8, fusion_size=4
    )
    obs = torch.as_tensor(np.stack([env.reset()[0], env.reset()[0]]))
    out = extractor(obs)
    assert out.shape == (2, 3 * 4 + 4 + 1)
    assert extractor.features_dim == 3 * 4 + 4 + 1


def test_extractor_is_permutation_equivariant_over_assets() -> None:
    torch.manual_seed(0)
    lookback, n, f = 4, 3, 2
    space_dim = lookback * n * f + n + 1
    import gymnasium as gym

    space = gym.spaces.Box(-np.inf, np.inf, shape=(space_dim,), dtype=np.float32)
    ext = AssetTemporalExtractor(space, lookback=lookback, num_assets=n, num_features=f, hidden_size=6, fusion_size=5)
    window = torch.randn(1, lookback, n, f)
    weights = torch.tensor([[0.2, 0.3, 0.1]])
    cash = torch.tensor([[0.4]])
    obs = torch.cat([window.reshape(1, -1), weights, cash], dim=1)
    perm = [2, 0, 1]
    obs_perm = torch.cat([window[:, :, perm, :].reshape(1, -1), weights[:, perm], cash], dim=1)
    fused = ext(obs)[:, : n * 5].reshape(n, 5)
    fused_perm = ext(obs_perm)[:, : n * 5].reshape(n, 5)
    assert torch.allclose(fused[perm], fused_perm, atol=1e-6)


def test_build_policy_kwargs_resolves_market_columns() -> None:
    kwargs = build_policy_kwargs(
        {"features_extractor": "asset_lstm", "market_feature_columns": ["mkt"], "hidden_size": 8, "net_arch": [16]},
        lookback=5,
        num_assets=3,
        feature_columns=["ret", "mkt"],
    )
    assert kwargs["features_extractor_class"] is AssetTemporalExtractor
    assert kwargs["features_extractor_kwargs"]["market_feature_indices"] == [1]
    assert kwargs["net_arch"] == [16]
    assert build_policy_kwargs(None, lookback=5, num_assets=3, feature_columns=["ret"]) == {}
    with pytest.raises(ValueError):
        build_policy_kwargs({"features_extractor": "asset_lstm", "market_feature_columns": ["nope"]}, lookback=5, num_assets=3, feature_columns=["ret"])
    with pytest.raises(ValueError):
        build_policy_kwargs({"features_extractor": "transformer"}, lookback=5, num_assets=3, feature_columns=["ret"])
    with pytest.raises(ValueError):
        build_policy_kwargs({"features_extractor": "asset_lstm", "layers": 3}, lookback=5, num_assets=3, feature_columns=["ret"])


def test_ppo_trains_with_fusion_extractor_and_softmax_actions() -> None:
    from stable_baselines3 import PPO

    env = LongTermTradingEnv(
        _features(), ["ret", "mkt"], lookback=5, action_transform="softmax", accounting="self_financing"
    )
    kwargs = build_policy_kwargs(
        {"features_extractor": "asset_lstm", "market_feature_columns": ["mkt"], "hidden_size": 8, "fusion_size": 4},
        lookback=5,
        num_assets=3,
        feature_columns=["ret", "mkt"],
    )
    model = PPO("MlpPolicy", env, n_steps=32, batch_size=16, n_epochs=1, seed=0, policy_kwargs=kwargs, verbose=0)
    model.learn(total_timesteps=64)
    action, _ = model.predict(env.reset()[0], deterministic=True)
    assert action.shape == (4,)
