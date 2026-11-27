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


def _space(lookback: int, n: int, f: int):
    import gymnasium as gym

    return gym.spaces.Box(-np.inf, np.inf, shape=(lookback * n * f + n + 1,), dtype=np.float32)


def test_lstm_stays_the_default_temporal_encoder() -> None:
    space = _space(4, 3, 3)
    kwargs = dict(lookback=4, num_assets=3, num_features=3, market_feature_indices=[2], hidden_size=6, fusion_size=5)
    torch.manual_seed(0)
    default = AssetTemporalExtractor(space, **kwargs)
    torch.manual_seed(0)
    explicit = AssetTemporalExtractor(space, **kwargs, temporal_encoder="lstm")
    assert isinstance(default.asset_encoder, torch.nn.LSTM) and isinstance(default.market_encoder, torch.nn.LSTM)
    assert list(default.state_dict()) == list(explicit.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(default.state_dict().values(), explicit.state_dict().values()))
    obs = torch.randn(2, 4 * 3 * 3 + 3 + 1)
    assert torch.equal(default(obs), explicit(obs))


def test_mlp_temporal_encoder_has_the_same_output_and_stays_permutation_equivariant() -> None:
    torch.manual_seed(0)
    lookback, n, f = 4, 3, 3
    ext = AssetTemporalExtractor(_space(lookback, n, f), lookback=lookback, num_assets=n, num_features=f,
                                 market_feature_indices=[2], hidden_size=6, fusion_size=5, temporal_encoder="mlp")
    assert not isinstance(ext.asset_encoder, torch.nn.LSTM)
    assert ext.asset_encoder[0].in_features == lookback * 2 and ext.market_encoder[0].in_features == lookback * 1
    window = torch.randn(1, lookback, n, f)
    window[:, :, :, 2] = window[:, :, :1, 2]  # market column identical across assets, as in the environment
    weights, cash = torch.tensor([[0.2, 0.3, 0.1]]), torch.tensor([[0.4]])
    obs = torch.cat([window.reshape(1, -1), weights, cash], dim=1)
    perm = [2, 0, 1]
    obs_perm = torch.cat([window[:, :, perm, :].reshape(1, -1), weights[:, perm], cash], dim=1)
    out, out_perm = ext(obs), ext(obs_perm)
    assert out.shape == (1, n * 5 + 5 + 1)
    assert torch.allclose(out[:, : n * 5].reshape(n, 5)[perm], out_perm[:, : n * 5].reshape(n, 5), atol=1e-6)
    # The encoder reads the whole window: changing the oldest session changes the embedding.
    older = window.clone()
    older[:, 0, 0, 0] += 1.0
    obs_older = torch.cat([older.reshape(1, -1), weights, cash], dim=1)
    assert not torch.allclose(ext(obs_older)[:, :5], out[:, :5])


def test_temporal_encoder_option_is_checked() -> None:
    kwargs = build_policy_kwargs({"features_extractor": "asset_lstm", "temporal_encoder": "mlp"}, lookback=4,
                                 num_assets=3, feature_columns=["a", "b"])
    assert kwargs["features_extractor_kwargs"]["temporal_encoder"] == "mlp"
    with pytest.raises(ValueError, match="unknown temporal_encoder"):
        AssetTemporalExtractor(_space(4, 3, 2), lookback=4, num_assets=3, num_features=2, temporal_encoder="gru")
    with pytest.raises(ValueError, match="bidirectional"):
        AssetTemporalExtractor(_space(4, 3, 2), lookback=4, num_assets=3, num_features=2, bidirectional=True,
                               temporal_encoder="mlp")
