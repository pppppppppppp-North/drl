"""Multi-source fusion feature extractors for Stable-Baselines3 policies.

The trading environment observation is ``[window (lookback x assets x features), weights (assets), cash (1)]``.
A flat MLP over ``lookback x assets x features`` grows linearly with the universe and has no notion of
time or of assets sharing structure. ``AssetTemporalExtractor`` instead:

1. encodes every asset's past window with one shared recurrent encoder (LSTM, optionally
   bidirectional *within the past window only*, so no future information is used);
2. encodes market-wide features (index, macro, investor flow, ...) once with a separate encoder;
3. fuses each asset embedding with the market embedding and the asset's current weight; and
4. returns per-asset embeddings plus a pooled cross-asset summary and the cash weight.
"""
from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AssetTemporalExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        lookback: int,
        num_assets: int,
        num_features: int,
        market_feature_indices: Sequence[int] | None = None,
        hidden_size: int = 32,
        fusion_size: int = 32,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        market = sorted({int(i) for i in (market_feature_indices or [])})
        if any(i < 0 or i >= num_features for i in market):
            raise ValueError("market_feature_indices must index the feature columns")
        asset_indices = [i for i in range(num_features) if i not in market]
        if not asset_indices:
            raise ValueError("at least one asset-level feature is required")
        expected = lookback * num_assets * num_features + num_assets + 1
        if int(np.prod(observation_space.shape)) != expected:
            raise ValueError(
                f"observation size {int(np.prod(observation_space.shape))} does not match "
                f"lookback*assets*features + assets + 1 = {expected}"
            )
        features_dim = num_assets * fusion_size + fusion_size + 1
        super().__init__(observation_space, features_dim)

        self.lookback = int(lookback)
        self.num_assets = int(num_assets)
        self.num_features = int(num_features)
        self.register_buffer("asset_index", torch.as_tensor(asset_indices, dtype=torch.long), persistent=False)
        self.register_buffer("market_index", torch.as_tensor(market, dtype=torch.long), persistent=False)
        directions = 2 if bidirectional else 1

        self.asset_encoder = nn.LSTM(
            input_size=len(asset_indices), hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional
        )
        self.market_encoder = (
            nn.LSTM(input_size=len(market), hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
            if market
            else None
        )
        market_dim = hidden_size * directions if market else 0
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * directions + market_dim + 1, fusion_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_size, fusion_size),
            nn.ReLU(),
        )

    @staticmethod
    def _last_state(encoder: nn.LSTM, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = encoder(sequence)
        # hidden: (num_layers * directions, batch, hidden); concatenate the final layer's directions.
        directions = 2 if encoder.bidirectional else 1
        return torch.cat([hidden[-d] for d in range(directions, 0, -1)], dim=-1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        window_size = self.lookback * self.num_assets * self.num_features
        window = observations[:, :window_size].reshape(batch, self.lookback, self.num_assets, self.num_features)
        weights = observations[:, window_size : window_size + self.num_assets]
        cash = observations[:, -1:]

        asset_seq = window.index_select(-1, self.asset_index)  # (B, L, N, Fa)
        asset_seq = asset_seq.permute(0, 2, 1, 3).reshape(batch * self.num_assets, self.lookback, -1)
        asset_state = self._last_state(self.asset_encoder, asset_seq).reshape(batch, self.num_assets, -1)

        parts = [asset_state]
        if self.market_encoder is not None:
            # Market-wide columns are identical across assets; read them from the first asset slot.
            market_seq = window[:, :, 0, :].index_select(-1, self.market_index)
            market_state = self._last_state(self.market_encoder, market_seq)
            parts.append(market_state.unsqueeze(1).expand(-1, self.num_assets, -1))
        parts.append(weights.unsqueeze(-1))
        fused = self.fusion(torch.cat(parts, dim=-1))  # (B, N, fusion)
        pooled = fused.mean(dim=1)
        return torch.cat([fused.reshape(batch, -1), pooled, cash], dim=-1)


FEATURE_EXTRACTORS = {"asset_lstm": AssetTemporalExtractor}


def build_policy_kwargs(
    policy_config: dict | None,
    *,
    lookback: int,
    num_assets: int,
    feature_columns: Sequence[str],
) -> dict:
    """Translate ``training.policy`` config into SB3 ``policy_kwargs``.

    Example config::

        training:
          policy:
            features_extractor: asset_lstm
            market_feature_columns: [set_return_1d, set_volatility_20d]
            hidden_size: 32
            fusion_size: 32
            bidirectional: false
            net_arch: [64, 64]
    """

    if not policy_config:
        return {}
    config = dict(policy_config)
    kwargs: dict = {}
    if "net_arch" in config:
        kwargs["net_arch"] = config.pop("net_arch")
    name = config.pop("features_extractor", None)
    if name is None:
        if config:
            raise ValueError(f"unknown training.policy keys without a features_extractor: {sorted(config)}")
        return kwargs
    if name not in FEATURE_EXTRACTORS:
        raise ValueError(f"unknown features_extractor {name!r}; choose from {sorted(FEATURE_EXTRACTORS)}")
    market_columns = list(config.pop("market_feature_columns", []))
    unknown = [c for c in market_columns if c not in feature_columns]
    if unknown:
        raise ValueError(f"market_feature_columns are not model features: {unknown}")
    extractor_kwargs = {
        "lookback": int(lookback),
        "num_assets": int(num_assets),
        "num_features": len(feature_columns),
        "market_feature_indices": [list(feature_columns).index(c) for c in market_columns],
    }
    allowed = {"hidden_size", "fusion_size", "bidirectional", "dropout"}
    extra = set(config) - allowed
    if extra:
        raise ValueError(f"unknown features_extractor options: {sorted(extra)}")
    extractor_kwargs.update(config)
    kwargs["features_extractor_class"] = FEATURE_EXTRACTORS[name]
    kwargs["features_extractor_kwargs"] = extractor_kwargs
    return kwargs
