"""Actor-critic policy that scores every asset with the same weights (milestone M5).

With ``AssetTemporalExtractor`` and SB3's default ``MlpPolicy``, each asset is encoded with shared
weights, but the actor is a dense layer over all slots: a stock's logit depends on which slot it
occupies. In the point-in-time SET50 environment, where members move between slots, the first run
traced most of its turnover to that dependence. ``SharedAssetPolicy`` removes it:

- **Shared standardization:** each feature is standardized with one mean and one standard deviation
  for all assets, taken from the training years (``feature_statistics``). ``VecNormalize`` keeps
  separate statistics for every observation element, so it would make the policy depend on the slot.
- **Actor:** every asset's logit comes from the same network, applied to the asset's own embedding
  and a summary of all assets in slots (the mean of their embeddings) plus the cash weight. The cash
  logit depends only on the summary and the cash weight. Permuting the slots permutes the logits.
- **Noise:** one log standard deviation for all asset logits and one for cash, so sampled actions
  are equivariant too.
- **Empty slots** (``slot_valid`` = 0) are left out of the summary and out of the action's
  log-probability and entropy. The environment already gives them no weight, so their sampled
  logits carry no information for the policy gradient.
- **Critic:** a function of the summary and the cash weight, so the value ignores the slot order.

Select it with ``training.policy.actor: shared_asset`` (see ``policy_from_config``).
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

from src.models.feature_extractors import AssetTemporalExtractor, build_policy_kwargs

VALID_COLUMN = "slot_valid"


class SlotSetExtractor(AssetTemporalExtractor):
    """``AssetTemporalExtractor`` with shared feature standardization and the slot-valid mask.

    Output: ``[per-asset embeddings (assets x fusion_size), valid flags (assets), cash weight (1)]``.
    """

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
        valid_feature_index: int | None = None,
        feature_mean: Sequence[float] | None = None,
        feature_std: Sequence[float] | None = None,
        clip: float = 10.0,
        temporal_encoder: str = "lstm",
    ) -> None:
        super().__init__(observation_space, lookback, num_assets, num_features, market_feature_indices,
                         hidden_size=hidden_size, fusion_size=fusion_size, bidirectional=bidirectional,
                         dropout=dropout, temporal_encoder=temporal_encoder)
        self.fusion_size = int(fusion_size)
        self._features_dim = self.num_assets * self.fusion_size + self.num_assets + 1
        if valid_feature_index is not None and not 0 <= int(valid_feature_index) < self.num_features:
            raise ValueError("valid_feature_index must index the feature columns")
        self.valid_feature_index = None if valid_feature_index is None else int(valid_feature_index)
        if (feature_mean is None) != (feature_std is None):
            raise ValueError("pass both feature_mean and feature_std, or neither")
        self.standardize = feature_mean is not None
        mean = torch.zeros(self.num_features) if feature_mean is None else torch.as_tensor(feature_mean, dtype=torch.float32)
        std = torch.ones(self.num_features) if feature_std is None else torch.as_tensor(feature_std, dtype=torch.float32)
        if mean.shape != (self.num_features,) or std.shape != (self.num_features,):
            raise ValueError(f"feature_mean and feature_std need one value per feature ({self.num_features})")
        if not bool(torch.isfinite(mean).all()) or not bool((std > 0).all()):
            raise ValueError("feature_mean must be finite and feature_std positive")
        self.register_buffer("feature_mean", mean, persistent=False)
        self.register_buffer("feature_std", std, persistent=False)
        self.clip = float(clip)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        window_size = self.lookback * self.num_assets * self.num_features
        window = observations[:, :window_size].reshape(batch, self.lookback, self.num_assets, self.num_features)
        if self.valid_feature_index is None:
            valid = observations.new_ones(batch, self.num_assets)
        else:
            valid = (window[:, -1, :, self.valid_feature_index] > 0.5).to(observations.dtype)
        if self.standardize:
            window = ((window - self.feature_mean) / self.feature_std).clamp(-self.clip, self.clip)
            observations = torch.cat([window.reshape(batch, -1), observations[:, window_size:]], dim=1)
        fused = super().forward(observations)[:, : self.num_assets * self.fusion_size]
        return torch.cat([fused, valid, observations[:, -1:]], dim=1)


def _mlp(input_dim: int, hidden: Sequence[int], activation_fn: type[nn.Module], output_dim: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for size in hidden:
        layers += [nn.Linear(last, int(size)), activation_fn()]
        last = int(size)
    if output_dim is not None:
        layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class SharedAssetHead(nn.Module):
    """Plays the role of SB3's ``MlpExtractor`` for ``SharedAssetPolicy``.

    ``forward_actor`` returns ``[action means (assets + 1), valid flags (assets)]``; the policy splits
    them. ``forward_critic`` returns the critic's latent vector.
    """

    def __init__(self, num_assets: int, embed_dim: int, pi_arch: Sequence[int], vf_arch: Sequence[int],
                 activation_fn: type[nn.Module]) -> None:
        super().__init__()
        self.num_assets = int(num_assets)
        self.embed_dim = int(embed_dim)
        self.asset_scorer = _mlp(2 * self.embed_dim + 1, pi_arch, activation_fn, 1)
        self.cash_scorer = _mlp(self.embed_dim + 1, pi_arch, activation_fn, 1)
        self.critic = _mlp(self.embed_dim + 1, vf_arch, activation_fn)
        self.latent_dim_pi = 2 * self.num_assets + 1
        self.latent_dim_vf = int(vf_arch[-1]) if len(vf_arch) else self.embed_dim + 1

    def output_layers(self) -> list[nn.Linear]:
        return [self.asset_scorer[-1], self.cash_scorer[-1]]

    def _split(self, features: torch.Tensor):
        batch = features.shape[0]
        size = self.num_assets * self.embed_dim
        embeddings = features[:, :size].reshape(batch, self.num_assets, self.embed_dim)
        valid = features[:, size : size + self.num_assets]
        cash = features[:, -1:]
        count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        summary = (embeddings * valid.unsqueeze(-1)).sum(dim=1) / count
        return embeddings, valid, summary, cash

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        embeddings, valid, summary, cash = self._split(features)
        shared = torch.cat([summary, cash], dim=1)
        per_asset = torch.cat([embeddings, shared.unsqueeze(1).expand(-1, self.num_assets, -1)], dim=-1)
        asset_logits = self.asset_scorer(per_asset).squeeze(-1)
        cash_logit = self.cash_scorer(shared)
        return torch.cat([asset_logits, cash_logit, valid], dim=1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        _, _, summary, cash = self._split(features)
        return self.critic(torch.cat([summary, cash], dim=1))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class MaskedDiagGaussianDistribution(DiagGaussianDistribution):
    """Diagonal Gaussian whose log-probability and entropy count only unmasked action dimensions."""

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor,
                           mask: torch.Tensor | None = None) -> "MaskedDiagGaussianDistribution":
        super().proba_distribution(mean_actions, log_std)
        self.mask = mask
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution.log_prob(actions)
        if self.mask is not None:
            log_prob = log_prob * self.mask
        return log_prob.sum(dim=1)

    def entropy(self) -> torch.Tensor:
        entropy = self.distribution.entropy()
        if self.mask is not None:
            entropy = entropy * self.mask
        return entropy.sum(dim=1)


class SharedAssetPolicy(ActorCriticPolicy):
    """PPO/A2C policy whose actor and critic do not depend on the order of the asset slots."""

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs) -> None:
        kwargs.setdefault("features_extractor_class", SlotSetExtractor)
        if kwargs.get("use_sde"):
            raise ValueError("SharedAssetPolicy does not support state-dependent exploration")
        if not issubclass(kwargs["features_extractor_class"], SlotSetExtractor):
            raise ValueError("SharedAssetPolicy needs a SlotSetExtractor")
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _architectures(self) -> tuple[list[int], list[int]]:
        arch = self.net_arch
        if isinstance(arch, dict):
            return list(arch.get("pi", [])), list(arch.get("vf", []))
        return list(arch or []), list(arch or [])

    def _build_mlp_extractor(self) -> None:
        extractor = self.pi_features_extractor
        pi_arch, vf_arch = self._architectures()
        self.mlp_extractor = SharedAssetHead(extractor.num_assets, extractor.fusion_size, pi_arch, vf_arch,
                                             self.activation_fn)

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        num_assets = self.mlp_extractor.num_assets
        if not isinstance(self.action_space, gym.spaces.Box) or self.action_space.shape != (num_assets + 1,):
            raise ValueError(f"SharedAssetPolicy needs a Box action space of {num_assets + 1} logits (assets, then cash)")
        self.action_dist = MaskedDiagGaussianDistribution(num_assets + 1)
        self.action_net = nn.Identity()
        self.asset_log_std = nn.Parameter(torch.full((1,), float(self.log_std_init)))
        self.cash_log_std = nn.Parameter(torch.full((1,), float(self.log_std_init)))
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            gains = {self.mlp_extractor: np.sqrt(2), self.value_net: 1.0}
            if self.share_features_extractor:
                gains[self.features_extractor] = np.sqrt(2)
            else:
                gains[self.pi_features_extractor] = np.sqrt(2)
                gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in gains.items():
                module.apply(partial(self.init_weights, gain=gain))
            # Small output layers, as SB3 does for its action net: the first allocations are near even.
            for layer in self.mlp_extractor.output_layers():
                self.init_weights(layer, gain=0.01)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    @property
    def log_std(self) -> torch.Tensor:
        """One value per action dimension, tied across assets (read by SB3's training logs)."""
        return torch.cat([self.asset_log_std.expand(self.mlp_extractor.num_assets), self.cash_log_std])

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        num_assets = self.mlp_extractor.num_assets
        mean_actions = latent_pi[:, : num_assets + 1]
        valid = latent_pi[:, num_assets + 1 :]
        mask = torch.cat([valid, torch.ones_like(valid[:, :1])], dim=1)
        return self.action_dist.proba_distribution(mean_actions, self.log_std, mask)


def feature_statistics(panel: pd.DataFrame, stock_columns: Sequence[str],
                       market_columns: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of each feature, shared by all stocks.

    Stock features pool every finite stock-session of ``panel``. Market features, which repeat for
    every stock, use one value per session. A constant feature gets a standard deviation of 1. Pass
    the training window only, so validation data never shape the statistics.
    """

    stats: dict[str, tuple[float, float]] = {}
    sessions = panel.drop_duplicates("date")
    for frame, columns in ((panel, stock_columns), (sessions, market_columns)):
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError(f"feature {column!r} has no finite values")
            std = float(values.std())
            stats[column] = (float(values.mean()), std if std > 1e-12 else 1.0)
    return stats


def policy_from_config(
    policy_config: dict | None,
    *,
    lookback: int,
    num_assets: int,
    feature_columns: Sequence[str],
    feature_stats: dict[str, tuple[float, float]] | None = None,
    normalize_observations: bool = False,
) -> tuple[str | type[ActorCriticPolicy], dict]:
    """SB3 ``(policy, policy_kwargs)`` from ``training.policy``.

    ``actor: dense`` (the default) is SB3's ``MlpPolicy`` with ``build_policy_kwargs``, unchanged.
    ``actor: shared_asset`` is ``SharedAssetPolicy``; it needs the ``asset_lstm`` extractor and, unless
    ``standardize_features: false``, ``feature_stats`` from ``feature_statistics``. It cannot be
    combined with ``VecNormalize``, whose statistics differ by slot.
    """

    config = dict(policy_config or {})
    actor = str(config.pop("actor", "dense"))
    if actor == "dense":
        if "standardize_features" in config:
            raise ValueError("standardize_features applies only to actor: shared_asset")
        return "MlpPolicy", build_policy_kwargs(config or None, lookback=lookback, num_assets=num_assets,
                                                feature_columns=feature_columns)
    standardize = bool(config.pop("standardize_features", True))
    if actor != "shared_asset":
        raise ValueError(f"unknown training.policy.actor {actor!r}; choose dense or shared_asset")
    if normalize_observations:
        raise ValueError("actor: shared_asset standardizes features itself; set training.normalize_observations: "
                         "false (VecNormalize keeps separate statistics for each slot)")
    kwargs = build_policy_kwargs(config, lookback=lookback, num_assets=num_assets, feature_columns=feature_columns)
    if kwargs.get("features_extractor_class") is not AssetTemporalExtractor:
        raise ValueError("actor: shared_asset needs features_extractor: asset_lstm")
    columns = list(feature_columns)
    extractor = dict(kwargs["features_extractor_kwargs"])
    extractor["valid_feature_index"] = columns.index(VALID_COLUMN) if VALID_COLUMN in columns else None
    if standardize:
        if feature_stats is None:
            raise ValueError("actor: shared_asset with standardize_features needs feature_stats from the training window")
        missing = [c for c in columns if c != VALID_COLUMN and c not in feature_stats]
        if missing:
            raise ValueError(f"no statistics for features: {missing}")
        extractor["feature_mean"] = [0.0 if c == VALID_COLUMN else float(feature_stats[c][0]) for c in columns]
        extractor["feature_std"] = [1.0 if c == VALID_COLUMN else float(feature_stats[c][1]) for c in columns]
    kwargs["features_extractor_class"] = SlotSetExtractor
    kwargs["features_extractor_kwargs"] = extractor
    return SharedAssetPolicy, kwargs
