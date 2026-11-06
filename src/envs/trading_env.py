from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - used only on minimal HPC smoke envs.
    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Env:
        def close(self) -> None:
            """Match the Gymnasium lifecycle API in minimal smoke runtimes."""

            return None

    class _Gym:
        Env = _Env

    class _Spaces:
        Box = _Box

    gym = _Gym()
    spaces = _Spaces()


REWARD_TYPES = ("simple", "log_return", "plan_main", "sortino")
DRAWDOWN_MODES = ("level", "increment")
ACCOUNTING_MODES = ("legacy", "self_financing")
ACTION_TRANSFORMS = ("clip", "softmax")


@dataclass(frozen=True)
class RewardConfig:
    """Reward definition.

    The defaults reproduce the historical V2 reward exactly:
    ``return_weight * net_return - turnover_penalty * turnover - drawdown_penalty * drawdown``.

    Long-term variants (implementation plan section 5):

    - ``log_return``: ``return_weight * log(1 + R)``; the episode sum equals log terminal wealth.
    - ``plan_main``: ``alpha * log(1 + R) + beta * dSharpe - gamma * drawdown_term - lambda * turnover``,
      where ``dSharpe`` is the Moody-Saffell differential Sharpe ratio.
    - ``sortino``: like ``plan_main`` but with the differential downside-deviation ratio.

    ``drawdown_mode="increment"`` penalizes only the step-to-step increase in drawdown instead of
    charging the drawdown level on every underwater step.
    """

    return_weight: float = 1.0
    turnover_penalty: float = 0.05
    drawdown_penalty: float = 0.1
    type: str = "simple"
    drawdown_mode: str = "level"
    sharpe_weight: float = 0.0
    sharpe_eta: float = 0.01
    sharpe_clip: float = 5.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.type not in REWARD_TYPES:
            raise ValueError(f"reward.type must be one of {REWARD_TYPES}; got {self.type!r}")
        if self.drawdown_mode not in DRAWDOWN_MODES:
            raise ValueError(f"reward.drawdown_mode must be one of {DRAWDOWN_MODES}; got {self.drawdown_mode!r}")
        if not 0.0 < self.sharpe_eta < 1.0:
            raise ValueError("reward.sharpe_eta must be in (0, 1)")
        if self.sharpe_clip <= 0:
            raise ValueError("reward.sharpe_clip must be positive")
        if self.scale <= 0:
            raise ValueError("reward.scale must be positive")


def differential_sharpe(net_return: float, mean: float, second_moment: float) -> float:
    """Moody & Saffell (2001) differential Sharpe ratio using the previous EMA moments."""

    variance = second_moment - mean**2
    if variance <= 1e-12:
        return 0.0
    delta_mean = net_return - mean
    delta_second = net_return**2 - second_moment
    return float((second_moment * delta_mean - 0.5 * mean * delta_second) / variance**1.5)


def differential_downside_ratio(net_return: float, mean: float, downside_second_moment: float) -> float:
    """Moody & Saffell (2001) differential downside-deviation ratio using previous EMA moments."""

    if downside_second_moment <= 1e-12:
        return 0.0
    downside_deviation = np.sqrt(downside_second_moment)
    if net_return > 0:
        return float((net_return - 0.5 * mean) / downside_deviation)
    return float(
        (downside_second_moment * (net_return - 0.5 * mean) - 0.5 * mean * net_return**2)
        / downside_deviation**3
    )


def load_feature_table(path: str | Path, fallback_csv: str | Path | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    if fallback_csv and Path(fallback_csv).exists():
        return pd.read_csv(fallback_csv)
    raise FileNotFoundError(path)


class ThaiStockTradingEnv(gym.Env):
    """Long-only Thai equity portfolio environment.

    Every new option defaults to the historical V2 behavior so locked V2 evidence stays reproducible.

    - ``accounting="self_financing"`` counts security-only turnover (no cash-leg double count),
      compounds costs multiplicatively, and lets weights drift with returns between decisions.
    - ``rebalance_every=k`` only accepts a new allocation every ``k`` sessions; positions are held
      (and drift under self-financing accounting) in between.
    - ``random_start`` / ``episode_length`` sample a random contiguous training episode.
    - ``action_transform="softmax"`` interprets an ``N + 1`` logit vector (last entry = cash)
      as portfolio weights. An ``N``-length weight vector is still accepted directly so baseline
      policies keep working.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        feature_columns: list[str],
        lookback: int = 30,
        initial_cash: float = 1_000_000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0,
        max_turnover: float | None = None,
        max_position_weight: float = 1.0,
        min_cash_weight: float = 0.0,
        reward_config: RewardConfig | None = None,
        accounting: str = "legacy",
        rebalance_every: int = 1,
        random_start: bool = False,
        episode_length: int | None = None,
        action_transform: str = "clip",
    ) -> None:
        super().__init__()
        self.features = features.copy()
        self.features["date"] = pd.to_datetime(self.features["date"])
        self.features = self.features.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.feature_columns = feature_columns
        self.lookback = lookback
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.slippage = float(slippage)
        self.max_turnover = float(max_turnover) if max_turnover is not None else None
        self.max_position_weight = float(max_position_weight)
        self.min_cash_weight = float(min_cash_weight)
        self.reward_config = reward_config or RewardConfig()
        self.accounting = str(accounting)
        self.rebalance_every = int(rebalance_every)
        self.random_start = bool(random_start)
        self.episode_length = int(episode_length) if episode_length is not None else None
        self.action_transform = str(action_transform)
        if self.transaction_cost < 0 or self.slippage < 0:
            raise ValueError("transaction_cost and slippage must be non-negative")
        if self.max_turnover is not None and self.max_turnover <= 0:
            raise ValueError("max_turnover must be positive when configured")
        if not 0.0 < self.max_position_weight <= 1.0:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not 0.0 <= self.min_cash_weight < 1.0:
            raise ValueError("min_cash_weight must be in [0, 1)")
        if self.accounting not in ACCOUNTING_MODES:
            raise ValueError(f"accounting must be one of {ACCOUNTING_MODES}; got {self.accounting!r}")
        if self.rebalance_every < 1:
            raise ValueError("rebalance_every must be a positive integer")
        if self.episode_length is not None and self.episode_length < 1:
            raise ValueError("episode_length must be positive when configured")
        if self.action_transform not in ACTION_TRANSFORMS:
            raise ValueError(f"action_transform must be one of {ACTION_TRANSFORMS}; got {self.action_transform!r}")

        self.dates = list(self.features["date"].drop_duplicates().sort_values())
        self.tickers = list(self.features["ticker"].drop_duplicates().sort_values())
        self.num_assets = len(self.tickers)
        self._validate_panel()

        self.panel = (
            self.features.set_index(["date", "ticker"])
            .reindex(pd.MultiIndex.from_product([self.dates, self.tickers], names=["date", "ticker"]))
            .sort_index()
        )
        self.panel[self.feature_columns] = self.panel[self.feature_columns].fillna(0.0)
        self.close_prices = self.panel["close"].unstack("ticker").to_numpy(dtype=np.float64)
        self.feature_tensor = np.stack(
            [
                self.panel[col].unstack("ticker").to_numpy(dtype=np.float32)
                for col in self.feature_columns
            ],
            axis=-1,
        )
        if self.episode_length is not None and self.lookback + self.episode_length > len(self.dates) - 1:
            raise ValueError("episode_length is longer than the available dates after the lookback")

        obs_size = self.lookback * self.num_assets * len(self.feature_columns) + self.num_assets + 1
        if self.action_transform == "softmax":
            self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.num_assets + 1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.reset()

    def _validate_panel(self) -> None:
        duplicates = self.features.duplicated(["date", "ticker"]).sum()
        if duplicates:
            raise ValueError(f"duplicate date/ticker rows: {duplicates}")
        missing = set(self.feature_columns + ["close", "date", "ticker"]) - set(self.features.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None and hasattr(super(), "reset"):
            try:
                super().reset(seed=seed)
            except TypeError:
                pass
        last_step = len(self.dates) - 1
        start = self.lookback
        if self.random_start:
            span = self.episode_length if self.episode_length is not None else 1
            latest_start = last_step - span
            if latest_start > self.lookback:
                rng = getattr(self, "np_random", None)
                if rng is None:
                    rng = np.random.default_rng(seed)
                start = int(rng.integers(self.lookback, latest_start + 1))
        self.start_step = start
        self.end_step = last_step if self.episode_length is None else min(last_step, start + self.episode_length)
        self.current_step = start
        self.cash_weight = 1.0
        self.weights = np.zeros(self.num_assets, dtype=np.float64)
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        self.previous_drawdown = 0.0
        self.steps_since_rebalance = 0
        self.decision_count = 0
        self._ema_mean = 0.0
        self._ema_second = 0.0
        self._ema_downside_second = 0.0
        return self._observation(), self._info(0.0, 0.0, 0.0, 0.0, 0.0)

    def step(self, action: np.ndarray):
        previous_weights = self.weights.copy()
        previous_cash = self.cash_weight
        rebalance = self.steps_since_rebalance % self.rebalance_every == 0
        if rebalance:
            target_weights = self._normalize_action(self._action_to_weights(action))
            target_weights = self._apply_turnover_limit(target_weights)
            self.decision_count += 1
        else:
            target_weights = previous_weights.copy()
        self.steps_since_rebalance += 1

        if self.accounting == "self_financing":
            turnover = float(np.abs(target_weights - previous_weights).sum())
        else:
            turnover = float(
                np.abs(target_weights - previous_weights).sum() + abs((1 - target_weights.sum()) - previous_cash)
            )

        current_prices = self.close_prices[self.current_step]
        next_prices = self.close_prices[self.current_step + 1]
        asset_returns = np.divide(next_prices, current_prices, out=np.ones_like(next_prices), where=current_prices != 0) - 1.0
        gross_return = float(np.dot(target_weights, asset_returns))
        cost = self.transaction_cost * turnover
        slippage_cost = self.slippage * turnover
        if self.accounting == "self_financing":
            net_return = (1.0 - cost - slippage_cost) * (1.0 + gross_return) - 1.0
        else:
            net_return = gross_return - cost - slippage_cost

        self.portfolio_value *= 1.0 + net_return
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = 1.0 - self.portfolio_value / self.peak_value if self.peak_value else 0.0

        reward, ratio_term = self._reward(net_return, turnover, drawdown)
        self.previous_drawdown = drawdown

        if self.accounting == "self_financing":
            growth = 1.0 + gross_return
            if growth > 0:
                self.weights = target_weights * (1.0 + asset_returns) / growth
            else:  # pragma: no cover - total loss; keep the accounting finite.
                self.weights = np.zeros_like(target_weights)
            self.weights = np.clip(self.weights, 0.0, None)
            self.cash_weight = max(0.0, 1.0 - float(self.weights.sum()))
        else:
            self.weights = target_weights
            self.cash_weight = 1.0 - float(target_weights.sum())
        self.current_step += 1
        terminated = self.current_step >= self.end_step
        truncated = False
        info = self._info(net_return, turnover, drawdown, cost, slippage_cost)
        info["gross_return"] = gross_return
        info["rebalanced"] = bool(rebalance)
        info["differential_ratio"] = ratio_term
        info["target_weights"] = target_weights.copy()
        return self._observation(), float(reward), terminated, truncated, info

    def _reward(self, net_return: float, turnover: float, drawdown: float) -> tuple[float, float]:
        cfg = self.reward_config
        if cfg.drawdown_mode == "increment":
            drawdown_term = max(0.0, drawdown - self.previous_drawdown)
        else:
            drawdown_term = drawdown

        if cfg.type == "simple":
            return_term = net_return
        else:
            return_term = float(np.log1p(max(net_return, -0.999999)))

        ratio_term = 0.0
        if cfg.type == "plan_main":
            ratio_term = differential_sharpe(net_return, self._ema_mean, self._ema_second)
        elif cfg.type == "sortino":
            ratio_term = differential_downside_ratio(net_return, self._ema_mean, self._ema_downside_second)
        ratio_term = float(np.clip(ratio_term, -cfg.sharpe_clip, cfg.sharpe_clip))

        eta = cfg.sharpe_eta
        self._ema_mean += eta * (net_return - self._ema_mean)
        self._ema_second += eta * (net_return**2 - self._ema_second)
        self._ema_downside_second += eta * (min(net_return, 0.0) ** 2 - self._ema_downside_second)

        reward = (
            cfg.return_weight * return_term
            + cfg.sharpe_weight * ratio_term
            - cfg.turnover_penalty * turnover
            - cfg.drawdown_penalty * drawdown_term
        )
        return float(cfg.scale * reward), ratio_term

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if self.action_transform == "softmax" and action.shape[0] == self.num_assets + 1:
            logits = np.nan_to_num(action, nan=0.0, posinf=5.0, neginf=-5.0)
            logits = logits - logits.max()
            exp = np.exp(logits)
            return (exp / exp.sum())[: self.num_assets]
        return action

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)
        action = np.clip(action, 0.0, self.max_position_weight)
        total = action.sum()
        investment_budget = 1.0 - self.min_cash_weight
        if total > investment_budget:
            action = action / total * investment_budget
        return action.astype(np.float64)

    def _apply_turnover_limit(self, target_weights: np.ndarray) -> np.ndarray:
        if self.max_turnover is None:
            return target_weights
        current_cash = 1.0 - float(self.weights.sum())
        target_cash = 1.0 - float(target_weights.sum())
        turnover = float(np.abs(target_weights - self.weights).sum())
        if self.accounting != "self_financing":
            turnover += abs(target_cash - current_cash)
        if turnover <= self.max_turnover:
            return target_weights
        scale = self.max_turnover / turnover
        constrained = self.weights + scale * (target_weights - self.weights)
        return np.clip(constrained, 0.0, self.max_position_weight)

    def _observation(self) -> np.ndarray:
        start = self.current_step - self.lookback
        window = self.feature_tensor[start : self.current_step]
        obs = np.concatenate(
            [
                window.reshape(-1),
                self.weights.astype(np.float32),
                np.array([self.cash_weight], dtype=np.float32),
            ]
        )
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _info(
        self,
        net_return: float,
        turnover: float,
        drawdown: float,
        transaction_cost_paid: float,
        slippage_paid: float,
    ) -> dict[str, Any]:
        return {
            "date": self.dates[self.current_step].date().isoformat(),
            "portfolio_value": float(self.portfolio_value),
            "net_return": float(net_return),
            "turnover": float(turnover),
            "transaction_cost_paid": float(transaction_cost_paid),
            "slippage_paid": float(slippage_paid),
            "drawdown": float(drawdown),
            "weights": self.weights.copy(),
            "cash_weight": float(self.cash_weight),
        }
