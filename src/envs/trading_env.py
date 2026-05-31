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
        pass

    class _Gym:
        Env = _Env

    class _Spaces:
        Box = _Box

    gym = _Gym()
    spaces = _Spaces()


@dataclass(frozen=True)
class RewardConfig:
    return_weight: float = 1.0
    turnover_penalty: float = 0.05
    drawdown_penalty: float = 0.1


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
    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        feature_columns: list[str],
        lookback: int = 30,
        initial_cash: float = 1_000_000.0,
        transaction_cost: float = 0.001,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.features = features.copy()
        self.features["date"] = pd.to_datetime(self.features["date"])
        self.features = self.features.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.feature_columns = feature_columns
        self.lookback = lookback
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.reward_config = reward_config or RewardConfig()

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
        self.close = self.panel["close"].unstack("ticker").to_numpy(dtype=np.float64)
        self.feature_tensor = np.stack(
            [
                self.panel[col].unstack("ticker").to_numpy(dtype=np.float32)
                for col in self.feature_columns
            ],
            axis=-1,
        )

        obs_size = self.lookback * self.num_assets * len(self.feature_columns) + self.num_assets + 1
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
        self.current_step = self.lookback
        self.cash_weight = 1.0
        self.weights = np.zeros(self.num_assets, dtype=np.float64)
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        return self._observation(), self._info(0.0, 0.0, 0.0)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        target_weights = self._normalize_action(action)
        previous_weights = self.weights.copy()
        previous_cash = self.cash_weight
        turnover = float(np.abs(target_weights - previous_weights).sum() + abs((1 - target_weights.sum()) - previous_cash))

        current_prices = self.close[self.current_step]
        next_prices = self.close[self.current_step + 1]
        asset_returns = np.divide(next_prices, current_prices, out=np.ones_like(next_prices), where=current_prices != 0) - 1.0
        gross_return = float(np.dot(target_weights, asset_returns))
        cost = self.transaction_cost * turnover
        net_return = gross_return - cost

        self.portfolio_value *= 1.0 + net_return
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = 1.0 - self.portfolio_value / self.peak_value if self.peak_value else 0.0

        reward = (
            self.reward_config.return_weight * net_return
            - self.reward_config.turnover_penalty * turnover
            - self.reward_config.drawdown_penalty * drawdown
        )

        self.weights = target_weights
        self.cash_weight = 1.0 - float(target_weights.sum())
        self.current_step += 1
        terminated = self.current_step >= len(self.dates) - 1
        truncated = False
        return self._observation(), float(reward), terminated, truncated, self._info(net_return, turnover, drawdown)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)
        action = np.clip(action, 0.0, 1.0)
        total = action.sum()
        if total > 1.0:
            action = action / total
        return action.astype(np.float64)

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

    def _info(self, net_return: float, turnover: float, drawdown: float) -> dict[str, Any]:
        return {
            "date": self.dates[self.current_step].date().isoformat(),
            "portfolio_value": float(self.portfolio_value),
            "net_return": float(net_return),
            "turnover": float(turnover),
            "drawdown": float(drawdown),
            "weights": self.weights.copy(),
            "cash_weight": float(self.cash_weight),
        }

