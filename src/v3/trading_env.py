"""Mask-aware close-to-close environment for isolated V3 development.

The active V2 environment remains unchanged.  This class consumes a separately
validated development-eligibility mask; it cannot accept the membership-only
panel directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.envs.trading_env import RewardConfig, gym, spaces
from src.v3.eligibility import validate_development_eligibility_mask
from src.v3.execution_ledger import (
    execute_self_financing_step,
    security_turnover,
)


class PointInTimeThaiStockTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        feature_columns: list[str],
        eligibility_mask: pd.DataFrame,
        asset_universe: Sequence[str],
        lookback: int = 30,
        initial_cash: float = 1_000_000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0,
        max_turnover: float | None = None,
        max_position_weight: float = 1.0,
        min_cash_weight: float = 0.0,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.feature_columns = list(feature_columns)
        self.lookback = int(lookback)
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.slippage = float(slippage)
        self.max_turnover = float(max_turnover) if max_turnover is not None else None
        self.max_position_weight = float(max_position_weight)
        self.min_cash_weight = float(min_cash_weight)
        self.reward_config = reward_config or RewardConfig()
        self.tickers = [str(value).strip().upper() for value in asset_universe]
        if not self.tickers or len(self.tickers) != len(set(self.tickers)):
            raise ValueError("asset_universe must contain unique tickers")
        self.num_assets = len(self.tickers)
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")
        if self.transaction_cost < 0 or self.slippage < 0:
            raise ValueError("transaction_cost and slippage must be non-negative")
        if self.max_turnover is not None and self.max_turnover <= 0:
            raise ValueError("max_turnover must be positive when configured")
        if not 0.0 < self.max_position_weight <= 1.0:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not 0.0 <= self.min_cash_weight < 1.0:
            raise ValueError("min_cash_weight must be in [0, 1)")

        self.features = features.copy()
        required_features = {"date", "ticker", "close", *self.feature_columns}
        missing = sorted(required_features - set(self.features.columns))
        if missing:
            raise ValueError(f"feature table is missing columns: {missing}")
        self.features["date"] = pd.to_datetime(
            self.features["date"],
            errors="coerce",
        ).dt.normalize()
        self.features["ticker"] = (
            self.features["ticker"].astype(str).str.strip().str.upper()
        )
        if self.features["date"].isna().any():
            raise ValueError("feature table contains invalid dates")
        if self.features.duplicated(["date", "ticker"]).any():
            raise ValueError("feature table has duplicate date/ticker rows")
        unknown = sorted(set(self.features["ticker"]) - set(self.tickers))
        if unknown:
            raise ValueError(f"feature table contains tickers outside asset_universe: {unknown}")

        self.dates = list(self.features["date"].drop_duplicates().sort_values())
        if len(self.dates) <= self.lookback + 1:
            raise ValueError(
                f"environment needs at least {self.lookback + 2} dates; "
                f"found {len(self.dates)}"
            )
        checked_mask = validate_development_eligibility_mask(
            eligibility_mask,
            expected_dates=self.dates,
            expected_tickers=self.tickers,
        )
        self.eligibility_frame = checked_mask
        self.eligibility_matrix = (
            checked_mask.pivot(index="date", columns="ticker", values="is_eligible")
            .reindex(index=self.dates, columns=self.tickers)
            .to_numpy(dtype=bool)
        )

        index = pd.MultiIndex.from_product(
            [self.dates, self.tickers],
            names=["date", "ticker"],
        )
        self.panel = self.features.set_index(["date", "ticker"]).reindex(index).sort_index()
        raw_feature_tensor = np.stack(
            [
                self.panel[column]
                .unstack("ticker")
                .reindex(index=self.dates, columns=self.tickers)
                .to_numpy(dtype=np.float64)
                for column in self.feature_columns
            ],
            axis=-1,
        )
        self.close_prices = (
            self.panel["close"]
            .unstack("ticker")
            .reindex(index=self.dates, columns=self.tickers)
            .to_numpy(dtype=np.float64)
        )
        self._validate_observation_coverage(raw_feature_tensor)
        self._validate_valuation_prices()
        self.feature_tensor = np.nan_to_num(
            raw_feature_tensor,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        obs_size = (
            self.lookback * self.num_assets * len(self.feature_columns)
            + self.num_assets
            + self.num_assets
            + 1
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_assets,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.reset()

    @property
    def current_action_mask(self) -> np.ndarray:
        return self.eligibility_matrix[self.current_step].copy()

    def _validate_observation_coverage(self, tensor: np.ndarray) -> None:
        for step in range(self.lookback, len(self.dates) - 1):
            active = self.eligibility_matrix[step]
            if not active.any():
                continue
            window = tensor[step - self.lookback : step, :, :][:, active, :]
            if not np.isfinite(window).all():
                raise ValueError(
                    "eligible assets have non-finite features in the lookback "
                    f"window ending before {self.dates[step].date()}"
                )

    def _validate_valuation_prices(self) -> None:
        for step in range(self.lookback, len(self.dates) - 1):
            active = self.eligibility_matrix[step]
            current = self.close_prices[step]
            following = self.close_prices[step + 1]
            invalid = active & (
                ~np.isfinite(current)
                | (current <= 0.0)
                | ~np.isfinite(following)
                | (following <= 0.0)
            )
            if invalid.any():
                tickers = [self.tickers[index] for index in np.flatnonzero(invalid)]
                raise ValueError(
                    "eligible assets lack finite positive current/next valuation "
                    f"prices at {self.dates[step].date()}: {tickers}"
                )

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
        return self._observation(), self._info(
            gross_return=0.0,
            net_return=0.0,
            turnover=0.0,
            drawdown=0.0,
            transaction_cost_paid=0.0,
            slippage_paid=0.0,
            decision_date=self.dates[self.current_step],
            decision_action_mask=self.current_action_mask,
            forced_exit_turnover=0.0,
            turnover_limit_overridden=False,
            executed_target_weights=np.zeros(self.num_assets, dtype=np.float64),
            trade_deltas=np.zeros(self.num_assets, dtype=np.float64),
            pre_return_cash_fraction=1.0,
            reconciliation_error=0.0,
            forced_exit_pending=np.zeros(self.num_assets, dtype=bool),
        )

    def step(self, action: np.ndarray):
        decision_step = self.current_step
        decision_date = self.dates[decision_step]
        action_mask = self.eligibility_matrix[decision_step]
        desired = self._normalize_action(action, action_mask)

        current_prices = self.close_prices[decision_step]
        next_prices = self.close_prices[decision_step + 1]
        valid = (
            np.isfinite(current_prices)
            & (current_prices > 0.0)
            & np.isfinite(next_prices)
            & (next_prices > 0.0)
        )
        asset_returns = np.zeros(self.num_assets, dtype=np.float64)
        asset_returns[valid] = (
            next_prices[valid] / current_prices[valid] - 1.0
        )
        execution = execute_self_financing_step(
            current_weights=self.weights,
            current_cash_weight=self.cash_weight,
            desired_weights=desired,
            asset_returns=asset_returns,
            buy_allowed=action_mask,
            # The Boolean development mask has no licensed status semantics.
            # It therefore models every mandatory development exit as fillable.
            # Final V3 must provide a separate sell-permission ledger.
            sell_allowed=np.ones(self.num_assets, dtype=bool),
            forced_exit_required=~action_mask,
            transaction_cost_rate=self.transaction_cost,
            slippage_rate=self.slippage,
            max_turnover=self.max_turnover,
        )
        self.portfolio_value *= execution.portfolio_multiplier
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (
            1.0 - self.portfolio_value / self.peak_value
            if self.peak_value
            else 0.0
        )
        reward = (
            self.reward_config.return_weight * execution.net_return
            - self.reward_config.turnover_penalty * execution.turnover
            - self.reward_config.drawdown_penalty * drawdown
        )

        self.weights = execution.ending_weights.copy()
        self.cash_weight = execution.ending_cash_weight
        self.current_step += 1
        terminated = self.current_step >= len(self.dates) - 1
        truncated = False
        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            self._info(
                gross_return=execution.gross_return,
                net_return=execution.net_return,
                turnover=execution.turnover,
                drawdown=drawdown,
                transaction_cost_paid=execution.transaction_cost_paid,
                slippage_paid=execution.slippage_paid,
                decision_date=decision_date,
                decision_action_mask=action_mask,
                forced_exit_turnover=execution.forced_exit_turnover,
                turnover_limit_overridden=execution.turnover_limit_overridden,
                executed_target_weights=execution.executed_target_weights,
                trade_deltas=execution.trade_deltas,
                pre_return_cash_fraction=execution.pre_return_cash_fraction,
                reconciliation_error=execution.reconciliation_error,
                forced_exit_pending=execution.forced_exit_pending,
            ),
        )

    def _normalize_action(
        self,
        action: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (self.num_assets,):
            raise ValueError(
                f"action shape must be {(self.num_assets,)}, got {values.shape}"
            )
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        values = np.clip(values, 0.0, self.max_position_weight)
        values[~action_mask] = 0.0
        total = float(values.sum())
        investment_budget = 1.0 - self.min_cash_weight
        if total > investment_budget:
            values = values / total * investment_budget
        return values

    @staticmethod
    def _turnover(
        previous_weights: np.ndarray,
        previous_cash: float,
        target_weights: np.ndarray,
        target_cash: float,
    ) -> float:
        del previous_cash, target_cash
        return security_turnover(previous_weights, target_weights)

    def _apply_turnover_limit(self, target_weights: np.ndarray) -> np.ndarray:
        if self.max_turnover is None:
            return target_weights
        current_cash = 1.0 - float(self.weights.sum())
        target_cash = 1.0 - float(target_weights.sum())
        turnover = self._turnover(
            self.weights,
            current_cash,
            target_weights,
            target_cash,
        )
        if turnover <= self.max_turnover:
            return target_weights
        scale = self.max_turnover / turnover
        constrained = self.weights + scale * (target_weights - self.weights)
        return np.clip(constrained, 0.0, self.max_position_weight)

    def _force_ineligible_to_cash(
        self,
        weights: np.ndarray,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        constrained = weights.copy()
        constrained[~action_mask] = 0.0
        forced_turnover = self._turnover(
            weights,
            1.0 - float(weights.sum()),
            constrained,
            1.0 - float(constrained.sum()),
        )
        return constrained, forced_turnover

    def _observation(self) -> np.ndarray:
        start = self.current_step - self.lookback
        window = self.feature_tensor[start : self.current_step]
        observation = np.concatenate(
            [
                window.reshape(-1),
                self.current_action_mask.astype(np.float32),
                self.weights.astype(np.float32),
                np.array([self.cash_weight], dtype=np.float32),
            ]
        )
        return observation.astype(np.float32)

    def _info(
        self,
        *,
        gross_return: float,
        net_return: float,
        turnover: float,
        drawdown: float,
        transaction_cost_paid: float,
        slippage_paid: float,
        decision_date: pd.Timestamp,
        decision_action_mask: np.ndarray,
        forced_exit_turnover: float,
        turnover_limit_overridden: bool,
        executed_target_weights: np.ndarray,
        trade_deltas: np.ndarray,
        pre_return_cash_fraction: float,
        reconciliation_error: float,
        forced_exit_pending: np.ndarray,
    ) -> dict[str, Any]:
        applied_mask = np.asarray(decision_action_mask, dtype=bool).copy()
        next_mask = self.current_action_mask
        return {
            "date": self.dates[self.current_step].date().isoformat(),
            "decision_date": decision_date.date().isoformat(),
            "portfolio_value": float(self.portfolio_value),
            "gross_return": float(gross_return),
            "net_return": float(net_return),
            "turnover": float(turnover),
            "transaction_cost_paid": float(transaction_cost_paid),
            "slippage_paid": float(slippage_paid),
            "drawdown": float(drawdown),
            "weights": self.weights.copy(),
            "cash_weight": float(self.cash_weight),
            "action_mask": applied_mask,
            "eligible_count": int(applied_mask.sum()),
            "next_action_mask": next_mask,
            "next_eligible_count": int(next_mask.sum()),
            "forced_exit_turnover": float(forced_exit_turnover),
            "turnover_limit_overridden": bool(turnover_limit_overridden),
            "executed_target_weights": np.asarray(
                executed_target_weights,
                dtype=np.float64,
            ).copy(),
            "trade_deltas": np.asarray(trade_deltas, dtype=np.float64).copy(),
            "pre_return_cash_fraction": float(pre_return_cash_fraction),
            "self_financing_reconciliation_error": float(reconciliation_error),
            "forced_exit_pending": np.asarray(
                forced_exit_pending,
                dtype=bool,
            ).copy(),
        }
