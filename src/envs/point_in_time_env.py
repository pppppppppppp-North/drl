"""Point-in-time index environment with a fixed number of asset slots (milestone M3).

``LongTermTradingEnv`` trades a fixed list of stocks. Here the investable set is the index's
membership on each session, which changes at every review. The agent sees ``K`` slots. On every
session the tradable members fill the slots in order of trailing traded value (as of the previous
session), and empty slots are masked out of the allocation.

- **Information lag:** a decision at session ``t`` uses features up to ``t-1`` and the membership
  list in force at ``t-1``.
- **Tradable:** a member is tradable at ``t`` if it has a price at ``t`` and ``t+1``, at least
  ``min_history_sessions`` sessions of price history, and finite features over the lookback
  window.
- **Forced exits:** a holding that stops being tradable (it left the index, or its series ends)
  is sold at the close of ``t``, with costs, on any session, not only on rebalancing sessions.
- **Accounting and reward:** the same self-financing accounting and reward options as
  ``LongTermTradingEnv`` (the reward function is inherited). Weights drift between decisions.

Observation (compatible with ``AssetTemporalExtractor`` with ``num_assets=K``): a
``lookback x K x F`` window, where F is the stock features, then the market features, then a
slot-valid flag; then the K slot weights and the cash weight. Action: ``K + 1`` logits (last =
cash) turned into weights by a softmax over valid slots and cash. A ``K``-length weight vector is
also accepted, so baseline policies can set weights directly.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.envs.long_term_env import LongTermRewardConfig, LongTermTradingEnv
from src.envs.trading_env import gym, spaces


class PointInTimeSlotEnv(LongTermTradingEnv):
    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: pd.DataFrame,
        membership: pd.DataFrame,
        stock_feature_columns: Sequence[str],
        market_feature_columns: Sequence[str],
        *,
        num_slots: int = 50,
        lookback: int = 30,
        min_history_sessions: int = 60,
        initial_cash: float = 1_000_000.0,
        transaction_cost: float = 0.0015,
        slippage: float = 0.0005,
        reward_config: LongTermRewardConfig | None = None,
        rebalance_every: int = 1,
        random_start: bool = False,
        episode_length: int | None = None,
    ) -> None:
        gym.Env.__init__(self)
        self.stock_feature_columns = list(stock_feature_columns)
        self.market_feature_columns = list(market_feature_columns)
        self.feature_columns = self.stock_feature_columns + self.market_feature_columns + ["slot_valid"]
        self.num_slots = int(num_slots)
        self.num_assets = self.num_slots  # what the feature extractor calls assets
        self.lookback = int(lookback)
        self.min_history_sessions = int(min_history_sessions)
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.slippage = float(slippage)
        self.reward_config = reward_config or LongTermRewardConfig()
        self.accounting = "self_financing"
        self.rebalance_every = int(rebalance_every)
        self.random_start = bool(random_start)
        self.episode_length = int(episode_length) if episode_length is not None else None
        self.action_transform = "softmax"
        self.max_turnover = None
        self.max_position_weight = 1.0
        self.min_cash_weight = 0.0
        if self.num_slots < 1 or self.lookback < 1 or self.rebalance_every < 1:
            raise ValueError("num_slots, lookback and rebalance_every must be positive")
        if self.transaction_cost < 0 or self.slippage < 0:
            raise ValueError("transaction_cost and slippage must be non-negative")

        frame = panel.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        needed = {"date", "ticker", "close", "history_sessions", "rank_value", *self.stock_feature_columns,
                  *self.market_feature_columns}
        missing = sorted(needed - set(frame.columns))
        if missing:
            raise ValueError(f"panel is missing columns: {missing}")
        if frame.duplicated(["date", "ticker"]).any():
            raise ValueError("panel has duplicate date/ticker rows")
        self.dates = list(pd.DatetimeIndex(sorted(frame["date"].unique())))
        if len(self.dates) <= self.lookback + 1:
            raise ValueError(f"need more than {self.lookback + 1} sessions, found {len(self.dates)}")

        members = membership.copy()
        members["date"] = pd.to_datetime(members["date"])
        members = members[members["date"].isin(self.dates) & members["price_ticker"].astype(str).ne("")]
        self.tickers = sorted(set(members["price_ticker"]) & set(frame["ticker"]))
        if not self.tickers:
            raise ValueError("no index member has prices in this window")
        self.num_series = len(self.tickers)
        date_index = pd.DatetimeIndex(self.dates)

        def pivot(column: str, fill: float = np.nan) -> np.ndarray:
            table = frame.pivot(index="date", columns="ticker", values=column)
            return table.reindex(index=date_index, columns=self.tickers).to_numpy(dtype=np.float64, na_value=fill)

        self.close_prices = pivot("close")
        history = pivot("history_sessions", 0.0)
        self.rank_values = np.nan_to_num(pivot("rank_value", 0.0), nan=0.0)
        self.stock_tensor = np.stack([pivot(c) for c in self.stock_feature_columns], axis=-1).astype(np.float32)
        market = frame.drop_duplicates("date").set_index("date")[self.market_feature_columns].reindex(date_index)
        self.market_tensor = np.nan_to_num(market.to_numpy(dtype=np.float32), nan=0.0)

        listed = (
            members.assign(member=True)
            .pivot_table(index="date", columns="price_ticker", values="member", aggfunc="any", fill_value=False)
            .reindex(index=date_index, columns=self.tickers, fill_value=False)
            .to_numpy(dtype=bool)
        )
        # The decision at t uses the list in force at the previous session.
        decision_member = np.zeros_like(listed)
        decision_member[1:] = listed[:-1]

        priced = np.isfinite(self.close_prices) & (self.close_prices > 0)
        next_priced = np.zeros_like(priced)
        next_priced[:-1] = priced[1:]
        row_ok = np.isfinite(self.stock_tensor).all(axis=-1)
        cumulative = np.vstack([np.zeros((1, self.num_series)), np.cumsum(row_ok, axis=0)])
        window_ok = np.zeros_like(row_ok)
        steps = np.arange(self.lookback, len(self.dates))
        window_ok[steps] = (cumulative[steps] - cumulative[steps - self.lookback]) == self.lookback
        self.tradable = (
            decision_member & priced & next_priced & window_ok & (history >= self.min_history_sessions)
        )
        self.tradable[: self.lookback] = False
        self.tradable[-1] = False
        self.stock_tensor = np.nan_to_num(self.stock_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        self.slots = np.full((len(self.dates), self.num_slots), -1, dtype=np.int64)
        for step in range(len(self.dates)):
            candidates = np.flatnonzero(self.tradable[step])
            if candidates.size > self.num_slots:
                raise ValueError(
                    f"{candidates.size} tradable members at {self.dates[step].date()} exceed {self.num_slots} slots"
                )
            order = sorted(candidates, key=lambda u: (-self.rank_values[step, u], self.tickers[u]))
            self.slots[step, : len(order)] = order

        if self.episode_length is not None and self.lookback + self.episode_length > len(self.dates) - 1:
            raise ValueError("episode_length is longer than the available sessions after the lookback")
        width = len(self.feature_columns)
        obs_size = self.lookback * self.num_slots * width + self.num_slots + 1
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.num_slots + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.reset()

    # ----------------------------------------------------------------------------- episode

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            gym.Env.reset(self, seed=seed)
        last_step = len(self.dates) - 1
        start = self.lookback
        if self.random_start:
            span = self.episode_length if self.episode_length is not None else 1
            latest_start = last_step - span
            if latest_start > self.lookback:
                rng = getattr(self, "np_random", None) or np.random.default_rng(seed)
                start = int(rng.integers(self.lookback, latest_start + 1))
        self.start_step = start
        self.end_step = last_step if self.episode_length is None else min(last_step, start + self.episode_length)
        self.current_step = start
        self.weights = np.zeros(self.num_series, dtype=np.float64)
        self.cash_weight = 1.0
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        self.previous_drawdown = 0.0
        self.steps_since_rebalance = 0
        self.decision_count = 0
        self._ema_mean = 0.0
        self._ema_second = 0.0
        self._ema_downside_second = 0.0
        return self._observation(), self._step_info(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def slot_tickers(self, step: int | None = None) -> list[str]:
        step = self.current_step if step is None else step
        return [self.tickers[u] for u in self.slots[step] if u >= 0]

    def _slot_weights(self, action: np.ndarray, slots: np.ndarray) -> np.ndarray:
        valid = slots >= 0
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.shape[0] == self.num_slots + 1:
            logits = np.nan_to_num(values, nan=0.0, posinf=5.0, neginf=-5.0)
            logits = np.where(np.r_[valid, True], logits, -np.inf)
            logits = logits - logits.max()
            exp = np.exp(logits)
            return (exp / exp.sum())[: self.num_slots]
        if values.shape[0] == self.num_slots:
            weights = np.clip(np.nan_to_num(values, nan=0.0), 0.0, None) * valid
            total = weights.sum()
            return weights / total if total > 1.0 else weights
        raise ValueError(f"action must have {self.num_slots + 1} logits or {self.num_slots} weights")

    def step(self, action: np.ndarray):
        step = self.current_step
        tradable = self.tradable[step]
        previous = self.weights.copy()
        forced = (previous > 0) & ~tradable
        rebalance = self.steps_since_rebalance % self.rebalance_every == 0
        if rebalance:
            slots = self.slots[step]
            slot_weights = self._slot_weights(action, slots)
            target = np.zeros(self.num_series, dtype=np.float64)
            target[slots[slots >= 0]] = slot_weights[slots >= 0]
            self.decision_count += 1
        else:
            target = previous.copy()
            target[forced] = 0.0
        self.steps_since_rebalance += 1

        turnover = float(np.abs(target - previous).sum())
        forced_turnover = float(previous[forced].sum())
        held = target > 0
        returns = np.zeros(self.num_series, dtype=np.float64)
        returns[held] = self.close_prices[step + 1, held] / self.close_prices[step, held] - 1.0
        gross = float(target @ returns)
        cost = self.transaction_cost * turnover
        slippage_cost = self.slippage * turnover
        net = (1.0 - cost - slippage_cost) * (1.0 + gross) - 1.0

        self.portfolio_value *= 1.0 + net
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = 1.0 - self.portfolio_value / self.peak_value if self.peak_value else 0.0
        reward, ratio_term = self._reward(net, turnover, drawdown)
        self.previous_drawdown = drawdown

        growth = 1.0 + gross
        self.weights = np.clip(target * (1.0 + returns) / growth, 0.0, None) if growth > 0 else np.zeros_like(target)
        self.cash_weight = max(0.0, 1.0 - float(self.weights.sum()))
        self.current_step += 1
        terminated = self.current_step >= self.end_step
        info = self._step_info(net, turnover, drawdown, cost, slippage_cost, forced_turnover)
        info.update(gross_return=gross, rebalanced=bool(rebalance), differential_ratio=ratio_term,
                    tradable_count=int(tradable.sum()))
        return self._observation(), float(reward), terminated, False, info

    # ----------------------------------------------------------------------------- views

    def _observation(self) -> np.ndarray:
        step = self.current_step
        slots = self.slots[step]
        valid = slots >= 0
        start = step - self.lookback
        width = len(self.feature_columns)
        n_stock = len(self.stock_feature_columns)
        window = np.zeros((self.lookback, self.num_slots, width), dtype=np.float32)
        if valid.any():
            window[:, valid, :n_stock] = self.stock_tensor[start:step][:, slots[valid], :]
        window[:, :, n_stock:-1] = self.market_tensor[start:step][:, None, :]
        window[:, valid, -1] = 1.0
        slot_weights = np.zeros(self.num_slots, dtype=np.float32)
        slot_weights[valid] = self.weights[slots[valid]]
        obs = np.concatenate([window.reshape(-1), slot_weights, np.array([self.cash_weight], dtype=np.float32)])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _step_info(
        self,
        net_return: float,
        turnover: float,
        drawdown: float,
        transaction_cost_paid: float,
        slippage_paid: float,
        forced_exit_turnover: float,
    ) -> dict[str, Any]:
        return {
            "date": self.dates[self.current_step].date().isoformat(),
            "portfolio_value": float(self.portfolio_value),
            "net_return": float(net_return),
            "turnover": float(turnover),
            "transaction_cost_paid": float(transaction_cost_paid),
            "slippage_paid": float(slippage_paid),
            "drawdown": float(drawdown),
            "cash_weight": float(self.cash_weight),
            "forced_exit_turnover": float(forced_exit_turnover),
            "holdings": {self.tickers[u]: float(w) for u, w in enumerate(self.weights) if w > 1e-9},
        }
