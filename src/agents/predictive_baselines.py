"""Prediction-based allocation baselines ("forecast, then allocate").

The implementation plan and proposal ask for ML forecasting baselines (LSTM / XGBoost / Random
Forest) so the DRL agent is compared against a supervised model that sees the same features.
``RidgeForecaster`` is the dependency-free first member of that family: a ridge regression on
standardized features predicting each stock's forward ``horizon``-session return. It is fitted on
training rows only, and its training targets never extend past the training end date.

``LSTMForecaster`` is the sequence model from the proposal (a Bi-LSTM over each stock's *past*
window). It uses the environment's timing convention: features through session ``t - 1`` forecast
the return from the close of ``t`` to ``t + horizon``. Early stopping uses the last block of
training dates, and training samples whose target overlaps that block are purged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RidgeForecaster:
    feature_columns: list[str]
    horizon: int = 21
    alpha: float = 10.0
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    n_train_: int = 0

    def fit(self, train_features: pd.DataFrame) -> "RidgeForecaster":
        frame = train_features.sort_values(["ticker", "date"]).copy()
        frame["target"] = frame.groupby("ticker")["close"].transform(lambda s: s.shift(-self.horizon) / s - 1.0)
        frame = frame.dropna(subset=["target"])  # drops the last `horizon` rows: no target past the training end
        if frame.empty:
            raise ValueError("not enough training rows to build forward-return targets")
        self.n_train_ = int(len(frame))
        x = frame[self.feature_columns].to_numpy(dtype=np.float64)
        y = frame["target"].to_numpy(dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        xs = (x - self.mean_) / self.scale_
        self.intercept_ = float(y.mean())
        gram = xs.T @ xs + self.alpha * np.eye(xs.shape[1])
        self.coef_ = np.linalg.solve(gram, xs.T @ (y - self.intercept_))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit the forecaster first")
        xs = (np.asarray(x, dtype=np.float64) - self.mean_) / self.scale_
        return xs @ self.coef_ + self.intercept_


def top_k_allocation(predictions: np.ndarray, k: int = 2) -> np.ndarray:
    """Equal-weight the ``k`` highest positive forecasts; hold cash when none is positive."""

    weights = np.zeros_like(predictions, dtype=np.float64)
    order = np.argsort(predictions)[::-1][:k]
    chosen = [i for i in order if predictions[i] > 0]
    if chosen:
        weights[chosen] = 1.0 / len(chosen)
    return weights


def forecaster_policy(forecaster: RidgeForecaster, k: int = 2):
    """Policy callable ``(obs, env, step) -> weights`` using the latest completed feature row.

    The environment's observation window ends at ``current_step - 1``; the forecaster uses that same
    row, so it sees exactly the information available to the DRL agent at decision time.
    """

    def _policy(obs, env, step):
        latest = env.feature_tensor[env.current_step - 1]  # (assets, features), ordered as env.feature_columns
        columns = [env.feature_columns.index(c) for c in forecaster.feature_columns]
        return top_k_allocation(forecaster.predict(latest[:, columns]), k=k)

    return _policy


@dataclass
class LSTMForecaster:
    feature_columns: list[str]
    lookback: int = 30
    horizon: int = 21
    hidden_size: int = 32
    bidirectional: bool = True
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    validation_fraction: float = 0.15
    patience: int = 5
    seed: int = 42
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    target_scale_: float = 1.0
    model_: object | None = None
    validation_start_: pd.Timestamp | None = None
    n_train_: int = 0
    n_validation_: int = 0
    best_validation_loss_: float = float("nan")

    def _samples(self, frame: pd.DataFrame):
        xs, ys, decision_dates, target_dates = [], [], [], []
        for _, part in frame.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
            feats = ((part[self.feature_columns].to_numpy(np.float64) - self.mean_) / self.scale_).astype(np.float32)
            close = part["close"].to_numpy(np.float64)
            dates = pd.to_datetime(part["date"]).to_numpy()
            # Decision at session t uses feature rows t-lookback .. t-1 (environment convention).
            for t in range(self.lookback, len(part) - self.horizon):
                xs.append(feats[t - self.lookback : t])
                ys.append(close[t + self.horizon] / close[t] - 1.0)
                decision_dates.append(dates[t])
                target_dates.append(dates[t + self.horizon])
        if not xs:
            raise ValueError("not enough rows per ticker for lookback + horizon")
        return np.stack(xs), np.asarray(ys, dtype=np.float32), np.asarray(decision_dates), np.asarray(target_dates)

    def fit(self, train_features: pd.DataFrame) -> "LSTMForecaster":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        values = train_features[self.feature_columns].to_numpy(dtype=np.float64)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        x, y, decision, target = self._samples(train_features)
        unique_dates = np.sort(np.unique(decision))
        self.validation_start_ = pd.Timestamp(unique_dates[int(len(unique_dates) * (1 - self.validation_fraction))])
        val_mask = decision >= self.validation_start_.to_datetime64()
        train_mask = target < self.validation_start_.to_datetime64()  # purge targets overlapping the block
        if train_mask.sum() < 10 or val_mask.sum() < 5:
            raise ValueError("too few samples after the time-ordered split")
        self.n_train_, self.n_validation_ = int(train_mask.sum()), int(val_mask.sum())
        self.target_scale_ = float(y[train_mask].std()) or 1.0

        directions = 2 if self.bidirectional else 1

        class _Regressor(nn.Module):
            def __init__(self, n_features: int, hidden: int, bidirectional: bool) -> None:
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=bidirectional)
                self.head = nn.Linear(hidden * directions, 1)

            def forward(self, inputs):
                _, (hidden, _) = self.lstm(inputs)
                state = torch.cat([hidden[-2], hidden[-1]], dim=-1) if directions == 2 else hidden[-1]
                return self.head(state).squeeze(-1)

        model = _Regressor(x.shape[-1], self.hidden_size, self.bidirectional)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        xt, yt = torch.as_tensor(x[train_mask]), torch.as_tensor(y[train_mask] / self.target_scale_)
        xv, yv = torch.as_tensor(x[val_mask]), torch.as_tensor(y[val_mask] / self.target_scale_)
        best, best_state, stale = float("inf"), None, 0
        for _ in range(self.epochs):
            model.train()
            order = rng.permutation(len(xt))
            for start in range(0, len(order), self.batch_size):
                idx = torch.as_tensor(order[start : start + self.batch_size])
                optimizer.zero_grad()
                loss = loss_fn(model(xt[idx]), yt[idx])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(xv), yv))
            if val_loss < best - 1e-6:
                best, stale = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.best_validation_loss_ = best
        return self

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """``windows``: (assets, lookback, features) raw feature rows, oldest first."""

        import torch

        if self.model_ is None:
            raise RuntimeError("fit the forecaster first")
        scaled = ((np.asarray(windows, dtype=np.float64) - self.mean_) / self.scale_).astype(np.float32)
        with torch.no_grad():
            return self.model_(torch.as_tensor(scaled)).numpy() * self.target_scale_


def sequence_forecaster_policy(forecaster: LSTMForecaster, k: int = 2):
    """Policy callable using the past ``lookback`` completed feature rows (rows before ``current_step``)."""

    def _policy(obs, env, step):
        start = env.current_step - forecaster.lookback
        if start < 0:
            return np.zeros(env.num_assets)
        columns = [env.feature_columns.index(c) for c in forecaster.feature_columns]
        window = env.feature_tensor[start : env.current_step][:, :, columns]  # (lookback, assets, features)
        return top_k_allocation(forecaster.predict(np.transpose(window, (1, 0, 2))), k=k)

    return _policy
