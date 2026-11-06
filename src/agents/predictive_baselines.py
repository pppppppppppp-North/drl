"""Prediction-based allocation baselines ("forecast, then allocate").

The implementation plan and proposal ask for ML forecasting baselines (LSTM / XGBoost / Random
Forest) so the DRL agent is compared against a supervised model that sees the same features.
``RidgeForecaster`` is the dependency-free first member of that family: a ridge regression on
standardized features predicting each stock's forward ``horizon``-session return. It is fitted on
training rows only, and its training targets never extend past the training end date.
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
