from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    losses = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gains / losses.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0) / 100.0


def add_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    frames = []
    for ticker, part in df.groupby("ticker", sort=False):
        part = part.copy()
        close = part["close"]
        high = part["high"]
        low = part["low"]
        volume = part["volume"]

        part["return_1d"] = close.pct_change().fillna(0.0)
        part["return_5d"] = close.pct_change(5).fillna(0.0)
        part["return_20d"] = close.pct_change(20).fillna(0.0)
        part["volatility_20d"] = part["return_1d"].rolling(20, min_periods=2).std().fillna(0.0)
        part["ma_ratio_10"] = close / close.rolling(10, min_periods=1).mean() - 1
        part["ma_ratio_30"] = close / close.rolling(30, min_periods=1).mean() - 1
        part["rsi_14"] = _rsi(close, 14)

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        part["macd"] = (ema_12 - ema_26) / close

        rolling_mean = close.rolling(20, min_periods=2).mean()
        rolling_std = close.rolling(20, min_periods=2).std()
        part["bollinger_z"] = ((close - rolling_mean) / rolling_std.replace(0, np.nan)).fillna(0.0)

        true_range = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        part["atr_14"] = true_range.rolling(14, min_periods=2).mean().fillna(0.0) / close
        part["volume_ratio_20"] = volume / volume.rolling(20, min_periods=1).mean() - 1
        part["ticker"] = ticker
        frames.append(part)

    features = pd.concat(frames, ignore_index=True)
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features.sort_values(["date", "ticker"]).reset_index(drop=True)

