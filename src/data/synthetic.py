from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ensure_dirs, load_config


def make_synthetic_prices(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    rng = np.random.default_rng(config["project"].get("seed", 42))
    dates = pd.bdate_range(data_cfg["start"], data_cfg["end"])
    tickers = list(data_cfg["tickers"])

    market_noise = rng.normal(0.00025, 0.008, len(dates))
    macro_rate_change = np.cumsum(rng.normal(0.0, 0.0005, len(dates)))

    rows: list[dict] = []
    for idx, ticker in enumerate(tickers):
        beta = 0.75 + 0.15 * idx
        idio = rng.normal(0.0001 + idx * 0.00002, 0.012 + idx * 0.001, len(dates))
        sentiment = np.clip(
            0.35 * np.roll(market_noise, 1) + rng.normal(0.0, 0.04, len(dates)),
            -1.0,
            1.0,
        )
        sentiment[0] = 0.0
        returns = beta * market_noise + idio + 0.0004 * sentiment
        close = data_cfg.get("initial_price", 100.0) * np.exp(np.cumsum(returns))
        open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.003, 0.002, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.003, 0.002, len(dates))))
        volume = rng.integers(800_000, 8_000_000, len(dates)) * (1 + idx * 0.1)

        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "ticker": ticker,
                    "open": float(open_[i]),
                    "high": float(high[i]),
                    "low": float(low[i]),
                    "close": float(close[i]),
                    "volume": int(volume[i]),
                    "market_return_1d": float(market_noise[i]),
                    "macro_rate_change": float(macro_rate_change[i]),
                    "sentiment_score": float(sentiment[i]),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output = Path(args.output or config["paths"]["raw_prices"])
    ensure_dirs(output.parent)
    make_synthetic_prices(config).to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

