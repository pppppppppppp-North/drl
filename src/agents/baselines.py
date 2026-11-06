from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import select_chronological_split, split_config_from_mapping
from src.envs.trading_env import RewardConfig, ThaiStockTradingEnv, load_feature_table
from src.evaluation.metrics import equity_metrics
from src.utils.config import ensure_dirs, load_config


def _buy_hold_ticker(policy_name: str) -> str | None:
    prefix = "buy_hold_"
    if policy_name.startswith(prefix):
        return policy_name[len(prefix) :]
    return None


def _moving_average_crossover_action(env: ThaiStockTradingEnv, short_window: int = 10, long_window: int = 30) -> np.ndarray:
    current = env.current_step
    start = max(0, current - long_window + 1)
    short_start = max(0, current - short_window + 1)
    short_ma = env.close_prices[short_start : current + 1].mean(axis=0)
    long_ma = env.close_prices[start : current + 1].mean(axis=0)
    signal = short_ma > long_ma
    if signal.any():
        return signal.astype(np.float64) / signal.sum()
    return np.zeros(env.num_assets, dtype=np.float64)


def _mean_variance_action(env: ThaiStockTradingEnv, window: int = 60, ridge: float = 1e-4) -> np.ndarray:
    current = env.current_step
    start = max(0, current - window)
    prices = env.close_prices[start : current + 1]
    if len(prices) < 3:
        return np.full(env.num_assets, 1.0 / env.num_assets)

    returns = np.divide(prices[1:], prices[:-1], out=np.ones_like(prices[1:]), where=prices[:-1] != 0) - 1.0
    expected_returns = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False)
    covariance = np.atleast_2d(covariance) + np.eye(env.num_assets) * ridge
    raw_weights = np.linalg.pinv(covariance) @ expected_returns
    weights = np.clip(raw_weights, 0.0, None)
    total = weights.sum()
    if total <= 0:
        return np.zeros(env.num_assets, dtype=np.float64)
    return weights / total


def _run_policy(env: ThaiStockTradingEnv, policy_name: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    rows = [
        {
            "date": info["date"],
            "portfolio_value": info["portfolio_value"],
            "policy": policy_name,
            "net_return": info["net_return"],
            "turnover": info["turnover"],
            "transaction_cost_paid": info.get("transaction_cost_paid", 0.0),
            "slippage_paid": info.get("slippage_paid", 0.0),
            "drawdown": info["drawdown"],
            "cash_weight": info["cash_weight"],
        }
    ]
    done = False

    while not done:
        if policy_name == "equal_weight":
            action = np.full(env.num_assets, 1.0 / env.num_assets)
        elif policy_name == "random":
            action = rng.random(env.num_assets)
            action = action / max(action.sum(), 1.0)
        elif policy_name == "momentum":
            lookback_returns = env.feature_tensor[env.current_step - 1, :, env.feature_columns.index("return_5d")]
            action = np.clip(lookback_returns, 0.0, None)
            action = action / action.sum() if action.sum() > 0 else np.zeros(env.num_assets)
        elif policy_name == "ma_crossover":
            action = _moving_average_crossover_action(env)
        elif policy_name == "mean_variance":
            action = _mean_variance_action(env)
        else:
            ticker = _buy_hold_ticker(policy_name)
            if ticker is None or ticker not in env.tickers:
                raise ValueError(f"unknown policy {policy_name}")
            action = np.zeros(env.num_assets, dtype=np.float64)
            action[env.tickers.index(ticker)] = 1.0

        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "date": info["date"],
                "portfolio_value": info["portfolio_value"],
                "policy": policy_name,
                "net_return": info["net_return"],
                "turnover": info["turnover"],
                "transaction_cost_paid": info.get("transaction_cost_paid", 0.0),
                "slippage_paid": info.get("slippage_paid", 0.0),
                "drawdown": info["drawdown"],
                "cash_weight": info["cash_weight"],
            }
        )
        done = terminated or truncated

    return pd.DataFrame(rows)


def _run_return_benchmark(
    features: pd.DataFrame,
    *,
    return_column: str,
    policy_name: str,
    lookback: int,
    initial_cash: float,
    transaction_cost: float,
    slippage: float = 0.0,
) -> pd.DataFrame:
    """Build a buy-and-hold benchmark from a daily index-return column.

    The curve starts on the same date as ``ThaiStockTradingEnv.reset`` and
    applies the same initial cash-to-asset turnover cost as the stock
    buy-and-hold policies. Subsequent index holding turnover is zero.
    """
    if return_column not in features.columns:
        raise ValueError(f"configured benchmark return column is missing: {return_column}")

    daily = (
        features.assign(date=pd.to_datetime(features["date"]))
        .sort_values(["date", "ticker"])
        .drop_duplicates("date")[["date", return_column]]
        .reset_index(drop=True)
    )
    if len(daily) <= lookback + 1:
        raise ValueError(
            f"benchmark requires more than lookback + 1 dates; found {len(daily)} with lookback={lookback}"
        )

    value = float(initial_cash)
    peak = value
    rows = [
        {
            "date": daily.iloc[lookback]["date"].date().isoformat(),
            "portfolio_value": value,
            "policy": policy_name,
            "net_return": 0.0,
            "turnover": 0.0,
            "transaction_cost_paid": 0.0,
            "slippage_paid": 0.0,
            "drawdown": 0.0,
            "cash_weight": 0.0,
        }
    ]
    for position in range(lookback + 1, len(daily)):
        gross_return = float(pd.to_numeric(daily.iloc[position][return_column], errors="coerce"))
        if not np.isfinite(gross_return):
            gross_return = 0.0
        turnover = 2.0 if position == lookback + 1 else 0.0
        transaction_cost_paid = float(transaction_cost) * turnover
        slippage_paid = float(slippage) * turnover
        net_return = gross_return - transaction_cost_paid - slippage_paid
        value *= 1.0 + net_return
        peak = max(peak, value)
        rows.append(
            {
                "date": daily.iloc[position]["date"].date().isoformat(),
                "portfolio_value": value,
                "policy": policy_name,
                "net_return": net_return,
                "turnover": turnover,
                "transaction_cost_paid": transaction_cost_paid,
                "slippage_paid": slippage_paid,
                "drawdown": 1.0 - value / peak if peak else 0.0,
                "cash_weight": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _baseline_split(config: dict) -> str:
    if "splits" not in config:
        return "all"
    return str(config.get("baseline_split", config["splits"].get("eval_split", "validation")))


def run_baselines_on_features(
    config: dict,
    features: pd.DataFrame,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    env_cfg = config["environment"]
    reward_cfg = RewardConfig(**env_cfg.get("reward", {}))
    tickers = sorted(features["ticker"].dropna().astype(str).unique())
    policies = ["equal_weight", "random", "momentum", "ma_crossover", "mean_variance"] + [
        f"buy_hold_{ticker}" for ticker in tickers
    ]
    curves = []
    metrics = []

    for policy in policies:
        env = ThaiStockTradingEnv(
            features=features,
            feature_columns=config["features"]["columns"],
            lookback=config["features"]["lookback"],
            initial_cash=env_cfg["initial_cash"],
            transaction_cost=env_cfg["transaction_cost"],
            slippage=env_cfg.get("slippage", 0.0),
            max_turnover=env_cfg.get("max_turnover"),
            max_position_weight=env_cfg.get("max_position_weight", 1.0),
            min_cash_weight=env_cfg.get("min_cash_weight", 0.0),
            reward_config=reward_cfg,
        )
        curve = _run_policy(env, policy, seed or config["project"].get("seed", 42))
        curves.append(curve)
        row = {"policy": policy}
        row.update(equity_metrics(curve))
        metrics.append(row)

    benchmark_cfg = config.get("baselines", {}).get("market_index")
    if benchmark_cfg:
        policy_name = str(benchmark_cfg.get("policy_name", "set_index"))
        curve = _run_return_benchmark(
            features,
            return_column=str(benchmark_cfg.get("return_column", "set_return_1d")),
            policy_name=policy_name,
            lookback=int(config["features"]["lookback"]),
            initial_cash=float(env_cfg["initial_cash"]),
            transaction_cost=float(env_cfg["transaction_cost"]),
            slippage=float(env_cfg.get("slippage", 0.0)),
        )
        curves.append(curve)
        row = {"policy": policy_name}
        row.update(equity_metrics(curve))
        metrics.append(row)

    return pd.concat(curves, ignore_index=True), pd.DataFrame(metrics)


def run_baselines(config: dict, split: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = load_feature_table(config["paths"]["features"], config["paths"].get("fallback_features_csv"))
    split = split or _baseline_split(config)
    features = select_chronological_split(features, split, split_config_from_mapping(config.get("splits")))
    return run_baselines_on_features(config, features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--split", choices=["all", "train", "validation", "test"], default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config["paths"]["results_dir"]) / "baselines"
    ensure_dirs(output_dir)
    curves, metrics = run_baselines(config, split=args.split)
    curves.to_csv(output_dir / "baseline_equity_curves.csv", index=False)
    metrics.to_csv(output_dir / "baselines_metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
