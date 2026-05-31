from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.agents.baselines import run_baselines_on_features
from src.envs.trading_env import RewardConfig, ThaiStockTradingEnv, load_feature_table
from src.evaluation.metrics import equity_metrics
from src.train import _stable_baselines_algorithm, ensure_feature_inputs, evaluate_model
from src.utils.config import ensure_dirs, load_config


def make_env_from_features(config: dict, features: pd.DataFrame) -> ThaiStockTradingEnv:
    env_cfg = config["environment"]
    return ThaiStockTradingEnv(
        features=features,
        feature_columns=config["features"]["columns"],
        lookback=config["features"]["lookback"],
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        reward_config=RewardConfig(**env_cfg.get("reward", {})),
    )


def filter_date_range(features: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    mask = df["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return df[mask].sort_values(["date", "ticker"]).reset_index(drop=True)


def _slice_summary_row(name: str, features: pd.DataFrame, note: str | None = None) -> dict:
    dates = pd.to_datetime(features["date"]) if not features.empty else pd.Series(dtype="datetime64[ns]")
    return {
        "regime": name,
        "start_date": dates.min().date().isoformat() if not features.empty else None,
        "end_date": dates.max().date().isoformat() if not features.empty else None,
        "rows": int(len(features)),
        "unique_dates": int(dates.nunique()) if not features.empty else 0,
        "tickers": int(features["ticker"].nunique()) if "ticker" in features and not features.empty else 0,
        "note": note or "",
    }


def configured_regime_slices(features: pd.DataFrame, config: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    regimes: dict[str, pd.DataFrame] = {}
    rows = []
    for item in config.get("regimes", {}).get("slices", []):
        name = str(item["name"])
        subset = filter_date_range(features, str(item["start"]), str(item["end"]))
        regimes[name] = subset
        rows.append(_slice_summary_row(name, subset, item.get("note")))
    return regimes, pd.DataFrame(rows)


def high_volatility_slices(features: pd.DataFrame, config: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    hv_config = config.get("regimes", {}).get("auto_high_volatility", {})
    if not hv_config.get("enabled", False):
        return {}, pd.DataFrame()

    window = int(hv_config.get("window", 63))
    top_n = int(hv_config.get("top_n", 2))
    min_gap = int(hv_config.get("min_gap", window))
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily_returns = df.groupby("date")["return_1d"].mean().sort_index()
    rolling_vol = daily_returns.rolling(window, min_periods=window).std().dropna()

    selected: list[pd.Timestamp] = []
    for end_date in rolling_vol.sort_values(ascending=False).index:
        if all(abs((end_date - existing).days) >= min_gap for existing in selected):
            selected.append(end_date)
        if len(selected) >= top_n:
            break

    regimes = {}
    rows = []
    all_dates = pd.Index(daily_returns.index)
    for idx, end_date in enumerate(selected, start=1):
        end_pos = all_dates.get_loc(end_date)
        start_pos = max(0, end_pos - window + 1)
        start_date = all_dates[start_pos]
        name = f"high_volatility_{idx}_{start_date.date().isoformat()}_{end_date.date().isoformat()}"
        subset = filter_date_range(df, start_date.date().isoformat(), end_date.date().isoformat())
        regimes[name] = subset
        rows.append(_slice_summary_row(name, subset, f"auto-selected {window}-date realized-volatility window"))
    return regimes, pd.DataFrame(rows)


def _metric_row(regime: str, policy: str, curve: pd.DataFrame) -> dict:
    row = {"regime": regime, "policy": policy}
    row.update(equity_metrics(curve))
    return row


def evaluate_loaded_models(config: dict, regime: str, features: pd.DataFrame, seed: int) -> tuple[list[pd.DataFrame], list[dict]]:
    curves = []
    metrics = []
    for model_config in config.get("regimes", {}).get("model_paths", []):
        path = Path(model_config["path"])
        if not path.exists():
            continue
        algo = str(model_config.get("algo", model_config.get("policy", "ppo"))).lower()
        model_cls = _stable_baselines_algorithm(algo)
        if model_cls is None:
            continue
        model = model_cls.load(path)
        policy = str(model_config.get("policy", algo))
        curve, actions = evaluate_model(
            model,
            make_env_from_features(config, features),
            seed,
            policy_name=policy,
            return_actions=True,
        )
        curves.append(curve)
        metrics.append(_metric_row(regime, policy, curve))
    return curves, metrics


def run_regime_tests(config: dict, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_feature_inputs(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = load_feature_table(config["paths"]["features"], config["paths"].get("fallback_features_csv"))

    configured, configured_summary = configured_regime_slices(features, config)
    high_vol, high_vol_summary = high_volatility_slices(features, config)
    regimes = {**configured, **high_vol}
    slice_summary = pd.concat([configured_summary, high_vol_summary], ignore_index=True)
    slice_summary.to_csv(output_dir / "regime_slices.csv", index=False)

    min_dates = int(config["features"]["lookback"]) + 2
    seed = int(config["project"].get("seed", 42))
    curve_frames = []
    metric_rows = []

    for regime, subset in regimes.items():
        if pd.to_datetime(subset["date"]).nunique() < min_dates:
            continue
        baseline_curves, baseline_metrics = run_baselines_on_features(config, subset, seed=seed)
        baseline_curves.insert(0, "regime", regime)
        baseline_metrics.insert(0, "regime", regime)
        curve_frames.append(baseline_curves)
        metric_rows.extend(baseline_metrics.to_dict("records"))

        model_curves, model_metrics = evaluate_loaded_models(config, regime, subset, seed)
        for curve in model_curves:
            curve.insert(0, "regime", regime)
            curve_frames.append(curve)
        metric_rows.extend(model_metrics)

    if not metric_rows:
        raise ValueError("no regime slices had enough dates to evaluate")

    metrics = pd.DataFrame(metric_rows).sort_values(["regime", "sharpe"], ascending=[True, False]).reset_index(drop=True)
    metrics.to_csv(output_dir / "regime_metrics.csv", index=False)
    if curve_frames:
        pd.concat(curve_frames, ignore_index=True).to_csv(output_dir / "regime_equity_curves.csv", index=False)

    write_regime_summary(metrics, output_dir / "regime_summary.md")
    return metrics, slice_summary


def write_regime_summary(metrics: pd.DataFrame, output_path: Path) -> None:
    best = (
        metrics.sort_values(["regime", "sharpe", "cumulative_return"], ascending=[True, False, False])
        .groupby("regime", as_index=False)
        .first()
        .sort_values("regime")
    )
    columns = ["regime", "policy", "cumulative_return", "annualized_return", "sharpe", "max_drawdown"]
    lines = [
        "# Regime Test Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in best[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/regime_tests_real_ohlcv.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config["paths"]["results_dir"])
    ensure_dirs(output_dir)
    metrics, slice_summary = run_regime_tests(config, output_dir)
    print(slice_summary.to_string(index=False))
    print(metrics[["regime", "policy", "cumulative_return", "annualized_return", "sharpe", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
