from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.envs.trading_env import load_feature_table
from src.features.build_features import write_features
from src.utils.config import load_config


METRIC_ALIASES = {
    "total_revenue": ["Total Revenue", "Operating Revenue"],
    "net_income": ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"],
    "total_assets": ["Total Assets"],
    "stockholders_equity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
    "total_debt": ["Total Debt", "Net Debt"],
    "free_cash_flow": ["Free Cash Flow"],
}


def _first_available(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return 0.0


def build_fundamental_context(fundamentals: pd.DataFrame, reporting_lag_days: int = 60) -> pd.DataFrame:
    required = {"ticker", "metric", "period_end", "value"}
    missing = required - set(fundamentals.columns)
    if missing:
        raise ValueError(f"fundamentals missing columns: {sorted(missing)}")

    df = fundamentals.copy()
    df["period_end"] = pd.to_datetime(df["period_end"])
    if "statement_type" in df.columns:
        df["statement_priority"] = df["statement_type"].astype(str).str.contains("quarterly").map({True: 0, False: 1})
        df = df.sort_values(["ticker", "period_end", "metric", "statement_priority"])
        df = df.drop_duplicates(["ticker", "period_end", "metric"], keep="first")
    pivot = (
        df.pivot_table(index=["ticker", "period_end"], columns="metric", values="value", aggfunc="last")
        .reset_index()
        .sort_values(["ticker", "period_end"])
    )

    rows = []
    for _, row in pivot.iterrows():
        total_revenue = _first_available(row, METRIC_ALIASES["total_revenue"])
        net_income = _first_available(row, METRIC_ALIASES["net_income"])
        total_assets = _first_available(row, METRIC_ALIASES["total_assets"])
        equity = _first_available(row, METRIC_ALIASES["stockholders_equity"])
        total_debt = _first_available(row, METRIC_ALIASES["total_debt"])
        free_cash_flow = _first_available(row, METRIC_ALIASES["free_cash_flow"])
        rows.append(
            {
                "ticker": row["ticker"],
                "period_end": row["period_end"],
                "effective_date": row["period_end"] + pd.Timedelta(days=reporting_lag_days),
                "fundamental_total_revenue": total_revenue,
                "fundamental_net_income": net_income,
                "fundamental_total_assets": total_assets,
                "fundamental_equity": equity,
                "fundamental_total_debt": total_debt,
                "fundamental_free_cash_flow": free_cash_flow,
                "fundamental_net_margin": net_income / total_revenue if total_revenue else 0.0,
                "fundamental_return_on_equity": net_income / equity if equity else 0.0,
                "fundamental_debt_to_equity": total_debt / equity if equity else 0.0,
                "fundamental_fcf_margin": free_cash_flow / total_revenue if total_revenue else 0.0,
            }
        )

    context = pd.DataFrame(rows).sort_values(["ticker", "period_end"]).reset_index(drop=True)
    context["fundamental_revenue_growth_qoq"] = (
        context.groupby("ticker")["fundamental_total_revenue"].pct_change().replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    )
    context["fundamental_assets_growth_qoq"] = (
        context.groupby("ticker")["fundamental_total_assets"].pct_change().replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    )
    numeric_columns = [column for column in context.columns if column.startswith("fundamental_")]
    context[numeric_columns] = context[numeric_columns].replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    return context


def merge_fundamental_context(
    features: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    reporting_lag_days: int = 60,
) -> pd.DataFrame:
    base = features.copy()
    base["date"] = pd.to_datetime(base["date"])
    context = build_fundamental_context(fundamentals, reporting_lag_days=reporting_lag_days)
    context["effective_date"] = pd.to_datetime(context["effective_date"])

    merged_frames = []
    for ticker, part in base.groupby("ticker", sort=True):
        ticker_context = context[context["ticker"].eq(ticker)].sort_values("effective_date")
        part = part.sort_values("date")
        if ticker_context.empty:
            merged = part.copy()
        else:
            merged = pd.merge_asof(
                part,
                ticker_context,
                left_on="date",
                right_on="effective_date",
                by="ticker",
                direction="backward",
            )
        merged_frames.append(merged)

    merged = pd.concat(merged_frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    fundamental_columns = [column for column in merged.columns if column.startswith("fundamental_")]
    merged[fundamental_columns] = merged[fundamental_columns].fillna(0.0)
    for column in ["period_end", "effective_date"]:
        if column in merged.columns:
            merged[column] = pd.to_datetime(merged[column]).dt.date
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_fundamentals.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--fundamentals", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    path_config = config["paths"]
    fundamental_config = config.get("fundamentals", {})
    feature_path = Path(args.features or path_config.get("base_features", path_config["features"]))
    fallback_csv = path_config.get("fallback_base_features_csv", path_config.get("fallback_features_csv"))
    output_path = Path(args.output or path_config["features"])
    output_csv = Path(path_config.get("fallback_features_csv", output_path.with_suffix(".csv")))
    fundamentals_path = Path(args.fundamentals or fundamental_config.get("raw_path", "data/raw/fundamentals_yahoo_quarterly.csv"))
    reporting_lag_days = int(fundamental_config.get("reporting_lag_days", 60))

    features = load_feature_table(feature_path, fallback_csv)
    fundamentals = pd.read_csv(fundamentals_path)
    merged = merge_fundamental_context(features, fundamentals, reporting_lag_days=reporting_lag_days)
    written = write_features(merged, output_path, output_csv)
    print(f"wrote {written} rows={len(merged)} columns={len(merged.columns)}")


if __name__ == "__main__":
    main()
