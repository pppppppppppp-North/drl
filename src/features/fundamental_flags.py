"""Missing-data flags for lagged fundamentals (milestone M4), additive to ``fundamental_context.py``.

``merge_fundamental_context`` zero-fills fundamentals before a stock's first lagged report, so a
zero cannot be told apart from "no report yet". ``merge_fundamental_context_with_flags`` runs the
locked merge unchanged and adds:

- ``fundamental_available``: 1 once a lagged report is available on that date, else 0;
- ``fundamental_report_age_years``: years since the report became available (0 when none);
- ``fundamental_metric_coverage``: share of the six core metrics actually reported in that period.

    python -m src.features.fundamental_flags --config config/real_ohlcv_fundamentals.yaml --output <path>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.envs.trading_env import load_feature_table
from src.features.build_features import write_features
from src.features.fundamental_context import METRIC_ALIASES, merge_fundamental_context
from src.utils.config import load_config


def fundamental_metric_coverage(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Per ticker and period, the share of the six core metrics present (same dedupe as the locked builder)."""

    df = fundamentals.copy()
    df["period_end"] = pd.to_datetime(df["period_end"])
    if "statement_type" in df.columns:
        df["statement_priority"] = df["statement_type"].astype(str).str.contains("quarterly").map({True: 0, False: 1})
        df = df.sort_values(["ticker", "period_end", "metric", "statement_priority"])
        df = df.drop_duplicates(["ticker", "period_end", "metric"], keep="first")
    pivot = df.pivot_table(index=["ticker", "period_end"], columns="metric", values="value", aggfunc="last").reset_index()

    def coverage(row: pd.Series) -> float:
        observed = sum(any(name in row and pd.notna(row[name]) for name in aliases) for aliases in METRIC_ALIASES.values())
        return observed / len(METRIC_ALIASES)

    pivot["fundamental_metric_coverage"] = pivot.apply(coverage, axis=1)
    return pivot[["ticker", "period_end", "fundamental_metric_coverage"]]


def merge_fundamental_context_with_flags(
    features: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    reporting_lag_days: int = 60,
) -> pd.DataFrame:
    merged = merge_fundamental_context(features, fundamentals, reporting_lag_days=reporting_lag_days)
    dates = pd.to_datetime(merged["date"])
    effective = (
        pd.to_datetime(merged["effective_date"]) if "effective_date" in merged.columns else pd.Series(pd.NaT, index=merged.index)
    )
    merged["fundamental_available"] = effective.notna().astype(float)
    merged["fundamental_report_age_years"] = ((dates - effective).dt.days / 365.25).where(effective.notna(), 0.0)
    if "period_end" in merged.columns:
        coverage = fundamental_metric_coverage(fundamentals)
        keys = pd.DataFrame({"ticker": merged["ticker"], "period_end": pd.to_datetime(merged["period_end"])})
        joined = keys.merge(coverage, on=["ticker", "period_end"], how="left")
        merged["fundamental_metric_coverage"] = joined["fundamental_metric_coverage"].fillna(0.0).to_numpy()
    else:
        merged["fundamental_metric_coverage"] = 0.0
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/real_ohlcv_fundamentals.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--fundamentals", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    fundamental_config = config.get("fundamentals", {})
    feature_path = Path(args.features or paths.get("base_features", paths["features"]))
    fallback_csv = paths.get("fallback_base_features_csv", paths.get("fallback_features_csv"))
    fundamentals_path = Path(args.fundamentals or fundamental_config.get("raw_path", "data/raw/fundamentals_yahoo_quarterly.csv"))
    merged = merge_fundamental_context_with_flags(
        load_feature_table(feature_path, fallback_csv),
        pd.read_csv(fundamentals_path),
        reporting_lag_days=int(fundamental_config.get("reporting_lag_days", 60)),
    )
    output = Path(args.output)
    written = write_features(merged, output, output.with_suffix(".csv"))
    print(f"wrote {written} rows={len(merged)} columns={len(merged.columns)}")


if __name__ == "__main__":
    main()
