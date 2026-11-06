"""Market-wide investor-flow features merged into the stock feature table without look-ahead.

For each investor group the builder computes, on the flow calendar:

- ``flow_<group>_net_share``: daily net value / total market value;
- ``flow_<group>_net_5d`` / ``_net_20d``: rolling net value / rolling market value;
- ``flow_<group>_net_z60``: z-score of the daily net share against the *previous* 60 sessions.

Flow data for date ``t`` is published after the close, so it is attached to the first stock
trading date strictly after ``t`` (``availability_lag_sessions=1``) and merged backward-only.
The merged columns are identical across tickers (market-level), so list them in
``training.policy.market_feature_columns`` when using the fusion extractor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingest_investor_flow import INVESTOR_TYPES

FLOW_PREFIX = "flow_"


def build_investor_flow_features(flow: pd.DataFrame, z_window: int = 60) -> pd.DataFrame:
    required = {"date", "investor_type", "buy_value", "net_value"}
    missing = required - set(flow.columns)
    if missing:
        raise ValueError(f"investor flow table is missing columns: {sorted(missing)}")
    data = flow.copy()
    data["date"] = pd.to_datetime(data["date"])
    market_value = data.groupby("date")["buy_value"].sum().sort_index()
    net = data.pivot_table(index="date", columns="investor_type", values="net_value", aggfunc="sum").sort_index()
    net = net.reindex(market_value.index)

    features = pd.DataFrame(index=market_value.index)
    safe_market = market_value.replace(0.0, np.nan)
    for group in INVESTOR_TYPES:
        if group not in net.columns:
            continue
        share = net[group] / safe_market
        features[f"{FLOW_PREFIX}{group}_net_share"] = share
        for window in (5, 20):
            features[f"{FLOW_PREFIX}{group}_net_{window}d"] = (
                net[group].rolling(window, min_periods=window).sum()
                / safe_market.rolling(window, min_periods=window).sum()
            )
        past = share.shift(1).rolling(z_window, min_periods=max(10, z_window // 3))
        features[f"{FLOW_PREFIX}{group}_net_z{z_window}"] = (share - past.mean()) / past.std(ddof=0)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.reset_index().rename(columns={"index": "date"})


def merge_investor_flow_context(
    features: pd.DataFrame,
    flow: pd.DataFrame,
    *,
    availability_lag_sessions: int = 1,
    max_age_days: int = 7,
    z_window: int = 60,
) -> pd.DataFrame:
    if availability_lag_sessions < 1:
        raise ValueError("availability_lag_sessions must be at least 1: flows are published after the close")
    base = features.copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.drop(columns=[c for c in base.columns if c.startswith(FLOW_PREFIX)])
    flow_features = build_investor_flow_features(flow, z_window=z_window)

    trading_dates = pd.Series(sorted(base["date"].unique()))
    positions = np.searchsorted(trading_dates.to_numpy(), flow_features["date"].to_numpy(), side="right")
    positions = positions + (availability_lag_sessions - 1)
    valid = positions < len(trading_dates)
    flow_features = flow_features.loc[valid].copy()
    flow_features["flow_available_date"] = trading_dates.to_numpy()[positions[valid]]
    flow_features = flow_features.rename(columns={"date": "flow_source_date"})
    flow_features = flow_features.sort_values("flow_available_date").drop_duplicates("flow_available_date", keep="last")

    merged = pd.merge_asof(
        base.sort_values("date"),
        flow_features,
        left_on="date",
        right_on="flow_available_date",
        direction="backward",
        tolerance=pd.Timedelta(days=max_age_days),
    )
    flow_columns = [c for c in merged.columns if c.startswith(FLOW_PREFIX) and c not in {"flow_available_date", "flow_source_date"}]
    merged["flow_data_available"] = merged["flow_source_date"].notna().astype(float)
    merged[flow_columns] = merged[flow_columns].fillna(0.0)
    if (merged["flow_source_date"].dropna() >= merged.loc[merged["flow_source_date"].notna(), "date"]).any():
        raise AssertionError("investor-flow merge used same-day or future flow data")
    return merged.drop(columns=["flow_available_date"]).sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", required=True, help="stock feature table (csv or parquet)")
    parser.add_argument("--flow", required=True, help="normalized output of src.data.ingest_investor_flow")
    parser.add_argument("--output", required=True)
    parser.add_argument("--availability-lag-sessions", type=int, default=1)
    args = parser.parse_args()

    feature_path = Path(args.features)
    features = pd.read_parquet(feature_path) if feature_path.suffix == ".parquet" else pd.read_csv(feature_path)
    flow = pd.read_csv(args.flow)
    merged = merge_investor_flow_context(features, flow, availability_lag_sessions=args.availability_lag_sessions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        merged.to_parquet(output, index=False)
    else:
        merged.to_csv(output, index=False)
    covered = merged["flow_data_available"].mean()
    print(f"wrote {output} rows={len(merged)} flow_coverage={covered:.1%}")


if __name__ == "__main__":
    main()
