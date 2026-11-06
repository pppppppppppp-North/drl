"""Point-in-time Bank of Thailand policy-rate features (milestone M4: official macro source).

Uses the normalized decision history from ``src.data.ingest_bot_policy_rate``. A decision announced
on day ``d`` becomes available from the first SET trading session strictly after ``d``
(``availability_lag_sessions=1``). The MPC announces in the early afternoon, so this is conservative.
Features (identical across tickers; list them as market-level columns for the fusion extractor):

- ``bot_policy_rate``: policy rate in effect (%);
- ``bot_policy_rate_change_1y``: change versus the rate known 365 days earlier (percentage points);
- ``bot_last_change_bp``: signed size of the most recent non-hold decision (basis points);
- ``bot_years_since_change``: time since that decision became available (years);
- ``bot_last_dissent_share``: share of MPC votes against the decision at the latest meeting (0 if unpublished);
- ``bot_policy_available``: 1 once any decision is available.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POLICY_COLUMNS = [
    "bot_policy_rate",
    "bot_policy_rate_change_1y",
    "bot_last_change_bp",
    "bot_years_since_change",
    "bot_last_dissent_share",
    "bot_policy_available",
]


def _available_dates(announcements: pd.Series, trading_dates: np.ndarray, lag: int) -> pd.Series:
    positions = np.searchsorted(trading_dates, announcements.to_numpy(dtype="datetime64[ns]"), side="right") + (lag - 1)
    out = pd.Series(pd.NaT, index=announcements.index, dtype="datetime64[ns]")
    valid = positions < len(trading_dates)
    out[valid] = trading_dates[positions[valid]]
    return out


def merge_policy_rate_context(
    features: pd.DataFrame,
    history: pd.DataFrame,
    *,
    availability_lag_sessions: int = 1,
) -> pd.DataFrame:
    if availability_lag_sessions < 1:
        raise ValueError("availability_lag_sessions must be at least 1")
    required = {"announcement_date", "policy_rate", "change_bp"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"policy-rate history is missing columns: {sorted(missing)}")
    base = features.copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.drop(columns=[c for c in POLICY_COLUMNS + ["bot_source_announcement"] if c in base.columns])

    hist = history.copy()
    hist["announcement_date"] = pd.to_datetime(hist["announcement_date"])
    hist = hist.sort_values("announcement_date").reset_index(drop=True)
    trading = np.sort(base["date"].unique()).astype("datetime64[ns]")
    hist["available_date"] = _available_dates(hist["announcement_date"], trading, availability_lag_sessions)
    hist = hist.dropna(subset=["available_date"])
    dissent = hist["dissent_share"] if "dissent_share" in hist.columns else pd.Series(0.0, index=hist.index)
    hist["bot_last_dissent_share"] = pd.to_numeric(dissent, errors="coerce").fillna(0.0)

    calendar = pd.DataFrame({"date": pd.to_datetime(trading)})
    level = pd.merge_asof(
        calendar,
        hist[["available_date", "announcement_date", "policy_rate", "bot_last_dissent_share"]].rename(
            columns={"announcement_date": "bot_source_announcement"}
        ),
        left_on="date",
        right_on="available_date",
        direction="backward",
    )
    year_ago = pd.merge_asof(
        pd.DataFrame({"date": calendar["date"] - pd.Timedelta(days=365)}),
        hist[["available_date", "policy_rate"]],
        left_on="date",
        right_on="available_date",
        direction="backward",
    )
    changes = hist[hist["change_bp"] != 0][["available_date", "change_bp"]].rename(columns={"available_date": "change_available"})
    last_change = pd.merge_asof(calendar, changes, left_on="date", right_on="change_available", direction="backward")

    context = pd.DataFrame({"date": calendar["date"]})
    context["bot_policy_available"] = level["policy_rate"].notna().astype(float)
    context["bot_policy_rate"] = level["policy_rate"].fillna(0.0)
    context["bot_policy_rate_change_1y"] = (level["policy_rate"] - year_ago["policy_rate"]).where(
        year_ago["policy_rate"].notna(), 0.0
    ).fillna(0.0)
    context["bot_last_change_bp"] = last_change["change_bp"].fillna(0.0).astype(float)
    context["bot_years_since_change"] = ((calendar["date"] - last_change["change_available"]).dt.days / 365.25).fillna(0.0)
    context["bot_last_dissent_share"] = level["bot_last_dissent_share"].fillna(0.0)
    context["bot_source_announcement"] = level["bot_source_announcement"]

    merged = base.merge(context, on="date", how="left", validate="many_to_one")
    used = merged["bot_source_announcement"].notna()
    if (merged.loc[used, "bot_source_announcement"] >= merged.loc[used, "date"]).any():
        raise AssertionError("policy-rate merge used a same-day or future announcement")
    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", required=True, help="stock feature table (csv or parquet)")
    parser.add_argument("--history", default="data/external/bot_policy_rate_history.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--availability-lag-sessions", type=int, default=1)
    args = parser.parse_args()

    path = Path(args.features)
    features = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    merged = merge_policy_rate_context(
        features, pd.read_csv(args.history), availability_lag_sessions=args.availability_lag_sessions
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        merged.to_parquet(output, index=False)
    else:
        merged.to_csv(output, index=False)
    print(f"wrote {output} rows={len(merged)} coverage={merged['bot_policy_available'].mean():.1%}")


if __name__ == "__main__":
    main()
