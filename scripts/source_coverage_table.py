"""Coverage of every input source of the point-in-time SET50 design (milestone M4).

For each source the title's "multi-source data" refers to, this measures how much of the modelling window
the data really cover, and how much is carried forward, stale or missing, instead of reporting only whether
a file exists (``src.data.source_readiness``):

- **stock-level sources** (prices, the technical inputs, the dated sector) per SET50 member-session: the
  share with a value from that session, and the share that is stale (a zero-volume price row);
- **market-level sources** (the SET50 ETF, the BOT policy rate, each daily macro proxy) per Thai session:
  the share with an observation dated that session, the share carried forward from an earlier one, and the
  longest carry in calendar days;
- **sources without historical data** (fundamentals, news sentiment, investor flow, a release-dated CPI)
  are listed with zero coverage and the file each would need.

The window is the training and validation years only, the first session of 2012 to ``splits.validation_end``
(2022-12-30), so the 2023-2024 test years and the sealed holdout are not read.

Usage::

    python -m scripts.source_coverage_table --config config/pit_source_ablation_cashcap.yaml \\
        --output-dir results/source_coverage_20261127
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.run_pit_walk_forward import load_inputs
from src.features.pit_sector_context import SECTOR_COLUMNS

MISSING_SOURCES = [
    ("Fundamentals (P/E, P/BV, ROE, D/E, yield)", "data/external/fundamentals.csv",
     "licensed SETSMART package (request ready since 2026-09-01, unsent)"),
    ("News sentiment", "data/external/historical_news.csv",
     "a news history for 2012-2022; the Yahoo latest-news probe has none"),
    ("Investor flow (foreign, institution, proprietary, retail)", "data/external/investor_flow.csv",
     "the SET daily trading-value-by-investor-type export"),
    ("Release-dated CPI", "data/external/official_macro_long.csv",
     "a CPI series with its release dates"),
]


def carried_forward(sessions: pd.DatetimeIndex, observed: pd.DatetimeIndex) -> dict:
    """Same-day share, carried share and longest carry (calendar days) of a series on ``sessions``."""
    observed = pd.DatetimeIndex(sorted(set(observed)))
    same_day = sessions.isin(observed)
    position = observed.searchsorted(sessions, side="right") - 1
    has_earlier = position >= 0
    last = observed[np.clip(position, 0, None)]
    gap = np.where(has_earlier, (sessions - last).days, np.nan)
    carried = has_earlier & ~same_day
    return {
        "units": int(len(sessions)),
        "covered_share": float(same_day.mean()),
        "carried_or_stale_share": float(carried.mean()),
        "missing_share": float((~has_earlier).mean()),
        "longest_carry_days": float(np.nanmax(gap)) if carried.any() else 0.0,
    }


def member_sessions(mask: pd.DataFrame, panel: pd.DataFrame, stock_columns: list[str]) -> pd.DataFrame:
    """One row per SET50 member-session with its price, staleness and input flags."""
    rows = mask[["date", "ticker", "price_ticker"]].rename(columns={"ticker": "member"})
    keep = ["date", "ticker", "close", "volume", "history_sessions", *stock_columns,
            *[c for c in SECTOR_COLUMNS if c in panel]]
    rows = rows.merge(panel[keep], left_on=["date", "price_ticker"], right_on=["date", "ticker"], how="left")
    rows["mapped"] = rows["price_ticker"].astype(str).ne("")
    rows["priced"] = rows["mapped"] & np.isfinite(rows["close"]) & (rows["close"] > 0)
    rows["stale"] = rows["priced"] & rows["volume"].fillna(0).le(0)
    rows["technical"] = rows["priced"] & np.isfinite(rows[stock_columns]).all(axis=1)
    # The technical builder fills the 20-session inputs of a series' first 20 sessions with 0.
    rows["warm_up"] = rows["priced"] & rows["history_sessions"].le(20)
    if "sector_peers_available" in rows:
        rows["sector_peer"] = rows["priced"] & rows["sector_peers_available"].eq(1.0)
    return rows


def coverage_table(spec: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = {**spec, "data": {**spec["data"], "sector": True}}
    end = pd.Timestamp(spec["splits"]["validation_end"])
    mask, panel, _ = load_inputs(spec, root)
    mask, panel = mask[mask["date"] <= end], panel[panel["date"] <= end]
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    stock_columns = list(spec["features"]["stock_columns"])
    members = member_sessions(mask, panel, stock_columns)
    n = len(members)
    rows = [
        {"source": "Stock prices (Yahoo .BK)", "level": "stock", "unit": "SET50 member-session", "units": n,
         "covered_share": members["priced"].mean(), "carried_or_stale_share": members["stale"].mean(),
         "missing_share": 1 - members["priced"].mean(),
         "detail": f"{(~members['mapped']).mean():.1%} have no Yahoo series (delisted or renamed members); "
                   "stale = a zero-volume row"},
        {"source": f"Technical inputs ({len(stock_columns)})", "level": "stock", "unit": "SET50 member-session",
         "units": n, "covered_share": members["technical"].mean() - members["warm_up"].mean(),
         "carried_or_stale_share": members["warm_up"].mean(), "missing_share": 1 - members["technical"].mean(),
         "detail": "filled = a series' first 20 sessions, whose 20-session inputs are 0; a member needs 60 "
                   "sessions to trade, so the agent's windows never contain them"},
        {"source": "Dated SET sector (membership archive)", "level": "stock", "unit": "SET50 member-session",
         "units": n, "covered_share": members["sector_peer"].mean(), "carried_or_stale_share": 0.0,
         "missing_share": 1 - members["sector_peer"].mean(),
         "detail": "every member has its list's sector; covered = at least one priced SET100 peer in it "
                   f"({members['sector_peer'].sum() / members['priced'].sum():.1%} of priced member-sessions)"},
    ]
    data = spec["data"]
    index = pd.read_csv(root / data["index"], parse_dates=["date"])
    index = index[index["date"] <= end]
    etf = carried_forward(sessions, pd.DatetimeIndex(index["date"]))
    traded = index[index["volume"] > 0]["date"]
    rows.append({"source": "SET50 ETF (TDEX), market inputs", "level": "market", "unit": "session", **etf,
                 "detail": f"{1 - sessions.isin(pd.DatetimeIndex(traded)).mean():.1%} of sessions without a traded row"})
    if data.get("policy_rate"):
        history = pd.read_csv(root / data["policy_rate"], parse_dates=["announcement_date"])
        decisions = history[(history["announcement_date"] <= end) & (history["announcement_date"] >= sessions[0])]
        rate = carried_forward(sessions, pd.DatetimeIndex(history["announcement_date"]))
        rows.append({"source": "BOT policy rate (MPC decisions)", "level": "market", "unit": "session",
                     **{**rate, "covered_share": 1.0 - rate["missing_share"], "carried_or_stale_share": 0.0},
                     "detail": f"{len(decisions)} decisions in the window; a step series, so every session has "
                               "the rate in force (no carry counted); longest carry = longest gap since a decision"})
    if data.get("macro"):
        macro = pd.read_csv(root / data["macro"], parse_dates=["date"])
        macro = macro[macro["date"] <= end]
        for ticker, name in data.get("macro_names", {}).items():
            series = macro[macro["ticker"] == ticker]
            rows.append({"source": f"Macro proxy {name} ({ticker})", "level": "market", "unit": "session",
                         **carried_forward(sessions, pd.DatetimeIndex(series["date"])),
                         "detail": "carried = the latest earlier close (holidays abroad)"})
    for source, path, need in MISSING_SOURCES:
        present = (root / path).is_file()
        rows.append({"source": source, "level": "-", "unit": "-", "units": 0, "covered_share": 0.0,
                     "carried_or_stale_share": np.nan, "missing_share": 1.0,
                     "detail": ("file present but not built into the inputs: " if present else "no historical file; needs ")
                               + need})
    table = pd.DataFrame(rows)
    members["year"] = members["date"].dt.year
    by_year = members.groupby("year").agg(
        member_sessions=("priced", "size"), no_yahoo_series=("mapped", lambda s: 1 - s.mean()),
        priced=("priced", "mean"), stale=("stale", "mean"), technical=("technical", "mean"),
        warm_up=("warm_up", "mean"), sector_peer=("sector_peer", "mean")).reset_index()
    return table, by_year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root", default=".", type=Path)
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.config).read_text())
    table, by_year = coverage_table(spec, args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "source_coverage.csv", index=False)
    by_year.to_csv(args.output_dir / "stock_coverage_by_year.csv", index=False)
    (args.output_dir / "source_coverage_receipt.json").write_text(json.dumps(
        {"config": args.config, "window_end": spec["splits"]["validation_end"], "rows": len(table)}, indent=2) + "\n")
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 70):
        print(table.round(4).to_string(index=False))
        print(by_year.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
