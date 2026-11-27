"""Point-in-time sector-relative inputs for the SET50 walk-forward (milestone M4, the sector source).

Each half-year SET50/SET100 list in the official 2012-2024 archive gives every member's SET sector. On a
session, a price series takes the sector of the list in force that session. A series that is not on that
session's list keeps the last sector an earlier list gave it (never a later one), and a series that no
earlier list contains has none. Its **peers** on a session are the other SET100 members on that session's
list that have a price series, the same sector and a real return over the horizon: a finite value from a
series with more sessions than the horizon (the technical builder fills the first sessions with 0).

Four stock-level inputs, computed from the same closes as the technical inputs. The environment's window
ends at the session before each decision, so a decision never sees its own session:

- ``sector_relative_return_5d`` and ``sector_relative_return_20d``: the series' 5- and 20-session return
  minus the equal-weighted mean of its peers' returns over the same sessions;
- ``sector_peer_return_20d``: that peer mean over 20 sessions (the sector's momentum without the stock);
- ``sector_peers_available``: 1 when the series has a sector and at least one peer with a 5-session return
  that session, otherwise 0.

A horizon with no peer gives 0, not a missing value. Wherever a series' own 5- and 20-session returns are
finite, all four inputs are finite, so adding them leaves the environment's tradable set unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_COLUMNS = [
    "sector_relative_return_5d",
    "sector_relative_return_20d",
    "sector_peer_return_20d",
    "sector_peers_available",
]
HORIZONS = (5, 20)


def session_sectors(mask: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """One row per session and listed price series: its dated sector.

    ``mask`` is the SET100 session mask of ``src.data.point_in_time_panel.session_membership_mask`` (before
    any SET50 filter); ``membership`` is the archive with ``record_id``, ``ticker`` and ``sector``.
    """
    labels = membership[["record_id", "ticker", "sector"]].astype(str)
    if labels["sector"].str.strip().eq("").any():
        raise ValueError("the membership archive has a member without a sector")
    listed = mask[["date", "record_id", "ticker", "price_ticker"]].merge(
        labels, on=["record_id", "ticker"], how="left", validate="many_to_one")
    if listed["sector"].isna().any():
        raise ValueError("a session's list member has no row in the membership archive")
    listed = listed[listed["price_ticker"].astype(str).ne("")]
    listed = listed[["date", "price_ticker", "sector"]].drop_duplicates()
    if listed.duplicated(["date", "price_ticker"]).any():
        raise ValueError("one price series has two sectors on the same session")
    return listed.rename(columns={"price_ticker": "ticker"}).reset_index(drop=True)


def add_sector_context(panel: pd.DataFrame, mask: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """``panel`` with the four ``SECTOR_COLUMNS`` added (see the module docstring); rows are unchanged."""
    missing = {"date", "ticker", "return_5d", "return_20d"} - set(panel.columns)
    if missing:
        raise ValueError(f"the panel is missing columns: {sorted(missing)}")
    listed = session_sectors(mask, membership).assign(listed=True)
    listed["date"] = pd.to_datetime(listed["date"])
    out = panel.drop(columns=[c for c in SECTOR_COLUMNS if c in panel.columns])
    out = out.merge(listed, on=["date", "ticker"], how="left", validate="one_to_one")
    out["listed"] = out["listed"].eq(True)
    # A series off the list keeps its last earlier sector; before its first listing it has none.
    out = out.sort_values(["ticker", "date"])
    out["sector"] = out.groupby("ticker", sort=False)["sector"].ffill()

    peer_means = {}
    # The technical builder fills a series' first sessions with 0 (src/features/technical.py), so a peer
    # counts towards a horizon only once its series has more sessions than the horizon.
    history = (pd.to_numeric(out["history_sessions"], errors="coerce") if "history_sessions" in out
               else pd.Series(np.inf, index=out.index))
    for horizon in HORIZONS:
        column = f"return_{horizon}d"
        own = pd.to_numeric(out[column], errors="coerce")
        counted = out["listed"] & np.isfinite(own) & (history > horizon)
        value = own.where(counted, 0.0)
        totals = (out.assign(_value=value, _count=counted.astype(np.int64))
                  .groupby(["date", "sector"])[["_value", "_count"]].transform("sum"))
        # Exclude the series itself where it is one of the session's counted members.
        total = totals["_value"] - value
        count = totals["_count"] - counted.astype(np.int64)
        has_sector = out["sector"].notna()
        count = count.where(has_sector, 0)
        mean = (total / count.where(count > 0)).where(count > 0)
        peer_means[horizon] = (mean, count)
        # Missing only where the series' own return is missing, which already makes the row untradable.
        out[f"sector_relative_return_{horizon}d"] = (own - mean).where(count > 0, 0.0)

    mean_20, count_20 = peer_means[20]
    out["sector_peer_return_20d"] = mean_20.where(count_20 > 0, 0.0)
    out["sector_peers_available"] = (peer_means[5][1] > 0).astype(np.float64)
    out = out.drop(columns=["sector", "listed"])
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)
