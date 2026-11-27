"""Trading diagnostics for walk-forward DRL agents: costs, turnover, cash and concentration.

For every fold and seed of one seeded variant, the agent's validation-year return is split into the
return before costs and the cost drag, and both are compared with a reference baseline on the same
sessions (the point-in-time monthly equal weight by default). Where the runner saved holdings
(``runs/<variant>_seed<S>/validation_holdings.csv``), it also measures, on rebalance sessions, how
spread out the weights are (effective number of holdings, largest weight, top-10 share) and their
distance from equal weight over the tradable members. With ``--pit-config`` it rebuilds the slot
assignment of each validation session and splits rebalance turnover between members that changed
slot since the previous decision and those that kept it.

Accounting follows ``PointInTimeSlotEnv.step`` and ``LongTermTradingEnv``: a session's net growth is
``(1 - cost - slippage) * (1 + gross)``, so the gross growth is the net growth divided by
``1 - cost - slippage``. Outputs, written to ``--output-dir`` (default ``--run-dir``):

- ``agent_diagnostics_runs.csv``: one row per fold x policy (every seed and the reference);
- ``agent_diagnostics_summary.csv``: per policy family, the mean over folds of the seed means.

    python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_walk_forward \\
        --variant pit_plan_weekly_lstm --reference pit_equal_weight_monthly --rebalance-every 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.aggregate_walk_forward_reward import load_fold

TRADING_DAYS = 252


def curve_diagnostics(curve: pd.DataFrame, rebalance_every: int | None = None) -> dict[str, float]:
    """Return, cost and trading statistics for one policy's validation curve (first row = start)."""
    value = curve["portfolio_value"].to_numpy(dtype=float)
    growth = value[1:] / value[:-1]
    paid = np.zeros_like(growth)
    for column in ("transaction_cost_paid", "slippage_paid"):
        if column in curve:
            paid += np.nan_to_num(curve[column].to_numpy(dtype=float)[1:])
    gross_growth = growth / (1.0 - paid)
    turnover = np.nan_to_num(curve["turnover"].to_numpy(dtype=float)[1:]) if "turnover" in curve else np.zeros_like(growth)
    net = float(value[-1] / value[0] - 1.0)
    gross = float(np.prod(gross_growth) - 1.0)
    out = {
        "net_return": net,
        "gross_return": gross,
        "cost_drag": gross - net,
        "costs_paid": float(paid.sum()),
        "turnover_total": float(turnover.sum()),
        "turnover_after_entry": float(turnover[1:].sum()),
        "avg_cash_weight": float(np.nanmean(curve["cash_weight"].to_numpy(dtype=float)[1:])) if "cash_weight" in curve else np.nan,
        "forced_exit_share": np.nan,
        "turnover_per_rebalance": np.nan,
    }
    if "forced_exit_turnover" in curve and turnover.sum() > 0:
        out["forced_exit_share"] = float(np.nan_to_num(curve["forced_exit_turnover"].to_numpy(dtype=float)[1:]).sum() / turnover.sum())
    if rebalance_every:
        later = turnover[rebalance_every::rebalance_every]  # rebalances after the entry on the first session
        out["turnover_per_rebalance"] = float(later.mean()) if len(later) else np.nan
    return out


def relative_diagnostics(curve: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """Daily-return closeness of a policy to the reference on the same sessions."""
    joined = curve[["date", "portfolio_value"]].merge(reference[["date", "portfolio_value"]], on="date", suffixes=("", "_ref"))
    mine = joined["portfolio_value"].pct_change().dropna().to_numpy()
    ref = joined["portfolio_value_ref"].pct_change().dropna().to_numpy()
    if len(mine) < 3:
        return {"correlation_vs_reference": np.nan, "tracking_error_vs_reference": np.nan}
    return {
        "correlation_vs_reference": float(np.corrcoef(mine, ref)[0, 1]),
        "tracking_error_vs_reference": float(np.std(mine - ref, ddof=1) * np.sqrt(TRADING_DAYS)),
    }


def holdings_diagnostics(holdings: pd.DataFrame, curve: pd.DataFrame, rebalance_every: int) -> dict[str, float]:
    """Weight spread and distance from equal weight, averaged over the rebalance sessions.

    Row ``k >= 1`` of a curve is the session after decision ``k - 1``; decisions ``0, r, 2r, ...``
    rebalance. The holdings recorded on that row are the targets after one session of drift.
    Distance from equal weight is the total-variation distance between the agent's weights (cash
    included) and 1/n on each of the n tradable members.
    """
    rows = curve.iloc[1::rebalance_every]
    by_date = {date: group["weight"].to_numpy(dtype=float) for date, group in holdings.groupby("date")}
    stats = []
    for _, row in rows.iterrows():
        weights = by_date.get(row["date"])
        if weights is None or weights.sum() <= 0:
            continue
        n = max(int(row.get("tradable_count", len(weights))), len(weights))
        invested = weights / weights.sum()
        distance = 0.5 * (np.abs(weights - 1.0 / n).sum() + (n - len(weights)) / n + float(row["cash_weight"]))
        stats.append({
            "effective_holdings": float(1.0 / np.square(invested).sum()),
            "tradable_members": float(n),
            "max_weight": float(weights.max()),
            "top10_weight": float(np.sort(weights)[::-1][:10].sum()),
            "distance_from_equal_weight": float(distance),
        })
    if not stats:
        return {}
    return pd.DataFrame(stats).mean().to_dict()


def slot_change_diagnostics(holdings: pd.DataFrame, curve: pd.DataFrame, slots_by_date: dict[str, dict[str, int]],
                            rebalance_every: int) -> dict[str, float]:
    """Split rebalance weight changes between members that changed slot since the previous decision and the rest.

    The point-in-time policy scores slots, not stocks, so a member that moves slot (because the order by
    trailing traded value changed) can get a new weight even if its own features did not change. A weight
    change is the holdings on a rebalance row minus those on the row before, both after one session of
    drift. Slots are those of the decision sessions, which are the rows before. Entries and exits count as
    slot changes.
    """
    dates = list(curve["date"])
    weights = {date: dict(zip(group["ticker"], group["weight"])) for date, group in holdings.groupby("date")}
    moved = {"turnover": 0.0, "names": 0}
    kept = {"turnover": 0.0, "names": 0}
    for k in range(1 + rebalance_every, len(dates), rebalance_every):
        now, before = slots_by_date.get(dates[k - 1]), slots_by_date.get(dates[k - 1 - rebalance_every])
        if now is None or before is None:
            continue
        new, old = weights.get(dates[k], {}), weights.get(dates[k - 1], {})
        for ticker in set(new) | set(old):
            bucket = kept if ticker in now and before.get(ticker) == now[ticker] else moved
            bucket["turnover"] += abs(new.get(ticker, 0.0) - old.get(ticker, 0.0))
            bucket["names"] += 1
    total = moved["turnover"] + kept["turnover"]
    names = moved["names"] + kept["names"]
    if not names or total <= 0:
        return {}
    return {
        "slot_moved_name_share": moved["names"] / names,
        "slot_moved_turnover_share": moved["turnover"] / total,
        "mean_weight_change_slot_moved": moved["turnover"] / moved["names"] if moved["names"] else np.nan,
        "mean_weight_change_slot_kept": kept["turnover"] / kept["names"] if kept["names"] else np.nan,
    }


def pit_slots_by_date(config_path: Path, root: Path, years: list[int],
                      variant: str | None = None) -> dict[int, dict[str, dict[str, int]]]:
    """Slot assignment of every validation session, rebuilt from the point-in-time config (no model needed).

    ``variant`` applies that variant's ``universe`` overrides (for example ``slot_order``)."""
    import yaml

    from scripts.run_pit_walk_forward import _deep_merge, load_inputs, make_env, window

    spec = yaml.safe_load(config_path.read_text())
    base = _deep_merge({"features": spec["features"], "universe": spec["universe"]}, spec["base"])
    for entry in spec.get("variants", []):
        if entry.get("name") == variant and "universe" in entry:
            base = _deep_merge(base, {"universe": entry["universe"]})
    mask, panel, _ = load_inputs(spec, root)
    limit = pd.Timestamp(spec["splits"]["validation_end"])
    lookback = int(spec["features"]["lookback"])
    out = {}
    for year in years:
        end = min(pd.Timestamp(f"{year}-12-31"), limit).date().isoformat()
        env = make_env(base, window(panel, f"{year}-01-01", end, lookback), mask, training=False)
        out[year] = {pd.Timestamp(date).date().isoformat(): {env.tickers[u]: j for j, u in enumerate(row) if u >= 0}
                     for date, row in zip(env.dates, env.slots)}
    return out


def load_holdings(fold_dir: Path, policy: str) -> pd.DataFrame | None:
    paths = sorted(fold_dir.glob(f"*/runs/{policy}/validation_holdings.csv"))
    return pd.read_csv(paths[0]) if paths else None


def diagnose(run_dir: Path, variant: str, reference: str, rebalance_every: int,
             reference_rebalance_every: int | None = 21,
             slots: dict[int, dict[str, dict[str, int]]] | None = None) -> pd.DataFrame:
    rows = []
    for fold_dir in sorted(p for p in run_dir.glob("fold_*") if p.is_dir()):
        year = int(fold_dir.name.split("_")[1])
        curves = load_fold(fold_dir)
        # Every seed of the variant, or the variant itself when it is one unseeded policy (a seed ensemble).
        policies = sorted(p for p in curves["policy"].unique() if p.startswith(f"{variant}_seed") or p == variant)
        if not policies:
            continue
        ref_curve = curves[curves["policy"] == reference].reset_index(drop=True)
        for policy in [*policies, reference]:
            curve = curves[curves["policy"] == policy].reset_index(drop=True)
            if curve.empty:
                continue
            every = rebalance_every if policy != reference else reference_rebalance_every
            row = {"fold": year, "policy": policy, "family": variant if policy != reference else reference,
                   **curve_diagnostics(curve, every)}
            if policy != reference and not ref_curve.empty:
                row.update(relative_diagnostics(curve, ref_curve))
            holdings = load_holdings(fold_dir, policy)
            if holdings is not None:
                row.update(holdings_diagnostics(holdings, curve, rebalance_every))
                if slots and year in slots:
                    row.update(slot_change_diagnostics(holdings, curve, slots[year], rebalance_every))
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no {variant}_seed* or {variant} curves under {run_dir}/fold_*")
    frame = pd.DataFrame(rows)
    ref = frame[frame["family"] == reference].set_index("fold")
    for column in ("net_return", "gross_return", "cost_drag"):
        frame[f"{column}_minus_reference"] = frame[column] - frame["fold"].map(ref[column])
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.drop(columns=["policy"]).groupby(["family", "fold"]).mean(numeric_only=True)
    summary = numeric.groupby("family").mean()
    summary.insert(0, "folds", numeric.groupby("family").size())
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant", required=True, help="seeded variant whose <variant>_seed<S> policies are diagnosed")
    parser.add_argument("--reference", default="pit_equal_weight_monthly")
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--reference-rebalance-every", type=int, default=21)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="agent_diagnostics")
    parser.add_argument("--pit-config", default=None,
                        help="point-in-time config; adds the split of turnover by slot changes (rebuilds the slots)")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.output_dir or run_dir)
    out.mkdir(parents=True, exist_ok=True)
    slots = None
    if args.pit_config:
        years = sorted(int(p.name.split("_")[1]) for p in run_dir.glob("fold_*") if p.is_dir())
        slots = pit_slots_by_date(Path(args.pit_config), Path(args.root), years, variant=args.variant)
    frame = diagnose(run_dir, args.variant, args.reference, args.rebalance_every, args.reference_rebalance_every, slots)
    summary = summarize(frame)
    frame.to_csv(out / f"{args.prefix}_runs.csv", index=False)
    summary.to_csv(out / f"{args.prefix}_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(4).T.to_string())


if __name__ == "__main__":
    main()
