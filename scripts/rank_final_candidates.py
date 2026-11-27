"""Rank walk-forward designs by the decision-7 default score, before the one test evaluation (M7/M8).

Decision 7 of ``docs/v3_author_decision_packet_20260904.md`` recommends this ranking rule; the
authors have not approved it yet (``docs/evaluation_metric_formulas_20261110.md``):

1. **Score of one model in one fold:** the annualized excess net log return over active equal
   weight on the same sessions, ``252 / N * (ln(V_N / V_0) - ln(E_N / E_0))``, where ``N`` is the
   fold's number of validation transitions (sessions minus one) and ``E`` the equal-weight curve.
2. **Score of a design:** the mean over its seeds within each fold, then the **median** of the fold
   values (the primary score). A seed ensemble is one curve per fold.
3. **Tie order:** worst-fold excess, then excess drawdown (the design's maximum drawdown minus equal
   weight's, mean over folds; larger is better), turnover, the smaller timestep budget, then the
   candidate id.
4. **Fallback:** if no design in a universe scores above zero, the rule selects active equal weight.

Candidates with ``kind: baseline`` (the forecast and heuristic baselines) are scored and ranked for
comparison but never selected; every other candidate is a DRL design (``kind: drl``, the default).

The same score against the SET50 ETF is reported beside it, because the reports so far have used the
ETF as the main reference (the reference is part of decision 7). Only validation curves are read
(``<run>/fold_<Y>/*/validation_equity_curves.csv``), so the test years and the holdout are not.

The candidate file lists, per universe, the reference policies and the designs::

    universes:
      point_in_time: {reference: pit_equal_weight_monthly, benchmark: set50_etf_buy_hold}
    candidates:
      - id: pit_shared_100k
        universe: point_in_time
        label: Shared-weight actor, 100k
        timesteps: 100000
        policy: pit_plan_weekly_lstm_shared      # seeded: <policy>_seed<S> for each seed
        seeds: [42, 43, 44, 45, 46]              # omit for one unseeded curve (an ensemble)
        runs: [pit_walk_forward_shared_20261110, pit_walk_forward_shared_100k_s4546_20261125]

Usage::

    python -m scripts.rank_final_candidates --candidates config/final_candidates_20261125.yaml \\
        --base ~/drl_runs --output-dir ~/drl_runs/final_candidate_ranking_20261125
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.aggregate_walk_forward_reward import load_fold

TRADING_DAYS = 252


def _values(curves: pd.DataFrame, policy: str) -> pd.Series:
    rows = curves[curves["policy"] == policy].sort_values("date")
    return pd.Series(rows["portfolio_value"].to_numpy(float), index=pd.to_datetime(rows["date"]))


def max_drawdown(values: pd.Series) -> float:
    return float((values / values.cummax() - 1.0).min())


def excess_log_return(values: pd.Series, reference: pd.Series) -> float:
    """``252 / N * (ln(V_N / V_0) - ln(E_N / E_0))`` on the sessions both curves share."""

    dates = values.index.intersection(reference.index)
    if len(dates) < 2:
        raise ValueError("the curves share fewer than two sessions")
    v, e = values.loc[dates], reference.loc[dates]
    n = len(dates) - 1
    return TRADING_DAYS / n * (np.log(v.iloc[-1] / v.iloc[0]) - np.log(e.iloc[-1] / e.iloc[0]))


def load_candidate_folds(candidate: dict, base: Path) -> dict[int, pd.DataFrame]:
    """Each fold's curves, merged over the candidate's run folders (first copy of a policy wins)."""

    folds: dict[int, list[pd.DataFrame]] = {}
    for run in candidate["runs"]:
        run_dir = (base / run).expanduser()
        if not run_dir.is_dir():
            raise SystemExit(f"{candidate['id']}: run folder {run_dir} not found")
        for fold_dir in sorted(run_dir.glob("fold_*")):
            if fold_dir.is_dir():
                folds.setdefault(int(fold_dir.name.split("_")[1]), []).append(load_fold(fold_dir))
    return {
        year: pd.concat(parts, ignore_index=True).drop_duplicates(["date", "policy"], keep="first")
        for year, parts in sorted(folds.items())
    }


def candidate_rows(candidate: dict, universe: dict, base: Path, years: list[int] | None) -> pd.DataFrame:
    folds = load_candidate_folds(candidate, base)
    if years is not None:
        missing = sorted(set(years) - set(folds))
        if missing:
            raise SystemExit(f"{candidate['id']}: folds {missing} missing")
        folds = {y: folds[y] for y in years}
    seeds = candidate.get("seeds")
    policies = [f"{candidate['policy']}_seed{s}" for s in seeds] if seeds else [candidate["policy"]]
    rows = []
    for year, curves in folds.items():
        reference = _values(curves, universe["reference"])
        benchmark = _values(curves, universe["benchmark"]) if universe.get("benchmark") else None
        if reference.empty:
            raise SystemExit(f"{candidate['id']} fold {year}: reference {universe['reference']} missing")
        for policy in policies:
            values = _values(curves, policy)
            if values.empty:
                raise SystemExit(f"{candidate['id']} fold {year}: {policy} missing")
            turnover = curves.loc[curves["policy"] == policy].sort_values("date")["turnover"].to_numpy(float)
            rows.append({
                "candidate": candidate["id"], "fold": year, "policy": policy,
                "excess_vs_reference": excess_log_return(values, reference),
                "excess_vs_benchmark": excess_log_return(values, benchmark) if benchmark is not None else np.nan,
                "annual_return": float(values.iloc[-1] / values.iloc[0] - 1.0),
                "max_drawdown": max_drawdown(values),
                "excess_drawdown": max_drawdown(values) - max_drawdown(reference),
                "turnover_per_transition": float(turnover[1:].mean()) if len(turnover) > 1 else 0.0,
            })
    return pd.DataFrame(rows)


def rank(candidates: list[dict], per_model: pd.DataFrame) -> pd.DataFrame:
    """Seed mean per fold, then the median over folds; sorted in the decision-7 tie order."""

    per_fold = per_model.groupby(["candidate", "fold"], as_index=False).mean(numeric_only=True)
    meta = pd.DataFrame(candidates).set_index("id")
    table = per_fold.groupby("candidate").agg(
        primary_score=("excess_vs_reference", "median"),
        mean_excess=("excess_vs_reference", "mean"),
        worst_fold_excess=("excess_vs_reference", "min"),
        folds_ahead=("excess_vs_reference", lambda s: int((s > 0).sum())),
        folds=("fold", "count"),
        median_excess_vs_benchmark=("excess_vs_benchmark", "median"),
        mean_annual_return=("annual_return", "mean"),
        excess_drawdown=("excess_drawdown", "mean"),
        turnover_per_transition=("turnover_per_transition", "mean"),
    )
    table["universe"] = meta.loc[table.index, "universe"]
    table["kind"] = meta["kind"].fillna("drl").loc[table.index] if "kind" in meta else "drl"
    table["label"] = meta.loc[table.index, "label"]
    table["timesteps"] = meta.loc[table.index, "timesteps"].astype(int)
    table["seeds"] = [len(meta.loc[c, "seeds"]) if isinstance(meta.loc[c, "seeds"], list) else 0 for c in table.index]
    table = table.reset_index()
    table = table.sort_values(
        ["universe", "primary_score", "worst_fold_excess", "excess_drawdown", "turnover_per_transition",
         "timesteps", "candidate"],
        ascending=[True, False, False, False, True, True, True],
    )
    table["rank"] = table.groupby("universe").cumcount() + 1
    return table.reset_index(drop=True)


def select(table: pd.DataFrame, universes: dict) -> dict[str, dict]:
    """The top DRL design per universe, or active equal weight if no DRL design scores above zero."""

    out = {}
    for name, group in table[table["kind"] == "drl"].groupby("universe"):
        top = group.iloc[0]
        if top["primary_score"] > 0:
            out[name] = {"selected": top["candidate"], "primary_score": float(top["primary_score"]),
                         "rule": "highest primary score"}
        else:
            out[name] = {"selected": universes[name]["reference"], "primary_score": 0.0,
                         "rule": "no design scored above zero, so active equal weight is selected",
                         "best_design": top["candidate"], "best_design_score": float(top["primary_score"])}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--base", default=".", help="folder the candidates' run folders are relative to")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--years", nargs="*", type=int, default=[2016, 2017, 2018, 2019, 2020, 2021, 2022])
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.candidates).read_text())
    if any(y > 2022 for y in args.years):
        raise SystemExit("the ranking uses validation years only (2022 or earlier)")
    base = Path(args.base).expanduser()
    universes, candidates = spec["universes"], spec["candidates"]
    ids = [c["id"] for c in candidates]
    if len(set(ids)) != len(ids):
        raise SystemExit("candidate ids must be unique")
    per_model = pd.concat([candidate_rows(c, universes[c["universe"]], base, args.years) for c in candidates],
                          ignore_index=True)
    table = rank(candidates, per_model)
    selection = select(table, universes)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_model.to_csv(out / "candidate_fold_scores.csv", index=False)
    table.to_csv(out / "candidate_ranking.csv", index=False)
    (out / "selection.json").write_text(json.dumps(
        {"candidates_file": str(args.candidates), "years": args.years, "selection": selection}, indent=2) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        cols = ["universe", "rank", "candidate", "kind", "primary_score", "mean_excess", "worst_fold_excess", "folds_ahead",
                "median_excess_vs_benchmark", "mean_annual_return", "excess_drawdown", "turnover_per_transition"]
        print(table[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
