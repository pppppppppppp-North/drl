"""The decision-7 ranking: the score formula, the median over folds, the tie order and the fallback."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import rank_final_candidates as rank


def _curve(dates, rate):
    return [1e6 * (1 + rate) ** i for i in range(len(dates))]


def _write_fold(run, year, policies):
    """``policies`` maps a policy name to a daily growth rate (or to a list of values)."""

    dates = pd.bdate_range(f"{year}-01-03", periods=11).strftime("%Y-%m-%d")
    rows = []
    for policy, rate in policies.items():
        values = rate if isinstance(rate, list) else _curve(dates, rate)
        rows += [{"date": d, "policy": policy, "portfolio_value": v, "turnover": 0.1 if i else 1.0}
                 for i, (d, v) in enumerate(zip(dates, values))]
    part = run / f"fold_{year}" / "part"
    part.mkdir(parents=True, exist_ok=True)
    path = part / "validation_equity_curves.csv"
    frame = pd.DataFrame(rows)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame])
    frame.to_csv(path, index=False)


def test_excess_log_return_formula():
    dates = pd.bdate_range("2020-01-01", periods=11)
    values = pd.Series(_curve(dates, 0.002), index=dates)
    reference = pd.Series(_curve(dates, 0.001), index=dates)
    expected = 252 / 10 * (10 * np.log(1.002) - 10 * np.log(1.001))
    assert rank.excess_log_return(values, reference) == pytest.approx(expected)


def test_max_drawdown():
    values = pd.Series([100.0, 120.0, 90.0, 130.0, 117.0])
    assert rank.max_drawdown(values) == pytest.approx(90 / 120 - 1)


def _spec(tmp_path, candidates):
    path = tmp_path / "candidates.yaml"
    path.write_text(yaml.safe_dump({
        "universes": {"u": {"reference": "ew", "benchmark": "etf"}},
        "candidates": candidates,
    }))
    return path


def _run(tmp_path, monkeypatch, candidates, years=("2016", "2017", "2018")):
    out = tmp_path / "out"
    argv = ["x", "--candidates", str(_spec(tmp_path, candidates)), "--base", str(tmp_path),
            "--output-dir", str(out), "--years", *years]
    monkeypatch.setattr(sys, "argv", argv)
    rank.main()
    return pd.read_csv(out / "candidate_ranking.csv"), json.loads((out / "selection.json").read_text())


def test_median_over_folds_seed_mean_and_selection(tmp_path, monkeypatch):
    # Design a beats equal weight in two of three folds by a little; design b in one fold by a lot.
    for year, (a42, a43, b) in zip([2016, 2017, 2018], [(0.0012, 0.0014, 0.0050), (0.0011, 0.0011, 0.0), (0.0, 0.0, 0.0)]):
        _write_fold(tmp_path / "run", year, {"ew": 0.001, "etf": 0.0005, "a_seed42": a42, "a_seed43": a43, "b_ens": b})
    table, selection = _run(tmp_path, monkeypatch, [
        {"id": "a", "universe": "u", "label": "A", "timesteps": 100, "policy": "a", "seeds": [42, 43], "runs": ["run"]},
        {"id": "b", "universe": "u", "label": "B", "timesteps": 100, "policy": "b_ens", "runs": ["run"]},
    ])
    scores = dict(zip(table["candidate"], table["primary_score"]))
    # a: each fold's value is the mean of its seeds' excess; the median fold is 2017.
    assert scores["a"] == pytest.approx(252 * (np.log(1.0011) - np.log(1.001)))
    assert scores["b"] == pytest.approx(252 * (np.log(1.0) - np.log(1.001)))  # median fold is a zero-growth fold
    assert table["candidate"].tolist() == ["a", "b"] and table["rank"].tolist() == [1, 2]
    assert selection["selection"]["u"]["selected"] == "a"


def test_fallback_to_equal_weight_and_baselines_never_selected(tmp_path, monkeypatch):
    for year in [2016, 2017, 2018]:
        _write_fold(tmp_path / "run", year, {"ew": 0.001, "etf": 0.0005, "drl_seed42": 0.0009, "fc": 0.003})
    table, selection = _run(tmp_path, monkeypatch, [
        {"id": "drl", "universe": "u", "label": "DRL", "timesteps": 100, "policy": "drl", "seeds": [42], "runs": ["run"]},
        {"id": "fc", "universe": "u", "kind": "baseline", "label": "Forecast", "timesteps": 0, "policy": "fc",
         "runs": ["run"]},
    ])
    assert table.iloc[0]["candidate"] == "fc"  # ranked first, but a baseline
    chosen = selection["selection"]["u"]
    assert chosen["selected"] == "ew" and chosen["best_design"] == "drl" and chosen["best_design_score"] < 0


def test_tie_order_prefers_smaller_budget(tmp_path, monkeypatch):
    for year in [2016, 2017, 2018]:
        _write_fold(tmp_path / "run", year, {"ew": 0.001, "etf": 0.0005, "x_seed42": 0.002})
    table, selection = _run(tmp_path, monkeypatch, [
        {"id": "big", "universe": "u", "label": "300k", "timesteps": 300, "policy": "x", "seeds": [42], "runs": ["run"]},
        {"id": "small", "universe": "u", "label": "100k", "timesteps": 100, "policy": "x", "seeds": [42], "runs": ["run"]},
    ])
    assert table["candidate"].tolist() == ["small", "big"]
    assert selection["selection"]["u"]["selected"] == "small"


def test_seeds_merge_across_runs_and_missing_seed_is_an_error(tmp_path, monkeypatch):
    for year in [2016, 2017, 2018]:
        _write_fold(tmp_path / "r1", year, {"ew": 0.001, "etf": 0.0005, "x_seed42": 0.002})
        _write_fold(tmp_path / "r2", year, {"ew": 0.001, "etf": 0.0005, "x_seed43": 0.003})
    table, _ = _run(tmp_path, monkeypatch, [
        {"id": "x", "universe": "u", "label": "X", "timesteps": 100, "policy": "x", "seeds": [42, 43],
         "runs": ["r1", "r2"]},
    ])
    assert table.loc[0, "primary_score"] == pytest.approx(252 * ((np.log(1.002) + np.log(1.003)) / 2 - np.log(1.001)))
    with pytest.raises(SystemExit, match="x_seed44 missing"):
        _run(tmp_path, monkeypatch, [
            {"id": "x", "universe": "u", "label": "X", "timesteps": 100, "policy": "x", "seeds": [42, 44],
             "runs": ["r1", "r2"]},
        ])


def test_refuses_test_years(tmp_path, monkeypatch):
    _write_fold(tmp_path / "run", 2016, {"ew": 0.001, "etf": 0.0005, "x_seed42": 0.002})
    with pytest.raises(SystemExit, match="validation years only"):
        _run(tmp_path, monkeypatch, [
            {"id": "x", "universe": "u", "label": "X", "timesteps": 100, "policy": "x", "seeds": [42], "runs": ["run"]},
        ], years=("2016", "2023"))
