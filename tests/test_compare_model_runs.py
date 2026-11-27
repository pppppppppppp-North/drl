"""Fold-by-fold comparison of the same policy in two run folders."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.compare_model_runs import compare_runs

V = "arm"


def _write(run, year, seed, values, weights):
    d = run / f"fold_{year}" / f"{V}_seed{seed}" / "runs" / f"{V}_seed{seed}"
    d.mkdir(parents=True)
    dates = pd.bdate_range("2016-01-04", periods=len(values))
    pd.DataFrame({"date": dates, "policy": f"{V}_seed{seed}", "portfolio_value": values,
                  "turnover": [0.0] + [0.1] * (len(values) - 1)}).to_csv(d / "validation_equity_curve.csv", index=False)
    rows = [{"date": dates[i], "policy": f"{V}_seed{seed}", "ticker": t, "weight": w}
            for i, row in enumerate(weights) for t, w in row.items()]
    pd.DataFrame(rows).to_csv(d / "validation_holdings.csv", index=False)


def test_identical_runs_give_zero_gaps_and_different_runs_do_not(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    weights = [{"A": 0.5, "B": 0.5}, {"A": 0.6, "B": 0.4}, {"A": 0.6, "B": 0.4}]
    _write(left, 2016, 42, [100.0, 101.0, 103.0], weights)
    _write(right, 2016, 42, [100.0, 101.0, 103.0], weights)
    _write(left, 2017, 42, [100.0, 99.0, 98.0], weights)
    _write(right, 2017, 42, [100.0, 99.0, 98.98], [{"A": 0.5, "B": 0.5}, {"A": 0.6, "C": 0.4}, {"A": 0.6, "B": 0.4}])
    _write(left, 2018, 42, [100.0, 100.0, 100.0], weights)  # only in one folder: skipped
    table = compare_runs(left, right, V, [42]).set_index("fold")
    assert list(table.index) == [2016, 2017]
    assert table.loc[2016, ["final_gap", "max_relative_value_gap", "max_weight_gap"]].abs().max() == 0.0
    assert table.loc[2017, "final_gap"] == pytest.approx(0.0098)
    assert table.loc[2017, "max_relative_value_gap"] == pytest.approx(0.01)
    assert table.loc[2017, "max_weight_gap"] == pytest.approx(0.4)
    assert table.loc[2016, "sessions"] == 3 and table.loc[2016, "turnover_left"] == pytest.approx(0.2)


def test_mismatched_sessions_and_empty_overlap_are_refused(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    _write(left, 2016, 42, [100.0, 101.0], [{"A": 1.0}, {"A": 1.0}])
    _write(right, 2016, 42, [100.0, 101.0, 102.0], [{"A": 1.0}] * 3)
    with pytest.raises(ValueError, match="different sessions"):
        compare_runs(left, right, V, [42])
    with pytest.raises(ValueError, match="no fold and seed"):
        compare_runs(left, right, V, [43])
