from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import benchmark_relative_metrics, equity_metrics
from src.evaluation.significance import (
    bootstrap_sharpe_ci,
    bootstrap_sharpe_difference,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    significance_table,
    stationary_bootstrap_indices,
)


def _curve(returns: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(returns) + 1)
    values = 1_000_000.0 * np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    return pd.DataFrame({"date": dates, "portfolio_value": values})


def test_equity_metrics_without_benchmark_keeps_legacy_keys() -> None:
    metrics = equity_metrics(_curve(np.full(50, 0.001)))
    assert "beta" not in metrics
    assert metrics["cumulative_return"] > 0


def test_benchmark_relative_metrics_identity_and_leverage() -> None:
    rng = np.random.default_rng(0)
    bench = rng.normal(0.0005, 0.01, 300)
    same = benchmark_relative_metrics(_curve(bench), _curve(bench))
    assert same["beta"] == pytest.approx(1.0)
    assert same["tracking_error"] == pytest.approx(0.0, abs=1e-12)
    assert same["information_ratio"] == 0.0
    levered = benchmark_relative_metrics(_curve(2 * bench), _curve(bench))
    assert levered["beta"] == pytest.approx(2.0, rel=1e-6)
    assert levered["tracking_error"] > 0
    with_bench = equity_metrics(_curve(2 * bench), benchmark_curve=_curve(bench))
    assert with_bench["beta"] == pytest.approx(2.0, rel=1e-6)


def test_benchmark_metrics_align_on_dates() -> None:
    bench = _curve(np.full(30, 0.001))
    portfolio = bench.iloc[5:].reset_index(drop=True)
    metrics = benchmark_relative_metrics(portfolio, bench)
    assert np.isfinite(metrics["beta"])
    with pytest.raises(ValueError):
        benchmark_relative_metrics(pd.Series([1.0, 1.1, 1.2]), pd.Series([1.0, 1.1]))


def test_stationary_bootstrap_indices_are_valid_and_seeded() -> None:
    a = stationary_bootstrap_indices(100, 10, np.random.default_rng(1))
    b = stationary_bootstrap_indices(100, 10, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < 100


def test_bootstrap_sharpe_intervals() -> None:
    rng = np.random.default_rng(2)
    good = rng.normal(0.002, 0.01, 500)
    ci = bootstrap_sharpe_ci(good, n_boot=300, seed=3)
    assert ci["sharpe_ci_low"] < ci["sharpe"] < ci["sharpe_ci_high"]
    assert ci["sharpe_ci_low"] > 0
    diff = bootstrap_sharpe_difference(good, rng.normal(0.0, 0.01, 500), n_boot=300, seed=3)
    assert diff["sharpe_difference"] > 0
    assert diff["p_value_not_better"] < 0.05


def test_probabilistic_and_deflated_sharpe() -> None:
    rng = np.random.default_rng(4)
    strong = rng.normal(0.002, 0.01, 750)
    noise = rng.normal(0.0, 0.01, 750)
    assert probabilistic_sharpe_ratio(strong) > 0.95
    assert 0.02 < probabilistic_sharpe_ratio(noise) < 0.98
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(540, 0.01) > expected_max_sharpe(30, 0.01) > 0
    assert deflated_sharpe_ratio(strong, n_trials=540) <= probabilistic_sharpe_ratio(strong)


def test_significance_table_from_long_curves() -> None:
    rng = np.random.default_rng(5)
    frames = []
    for name, mu in (("ppo", 0.001), ("equal_weight", 0.0003)):
        curve = _curve(rng.normal(mu, 0.01, 200))
        frames.append(curve.assign(policy=name))
    table = significance_table(pd.concat(frames), reference="equal_weight", n_trials=5, n_boot=200)
    assert set(table["policy"]) == {"ppo", "equal_weight"}
    ppo = table.set_index("policy").loc["ppo"]
    assert "sharpe_difference" in ppo and np.isfinite(ppo["deflated_sharpe"])
