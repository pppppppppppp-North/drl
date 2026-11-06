"""Statistical significance tools for comparing trading policies.

- Stationary (Politis-Romano) block bootstrap confidence intervals for the Sharpe ratio and for the
  paired Sharpe difference between two policies evaluated on the same dates.
- Probabilistic Sharpe Ratio (Bailey and Lopez de Prado, 2012).
- Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014), which corrects the best observed Sharpe
  for the number of configurations tried. The V3 grid plans 540 fits, so the best cell is expected
  to look good partly by luck.

Usage on a long-format equity-curve CSV (columns ``date, policy, portfolio_value``)::

    python -m src.evaluation.significance --curves results/run/equity_curves.csv \
        --reference equal_weight --n-trials 30 --output results/run/significance.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

EULER_MASCHERONI = 0.5772156649015329
_NORMAL = NormalDist()


def _as_returns(returns: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError("at least three finite returns are required")
    return values


def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized mean/standard-deviation Sharpe ratio with a zero risk-free rate."""

    values = _as_returns(returns)
    std = values.std(ddof=1)
    if std <= 0:
        return 0.0
    return float(values.mean() / std * math.sqrt(periods_per_year))


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One resample of ``n`` indices using geometric block lengths with the given mean."""

    if n <= 0:
        raise ValueError("n must be positive")
    if mean_block < 1:
        raise ValueError("mean_block must be at least 1")
    restart_probability = 1.0 / mean_block
    indices = np.empty(n, dtype=np.int64)
    indices[0] = rng.integers(n)
    restarts = rng.random(n) < restart_probability
    fresh = rng.integers(n, size=n)
    for i in range(1, n):
        indices[i] = fresh[i] if restarts[i] else (indices[i - 1] + 1) % n
    return indices


def bootstrap_sharpe_ci(
    returns: pd.Series | np.ndarray,
    n_boot: int = 2000,
    mean_block: float = 20.0,
    alpha: float = 0.05,
    periods_per_year: int = 252,
    seed: int = 42,
) -> dict[str, float]:
    values = _as_returns(returns)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = values[stationary_bootstrap_indices(values.size, mean_block, rng)]
        std = sample.std(ddof=1)
        draws[b] = sample.mean() / std * math.sqrt(periods_per_year) if std > 0 else 0.0
    return {
        "sharpe": sharpe_ratio(values, periods_per_year),
        "sharpe_ci_low": float(np.quantile(draws, alpha / 2)),
        "sharpe_ci_high": float(np.quantile(draws, 1 - alpha / 2)),
    }


def bootstrap_sharpe_difference(
    returns_a: pd.Series | np.ndarray,
    returns_b: pd.Series | np.ndarray,
    n_boot: int = 2000,
    mean_block: float = 20.0,
    alpha: float = 0.05,
    periods_per_year: int = 252,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap of Sharpe(a) - Sharpe(b); both series must cover the same dates."""

    a = np.asarray(returns_a, dtype=np.float64).reshape(-1)
    b = np.asarray(returns_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("paired returns must have the same length")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        raise ValueError("at least three paired finite returns are required")
    rng = np.random.default_rng(seed)
    scale = math.sqrt(periods_per_year)
    draws = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = stationary_bootstrap_indices(a.size, mean_block, rng)
        sa, sb = a[idx], b[idx]
        sd_a, sd_b = sa.std(ddof=1), sb.std(ddof=1)
        sharpe_a = sa.mean() / sd_a * scale if sd_a > 0 else 0.0
        sharpe_b = sb.mean() / sd_b * scale if sd_b > 0 else 0.0
        draws[i] = sharpe_a - sharpe_b
    observed = sharpe_ratio(a, periods_per_year) - sharpe_ratio(b, periods_per_year)
    return {
        "sharpe_difference": float(observed),
        "difference_ci_low": float(np.quantile(draws, alpha / 2)),
        "difference_ci_high": float(np.quantile(draws, 1 - alpha / 2)),
        "p_value_not_better": float((draws <= 0).mean()),
    }


def probabilistic_sharpe_ratio(returns: pd.Series | np.ndarray, benchmark_sharpe: float = 0.0) -> float:
    """Probability that the true per-period Sharpe exceeds ``benchmark_sharpe`` (per-period units)."""

    values = _as_returns(returns)
    n = values.size
    std = values.std(ddof=1)
    if std <= 0:
        return 0.0
    sr = values.mean() / std
    centered = values - values.mean()
    m2 = (centered**2).mean()
    skew = (centered**3).mean() / m2**1.5 if m2 > 0 else 0.0
    kurt = (centered**4).mean() / m2**2 if m2 > 0 else 3.0
    denominator = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denominator <= 0:
        return 0.0
    z = (sr - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(denominator)
    return float(_NORMAL.cdf(z))


def expected_max_sharpe(n_trials: int, trial_sharpe_variance: float) -> float:
    """Expected maximum of ``n_trials`` per-period Sharpe estimates under the null of no skill."""

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1 or trial_sharpe_variance <= 0:
        return 0.0
    first = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    second = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(trial_sharpe_variance) * ((1 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second))


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int,
    trial_sharpe_variance: float | None = None,
) -> float:
    """PSR evaluated against the expected maximum Sharpe of ``n_trials`` unskilled configurations.

    ``trial_sharpe_variance`` is the cross-trial variance of per-period Sharpe estimates. When it is
    unknown, the sampling variance of a single Sharpe estimate (about ``1 / (n - 1)``) is used.
    """

    values = _as_returns(returns)
    if trial_sharpe_variance is None:
        trial_sharpe_variance = 1.0 / (values.size - 1)
    threshold = expected_max_sharpe(n_trials, trial_sharpe_variance)
    return probabilistic_sharpe_ratio(values, benchmark_sharpe=threshold)


def policy_returns(curves: pd.DataFrame) -> pd.DataFrame:
    """Wide date x policy table of simple returns from a long equity-curve table."""

    required = {"date", "policy", "portfolio_value"}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"equity curves are missing columns: {sorted(missing)}")
    wide = curves.assign(date=pd.to_datetime(curves["date"])).pivot_table(
        index="date", columns="policy", values="portfolio_value", aggfunc="last"
    )
    return wide.sort_index().pct_change().iloc[1:]


def significance_table(
    curves: pd.DataFrame,
    reference: str | None = None,
    n_trials: int = 1,
    n_boot: int = 2000,
    mean_block: float = 20.0,
    seed: int = 42,
) -> pd.DataFrame:
    returns = policy_returns(curves)
    rows = []
    for policy in returns.columns:
        series = returns[policy].dropna()
        row = {"policy": policy, "observations": int(series.size)}
        row.update(bootstrap_sharpe_ci(series, n_boot=n_boot, mean_block=mean_block, seed=seed))
        row["probabilistic_sharpe_vs_zero"] = probabilistic_sharpe_ratio(series)
        row["deflated_sharpe"] = deflated_sharpe_ratio(series, n_trials=n_trials)
        if reference is not None and reference in returns.columns and policy != reference:
            paired = returns[[policy, reference]].dropna()
            row.update(
                bootstrap_sharpe_difference(
                    paired[policy], paired[reference], n_boot=n_boot, mean_block=mean_block, seed=seed
                )
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--curves", required=True, help="long CSV with date, policy, portfolio_value")
    parser.add_argument("--reference", default=None, help="policy used for paired Sharpe differences")
    parser.add_argument("--n-trials", type=int, default=1, help="configurations tried, for the deflated Sharpe")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--mean-block", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    curves = pd.read_csv(args.curves)
    table = significance_table(
        curves,
        reference=args.reference,
        n_trials=args.n_trials,
        n_boot=args.n_boot,
        mean_block=args.mean_block,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
