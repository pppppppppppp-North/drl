# Repeated-Seed Walk-Forward Robustness Experiment

## Purpose

This experiment tests whether PPO and A2C results are stable across training randomness and chronological market windows. It uses the existing five-stock Thai pilot, SET market context, transaction costs, and fixed model settings. It does not depend on the missing historical sentiment, official macro, sector, licensed-fundamental, or broader-universe files.

The final research conclusion must use only the chronological `test` rows. Validation rows are retained for diagnostics and must not be mixed into the final performance table.

## Fixed Design

- Candidate algorithms: PPO and A2C.
- Algorithm settings are fixed before test evaluation in `robustness.algorithm_parameters`; PPO uses the previously selected validation configuration and A2C uses an explicit frozen baseline configuration.
- Training seeds: 42, 43, 44, 45, and 46.
- Walk-forward windows: 0 and 1.
- Training budget: 20,000 timesteps for every algorithm/seed/window combination.
- Training environments: eight distinct chronological slices per run using subprocess vectorization.
- Primary baselines: equal-weight portfolio and SET index buy-and-hold.
- Secondary descriptive baselines: individual-stock buy-and-hold, momentum, moving-average crossover, mean-variance, and random allocation.
- Costs: the configured 0.1% transaction cost. The SET index benchmark pays the same initial cash-to-asset turnover cost as stock buy-and-hold.

This produces 20 DRL runs: 2 algorithms x 5 seeds x 2 windows. Baselines are evaluated on the identical test dates.

## BistKA Execution

Run the source-readiness checks and confirm that the existing real OHLCV plus SET market feature table is present. The robustness experiment needs `data/processed/features_real_ohlcv_market.parquet`; it does not need the five blocked provider files.

Submit the four-task array. Each array task owns one algorithm/window combination and runs its five seeds sequentially. At most two tasks run at once.

```bash
cd /lustrefs/project/25sfcs03/drl_thai_stock

ARRAY_JOB_ID=$(sbatch --parsable slurm/robustness_array.sbatch)
RUN_DIR="results/robustness_real_ohlcv_market_${ARRAY_JOB_ID}"

sbatch \
  --dependency="afterok:${ARRAY_JOB_ID}" \
  --export="ALL,RUN_DIR=${RUN_DIR}" \
  slurm/robustness_aggregate.sbatch
```

For a short wiring check, lower the training budget without changing the committed research configuration:

```bash
TOTAL_TIMESTEPS=1000 RUN_PREFIX=robustness_smoke sbatch slurm/robustness_array.sbatch
```

Do not use smoke-run metrics in the final report.

## Task Mapping

The default array indices are:

| task | algorithm | window |
| ---: | --- | ---: |
| 0 | PPO | 0 |
| 1 | PPO | 1 |
| 2 | A2C | 0 |
| 3 | A2C | 1 |

The mapping is generated from `robustness.algorithms` and `robustness.window_ids` in `config/robustness_real_ohlcv_market.yaml`.

## Output Artifacts

The aggregation job writes these files under `<run_dir>/aggregate/`:

- `robustness_all_metrics.csv`: validation and test metrics for auditability.
- `robustness_test_metrics.csv`: test-only metrics used for conclusions.
- `robustness_policy_summary.csv`: mean, median, sample standard deviation, and 95% bootstrap confidence interval by policy.
- `robustness_paired_excess.csv`: run-matched differences between each DRL policy and each primary baseline.
- `robustness_paired_summary.csv`: mean paired differences, confidence intervals, and positive-run rates.
- `robustness_completeness.csv`: expected algorithm/seed/window grid with a hard completeness gate.
- `experiment_manifest.csv`: collected run status, provenance, runtime, and SLURM identifiers.
- `robustness_confidence_intervals.png`: test-only mean and confidence-interval chart.
- `robustness_sharpe_heatmap.png`: candidate-policy Sharpe by seed and window.
- `robustness_summary.md`: report-ready compact summary.

## Interpretation Rules

1. Lock the configuration before examining final test results.
2. Do not select the best stock after viewing the test period and call it the primary baseline. Equal weight and the SET index are predeclared primary baselines.
3. Treat a positive paired difference as candidate-policy improvement. For maximum drawdown, a positive difference means the candidate drawdown is less negative and therefore better.
4. Deterministic baselines are summarized once per temporal window, not once per repeated DRL seed. The random-allocation baseline remains seed-specific.
5. Do not claim reliable outperformance when the 95% interval for excess cumulative return crosses zero.
6. Report failure honestly. Consistent underperformance is a valid negative result; consistently smaller drawdown supports a narrower risk-control contribution.
7. The intervals quantify the observed seed/window experiment. With only two temporal windows, they do not establish whole-market generalization.

## Recommended Research Question

Are the return and drawdown differences between PPO/A2C and predeclared simple baselines robust across training seeds and chronological out-of-sample periods, or are they artifacts of stochastic training and one particular split?
