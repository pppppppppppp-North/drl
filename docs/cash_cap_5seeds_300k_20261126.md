# The Locked Design With Five Seeds, and at 300k Timesteps (M6, validation years only)

Date: 2026-11-26
Evidence tier: **exploratory**. Yahoo data, validation years 2016–2022 only. The test years 2023–2024 were not read again, and the sealed 2025+ holdout was not read at all. Nothing here changes the locked design or the test results already recorded ([test report](final_test_results_20261125.md)).
Outputs (git-ignored): `results/pit_walk_forward_cash_cap_100k_s4546_20261125/` (training), `results/pit_cash_cap_ensemble_100k_5seeds_20261125/` and `results/pit_walk_forward_combined_cash_cap_5seeds_20261125/` (analysis).

## Why

The design locked for the test is the shared-weight point-in-time actor with cash capped at 20%, as a seed ensemble with the 0.5-point no-trade band ([decision](final_design_decision_20261125.md)). It was chosen with three seeds (42–44) at 100k timesteps. Two questions were left open:

1. **Does the choice hold with five seeds?** The uncapped design already has five seeds at 100k ([report](bistka_300k_results_20261125.md)).
2. **Does longer training help once the cap blocks the drift to cash?** Without the cap, 300k timesteps pushed 17 of 35 models to 45–98% cash and lowered the return. BistKA job `4518` trains the capped design at 300k with five seeds (section below).

## Five seeds at 100k

Seeds 45 and 46 were trained on the Mac: 14 models, 3 workers, from 22:10 on 2026-11-25 to 00:23 on 2026-11-26 (`slurm/pools/local_20261125b.tasks`). No errors. The folder `pit_walk_forward_cash_cap_100k_5seeds_20261125` links seeds 42–44 (2026-11-19) and 45–46, so the ensemble reads all five from one place. Every model replayed its saved validation curve to within 2.2 × 10⁻¹⁶.

| Design (100k, validation 2016–2022) | Seeds | Mean annual return | Average cash | Mean max drawdown | Sharpe | 1 baht became | Decision-7 score | Years ahead of equal weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Cash capped, ensemble + band (locked)** | **3** | **5.0%** | **7.6%** | **−12.0%** | **0.36** | **1.36** | **−0.01** | **3/7** |
| Cash capped, ensemble + band | 5 | 5.0% | 5.4% | −12.2% | 0.36 | 1.36 | −0.16 | 1/7 |
| Cash capped, single models | 5 | 4.6% | 5.1% | −12.5% | 0.33 | 1.33 | −0.96 | 1/7 |
| No cap, ensemble + band | 5 | 5.2% | 6.9% | −12.0% | 0.38 | 1.38 | −0.78 | 2/7 |
| No cap, single models | 5 | 4.5% | 7.2% | −12.2% | 0.34 | 1.29 | −1.45 | 1/7 |
| Point-in-time equal weight, monthly | | 5.6% | 0.7% | −12.7% | 0.38 | 1.41 | 0 | |
| SET50 ETF, buy and hold | | 7.1% | | −11.2% | 0.50 | 1.55 | | |

The decision-7 score is the median over years of the annualized log return above point-in-time equal weight, in points a year, after averaging the seeds (the rule in the [decision](final_design_decision_20261125.md)). The single models' chained value is the mean of their seed curves.

![Point-in-time SET50, cash cap with five seeds at 100k](figures/pit_cash_cap_100k_5seeds_chained_equity_20261126.png)

- **The locked choice holds up.** With five seeds, the capped ensemble is still the best-scoring DRL design at 100k (−0.16 points a year, against −0.78 without the cap). It still scores below zero, so the rule's fallback to equal weight is unchanged.
- **Two more seeds moved the score, not the result.** The ensemble's return (5.0%), Sharpe ratio (0.36) and chained value (1.36) are the same with three or five seeds. The two Sharpe ratios differ by −0.004 (−0.023 to +0.015). Its median moved from −0.01 to −0.16 because two years near zero, 2020 and 2022, turned slightly negative. Its mean barely moved (−0.56 to −0.54).
- **At 100k the cap barely binds.** Average cash was 5% with the cap and 7% without it. Only 3 of the 35 capped models held more than 15% on average, all seed 43 (2017, 2019, 2020), at about the 20% cap. Without the cap, 2 of 35 held more than 45%. Capped against uncapped ensemble: Sharpe −0.02 (−0.06 to +0.04), not significant.
- **Nothing beats the benchmarks significantly.**
  - Capped ensemble against equal weight: −0.02 (−0.05 to +0.02).
  - Capped ensemble against the ETF: −0.14 (−0.39 to +0.10).
  - Deflated Sharpe at 22 designs: 0.16 for the capped ensemble (five or three seeds), 0.17 for equal weight, 0.26 for the ETF.
- **The ensemble's gain over its own seeds is borderline with the cap.**
  - Capped ensemble against its five seeds: +0.03 (−0.001 to +0.06), not quite significant.
  - Uncapped ensemble against its seeds: +0.04 (+0.02 to +0.06), significant.
  - The cap already removes part of what the ensemble averaged away: the cash-heavy seeds.

**For the test results:** the test used the locked three-seed ensemble, and that does not change. The five-seed result says that choice was not an accident of three seeds. On validation, the five-seed ensemble earns the same and ranks the same.

## 300k timesteps (BistKA job 4518)

*Pending.* The job started at 22:09 on 2026-11-25 (35 tasks: seeds 42–46, folds 2016–2022, the same capped design and validation years). When it ends, its curves and its 35 model files (needed for the ensemble) are copied to the Mac, and the same table is built at 300k: return, cash, drawdown, Sharpe ratio, decision-7 score, and the 300k − 100k Sharpe difference with five seeds on each side.

## Run it

```bash
R=~/drl_runs
# Training (local, 3 workers; 14 tasks, about 2.5 hours)
PYTHON=~/.venvs/drl_thai_py311/bin/python bash scripts/run_task_pool.sh slurm/pools/local_20261125b.tasks 3
# One folder with all five seeds (links)
V=pit_plan_weekly_lstm_shared_cashcap; S=$R/pit_walk_forward_cash_cap_100k_5seeds_20261125
for y in 2016 2017 2018 2019 2020 2021 2022; do mkdir -p $S/fold_$y
  for s in 42 43 44; do ln -s ../../pit_walk_forward_cash_cap_20261119/fold_$y/${V}_seed$s $S/fold_$y/; done
  for s in 45 46; do ln -s ../../pit_walk_forward_cash_cap_100k_s4546_20261125/fold_$y/${V}_seed$s $S/fold_$y/; done
done
python -m scripts.evaluate_pit_seed_ensemble --config config/pit_walk_forward_experiment.yaml \
  --source-run $S --variant $V --seeds 42 43 44 45 46 \
  --output-dir $R/pit_cash_cap_ensemble_100k_5seeds_20261125 --band 0.005 --modes ensemble_band
python -m scripts.aggregate_walk_forward_reward --run-dir $S \
  --extra-run-dirs $R/pit_cash_cap_ensemble_100k_5seeds_20261125 $R/pit_cash_cap_ensemble_20261119 \
  $R/pit_walk_forward_shared_100k_5seeds_20261125 $R/pit_seed_ensemble_100k_5seeds_20261125 \
  --rename $R/pit_cash_cap_ensemble_20261119:${V}_ensemble_band=${V}_ensemble_band_3seeds \
  --variants $V pit_plan_weekly_lstm_shared --seeds 42 43 44 45 46 --n-trials 22 \
  --output-dir $R/pit_walk_forward_combined_cash_cap_5seeds_20261125 \
  --references pit_equal_weight_monthly ${V}_ensemble_band pit_plan_weekly_lstm_shared_ensemble_band \
  --title "Point-in-time SET50, 100k, validation years 2016-2022" \
  --labels "$V=Cash capped at 20%, single models" "${V}_ensemble_band=Cash capped, ensemble + band (5 seeds)" \
  "${V}_ensemble_band_3seeds=Cash capped, ensemble + band (3 seeds, locked)" "pit_plan_weekly_lstm_shared=No cap, single models" \
  "pit_plan_weekly_lstm_shared_ensemble_band=No cap, ensemble + band (5 seeds)"
O=$R/pit_walk_forward_combined_cash_cap_5seeds_20261125
python -m scripts.walk_forward_agent_diagnostics --run-dir $R/pit_cash_cap_ensemble_100k_5seeds_20261125 \
  --variant ${V}_ensemble_band --prefix diagnostics_ensemble_band
python -m scripts.walk_forward_agent_diagnostics --run-dir $S --variant $V --output-dir $O --prefix diagnostics_single
python -m scripts.walk_forward_agent_diagnostics --run-dir $R/pit_seed_ensemble_100k_5seeds_20261125 \
  --variant pit_plan_weekly_lstm_shared_ensemble_band --output-dir $O --prefix diagnostics_uncapped_ensemble_band
python -m scripts.walk_forward_agent_diagnostics --run-dir $R/pit_walk_forward_shared_100k_5seeds_20261125 \
  --variant pit_plan_weekly_lstm_shared --output-dir $O --prefix diagnostics_uncapped_single
# Decision-7 scores (validation years only)
python -m scripts.rank_final_candidates --candidates config/cash_cap_5seeds_candidates_20261126.yaml --base $R \
  --output-dir $O/decision7
```
