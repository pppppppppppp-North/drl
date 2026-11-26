# Reproducing the Results (M8)

Date: 2026-11-11; updated 2026-11-26 with the HPC runs, the design choice, the test years and the sealed holdout (sections 9–12)
Scope: every result reported since 2026-11-06 in:
- the five-stock walk-forward;
- the source ablation;
- the algorithm comparison;
- the point-in-time SET50 study ([report](pit_walk_forward_20261107.md)) and its benchmarks;
- the longer training on BistKA and the five-seed runs ([300k](bistka_300k_results_20261125.md), [cash cap](cash_cap_5seeds_300k_20261126.md));
- the choice of the final designs ([decision](final_design_decision_20261125.md));
- the single evaluations on the test years ([results](final_test_results_20261125.md)) and on the sealed holdout ([results](holdout_results_20261126.md)).

It gives one command block per stage, in order, from the repository root. Each report's own "Run it" or "Reproduce" section has the same commands with more options. Run folders under `results/` are ignored by git; the reports cite them by their dated names, for example `results/pit_walk_forward_shared_20261110/`.

**Data limits that apply to every stage:**
- **Validation runs** (sections 2–9) never read past `splits.validation_end` (2022-12-30).
- **The test years** (2023–2024) were read once per universe, on 2026-11-25, only through the run-once gate of section 11.
- **The sealed holdout** (2025 to 4 November 2026): files under `data/holdout_sealed/` are written by the download step and opened only by the holdout stage of section 12, under an approved lock. It ran once per universe on 2026-11-26.
- **The ledger** `results/FINAL_TEST_LEDGER.jsonl` records every start of those stages. A project that already has a stage in it refuses to run that stage again; a fresh clone without `results/` can repeat it as a reproduction.

## 0. Environment

The reported runs used Python 3.11.13 on macOS (8-core Apple silicon, 8 GB of memory) with these package versions:

| Package | Version |
| --- | --- |
| torch | 2.15.0 |
| stable-baselines3 | 2.9.2 |
| gymnasium | 1.4.0 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scipy | 1.17.1 |
| PyYAML | 6.0.3 |
| matplotlib | 3.11.2 |
| yfinance | 1.8.0 |

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                   # torch comes with stable-baselines3
pip install -r requirements-data-extensions.txt   # xlrd, only to rebuild the 2012-2014 membership lists
```

**Determinism:**
- **Threads:** every training process runs with one thread (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`), so a seed gives the same model on the same machine and library versions.
- **Seeds:** 42, 43 and 44 for every three-seed run; 42–46 for the five-stock walk-forward.
- **Replay check:** the seed-ensemble script replays every saved model and stops if a curve differs.

## 1. Inputs

Git tracks the membership lists and the BOT policy-rate history. The price files come from Yahoo Finance and are not tracked; their receipts are, with each file's SHA-256:

| File | Made by | SHA-256 (first 12) |
| --- | --- | --- |
| `data/reference/set50_set100_membership_2012_2024.csv` | `python -m src.data.ingest_set_constituent_history_extended` (official SET files) | in git |
| `data/raw/extended/prices_set100_union_2012_2024.csv.gz` | `scripts.download_extended_history`, 2026-11-06 | `a2445d075025` |
| `data/raw/extended/set100_2012_2024_additions/prices_set100_2012_2024_additions.csv.gz` | the same, 2026-11-07, for the members without a price file | `ec66238af043` |
| `data/raw/extended/set100_2012_2024_additions/index_set50_etf_2012_2024.csv` | the same (TDEX, the SET50 ETF) | `dd1ddbff433e` |
| `data/raw/extended/set100_2012_2024_additions/member_price_map_2012_2024.csv` | `python -m scripts.build_member_price_map` (rebuilds the map of 2026-11-07 byte for byte) | `0f19820c1192` |
| `data/raw/extended/macro_proxies_2012_2024.csv` | `scripts.download_extended_history`, 2026-11-06 | `bbac4b3e356f` |
| `data/external/bot_policy_rate_history.csv` | `python -m src.data.ingest_bot_policy_rate` (official BOT table) | in git |

**Yahoo revises and drops series**, so a new download may not match these checksums. To reproduce the reported numbers exactly, use the original files, which the receipts identify.

**The price map** records, for each list and member, which price series to use: the member's own symbol, its current symbol after a rename, or none. It was first made in a one-off step on 2026-11-07. `scripts/build_member_price_map.py` applies the rule in that map's receipt to the two downloads' coverage files and reproduces the map byte for byte. `--check` compares without writing.

```bash
python -m src.data.ingest_set_constituent_history_extended
python -m scripts.download_extended_history --tickers-csv data/reference/set50_or_set100_universe.csv \
  --dev-output data/raw/extended/prices.csv.gz --index-dev-output data/raw/extended/index.csv \
  --holdout-dir data/holdout_sealed/<name>
python -m scripts.build_member_price_map --check   # or without --check, to rebuild it
python -m src.data.ingest_bot_policy_rate --input data/external/original/bot_policy_rate_mpc_table_asof_20261028.xlsx \
  --output data/external/bot_policy_rate_history.csv --source-url <url> --retrieved-at <utc>
```

## 2. Five-stock walk-forward (RQ5)

```bash
PYTHON=python scripts/run_walk_forward_reward.sh results/walk_forward_reward_20261106 5 100000
python -m scripts.evaluate_forecast_baselines --config config/long_term_reward_experiment.yaml \
  --run-dir results/walk_forward_reward_20261106 --years 2016 2017 2018 2019 2020 2021 2022
python -m scripts.aggregate_walk_forward_reward --run-dir results/walk_forward_reward_20261106
```

## 3. Source ablation (RQ2/RQ3)

```bash
CONFIG=config/source_ablation_experiment.yaml VARIANTS="src_technical src_technical_market_macro" SEEDS="42 43 44" \
  PYTHON=python scripts/run_walk_forward_reward.sh results/source_ablation_20261106 5 100000
python -m scripts.aggregate_walk_forward_reward --run-dir results/source_ablation_20261106 \
  --extra-run-dirs results/walk_forward_reward_20261106 \
  --variants src_technical plan_main_weekly_lstm src_technical_market_macro --seeds 42 43 44 --n-trials 3 \
  --output-dir results/source_ablation_combined_20261106
```

## 4. Algorithm comparison (RQ1/RQ4)

PPO comes from stage 2 and A2C runs locally. SAC, TD3 and DDPG need BistKA (`slurm/walk_forward_reward_array.sbatch`); see the [report](algorithm_comparison_20261107.md).

```bash
CONFIG=config/algorithm_comparison_experiment.yaml VARIANTS=algo_a2c SEEDS="42 43 44" \
  PYTHON=python scripts/run_walk_forward_reward.sh results/algorithm_comparison_20261106 5 100000
```

Aggregate with the report's command, keeping the variants that ran (`--variants plan_main_weekly_lstm algo_a2c`).

## 5. Point-in-time SET50: the DRL designs (M3, M5)

**Training:** one launcher call per variant, three workers.
- **Variants:** `pit_plan_weekly_lstm` (first run), `pit_plan_weekly_lstm_sticky`, `pit_plan_weekly_lstm_shared` and `pit_plan_weekly_lstm_shared_band`.
- **Cash cap:** `pit_plan_weekly_lstm_shared_cashcap`, from the main config, caps the cash target at 20%.
- **Source arms:** the BOT policy-rate arm uses its own config, `config/pit_walk_forward_policy_rate.yaml`, with `VARIANTS=pit_plan_weekly_lstm_shared_policy`. The macro-proxy arm uses `config/pit_walk_forward_macro.yaml` with `VARIANTS=pit_plan_weekly_lstm_shared_macro`.
- **Time:** each variant is 21 models of about 30–60 minutes of one core.

```bash
RUNNER=scripts.run_pit_walk_forward CONFIG=config/pit_walk_forward_experiment.yaml VARIANTS=pit_plan_weekly_lstm_shared \
  SEEDS="42 43 44" PYTHON=python scripts/run_walk_forward_reward.sh results/pit_walk_forward_shared_20261110 3 100000
```

The seed ensembles and the no-trade band reuse trained models without training:

```bash
python -m scripts.evaluate_pit_seed_ensemble --config config/pit_walk_forward_experiment.yaml \
  --source-run results/pit_walk_forward_shared_20261110 --variant pit_plan_weekly_lstm_shared \
  --output-dir results/pit_seed_ensemble_20261110 --band 0.005
python -m scripts.evaluate_pit_seed_ensemble --config config/pit_walk_forward_experiment.yaml \
  --source-run results/pit_walk_forward_band_20261110 --variant pit_plan_weekly_lstm_shared_band \
  --output-dir results/pit_band_ensemble_20261111 --modes ensemble
python -m scripts.evaluate_pit_seed_ensemble --config config/pit_walk_forward_policy_rate.yaml \
  --source-run results/pit_walk_forward_policy_20261111 --variant pit_plan_weekly_lstm_shared_policy \
  --output-dir results/pit_policy_ensemble_20261111 --band 0.005 --modes ensemble_band
python -m scripts.evaluate_pit_seed_ensemble --config config/pit_walk_forward_macro.yaml \
  --source-run results/pit_walk_forward_macro_20261111 --variant pit_plan_weekly_lstm_shared_macro \
  --output-dir results/pit_macro_ensemble_20261111 --band 0.005 --modes ensemble_band
python -m scripts.evaluate_pit_seed_ensemble --config config/pit_walk_forward_experiment.yaml \
  --source-run results/pit_walk_forward_cash_cap_20261119 --variant pit_plan_weekly_lstm_shared_cashcap \
  --output-dir results/pit_cash_cap_ensemble_20261119 --band 0.005 --modes ensemble_band
```

## 6. Point-in-time SET50: benchmarks (M3, M6)

None of these trains a DRL model:
- **Universes and ridge:** equal weight on three universes (point-in-time, today's members with hindsight, the five starter stocks) and the ridge forecaster.
- **Heuristics:** momentum, inverse volatility and minimum variance.
- **LSTM forecaster:** one fit per fold and seed, 21 in all.

```bash
YEARS="2016 2017 2018 2019 2020 2021 2022"
python -m scripts.pit_universe_baselines --config config/pit_walk_forward_experiment.yaml \
  --run-dir results/pit_walk_forward_20261107 --years ${=YEARS}   # zsh; in bash: --years $YEARS
python -m scripts.pit_heuristic_baselines --config config/pit_walk_forward_experiment.yaml \
  --run-dir results/pit_heuristic_baselines_20261110 --years ${=YEARS}
python -m scripts.pit_lstm_forecast_baseline --config config/pit_walk_forward_experiment.yaml \
  --run-dir results/pit_lstm_forecast_20261111 --years ${=YEARS}
```

## 7. Aggregation, significance and diagnostics

`scripts.aggregate_walk_forward_reward` merges run folders and writes:
- the per-fold and summary tables;
- the chained equity figure;
- the bootstrap Sharpe tests against the ETF, equal weight and every `--references` variant.

`--n-trials` is the number of designs counted by the deflated Sharpe ratio on the point-in-time results: 13 up to the band training, 15 with the policy-rate arm, 17 with the macro-proxy arm, 19 with the cash-capped arm, 21 with the 300k runs, 22 with the five-seed 100k ensemble (the count in the test and holdout locks) and 24 with the cash-capped design at 300k. On the five starter stocks it is 14. The exact combinations behind each table are in each report's "Run it" section. For example, the band-training tables:

```bash
python -m scripts.aggregate_walk_forward_reward --run-dir results/pit_walk_forward_band_20261110 \
  --extra-run-dirs results/pit_band_ensemble_20261111 results/pit_seed_ensemble_20261110 results/pit_walk_forward_shared_20261110 \
  --n-trials 13 --variants pit_plan_weekly_lstm_shared pit_plan_weekly_lstm_shared_band \
  --references pit_plan_weekly_lstm_shared pit_plan_weekly_lstm_shared_ensemble_band pit_plan_weekly_lstm_shared_band \
  pit_plan_weekly_lstm_shared_band_ensemble --output-dir results/pit_walk_forward_combined_band_20261111
python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_walk_forward_band_20261110 \
  --variant pit_plan_weekly_lstm_shared_band --pit-config config/pit_walk_forward_experiment.yaml
```

The input attribution re-runs every saved model with one input group shuffled or held at its training mean. It takes a few minutes per run:

```bash
python -m scripts.pit_feature_group_attribution --config config/pit_walk_forward_policy_rate.yaml \
  --run-dir results/pit_walk_forward_policy_20261111 --variant pit_plan_weekly_lstm_shared_policy
python -m scripts.pit_feature_group_attribution --config config/pit_walk_forward_experiment.yaml \
  --run-dir results/pit_walk_forward_shared_20261110 --variant pit_plan_weekly_lstm_shared \
  --output-dir results/pit_attribution_shared_20261111
python -m scripts.pit_feature_group_attribution --config config/pit_walk_forward_macro.yaml \
  --run-dir results/pit_walk_forward_macro_20261111 --variant pit_plan_weekly_lstm_shared_macro
python -m scripts.pit_feature_group_attribution --config config/pit_walk_forward_experiment.yaml \
  --run-dir results/pit_walk_forward_band_20261110 --variant pit_plan_weekly_lstm_shared_band \
  --output-dir results/pit_attribution_band_20261111
```

The formulas are in [`evaluation_metric_formulas_20261110.md`](evaluation_metric_formulas_20261110.md).

## 8. Compute accounting (RQ6, local part)

```bash
python -m scripts.local_compute_accounting --output results/local_compute_accounting_20261111.csv --run-dirs \
  results/walk_forward_reward_20261106 results/source_ablation_20261106 results/algorithm_comparison_20261106 \
  results/pit_walk_forward_20261107 results/pit_walk_forward_sticky_20261109 results/pit_walk_forward_shared_20261110 \
  results/pit_walk_forward_band_20261110 results/pit_walk_forward_policy_20261111 results/pit_walk_forward_macro_20261111
```

## 9. Longer training on BistKA, and five seeds (M6, RQ4/RQ6)

Two pool jobs on BistKA, each one 32-CPU job working through a task list (`slurm/walk_forward_pool.sbatch`), in the cluster copy of the code (`/lustrefs/project/25sfcs03/drl_thai_stock_20261122`; [HPC record](../hpc/hpc_sync_status.md)):

```bash
# Job 4409: the five algorithms and the point-in-time design at 300k, five seeds (210 models, 56 hours)
sbatch --export=ALL,PROJECT_DIR=$PWD,POOL=slurm/pools/bistka_20261122.tasks slurm/walk_forward_pool.sbatch
# Job 4518: the cash-capped point-in-time design at 300k, five seeds (35 models, 11.2 hours)
sbatch --mem=80G --time=2-00:00:00 --job-name=drl_pool_cap \
  --export=ALL,PROJECT_DIR=$PWD,POOL=slurm/pools/bistka_20261125.tasks slurm/walk_forward_pool.sbatch
```

Copy the run folders back with `rsync`, leaving out `*.zip` and `*.pkl`, except for point-in-time designs evaluated as seed ensembles: the ensemble needs their 35 `ppo_model.zip` files. Seeds 45 and 46 of the 100k designs were trained locally:

```bash
PYTHON=python bash scripts/run_task_pool.sh slurm/pools/local_20261125.tasks 3    # A2C and the shared design
PYTHON=python bash scripts/run_task_pool.sh slurm/pools/local_20261125b.tasks 3   # the cash-capped design
```

The ensembles, aggregation, diagnostics and the five-seed link folders are in the "Run it" sections of the [300k report](bistka_300k_results_20261125.md#run-it) and the [cash-cap report](cash_cap_5seeds_300k_20261126.md#run-it).

## 10. Choosing the final designs (decision 7)

Every design with complete validation curves is ranked by the median over the seven validation years of its annualised log return above equal weight of the same universe. Validation curves only; the script refuses any year after 2022.

```bash
python -m scripts.rank_final_candidates --candidates config/final_candidates_20261125.yaml --base results \
  --output-dir results/final_candidate_ranking_20261125
```

`--base` is the folder that holds the run folders (`results` in the project, `~/drl_runs` in the working clone). `config/cash_cap_5seeds_candidates_20261126.yaml` scores the locked design's later five-seed and 300k checks the same way.

## 11. The test years, once

The locks `config/final_test_protocol_pit.yaml` and `config/final_test_protocol_five_stock.yaml` name each design, its config hash, seeds, ensemble, references and trial count. The gate checks them, then trains one fold per test year and seed and writes receipts and a ledger line:

```bash
python -m scripts.run_final_test_evaluation --lock config/final_test_protocol_pit.yaml --dry-run
python -m scripts.run_final_test_evaluation --lock config/final_test_protocol_pit.yaml --workers 3
python -m scripts.run_final_test_evaluation --lock config/final_test_protocol_five_stock.yaml --workers 3
```

About an hour and 20 minutes on the Mac with three workers. The report's figures were redrawn from the stage outputs with legend labels (`scripts.aggregate_walk_forward_reward --labels ...`). The curves are identical to the stage summaries.

## 12. The sealed holdout, once

The locks `config/holdout_protocol_pit.yaml` and `config/holdout_protocol_five_stock.yaml` carry the same designs, the holdout end (4 November 2026), the folds (2025 and 2026), which sealed part continues each development file, the membership rule after 2024 and an optional stale-price stop. The gate ([stage note](holdout_stage_20261126.md)) checks the seals without parsing them and needs the matching test stage in the ledger. It opens the sealed files, continues the development files inside the output folder, and counts stale prices before any training. It then runs like the test stage:

```bash
python -m scripts.run_holdout_evaluation --lock config/holdout_protocol_pit.yaml --dry-run
python -m scripts.run_holdout_evaluation --lock config/holdout_protocol_five_stock.yaml --workers 3
python -m scripts.run_holdout_evaluation --lock config/holdout_protocol_pit.yaml --workers 3
```

About 20 minutes (five stocks) and 70 minutes (point-in-time). If an attempt stops before any model is evaluated, `--record-failed-attempt "<reason>"` renames its folder and records the failure in the ledger. A new attempt needs `retry_of` in an approved lock. The point-in-time stage of 2026-11-26 needed this once ([results](holdout_results_20261126.md#the-first-attempt-failed-before-training)). In a fresh clone the lock runs as it stands, because `retry_of` is checked only when the ledger already has the stage.

## 13. Report chapter

`docs/report_drafts/results_title_alignment_20261106.tex` is a drop-in chapter for `kvis_final_research_report.tex`. It builds with XeLaTeX (`latexmk -xelatex`). The figures it includes are in `docs/figures/`. The full report's own figures are in `reports/figures/` of the project folder, which git does not track; when building from a clone, put the project folder on TeX's search path (`TEXINPUTS=".:<project folder>:" latexmk -xelatex ...`).

## 14. Slim archive

The archive holds the code, configs, documents and reference inputs, and each run folder's metrics, curves, receipts, timings and figures. It leaves out:
- **Models:** trained models (`*.zip`) and `VecNormalize` statistics (`*.pkl`), which retraining rebuilds from the configs and seeds;
- **Logs:** per-task worker logs and lock folders;
- **Prices:** the Yahoo price files, which the receipts identify.

It carries a SHA-256 manifest and is verified after it is written:

```bash
python -m scripts.build_slim_results_archive --output results/slim_archive_20261120.tar.gz \
  $(for d in walk_forward_reward_20261106 source_ablation_20261106 source_ablation_combined_20261106 \
      algorithm_comparison_20261106 algorithm_comparison_combined_20261107 \
      pit_walk_forward_20261107 pit_walk_forward_sticky_20261109 pit_walk_forward_shared_20261110 \
      pit_walk_forward_combined_20261109 pit_walk_forward_combined_20261110 \
      pit_seed_ensemble_20261110 pit_seed_ensemble_combined_20261110 pit_band_sensitivity_20261110 \
      pit_heuristic_baselines_20261110 pit_walk_forward_band_20261110 pit_band_ensemble_20261111 \
      pit_walk_forward_combined_band_20261111 pit_lstm_forecast_20261111 pit_lstm_forecast_combined_20261111 \
      pit_walk_forward_policy_20261111 pit_policy_ensemble_20261111 pit_walk_forward_combined_policy_20261111 \
      pit_attribution_shared_20261111 pit_attribution_band_20261111 pit_walk_forward_macro_20261111 \
      pit_macro_ensemble_20261111 pit_walk_forward_combined_macro_20261111 pit_source_arms_combined_20261111 \
      pit_walk_forward_cash_cap_20261119 pit_cash_cap_ensemble_20261119 pit_walk_forward_combined_cash_cap_20261119; \
    do echo --run-dir results/$d; done)
```

**Rebuilt 2026-11-20** from these 31 run folders, after the cash-capped run, at code commit `9920591`:
- **Archive:** `results/slim_archive_20261120.tar.gz`, 59.1 MB (301 MB unpacked, 3,260 files), SHA-256 `5d9373933a7e…`. The full digest is in `slim_archive_20261120.tar.gz.receipt.json`, next to it.
- **Earlier build:** `results/slim_archive_20261111.tar.gz` (28 run folders, before the cash-capped run, commit `d23572a`) is kept with its receipt.
- **Verified:** the build reopened the archive and checked every file against the manifest.
- **Left out:** the first long-term-reward experiment (`results/long_term_reward_experiment_20261106/`, superseded by the walk-forward). Two of its files exist only in iCloud, and macOS refused to download them (`Errno 81`).

**Next rebuild (after 2026-11-26).** The run folders since 2026-11-20, in the project's `results/`, are:
- **BistKA and five seeds:** `algorithm_comparison_bistka_20261122`, `walk_forward_reward_bistka_20261122`, `pit_walk_forward_bistka_20261122`, `pit_walk_forward_cash_cap_bistka_20261125` and their combined and ensemble folders;
- **Design choice:** `final_candidate_ranking_20261125`;
- **Test years:** `final_test_point_in_time`, `final_test_five_stock`;
- **Holdout:** `holdout_point_in_time`, `holdout_point_in_time_attempt1_failed`, `holdout_five_stock`.

The archive now also takes the ledger (`results/FINAL_TEST_LEDGER.jsonl`) and leaves out the holdout stages' price copies (`holdout_data/`), like every other price file.
