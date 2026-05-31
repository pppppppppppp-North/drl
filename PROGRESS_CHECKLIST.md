# Project Progress Checklist

Last updated: 2026-05-31

Use this file as the single running checklist for what has been completed and what remains. Update it whenever code, data, HPC jobs, results, or report artifacts change.

## Status Legend

- `[x]` Done and verified
- `[~]` Started or partially complete
- `[ ]` Not started
- `[!]` Blocked or needs attention

## Current Snapshot

- `[x]` Local project scaffold exists.
- `[x]` BistKA HPC project directory created: `/lustrefs/project/25sfcs03/drl_thai_stock`.
- `[x]` BistKA conda environment created: `/lustrefs/project/25sfcs03/envs/drl_env`.
- `[x]` Synthetic pilot pipeline runs end-to-end.
- `[x]` Trading environment, baselines, PPO training, metrics, action logging, and comparison summaries are implemented.
- `[x]` Latest verified HPC PPO technical run: `results/run_2850`.
- `[x]` Latest verified HPC tests: `40 passed` on 2026-05-31 after adding official macro ingestion and merge coverage.
- `[x]` Buy-and-hold and moving-average crossover baselines are implemented and verified on HPC.
- `[x]` Mean-variance baseline is implemented and verified on HPC.
- `[x]` Result plotting utility for equity curves, drawdowns, turnover, and action weights is implemented and verified on HPC.
- `[x]` Latest local syntax check: `python3 -m compileall -q src tests` passed on 2026-05-30.
- `[x]` Latest local syntax check after Optuna trial-runner edits: `python3 -m compileall -q src tests` passed on 2026-05-30.
- `[x]` Latest local syntax check after walk-forward/report edits: `python3 -m compileall -q src tests` passed on 2026-05-30.
- `[x]` Latest local syntax check after file-separation helper script: `python3 -m compileall -q src tests scripts` passed on 2026-05-30.
- `[x]` Latest local syntax check after walk-forward runner edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-30.
- `[x]` Latest local syntax check after walk-forward aggregation edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-30.
- `[x]` Latest local syntax check after ablation tooling edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-30.
- `[x]` Latest local syntax check after regime-test edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-31.
- `[x]` Latest local syntax check after SET index feature edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-31.
- `[x]` Latest local syntax check after market-context config activation: `python3 -m compileall -q src tests scripts` passed on 2026-05-31.
- `[x]` Latest local syntax check after sentiment ingestion/merge edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-31.
- `[x]` Latest local syntax check after official macro ingestion/merge edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-31.
- `[x]` SET market-index ingestion and market-context feature merge implemented and verified on HPC.
- `[x]` SET index raw data collected on HPC: `data/raw/prices_market_indices.csv`.
- `[x]` Real OHLCV + SET market-context feature table built on HPC: `data/processed/features_real_ohlcv_market.parquet`.
- `[x]` Latest SET market-context pilot completed on HPC: `results/real_ohlcv_market_pilot_20260531`.
- `[x]` Pilot sector mapping added for the five starter tickers: `data/reference/sector_mapping_thai_pilot.csv`.
- `[x]` Real OHLCV + SET + sector-context feature table built on HPC: `data/processed/features_real_ohlcv_sector.parquet`.
- `[x]` Latest sector-context pilot completed on HPC: `results/real_ohlcv_sector_pilot_20260531`.
- `[x]` Yahoo Finance macro proxy ingestion added and verified on HPC: `data/raw/prices_macro_yahoo.csv`.
- `[x]` Real OHLCV + SET + sector + macro-proxy feature table built on HPC: `data/processed/features_real_ohlcv_macro.parquet`.
- `[x]` Latest macro-proxy pilot completed on HPC: `results/real_ohlcv_macro_pilot_20260531`.
- `[x]` Yahoo Finance annual and quarterly fundamentals ingestion added and verified on HPC: `data/raw/fundamentals_yahoo_quarterly.csv`.
- `[x]` Real OHLCV + SET + sector + macro-proxy + fundamentals feature table built on HPC: `data/processed/features_real_ohlcv_fundamentals.parquet`.
- `[x]` Latest fundamentals pilot completed on HPC: `results/real_ohlcv_fundamentals_pilot_20260531`.
- `[x]` Yahoo Finance latest-news probe added and run on HPC: `data/raw/news_yahoo_latest.csv`.
- `[x]` Daily sentiment extraction and no-lookahead sentiment merge scaffold implemented and verified on HPC.
- `[x]` Real OHLCV + SET + sector + macro-proxy + fundamentals + sentiment scaffold feature table built on HPC: `data/processed/features_real_ohlcv_sentiment.parquet`.
- `[!]` Yahoo latest-news probe returned only 11 rows dated 2025-06-24 to 2026-03-31, so it does not overlap the 2021-2024 modeling window and cannot support a valid historical sentiment pilot.
- `[x]` Bank of Thailand official macro source probe and release-date merge scaffold implemented and verified on HPC.
- `[x]` BOT Leading Economic Indicator probe collected on HPC: `data/raw/bot_official_macro.csv`.
- `[x]` Real OHLCV + SET + sector + macro-proxy + fundamentals + official macro scaffold feature table built on HPC: `data/processed/features_real_ohlcv_official_macro.parquet`.
- `[!]` BOT web table probe returned only 54 rows dated 2025-11-01 to 2026-04-01, so it does not overlap the 2021-2024 modeling window and cannot support a valid historical official-macro pilot without a historical export/API source.
- `[x]` Regime/stress-test config and runner implemented and verified on HPC.
- `[x]` Latest regime/stress test completed on HPC: `results/regime_tests_real_ohlcv_20260531`.
- `[x]` Real-OHLCV ablation configs, SLURM array script, and aggregation utility implemented and verified on HPC.
- `[x]` Latest ablation pilot completed on HPC: `results/ablation_pilot_2871`.
- `[x]` Latest algorithm comparison job completed on HPC: `results/algo_compare_2853`.
- `[x]` Train/validation/test split wiring added for training and baseline evaluation, verified on HPC.
- `[x]` Latest split-aware PPO/A2C validation comparison completed on HPC: `results/algo_compare_2853`.
- `[x]` Leakage check utilities added for date/ticker duplicates, split boundaries, and future release dates; verified on HPC.
- `[x]` Real Thai OHLCV starter ingestion added via Yahoo Finance/yfinance schema and verified on HPC.
- `[x]` Real OHLCV feature table built on HPC: `data/processed/features_real_ohlcv.parquet`.
- `[x]` Real OHLCV feature table passed leakage and split-boundary checks on HPC.
- `[x]` Synthetic-regeneration guard added and verified so non-synthetic configs cannot overwrite real feature tables.
- `[x]` Latest real-OHLCV PPO/A2C validation comparison completed on HPC: `results/real_ohlcv_compare_2855`.
- `[x]` Real-data Optuna trial runner implemented, uploaded, and verified through SLURM sanity job `2856`.
- `[x]` Optuna trial aggregation utility implemented and verified on HPC.
- `[x]` Latest real-OHLCV Optuna search completed on HPC: `results/optuna_search_fixed_2863`.
- `[x]` Best Optuna PPO config prepared and run on HPC: `config/best_ppo_real_ohlcv.yaml`, job `2868`.
- `[x]` Final methodology/results writeup drafted and uploaded: `docs/final_methodology_results.md`.
- `[x]` Real OHLCV data-quality report generated on HPC: `reports/data_quality/real_ohlcv_data_quality.md`.
- `[x]` Walk-forward window config and summary writer implemented and verified on HPC: `results/walk_forward_summary_check/walk_forward_summary.csv`.
- `[x]` Walk-forward per-window training runner and SLURM array script implemented and verified on HPC.
- `[x]` Walk-forward aggregation utility implemented and verified on HPC.
- `[x]` Latest walk-forward pilot completed on HPC: `results/walk_forward_pilot_2869`.
- `[x]` File-separation snapshot helper added: `scripts/file_separation_snapshot.py`.
- `[x]` Final artifact manifest uploaded: `docs/final_artifacts_manifest.md`.
- `[x]` Final methodology/results report converted to LaTeX locally: `final_methodology_results.tex`.
- `[x]` Final pilot archive created on HPC: `final_archive/20260530_final_project_pilot.tar.gz`.
- `[x]` Updated final pilot archive with walk-forward, ablation, and regime/stress-test artifacts: `final_archive/20260531_final_project_pilot.tar.gz`.
- `[x]` Updated final pilot archive with SET market-context artifacts: `final_archive/20260531_final_project_pilot_with_set_context.tar.gz`.
- `[x]` Updated final pilot archive with sector-context artifacts: `final_archive/20260531_final_project_pilot_with_sector_context.tar.gz`.
- `[x]` Updated final pilot archive with macro-proxy artifacts: `final_archive/20260531_final_project_pilot_with_macro_context.tar.gz`.
- `[x]` Updated final pilot archive with fundamentals artifacts: `final_archive/20260531_final_project_pilot_with_fundamentals.tar.gz`.
- `[x]` Updated final pilot archive with sentiment-scaffold artifacts: `final_archive/20260531_final_project_pilot_with_sentiment_scaffold.tar.gz`.
- `[x]` Updated final pilot archive with official-macro-scaffold artifacts: `final_archive/20260531_final_project_pilot_with_official_macro_scaffold.tar.gz`.
- `[x]` Latest file-separation snapshot on HPC: `_file_separation/20260531_160247`.
- `[!]` Real-OHLCV comparison job `2854` is invalid because the old training entrypoint regenerated synthetic data into real config paths; use corrected job `2855`.
- `[!]` Real OHLCV starter data, SET index context, pilot sector mapping, daily macro proxies, Yahoo statement fundamentals, sentiment merge scaffolding, and official macro merge scaffolding exist, but broad sector indices, historical official release-aligned macro, licensed fundamentals, and historical sentiment data are still incomplete.

## Phase 1: Foundation And Repository Setup

- `[x]` Create repository-style folder structure.
- `[x]` Add `README.md`.
- `[x]` Add `requirements.txt`.
- `[x]` Add `environment.yml`.
- `[x]` Add base configs: `config/default.yaml`, `config/debug.yaml`, `config/ppo_technical.yaml`.
- `[x]` Add source package folders under `src/`.
- `[x]` Add tests folder.
- `[x]` Add HPC notes and SLURM templates.
- `[x]` Upload project to BistKA.
- `[x]` Separate uploaded files from pre-existing/generated files on HPC with `_file_separation/latest`.

## Phase 2: Data Manifest And Pilot Data

- `[x]` Add `data/data_manifest.csv`.
- `[x]` Add `docs/data_sources.md`.
- `[x]` Implement deterministic synthetic price generator.
- `[x]` Generate synthetic pilot raw prices on HPC.
- `[x]` Generate synthetic pilot feature table on HPC.
- `[x]` Add validation so stale pilot feature files are regenerated when config ticker/date/column coverage changes.
- `[x]` Collect real SET50/SET100 OHLCV data; starter Yahoo Finance download completed for 5 `.BK` tickers.
- `[~]` Collect SET index and sector index data; SET index collected with Yahoo symbol `^SET.BK`, sector index source still pending.
- `[x]` Collect daily macro proxy data from Yahoo Finance.
- `[~]` Collect official real macro data with release-date alignment; BOT latest table probe works but has no 2021-2024 overlap.
- `[~]` Collect real news/sentiment source data; Yahoo latest-news probe works but has no 2021-2024 overlap.
- `[ ]` Create data-quality notebook.
- `[x]` Add missing value and coverage reports for real data.

## Phase 3: Feature Engineering

- `[x]` Implement technical indicators.
- `[x]` Add returns: 1-day, 5-day, 20-day.
- `[x]` Add rolling volatility.
- `[x]` Add moving-average ratios.
- `[x]` Add RSI.
- `[x]` Add MACD.
- `[x]` Add Bollinger z-score.
- `[x]` Add volume ratio.
- `[x]` Include synthetic market return, macro change, and sentiment score placeholders.
- `[x]` Add ATR.
- `[~]` Add sector-relative features; SET market-index context features are implemented and verified, true sector-relative features still pending sector data.
- `[x]` Add daily macro proxy features.
- `[~]` Add official release-aligned macro features; BOT release-date merge scaffold is implemented, but a historical official export/API source is still needed.
- `[x]` Add Yahoo annual/quarterly statement fundamentals with reporting lag.
- `[~]` Add real sentiment embeddings or scores; daily extraction and merge scaffolding are implemented, but a historical licensed/news source is still needed.
- `[x]` Add train/validation/test split utilities.
- `[x]` Wire train/validation/test splits into training and baseline evaluation entrypoints.
- `[x]` Add leakage checks for real mixed-frequency data.

## Phase 4: Trading Environment And Baselines

- `[x]` Implement Gymnasium-compatible `ThaiStockTradingEnv`.
- `[x]` Continuous long-only action vector.
- `[x]` Portfolio accounting.
- `[x]` Transaction cost and turnover handling.
- `[x]` Reward with return, turnover penalty, and drawdown penalty.
- `[x]` Stable observation shape.
- `[x]` Info dict includes portfolio value, return, turnover, drawdown, weights, and cash weight.
- `[x]` Add environment tests.
- `[x]` Implement equal-weight baseline.
- `[x]` Implement random baseline.
- `[x]` Implement momentum baseline.
- `[x]` Implement buy-and-hold per ticker baseline.
- `[x]` Implement moving-average crossover baseline.
- `[x]` Implement mean-variance baseline.

## Phase 5: PPO Technical Agent

- `[x]` Integrate Stable-Baselines3 PPO.
- `[x]` Run debug PPO job on BistKA.
- `[x]` Run technical PPO job on BistKA.
- `[x]` Save PPO model.
- `[x]` Save PPO equity curve.
- `[x]` Save PPO metrics.
- `[x]` Save PPO action weights.
- `[x]` Save PPO action summary.
- `[x]` Save TensorBoard logs.
- `[x]` Compare PPO against baselines for latest verified run.
- `[x]` Latest verified run: `/lustrefs/project/25sfcs03/drl_thai_stock/results/run_2850`.
- `[x]` Add checkpoint callbacks during training.
- `[x]` Add action distribution plots via `src/evaluation/plot_results.py`.
- `[x]` Add drawdown plots via `src/evaluation/plot_results.py`.
- `[x]` Add turnover plots via `src/evaluation/plot_results.py`.

## Phase 6: Algorithm Comparison

- `[x]` Refactor training script to support multiple Stable-Baselines3 algorithms.
- `[x]` Add support paths for PPO, A2C, DDPG, TD3, and SAC.
- `[x]` Update comparison utility to aggregate multiple model metric files.
- `[x]` Add `slurm/algorithm_compare.sbatch`.
- `[x]` Upload and run `slurm/algorithm_compare.sbatch` on HPC.
- `[x]` Verify A2C run on HPC.
- `[x]` Compare PPO vs A2C vs baselines.
- `[x]` Run split-aware PPO vs A2C vs baselines validation comparison on HPC.
- `[x]` Decide whether DDPG, TD3, and SAC are worth running on the current environment: defer until real data and out-of-sample splits are active.

## Phase 7: Hyperparameter Search

- `[x]` Add initial Optuna/job-array SLURM template.
- `[x]` Convert Optuna trial runner to tune real PPO/A2C hyperparameters.
- `[x]` Add search space for learning rate, gamma, GAE lambda, clip range, entropy coefficient, rollout length, and batch size.
- `[x]` Add `config/optuna_search.yaml` for real-OHLCV PPO/A2C search.
- `[x]` Run small sanity search: SLURM array job `2856`, output `results/optuna_sanity_2856`.
- `[x]` Run HPC job array: replacement job `2863`, output `results/optuna_search_fixed_2863`.
- `[x]` Aggregate hyperparameter search results: `results/optuna_search_fixed_2863/trial_results.csv`.
- `[x]` Promote best Optuna PPO parameters into a dedicated config and run longer validation job.

## Phase 8: Multi-Source Features And Ablations

- `[~]` Synthetic placeholders exist for macro and sentiment.
- `[~]` Build real sentiment extraction input dataset; Yahoo latest-news probe produced `data/raw/news_yahoo_latest.csv` with no historical overlap.
- `[~]` Run WangchanBERTa embedding or sentiment extraction on GPU; deterministic sentiment scorer is verified as a pipeline placeholder, GPU transformer extraction still pending.
- `[~]` Merge real sentiment features by date/ticker; no-lookahead merge scaffold is verified, but merged 2021-2024 sentiment values are zero because source news is outside the modeling window.
- `[x]` Merge daily macro proxy features by no-lookahead forward fill.
- `[~]` Merge official real macro features by release date; no-lookahead merge scaffold is verified, but merged 2021-2024 BOT values are zero because the web-table probe is outside the modeling window.
- `[x]` Merge Yahoo annual/quarterly statement fundamentals with reporting lag.
- `[~]` Create feature-group configs: returns-only, technical, technical+context, index, sector, macro-proxy, fundamentals, sentiment-scaffold, and official-macro-scaffold configs added and verified; licensed fundamentals, historical official macro, historical sentiment, and all-real-feature configs still depend on missing external data.
- `[x]` Run ablation pilot experiments: SLURM array job `2871`.
- `[x]` Write pilot ablation metrics: `results/ablation_pilot_2871/aggregate/ablation_metrics.csv`.

## Phase 9: Walk-Forward And Regime Testing

- `[x]` Implement walk-forward window config.
- `[x]` Add training/evaluation split support to environment or data loader.
- `[x]` Run one job per walk-forward window; pilot SLURM array job `2869` completed for windows 0 and 1.
- `[x]` Aggregate out-of-sample equity curves; output `results/walk_forward_pilot_2869/aggregate/walk_forward_test_equity_curves.csv`.
- `[x]` Compute pilot walk-forward metrics; output `results/walk_forward_pilot_2869/aggregate/walk_forward_policy_summary.csv`.
- `[x]` Define COVID/crisis period slices; acute 2020 COVID crash is marked unavailable because the starter real-OHLCV file begins in 2021.
- `[x]` Run regime/stress tests: `results/regime_tests_real_ohlcv_20260531`.

## Phase 10: Reporting And Final Artifacts

- `[x]` Main implementation plan exists: `implementation_plan.md`, `.tex`, `.pdf`.
- `[x]` Literature review exists: `literature_review/literature_review.md`, `.tex`, `.pdf`.
- `[x]` HPC guide exists.
- `[x]` SLURM guide exists.
- `[x]` Add final result table from synthetic algorithm comparison pilot.
- `[x]` Add equity curve figure generator.
- `[x]` Add drawdown figure generator.
- `[x]` Add action distribution figure generator.
- `[x]` Add ablation bar chart: `results/ablation_pilot_2871/aggregate/ablation_ppo_sharpe.png`.
- `[x]` Add final methodology/results writeup.
- `[x]` Add final methodology/results LaTeX report: `final_methodology_results.tex`.
- `[x]` Archive final config, logs, metrics, and model summaries.

## Latest Verified Results

Latest verified real-OHLCV split-aware algorithm comparison run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_compare_2855
```

Job `2855` completed on BistKA with an empty error log and produced PPO/A2C models, checkpoints, baseline comparisons, comparison summaries, action summaries, and figures.

Real OHLCV split used by `real_ohlcv_compare_2855`:

| split | start_date | end_date | rows | unique_dates | tickers |
| --- | --- | --- | ---: | ---: | ---: |
| train | 2021-01-04 | 2023-05-29 | 2900 | 580 | 5 |
| validation | 2023-05-30 | 2024-03-12 | 970 | 194 | 5 |
| test | 2024-03-13 | 2024-12-30 | 970 | 194 | 5 |

Main real-OHLCV validation comparison from `real_ohlcv_compare_2855`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 | 1043523.041075 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 | 1009998.557967 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 | 969426.916197 |
| a2c | -0.038061 | -0.057884 | -0.434028 | -0.120042 | 961938.712210 |
| ppo | -0.035900 | -0.054629 | -0.439532 | -0.066219 | 964100.020195 |
| buy_hold_KBANK.BK | -0.063331 | -0.095643 | -0.517619 | -0.125000 | 936669.309842 |
| buy_hold_AOT.BK | -0.077056 | -0.115925 | -0.569733 | -0.201365 | 922944.160998 |
| buy_hold_CPALL.BK | -0.093831 | -0.140495 | -0.658999 | -0.224335 | 906169.402777 |
| mean_variance | -0.099682 | -0.149008 | -1.097376 | -0.166737 | 900318.191064 |
| ma_crossover | -0.130706 | -0.193648 | -1.581191 | -0.178892 | 869294.229253 |
| random | -0.150932 | -0.222296 | -1.802937 | -0.191240 | 849068.332576 |
| momentum | -0.186758 | -0.272144 | -1.815650 | -0.216477 | 813242.480998 |

Invalidated run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_compare_2854
```

Run `2854` should not be used because the pre-fix training entrypoint regenerated synthetic data into the real config paths.

Latest verified real-OHLCV Optuna sanity run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/optuna_sanity_2856
```

Job `2856` ran two short 1000-timestep array tasks through SLURM:

| trial_id | algorithm | objective_sharpe | cumulative_return | max_drawdown |
| ---: | --- | ---: | ---: | ---: |
| 0 | ppo | -1.397044 | -0.003183 | -0.004519 |
| 1 | a2c | -2.325976 | -0.002303 | -0.002564 |

Invalidated Optuna search run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/optuna_search_2858
```

Job `2858` completed, but should not be used as a hyperparameter search because the pre-fix sampler produced identical sampled parameters for repeated PPO/A2C trial numbers.

Latest verified real-OHLCV Optuna search run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/optuna_search_fixed_2863
```

Job `2863` completed five 4000-timestep search tasks and wrote `trial_results.csv`.

| trial_id | algorithm | objective_sharpe | cumulative_return | max_drawdown | learning_rate | gamma | n_steps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | ppo | 0.279954 | 0.020408 | -0.068810 | 0.000389 | 0.98 | 256 |
| 0 | ppo | -0.305640 | -0.002649 | -0.008809 | 0.000017 | 0.90 | 512 |
| 1 | a2c | -0.519583 | -0.042686 | -0.127149 | 0.001210 | 0.99 | 256 |
| 2 | ppo | -1.079920 | -0.088712 | -0.115319 | 0.002160 | 0.95 | 256 |
| 3 | a2c | -1.708120 | -0.003138 | -0.004297 | 0.000029 | 0.90 | 128 |

Longer best-parameter PPO validation run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/best_ppo_real_ohlcv_2868
```

Job `2868` used `config/best_ppo_real_ohlcv.yaml` for 20000 timesteps. The error log was empty, but validation performance did not improve over the short Optuna trial:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| ppo | -0.030269 | -0.046132 | -0.403028 | -0.099395 | 969730.884751 |

Latest verified real-OHLCV walk-forward pilot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/walk_forward_pilot_2869
```

SLURM array job `2869` completed one short 1000-timestep PPO training/evaluation task per walk-forward window. Both error logs were empty:

- `logs/walk_forward_2869_0.err`
- `logs/walk_forward_2869_1.err`

Aggregated outputs:

- `results/walk_forward_pilot_2869/aggregate/walk_forward_all_metrics.csv`
- `results/walk_forward_pilot_2869/aggregate/walk_forward_policy_summary.csv`
- `results/walk_forward_pilot_2869/aggregate/walk_forward_test_equity_curves.csv`

Pilot test-window policy summary:

| policy | windows | mean_cumulative_return | mean_annualized_return | mean_sharpe | mean_max_drawdown | mean_final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| buy_hold_ADVANC.BK | 2 | 0.083927 | 0.281661 | 1.467928 | -0.064073 | 1083926.536029 |
| buy_hold_KBANK.BK | 2 | 0.064203 | 0.202352 | 1.157021 | -0.096110 | 1064202.767969 |
| buy_hold_PTT.BK | 2 | 0.040690 | 0.111308 | 0.672495 | -0.081828 | 1040690.202797 |
| equal_weight | 2 | 0.021133 | 0.066667 | 0.633491 | -0.068208 | 1021133.458705 |
| ppo | 2 | 0.004260 | 0.011669 | 0.057644 | -0.026071 | 1004259.896894 |

Latest verified real-OHLCV ablation pilot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/ablation_pilot_2871
```

SLURM array job `2871` completed three short 1000-timestep PPO ablation tasks:

- `config/ablation_returns_real_ohlcv.yaml`
- `config/ablation_technical_real_ohlcv.yaml`
- `config/ablation_technical_context_real_ohlcv.yaml`

All three error logs were empty:

- `logs/ablation_2871_0.err`
- `logs/ablation_2871_1.err`
- `logs/ablation_2871_2.err`

Aggregated outputs:

- `results/ablation_pilot_2871/aggregate/ablation_metrics.csv`
- `results/ablation_pilot_2871/aggregate/ablation_best_policy_summary.csv`
- `results/ablation_pilot_2871/aggregate/ablation_ppo_summary.csv`
- `results/ablation_pilot_2871/aggregate/ablation_ppo_sharpe.png`

PPO ablation summary:

| feature_group | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| ablation_returns_real_ohlcv | -0.010482 | -0.016061 | -0.453371 | -0.040127 | 989517.774541 |
| ablation_technical_context_real_ohlcv | -0.055886 | -0.084574 | -0.823513 | -0.099160 | 944114.266259 |
| ablation_technical_real_ohlcv | -0.078913 | -0.118657 | -1.611158 | -0.109727 | 921086.783720 |

The best policy in each pilot ablation comparison was still `buy_hold_PTT.BK` with validation Sharpe `0.405480`, so these short ablation runs are implementation checks rather than evidence that PPO is competitive yet.

Latest verified real-OHLCV regime/stress test:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/regime_tests_real_ohlcv_20260531
```

The stress-test runner evaluated baselines plus the saved PPO/A2C models from `results/real_ohlcv_compare_2855` across named slices and two auto-selected 63-trading-date high-volatility windows. It also records that the acute 2020 COVID crash cannot be evaluated from the current starter data because the OHLCV file begins on 2021-01-04.

Outputs:

- `results/regime_tests_real_ohlcv_20260531/regime_slices.csv`
- `results/regime_tests_real_ohlcv_20260531/regime_metrics.csv`
- `results/regime_tests_real_ohlcv_20260531/regime_equity_curves.csv`
- `results/regime_tests_real_ohlcv_20260531/regime_summary.md`

Best policy by evaluated regime:

| regime | policy | cumulative_return | annualized_return | sharpe | max_drawdown |
| --- | --- | ---: | ---: | ---: | ---: |
| covid_recovery_2021 | buy_hold_ADVANC.BK | 0.377423 | 0.465851 | 2.375753 | -0.061453 |
| high_volatility_1_2021-01-28_2021-05-06 | buy_hold_ADVANC.BK | -0.016542 | -0.119601 | -0.735410 | -0.055866 |
| high_volatility_2_2021-06-01_2021-09-01 | a2c | 0.093690 | 0.981571 | 7.220885 | -0.027634 |
| inflation_hike_drawdown_2022 | buy_hold_AOT.BK | 0.160483 | 0.194535 | 1.366575 | -0.047458 |
| test_period | buy_hold_ADVANC.BK | 0.483925 | 0.833957 | 3.681822 | -0.074324 |
| validation_period | buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 |

Latest verified SET market-context feature pilot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_market_pilot_20260531
```

The market-context pipeline uses Yahoo Finance symbol `^SET.BK` to collect the SET index, then merges SET-derived features into the five-stock OHLCV table. The obvious Yahoo Finance variants probed for SET50/SET100 did not return usable rows in the HPC yfinance check, so sector/SET50/SET100 source work remains open.

Data artifacts:

- `data/raw/prices_market_indices.csv`: 965 SET index rows, 2021-01-04 to 2024-12-30, no missing values.
- `data/processed/features_real_ohlcv_market.parquet`: 4,840 stock rows, 26 columns.
- `reports/data_quality_market/real_ohlcv_data_quality.md`: complete five-ticker panel coverage and zero missing values in the active market-context table.

Added market-context columns:

- `set_return_1d`
- `set_return_5d`
- `set_return_20d`
- `set_volatility_20d`
- `set_ma_ratio_20`

Short 1000-timestep validation comparison using `config/real_ohlcv_market.yaml`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 | 1043523.041075 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 | 1009998.557967 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 | 969426.916197 |
| ppo | -0.054013 | -0.081783 | -0.952524 | -0.104711 | 945986.972584 |

Latest verified sector-context feature pilot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_sector_pilot_20260531
```

The sector-context pipeline uses `data/reference/sector_mapping_thai_pilot.csv`, a static five-ticker mapping from SET public listed-company and constituent classifications. It is a pilot fallback because the Yahoo SET50/SET100 and sector-index candidates probed from HPC did not return reliable history.

Sector data artifacts:

- `data/reference/sector_mapping_thai_pilot.csv`: ADVANC.BK, AOT.BK, CPALL.BK, KBANK.BK, and PTT.BK sector labels.
- `data/processed/features_real_ohlcv_sector.parquet`: 4,840 stock rows, 38 columns.
- `reports/data_quality_sector/real_ohlcv_data_quality.md`: complete five-ticker panel coverage and zero missing values.

Short 1000-timestep validation comparison using `config/real_ohlcv_sector.yaml`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 | 1043523.041075 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 | 1009998.557967 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 | 969426.916197 |
| ppo | -0.031097 | -0.047383 | -0.626349 | -0.068484 | 968902.974943 |

The sector-context PPO pilot improves over the SET-market-context PPO pilot on cumulative return, Sharpe, and max drawdown, but it still does not beat the PTT buy-and-hold baseline.

Latest verified macro-proxy feature pilot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_macro_pilot_20260531
```

The macro-proxy pipeline uses Yahoo Finance daily series for USD/THB, Brent crude, WTI crude, gold, and the U.S. 10-year yield. These are liquid market proxies, not official macroeconomic release-date series. The merge uses forward fill from each macro proxy's latest available observation to each Thai stock trading date, avoiding use of future observations.

Macro proxy artifacts:

- `data/raw/prices_macro_yahoo.csv`: 5,059 rows across `usdthb`, `brent`, `wti`, `gold`, and `us10y`.
- `data/processed/features_real_ohlcv_macro.parquet`: 4,840 stock rows, 68 columns.
- `reports/data_quality_macro/real_ohlcv_data_quality.md`: complete five-ticker panel coverage and zero missing values.

Short 1000-timestep validation comparison using `config/real_ohlcv_macro.yaml`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 | 1043523.041075 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 | 1009998.557967 |
| ppo | -0.022360 | -0.034152 | -0.344318 | -0.085930 | 977639.781729 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 | 969426.916197 |

The macro-proxy PPO pilot improves over the sector-context PPO pilot and beats equal weight on this short validation run, but it still does not beat the PTT or ADVANC buy-and-hold baselines.

Latest verified fundamentals feature pilot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_fundamentals_pilot_20260531
```

The fundamentals pipeline uses Yahoo Finance annual and quarterly financial statements for the five starter tickers. Quarterly statements are preferred when available for a duplicate ticker/period/metric, and annual statements provide older fallback coverage. Features are merged only after a 60-day reporting lag.

Fundamentals artifacts:

- `data/raw/fundamentals_yahoo_quarterly.csv`: 8,701 statement rows across 5 tickers and 13 period ends.
- `data/processed/features_real_ohlcv_fundamentals.parquet`: 4,840 stock rows, 82 columns.
- `reports/data_quality_fundamentals/real_ohlcv_data_quality.md`: complete five-ticker panel coverage; numeric fundamental features are zero-filled before the first lagged report, while `period_end`/`effective_date` are missing for 27.7% of rows before coverage begins.

Short 1000-timestep validation comparison using `config/real_ohlcv_fundamentals.yaml`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 | 1043523.041075 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 | 1009998.557967 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 | 969426.916197 |
| ppo | -0.025996 | -0.039666 | -0.416967 | -0.067777 | 974003.500341 |

The fundamentals PPO pilot has lower drawdown than equal weight and similar Sharpe, but it still does not beat the PTT or ADVANC buy-and-hold baselines. It also underperforms the macro-proxy-only pilot, so fundamentals need a wider licensed history before strong conclusions.

Latest real-OHLCV data-quality report:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/reports/data_quality/real_ohlcv_data_quality.md
```

The report shows 968 unique trading dates per ticker, 0 missing panel dates for each of the five `.BK` tickers, and 0 missing values across the active feature columns.

Final pilot archives:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260530_final_project_pilot.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_set_context.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_sector_context.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_macro_context.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_fundamentals.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_sentiment_scaffold.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_official_macro_scaffold.tar.gz
```

Latest archive size: 245 MB. It contains the final writeups, configs, source code, tests, real OHLCV data, SET index data, pilot sector mapping, macro proxy data, BOT official macro source probe data, Yahoo statement fundamentals, Yahoo latest-news probe data, market-context, sector-context, macro-proxy, fundamentals, sentiment-scaffold, and official-macro-scaffold features, data-quality reports, comparison summaries, Optuna summary, walk-forward aggregates, ablation aggregates, regime/stress-test outputs, selected figures, and relevant run logs.

Latest HPC file-separation snapshot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/_file_separation/20260531_160247
```

`uploaded_current/` contains the files uploaded in the latest pass, while `preexisting_reference/generated_file_index.txt` records generated and previously present data/result artifacts.

Latest verified split-aware algorithm comparison run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/algo_compare_2853
```

Job `2853` completed on BistKA with an empty error log and produced:

- `ppo_model.zip`, `ppo_equity_curve.csv`, `ppo_actions.csv`, `ppo_metrics.csv`
- `a2c_model.zip`, `a2c_equity_curve.csv`, `a2c_actions.csv`, `a2c_metrics.csv`
- checkpoint files under `checkpoints/ppo/` and `checkpoints/a2c/` at 2k-step intervals
- `baselines/baseline_equity_curves.csv`, `baselines/baselines_metrics.csv`
- `comparison_metrics.csv`, `comparison_summary.md`, `action_summary.csv`, `ppo_action_summary.csv`
- `split_summary.csv`
- figures under `figures/`: `equity_curves.png`, `drawdowns.png`, `turnover.png`, `action_mean_weights.png`

Chronological split used by `algo_compare_2853`:

| split | start_date | end_date | rows | unique_dates | tickers |
| --- | --- | --- | ---: | ---: | ---: |
| train | 2021-01-01 | 2023-05-25 | 3125 | 625 | 5 |
| validation | 2023-05-26 | 2024-03-13 | 1045 | 209 | 5 |
| test | 2024-03-14 | 2024-12-31 | 1045 | 209 | 5 |

Main validation comparison from `algo_compare_2853`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_KBANK.BK | 0.580296 | 0.904531 | 3.218974 | -0.143165 | 1580295.743752 |
| ma_crossover | 0.257937 | 0.381344 | 1.880703 | -0.091600 | 1257936.712797 |
| equal_weight | 0.152812 | 0.221644 | 1.397638 | -0.095300 | 1152811.724538 |
| a2c | 0.167278 | 0.243282 | 1.372647 | -0.103622 | 1167278.283851 |
| mean_variance | 0.122426 | 0.176558 | 0.860456 | -0.109790 | 1122426.235382 |
| buy_hold_PTT.BK | 0.078909 | 0.112851 | 0.524486 | -0.151637 | 1078909.487165 |
| random | 0.053055 | 0.075491 | 0.457392 | -0.110405 | 1053054.552875 |
| buy_hold_ADVANC.BK | 0.090772 | 0.130115 | 0.428325 | -0.159049 | 1090772.121175 |
| buy_hold_CPALL.BK | 0.015603 | 0.022035 | 0.084316 | -0.224856 | 1015602.576445 |
| buy_hold_AOT.BK | 0.000795 | 0.001119 | 0.005067 | -0.182746 | 1000794.673700 |
| ppo | -0.012888 | -0.018096 | -0.108208 | -0.100581 | 987112.489101 |
| momentum | -0.131968 | -0.180650 | -0.900621 | -0.152156 | 868032.024449 |

Previous full-period synthetic comparison:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/algo_compare_2852
```

Previous verified PPO-only run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/run_2850
```

Run `2850` used the 5-ticker synthetic technical config:

- `ADVANC.BK`
- `AOT.BK`
- `CPALL.BK`
- `KBANK.BK`
- `PTT.BK`

Feature period:

```text
2021-01-01 to 2024-12-31
```

Main comparison from `run_2850`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| equal_weight | 1.359177 | 0.238027 | 1.438354 | -0.145711 | 2359177.198539 |
| ppo | 0.529157 | 0.111438 | 0.577816 | -0.200052 | 1529156.907643 |
| random | 0.277048 | 0.062725 | 0.358815 | -0.215310 | 1277047.942745 |
| momentum | -0.015674 | -0.003922 | -0.018431 | -0.336980 | 984326.057598 |

## Current Blockers

- `[!]` Local Python is missing `numpy` and `pytest`, so local validation is limited to syntax compilation.
- `[!]` Real OHLCV starter results exist, but they use only five `.BK` tickers and should be treated as pilot results.

## Immediate Next Actions

1. Replace the static five-ticker pilot sector mapping with a broader, licensed sector-index or constituent source.
2. Expand real data sources: official macro releases, licensed fundamentals, and sentiment.
3. Run broader ablations once real sector, official macro, sentiment, and licensed fundamentals are available.
