# Final Methodology And Results

Last updated: 2026-05-31

## Objective

This project builds a reproducible pilot pipeline for deep reinforcement learning portfolio allocation in Thai stocks. The current implementation focuses on a five-stock Thai OHLCV starter universe and a synthetic pilot universe, with the same environment, metrics, and chronological train/validation/test split logic used for both.

The current results are pilot results. They are useful for validating the experimental workflow, but they should not be interpreted as a complete Thai-market performance claim because only starter SET index, sector-context, daily macro-proxy, and Yahoo statement fundamental features are integrated with overlapping history. Sentiment and official macro ingestion/merge scaffolding are implemented, but the available Yahoo latest-news probe and BOT web-table probe do not overlap the 2021-2024 modeling window. Historical official macro releases, licensed fundamentals, historical sentiment, and a broader Thai universe remain incomplete.

## Data

The real-data starter run uses Yahoo Finance OHLCV data for:

- `ADVANC.BK`
- `AOT.BK`
- `CPALL.BK`
- `KBANK.BK`
- `PTT.BK`

The real feature table is:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/data/processed/features_real_ohlcv.parquet
```

It contains 4840 rows, 5 tickers, and 21 columns from 2021-01-04 to 2024-12-30. The chronological split used by the latest verified real comparison was:

| split | start_date | end_date | rows | unique_dates | tickers |
| --- | --- | --- | ---: | ---: | ---: |
| train | 2021-01-04 | 2023-05-29 | 2900 | 580 | 5 |
| validation | 2023-05-30 | 2024-03-12 | 970 | 194 | 5 |
| test | 2024-03-13 | 2024-12-30 | 970 | 194 | 5 |

The generated data-quality report is written to:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/reports/data_quality/real_ohlcv_data_quality.md
```

## Features

The active technical feature set includes 1-day, 5-day, and 20-day returns; 20-day volatility; 10-day and 30-day moving-average ratios; RSI; MACD; Bollinger z-score; volume ratio; and ATR. The expanded pilot feature tables add SET index features, sector-context features, daily macro-proxy features, and lagged Yahoo statement fundamentals. Sentiment and official macro feature tables are also buildable through no-lookahead merge scaffolds, but current Yahoo latest-news and BOT web-table rows are outside the historical validation window.

## Environment

The trading environment is a Gymnasium-compatible long-only portfolio allocator. Actions are continuous portfolio weights over the stock universe. The environment applies transaction costs, tracks cash, portfolio value, net return, turnover, drawdown, and per-ticker weights, and exposes a fixed lookback observation tensor.

The reward combines portfolio return with turnover and drawdown penalties:

```text
reward = return_weight * return - turnover_penalty * turnover - drawdown_penalty * drawdown_penalty
```

The current real-data configs use `return_weight=1.0`, `turnover_penalty=0.05`, and `drawdown_penalty=0.1`.

## Models And Baselines

Implemented DRL models:

- PPO
- A2C
- Support paths for DDPG, TD3, and SAC

Implemented baselines:

- Equal weight
- Random allocation
- Momentum
- Buy-and-hold per ticker
- Moving-average crossover
- Mean-variance

Hyperparameter search uses SLURM job arrays and the trial runner in `src/experiments/run_trial.py`. The search space covers learning rate, gamma, GAE lambda, entropy coefficient, rollout length, PPO clip range, and PPO batch size.

## Verified Runs

| purpose | HPC path | status |
| --- | --- | --- |
| Synthetic split-aware comparison | `/lustrefs/project/25sfcs03/drl_thai_stock/results/algo_compare_2853` | Valid |
| Real OHLCV PPO/A2C comparison | `/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_compare_2855` | Valid |
| Optuna sanity array | `/lustrefs/project/25sfcs03/drl_thai_stock/results/optuna_sanity_2856` | Valid |
| Fixed Optuna search | `/lustrefs/project/25sfcs03/drl_thai_stock/results/optuna_search_fixed_2863` | Valid |
| Longer best-parameter PPO run | `/lustrefs/project/25sfcs03/drl_thai_stock/results/best_ppo_real_ohlcv_2868` | Valid |
| Real OHLCV comparison | `/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_compare_2854` | Invalidated |
| Optuna search | `/lustrefs/project/25sfcs03/drl_thai_stock/results/optuna_search_2858` | Invalidated |

Run `2854` was invalidated because an earlier training entrypoint regenerated synthetic data into real-data paths. Run `2858` was invalidated because the pre-fix sampler repeated parameters across repeated algorithm trials.

## Synthetic Pilot Results

The latest valid split-aware synthetic comparison is `algo_compare_2853`.

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_KBANK.BK | 0.580296 | 0.904531 | 3.218974 | -0.143165 | 1580295.743752 |
| ma_crossover | 0.257937 | 0.381344 | 1.880703 | -0.091600 | 1257936.712797 |
| equal_weight | 0.152812 | 0.221644 | 1.397638 | -0.095300 | 1152811.724538 |
| a2c | 0.167278 | 0.243282 | 1.372647 | -0.103622 | 1167278.283851 |
| mean_variance | 0.122426 | 0.176558 | 0.860456 | -0.109790 | 1122426.235382 |
| ppo | -0.012888 | -0.018096 | -0.108208 | -0.100581 | 987112.489101 |

The synthetic result confirms that the pipeline can run end-to-end, but the synthetic generator creates simplified market structure and should not be used for market conclusions.

## Real OHLCV Validation Results

The latest valid real-OHLCV comparison is `real_ohlcv_compare_2855`.

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 | 1043523.041075 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 | 1009998.557967 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 | 969426.916197 |
| a2c | -0.038061 | -0.057884 | -0.434028 | -0.120042 | 961938.712210 |
| ppo | -0.035900 | -0.054629 | -0.439532 | -0.066219 | 964100.020195 |
| mean_variance | -0.099682 | -0.149008 | -1.097376 | -0.166737 | 900318.191064 |

On this validation split, the best policy was buy-and-hold `PTT.BK`. PPO and A2C underperformed equal weight and the strongest single-stock buy-and-hold baselines. PPO did have a smaller max drawdown than A2C and equal weight in the comparison run.

## Hyperparameter Search Results

The fixed five-task Optuna search is `optuna_search_fixed_2863`.

| trial_id | algorithm | objective_sharpe | cumulative_return | max_drawdown | learning_rate | gamma | n_steps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | ppo | 0.279954 | 0.020408 | -0.068810 | 0.000389 | 0.98 | 256 |
| 0 | ppo | -0.305640 | -0.002649 | -0.008809 | 0.000017 | 0.90 | 512 |
| 1 | a2c | -0.519583 | -0.042686 | -0.127149 | 0.001210 | 0.99 | 256 |
| 2 | ppo | -1.079920 | -0.088712 | -0.115319 | 0.002160 | 0.95 | 256 |
| 3 | a2c | -1.708120 | -0.003138 | -0.004297 | 0.000029 | 0.90 | 128 |

The best short search trial was PPO trial `4`. A longer 20000-timestep run using those parameters produced:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| ppo | -0.030269 | -0.046132 | -0.403028 | -0.099395 | 969730.884751 |

This means the tuned PPO parameters are not yet robust. The short trial improved validation Sharpe, but the longer run regressed and should be treated as evidence that more tuning, walk-forward validation, and broader data are needed before making a model-performance claim.

## Walk-Forward Extension

Walk-forward window generation is now implemented in `src/data/splits.py`. The starter real-OHLCV config uses 504 training dates, 126 validation dates, 126 test dates, and a 126-date step size. This creates rolling windows for future one-job-per-window SLURM runs and out-of-sample aggregation.

The first walk-forward pilot is `results/walk_forward_pilot_2869`. It ran two 1000-timestep PPO jobs through a SLURM array, one task for each generated window. Both error logs were empty, and aggregation wrote:

- `results/walk_forward_pilot_2869/aggregate/walk_forward_all_metrics.csv`
- `results/walk_forward_pilot_2869/aggregate/walk_forward_policy_summary.csv`
- `results/walk_forward_pilot_2869/aggregate/walk_forward_test_equity_curves.csv`

Mean test-window results from the pilot:

| policy | windows | mean_cumulative_return | mean_annualized_return | mean_sharpe | mean_max_drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_ADVANC.BK | 2 | 0.083927 | 0.281661 | 1.467928 | -0.064073 |
| buy_hold_KBANK.BK | 2 | 0.064203 | 0.202352 | 1.157021 | -0.096110 |
| buy_hold_PTT.BK | 2 | 0.040690 | 0.111308 | 0.672495 | -0.081828 |
| equal_weight | 2 | 0.021133 | 0.066667 | 0.633491 | -0.068208 |
| ppo | 2 | 0.004260 | 0.011669 | 0.057644 | -0.026071 |

## Ablation Pilot

The first real-OHLCV ablation pilot is `results/ablation_pilot_2871`. It ran three short 1000-timestep PPO ablations through a SLURM array:

- `config/ablation_returns_real_ohlcv.yaml`
- `config/ablation_technical_real_ohlcv.yaml`
- `config/ablation_technical_context_real_ohlcv.yaml`

Aggregated outputs are:

- `results/ablation_pilot_2871/aggregate/ablation_metrics.csv`
- `results/ablation_pilot_2871/aggregate/ablation_best_policy_summary.csv`
- `results/ablation_pilot_2871/aggregate/ablation_ppo_summary.csv`
- `results/ablation_pilot_2871/aggregate/ablation_ppo_sharpe.png`

PPO ablation summary:

| feature_group | cumulative_return | annualized_return | sharpe | max_drawdown |
| --- | ---: | ---: | ---: | ---: |
| ablation_returns_real_ohlcv | -0.010482 | -0.016061 | -0.453371 | -0.040127 |
| ablation_technical_context_real_ohlcv | -0.055886 | -0.084574 | -0.823513 | -0.099160 |
| ablation_technical_real_ohlcv | -0.078913 | -0.118657 | -1.611158 | -0.109727 |

The best policy in each pilot ablation comparison remained `buy_hold_PTT.BK`, with validation Sharpe `0.405480`. This means the ablation machinery is now verified, but stronger claims should wait for broader sector, official macro, sentiment, and fundamentals features.

## Regime And Stress Tests

The first real-OHLCV regime/stress test is `results/regime_tests_real_ohlcv_20260531`. It evaluates baselines plus the saved PPO/A2C models from `results/real_ohlcv_compare_2855` across named slices and two auto-selected 63-trading-date high-volatility windows.

The acute 2020 COVID crash is explicitly recorded as unavailable because the current starter OHLCV table begins on 2021-01-04. The available COVID-related slice is therefore `covid_recovery_2021`.

Outputs are:

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

## SET Market-Context Features

The first verified market-context feature extension uses Yahoo Finance symbol `^SET.BK` for the SET index. The ingestion output is `data/raw/prices_market_indices.csv`, with 965 rows from 2021-01-04 to 2024-12-30 and no missing values. The merged feature table is `data/processed/features_real_ohlcv_market.parquet`, with 4,840 stock rows and 26 columns.

The merged table adds:

- `set_return_1d`
- `set_return_5d`
- `set_return_20d`
- `set_volatility_20d`
- `set_ma_ratio_20`

It also replaces the placeholder `market_return_1d` with the SET index daily return. The market-context data-quality report is `reports/data_quality_market/real_ohlcv_data_quality.md`, and the first short 1000-timestep validation run is `results/real_ohlcv_market_pilot_20260531`.

| policy | cumulative_return | annualized_return | sharpe | max_drawdown |
| --- | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 |
| ppo | -0.054013 | -0.081783 | -0.952524 | -0.104711 |

This verifies the real SET-index feature path end to end. It does not yet satisfy sector-relative features because the Yahoo SET50/SET100 candidates probed from HPC did not return usable rows; a more reliable sector-index source is still needed.

## Sector Context Features

Because Yahoo SET50/SET100 and sector-index candidates were not reliable through the HPC yfinance check, the first sector extension uses a static pilot mapping from SET public listed-company and index constituent classifications. The mapping file is `data/reference/sector_mapping_thai_pilot.csv` and covers the five starter tickers:

| ticker | sector |
| --- | --- |
| ADVANC.BK | Information & Communication Technology |
| AOT.BK | Transportation & Logistics |
| CPALL.BK | Commerce |
| KBANK.BK | Banking |
| PTT.BK | Energy & Utilities |

The sector feature builder writes `data/processed/features_real_ohlcv_sector.parquet`, with 4,840 stock rows and 38 columns. The sector data-quality report is `reports/data_quality_sector/real_ohlcv_data_quality.md`, with complete panel coverage and zero missing values. Added numeric training columns are sector equal-weight return, sector-relative return, sector peer count, and sector one-hot flags.

The first short 1000-timestep validation run is `results/real_ohlcv_sector_pilot_20260531`.

| policy | cumulative_return | annualized_return | sharpe | max_drawdown |
| --- | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 |
| ppo | -0.031097 | -0.047383 | -0.626349 | -0.068484 |

The sector-context PPO pilot improves over the SET-market-context PPO pilot on cumulative return, Sharpe, and max drawdown, but it still does not beat the PTT buy-and-hold baseline.

## Macro Proxy Features

The first macro extension uses daily Yahoo Finance proxy series rather than official release-date macroeconomic data. The raw macro proxy file is `data/raw/prices_macro_yahoo.csv`, with 5,059 rows across:

- `usdthb`: `USDTHB=X`
- `brent`: `BZ=F`
- `wti`: `CL=F`
- `gold`: `GC=F`
- `us10y`: `^TNX`

The merge uses forward fill from each proxy's latest available observation to each Thai stock trading date, so no future macro proxy observations are used. The macro feature table is `data/processed/features_real_ohlcv_macro.parquet`, with 4,840 stock rows and 68 columns. The data-quality report is `reports/data_quality_macro/real_ohlcv_data_quality.md`, with complete panel coverage and zero missing values.

The first short 1000-timestep validation run is `results/real_ohlcv_macro_pilot_20260531`.

| policy | cumulative_return | annualized_return | sharpe | max_drawdown |
| --- | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 |
| ppo | -0.022360 | -0.034152 | -0.344318 | -0.085930 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 |

The macro-proxy PPO pilot improves over the sector-context PPO pilot and beats equal weight on this short validation run. It still does not beat the PTT or ADVANC buy-and-hold baselines, so it remains a pipeline validation result rather than a model-performance claim.

## Fundamentals Features

The first fundamentals extension uses Yahoo Finance annual and quarterly financial statements. Quarterly statements are preferred when both annual and quarterly rows exist for the same ticker, period, and metric; annual statements provide older fallback coverage. Features are available only after a 60-day reporting lag.

The raw fundamentals file is `data/raw/fundamentals_yahoo_quarterly.csv`, with 8,701 statement rows across five tickers and 13 period ends. The merged feature table is `data/processed/features_real_ohlcv_fundamentals.parquet`, with 4,840 stock rows and 82 columns. The data-quality report is `reports/data_quality_fundamentals/real_ohlcv_data_quality.md`. Numeric fundamental features are zero-filled before the first lagged report; `period_end` and `effective_date` are missing for 27.7% of rows before coverage begins.

The first short 1000-timestep validation run is `results/real_ohlcv_fundamentals_pilot_20260531`.

| policy | cumulative_return | annualized_return | sharpe | max_drawdown |
| --- | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 |
| ppo | -0.025996 | -0.039666 | -0.416967 | -0.067777 |

The fundamentals PPO pilot has lower max drawdown than equal weight and nearly identical Sharpe, but it still does not beat the PTT or ADVANC buy-and-hold baselines. It also underperforms the macro-proxy-only pilot, so these fundamentals should be treated as a pipeline validation source until a broader licensed history is available.

## Official Macro Scaffold

The official macro extension now has a Bank of Thailand source probe and release-date-aware feature merge:

- Raw BOT probe: `data/raw/bot_official_macro.csv`
- Merged feature table: `data/processed/features_real_ohlcv_official_macro.parquet`
- Data-quality report: `reports/data_quality_official_macro/real_ohlcv_data_quality.md`

The source probe targets the BOTWEBSTAT `EC_EI_002_S2` Leading Economic Indicator table. Monthly observations are assigned release dates using a last-business-day-of-following-month rule, then merged backward-only to stock trading dates. On the current HPC probe, the public web table returned 54 rows across nine indicators, dated 2025-11-01 to 2026-04-01. The modeling table spans 2021-01-04 to 2024-12-30, so there is no overlap. The merged 4,840-row official macro feature table therefore has BOT macro training columns equal to zero throughout the validation period.

Because of that date mismatch, no official macro model-performance claim is made from this probe. The code path is verified by 40 passing HPC tests, but a historical BOT export/API source or downloaded historical file is still required before running a valid official macro ablation.

## Sentiment Scaffold

The sentiment extension now has source probing, daily scoring, and feature merging paths:

- Raw latest-news probe: `data/raw/news_yahoo_latest.csv`
- Daily sentiment table: `data/processed/sentiment_daily.parquet`
- Merged feature table: `data/processed/features_real_ohlcv_sentiment.parquet`
- Data-quality report: `reports/data_quality_sentiment/real_ohlcv_data_quality.md`

The merge is backward-only by ticker/date with a seven-day maximum news age. This prevents future news from being used by earlier stock rows. On the current HPC probe, Yahoo Finance returned 11 latest-news rows across four starter tickers, dated 2025-06-24 to 2026-03-31. The modeling feature table spans 2021-01-04 to 2024-12-30, so there is no overlap. The merged 4,840-row sentiment feature table therefore has `sentiment_score`, `news_count`, and `sentiment_news_age_days` equal to zero throughout the validation period.

Because of that date mismatch, no sentiment model-performance claim is made from this source. The code path is verified by 38 passing HPC tests, but a licensed or otherwise accessible historical Thai news/disclosure source is still required before running a valid sentiment ablation.

## Current Conclusion

The implementation is now strong enough for a reproducible pilot: ingestion, feature building, leakage checks, chronological splits, DRL training, baselines, plots, hyperparameter arrays, result aggregation, sentiment merge scaffolding, and official macro release-date merge scaffolding all run on BistKA. The current real-data results do not show PPO or A2C beating the strongest simple buy-and-hold baseline. This is a useful research outcome because it sets a realistic baseline and shows that the next work should focus on data coverage, multi-source features, and walk-forward validation rather than only increasing training time.

## Remaining Work

- Expand the universe beyond five starter `.BK` tickers.
- Replace pilot sector mapping with a broader, licensed sector-index or constituent source.
- Add historical official macro data with release-date alignment.
- Replace Yahoo statement fundamentals with licensed SET/SETSMART fundamentals where possible.
- Add historical real news/sentiment features with coverage inside the modeling window.
- Extend walk-forward windows after adding more history and features.
- Run a larger hyperparameter search after the feature set is expanded.
- Add ablation experiments for price-only, technical, index, macro, sentiment, and all-feature configs.
