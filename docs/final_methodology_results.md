# Final Methodology And Results

Last updated: 2026-11-27. The May 2026 pilot write-up is kept unchanged as an appendix at the end.

This is the short account of what was built, how it was evaluated and what it found. The numbers come from the dated reports linked in each section. The report chapter `docs/report_drafts/results_title_alignment_20261106.tex` has the same results with tables and figures, and [the research-question answers](research_question_answers_20261111.md) take each question in turn.

**Evidence tier: exploratory.** Prices, the market series and the macro proxies come from Yahoo Finance, not the licensed SETSMART package the project requested. The SET50 membership is official.

## Question

Can a deep reinforcement learning agent with a long-term objective and several data sources earn more over years than simple benchmarks, after costs, on the Thai stock market?

## Data and universes

- **Five starter stocks** (ADVANC, AOT, CPALL, KBANK, PTT), 2012–2024 for development. They were chosen with hindsight: today's large caps.
- **Point-in-time SET50:** on every session, the official SET50 members of that session (archive 2012–2024), restricted to members with Yahoo prices. Between 39 and 48 members are tradable a session; 13% of member-sessions have no prices, mostly delisted or merged names ([universe](point_in_time_universe_2012_2024_20261107.md)).
- **Market series:** the SET50 ETF (TDEX), also the investable benchmark.
- **Other sources tried:** SET50 market context, the Bank of Thailand policy rate (point-in-time decisions), daily macro proxies (USD/THB, oil, gold, US 10-year yield). Fundamentals, news sentiment and investor flow are built as code paths but lack historical data inside the modelling window, so they are not in any result.
- **Splits:** validation years 2016–2022 as seven walk-forward folds; test years 2023–2024; a sealed holdout from 2025 to 4 November 2026, written and hashed at download and opened once.

## Method

- **Environment:** daily data, weekly decisions, long-only weights with cash, 0.15% commission plus 0.05% slippage per side, one-session information lag ([environment and reward](long_term_reward_experiment_20261106.md)).
- **Reward ("plan" reward):** log wealth growth, plus a small differential-Sharpe term, minus half of each new drawdown. It replaced the one-day reward of the earlier pilot, which overtraded.
- **Policy:** PPO with a per-asset LSTM encoder. On the point-in-time universe, a shared-weight actor scores every member with the same weights, so the slot order does not matter.
- **Variants on the point-in-time universe:** sticky slots; the shared-weight actor; a seed ensemble (the mean of the seeds' target weights); a 0.5-point no-trade band; training with the band; the policy rate; the macro proxies; and the cash weight capped at 20%.
- **Algorithms (five stocks):** PPO, A2C, SAC, TD3 and DDPG with the same reward, model and inputs.
- **Benchmarks:** equal weight of the same universe (monthly), the SET50 ETF, momentum, inverse volatility, minimum variance, and ridge and LSTM forecast-then-allocate strategies.
- **Statistics:** Sharpe ratios on chained daily returns, paired stationary-bootstrap 95% intervals, and the deflated Sharpe ratio for the number of designs tried ([formulas](evaluation_metric_formulas_20261110.md)).
- **Choosing the design:** a rule fixed before the test years: the median over the validation years of the annualised log return above equal weight of the same universe, averaged over seeds. If no design scores above zero, it selects equal weight ([decision](final_design_decision_20261125.md)).
- **Single evaluations:** the chosen designs, their seeds and statistics were fixed in lock files approved before each run. Run-once gates refuse a changed config, uncommitted code or a second run, and write receipts and a ledger: `scripts/run_final_test_evaluation.py` (test years) and `scripts/run_holdout_evaluation.py` (holdout).

## Results on the validation years (2016–2022)

| Design or benchmark | Universe | Annual return | Note |
| --- | --- | ---: | --- |
| PPO, plan reward, 100k timesteps | Five stocks | 7.6% | five seeds |
| A2C, 100k / 300k | Five stocks | 7.5% / 8.7% | best algorithm; not significantly better than PPO |
| PPO, 300k | Five stocks | 4.4% | held 39% cash; significantly worse than at 100k |
| SAC, TD3, DDPG, 300k | Five stocks | 0.3%, −0.9%, −1.4% | about 30× turnover a year |
| Equal weight / ETF | Five stocks | 10.1% / 7.1% | |
| DRL, slots by traded value | Point-in-time | −1.1% | overtraded as the slot order changed |
| DRL, sticky slots | Point-in-time | 3.7% | |
| DRL, shared-weight actor (5 seeds) | Point-in-time | 4.5% | |
| Seed ensemble + band (5 seeds) | Point-in-time | 5.2% | 5.5% with three seeds |
| **Cash-capped ensemble + band (locked)** | Point-in-time | **5.0%** | same with five seeds; 4.9% at 300k |
| Shared actor at 300k, ensemble + band | Point-in-time | 3.9% | 41% cash, smaller drawdown |
| Equal weight / ETF | Point-in-time | 5.6% / 7.1% | |
| Momentum, inverse volatility, minimum variance | Point-in-time | 5.3%, 5.0%, 4.0% | |

- **Survivorship:** equal weight over today's SET50 names made 12.2% a year over the same years, against 5.6% over each session's actual members ([report](pit_walk_forward_20261107.md)).
- **What the point-in-time agent learned:** close to equal weight (daily correlation 0.998 for the seed ensemble).
- **Sources:** without a cash cap, adding the policy rate or the macro proxies lowered the point-in-time return; with the locked design's cap and five seeds, no input set changed it measurably (below). Only SET50 market context ever helped, borderline and on five stocks ([ablation](source_ablation_20261106.md)).
- **Training budget:** 300k timesteps never beat 100k significantly. Without a cap, PPO drifted to cash; with the 20% cap, 10 of 35 models still sat near it ([300k](bistka_300k_results_20261125.md), [cash cap](cash_cap_5seeds_300k_20261126.md)).
- **Selection:** no DRL design scored above zero against equal weight in either universe, so the rule as written selects equal weight. The top DRL designs were then evaluated anyway: the point-in-time cash-capped ensemble with the band (score −0.01) and five-stock A2C at 300k (−0.16).

## Results on the test years (2023–2024) and the sealed holdout (2025 to 4 November 2026)

| Policy | 2023 | 2024 | 2025 | 2026 to 4 Nov | Test, chained | Holdout, chained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Point-in-time DRL (locked)** | −11.4% | −6.7% | −8.8% | +23.7% | 0.827 | 1.128 |
| Point-in-time equal weight | −13.5% | −7.6% | −8.8% | +29.7% | 0.799 | 1.183 |
| **Five-stock A2C, 300k** | −1.9% | +4.4% | +0.7% | +14.3% | 1.026 | 1.153 |
| Five-stock equal weight | −3.6% | +10.1% | +6.0% | +20.8% | 1.061 | 1.281 |
| SET50 ETF | −9.1% | +8.2% | −2.8% | +24.9% | 0.984 | 1.214 |

- **Test years** ([results](final_test_results_20261125.md)): the point-in-time design was ahead of equal weight in both years (Sharpe difference +0.06, −0.06 to +0.17, not significant) and behind the ETF. Five-stock A2C trailed equal weight. Yahoo's 2024 prices turned out to have stale rows on 17.5% of stock rows; the results stand as locked and the defect is disclosed.
- **Holdout** ([results](holdout_results_20261126.md)): the point-in-time design matched equal weight in 2025 and trailed it in 2026, holding up to its 20% cash cap through a rally. Its Sharpe shortfall against equal weight is significant: −0.12 (−0.23 to −0.01). Five-stock A2C trailed equal weight and the ETF, not significantly. The point-in-time stage's first attempt stopped at loading before any training and was rerun once with the authors' approval. Stale prices are rare from 2025 on (0.2–1.8% of rows).

## Checks after the holdout (validation years only, 2026-11-27)

These used the validation years 2016–2022 only, after both single evaluations, so they cannot change the locked design or its results.

| Design (5 seeds, 100k) | Single models | Ensemble + band | Single-model Sharpe difference (95% CI) |
| --- | ---: | ---: | --- |
| Locked: LSTM encoder, technical + ETF market inputs | 4.6% | 5.0% | — |
| MLP encoder | 4.4% | 5.1% | −0.006 vs locked (−0.040 to +0.022) |
| Technical inputs only | 4.5% | 5.2% | −0.001 vs locked (−0.028 to +0.028) |
| + policy rate | 4.6% | 5.2% | +0.009 vs technical only (−0.027 to +0.048) |
| + macro proxies | 4.4% | 5.0% | −0.011 vs technical only (−0.051 to +0.033) |
| + policy rate + macro proxies | 4.6% | 5.5% | +0.019 vs technical only (−0.027 to +0.062) |

- **Encoder** ([report](architecture_ablation_20261127.md)): the MLP did as well as the LSTM and trained about seven times faster.
- **Sources** ([report](source_ablation_20261127.md)): no source changed the result measurably. The all-sources ensemble (5.5%, Sharpe 0.41) is level with equal weight and not significantly ahead of the technical-only ensemble; its lead came from cash.
- **Retrain on BistKA:** the same seed gave a different model with the same result (4.8% against 5.0% a year); inference carries across machines to within 4 × 10⁻⁸.

## Conclusion

The long-term reward fixed the overtrading of the earlier one-day objective, and on the point-in-time SET50 the agent came to hold close to equal weight at lower turnover. No design beat equal weight of its own universe on the validation years. The design chosen by a fixed rule was slightly ahead of equal weight in the test years and significantly behind it, by Sharpe ratio, on the sealed holdout. Neither DRL design beat the SET50 ETF on years it had not seen. Neither extra source tried on the point-in-time universe (the policy rate, the macro proxies) helped, with or without the cash cap, and a simpler encoder did as well as the LSTM. The evidence does not support a claim that this DRL design improves long-term profit in the Thai stock market. It does support the protocol: a survivorship-aware universe, walk-forward folds, a selection rule fixed in advance, and locked single evaluations that report a negative result as it came.

## Limitations

- **Data tier:** Yahoo prices carry no trading-status or corporate-action records; 13% of point-in-time member-sessions have no prices; 2024 has stale rows.
- **Membership after 2024:** the holdout holds the 2024 H2 SET50 list fixed, so the 2025–2026 index revisions are ignored.
- **Costs:** 0.20% per side, with no market impact, spreads, board lots or price limits.
- **Sources:** fundamentals, news sentiment and investor flow have no historical data in the modelling window, so the multi-source claim rests on price, technical, market, macro-proxy and policy-rate inputs.
- **Selection:** many designs were tried on the validation years; the deflated Sharpe corrects for this only in part.
- **Budget and settings:** most models trained for 100k timesteps; the algorithms' settings were not tuned.

## Reproducing

[`reproduce_20261111.md`](reproduce_20261111.md) gives one command block per stage, including the HPC runs, the design choice, the test years and the holdout. The slim archive (`scripts/build_slim_results_archive.py`; latest `slim_archive_20261127.tar.gz`, 74 run folders) holds the configs, receipts and metrics without trained models or price files.

## What remains

- Licensed SETSMART data (prices with trading status, corporate actions, fundamentals) and a historical news source, then the multi-source ablation on them.
- Investor-flow features from SET's daily trading value by investor type.
- The authors' choice of title wording ([title check](title_check_draft_20261110.md)), the advisor's review of the locks and decisions, and the Beamer deck and posters (the chapter is in `kvis_final_research_report.tex`).
- Any new claim needs years this work has not used.

---

# Appendix: the May 2026 pilot (as written on 2026-05-31)

The text below is the earlier version of this document, unchanged except that its headings are one level lower. It describes the five-stock pilot and the data scaffolds before the work above.

### Objective

This project builds a reproducible pilot pipeline for deep reinforcement learning portfolio allocation in Thai stocks. The current implementation focuses on a five-stock Thai OHLCV starter universe and a synthetic pilot universe, with the same environment, metrics, and chronological train/validation/test split logic used for both.

The current results are pilot results. They are useful for validating the experimental workflow, but they should not be interpreted as a complete Thai-market performance claim because only starter SET index, sector-context, daily macro-proxy, and Yahoo statement fundamental features are integrated with overlapping history. Sentiment and official macro ingestion/merge scaffolding are implemented, but the available Yahoo latest-news probe and BOT web-table probe do not overlap the 2021-2024 modeling window. Historical official macro releases, licensed fundamentals, historical sentiment, and a broader Thai universe remain incomplete.

### Data

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

### Features

The active technical feature set includes 1-day, 5-day, and 20-day returns; 20-day volatility; 10-day and 30-day moving-average ratios; RSI; MACD; Bollinger z-score; volume ratio; and ATR. The expanded pilot feature tables add SET index features, sector-context features, daily macro-proxy features, and lagged Yahoo statement fundamentals. Sentiment and official macro feature tables are also buildable through no-lookahead merge scaffolds, but current Yahoo latest-news and BOT web-table rows are outside the historical validation window.

### Environment

The trading environment is a Gymnasium-compatible long-only portfolio allocator. Actions are continuous portfolio weights over the stock universe. The environment applies transaction costs, tracks cash, portfolio value, net return, turnover, drawdown, and per-ticker weights, and exposes a fixed lookback observation tensor.

The reward combines portfolio return with turnover and drawdown penalties:

```text
reward = return_weight * return - turnover_penalty * turnover - drawdown_penalty * drawdown_penalty
```

The current real-data configs use `return_weight=1.0`, `turnover_penalty=0.05`, and `drawdown_penalty=0.1`.

### Models And Baselines

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

### Verified Runs

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

### Synthetic Pilot Results

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

### Real OHLCV Validation Results

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

### Hyperparameter Search Results

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

### Walk-Forward Extension

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

### Ablation Pilot

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

### Regime And Stress Tests

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

### SET Market-Context Features

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

### Sector Context Features

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

### Macro Proxy Features

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

### Fundamentals Features

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

### Official Macro Scaffold

The official macro extension now has a Bank of Thailand source probe and release-date-aware feature merge:

- Raw BOT probe: `data/raw/bot_official_macro.csv`
- Merged feature table: `data/processed/features_real_ohlcv_official_macro.parquet`
- Data-quality report: `reports/data_quality_official_macro/real_ohlcv_data_quality.md`

The source probe targets the BOTWEBSTAT `EC_EI_002_S2` Leading Economic Indicator table. Monthly observations are assigned release dates using a last-business-day-of-following-month rule, then merged backward-only to stock trading dates. On the current HPC probe, the public web table returned 54 rows across nine indicators, dated 2025-11-01 to 2026-04-01. The modeling table spans 2021-01-04 to 2024-12-30, so there is no overlap. The merged 4,840-row official macro feature table therefore has BOT macro training columns equal to zero throughout the validation period.

Because of that date mismatch, no official macro model-performance claim is made from this probe. The code path is verified by 40 passing HPC tests, but a historical BOT export/API source or downloaded historical file is still required before running a valid official macro ablation.

### Sentiment Scaffold

The sentiment extension now has source probing, daily scoring, and feature merging paths:

- Raw latest-news probe: `data/raw/news_yahoo_latest.csv`
- Daily sentiment table: `data/processed/sentiment_daily.parquet`
- Merged feature table: `data/processed/features_real_ohlcv_sentiment.parquet`
- Data-quality report: `reports/data_quality_sentiment/real_ohlcv_data_quality.md`

The merge is backward-only by ticker/date with a seven-day maximum news age. This prevents future news from being used by earlier stock rows. On the current HPC probe, Yahoo Finance returned 11 latest-news rows across four starter tickers, dated 2025-06-24 to 2026-03-31. The modeling feature table spans 2021-01-04 to 2024-12-30, so there is no overlap. The merged 4,840-row sentiment feature table therefore has `sentiment_score`, `news_count`, and `sentiment_news_age_days` equal to zero throughout the validation period.

Because of that date mismatch, no sentiment model-performance claim is made from this source. The code path is verified by 38 passing HPC tests, but a licensed or otherwise accessible historical Thai news/disclosure source is still required before running a valid sentiment ablation.

### Current Conclusion

The implementation is now strong enough for a reproducible pilot: ingestion, feature building, leakage checks, chronological splits, DRL training, baselines, plots, hyperparameter arrays, result aggregation, sentiment merge scaffolding, and official macro release-date merge scaffolding all run on BistKA. The current real-data results do not show PPO or A2C beating the strongest simple buy-and-hold baseline. This is a useful research outcome because it sets a realistic baseline and shows that the next work should focus on data coverage, multi-source features, and walk-forward validation rather than only increasing training time.

### Remaining Work

- Expand the universe beyond five starter `.BK` tickers.
- Replace pilot sector mapping with a broader, licensed sector-index or constituent source.
- Add historical official macro data with release-date alignment.
- Replace Yahoo statement fundamentals with licensed SET/SETSMART fundamentals where possible.
- Add historical real news/sentiment features with coverage inside the modeling window.
- Extend walk-forward windows after adding more history and features.
- Run a larger hyperparameter search after the feature set is expanded.
- Add ablation experiments for price-only, technical, index, macro, sentiment, and all-feature configs.