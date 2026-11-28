# Answers to the Research Questions (draft)

Date: 2026-11-11; cash-cap arm added 2026-11-20; BistKA 300k-timestep runs (RQ1, RQ4, RQ6) added 2026-11-25; the single test-year evaluation added 2026-11-25 (evening); the cash-capped design at five seeds and 300k (RQ1, RQ6) and the sealed holdout added 2026-11-26; the checks after the holdout (RQ2, RQ3, RQ6 and the encoder) added 2026-11-27; the sector and price-only arms (RQ2, RQ3) added 2026-11-28
Status: **draft** for the M8 item "answer each research question explicitly, including negative results". It answers the six questions of the implementation plan (`implementation_plan.md`, section 2.2) from the evidence in the repository today. Each answer links the report it comes from.

> **Checks after the holdout, 2026-11-27 (validation years 2016–2022 only; they cannot change the locked design or its results):**
> - **Data sources** ([report](source_ablation_20261127.md); BistKA job 4528, the locked cash-capped design, 5 seeds): technical inputs alone, + the ETF market inputs (the locked design), + the policy rate, + the macro proxies, and all of them. No source changed the Sharpe ratio significantly; against technical inputs alone in the same run, the single models' differences are +0.009 (policy rate), −0.011 (macro) and +0.019 (all), and the ETF inputs add +0.001. The all-sources ensemble with the band made 5.5% a year (Sharpe 0.41), level with equal weight, but not significantly ahead of the technical-only ensemble (+0.038, −0.023 to +0.085); its lead came from cash. Under the cap, the earlier policy-rate shortfall is gone.
> - **Encoder** ([report](architecture_ablation_20261127.md)): an MLP over the 30-session window did as well as the locked LSTM (single models 4.4% against 4.6% a year; ensembles 5.1% against 5.0%; neither difference significant) and trained about seven times faster.
> - **Retrain on BistKA:** the locked design with seed 42 made 4.8% a year there against 5.0% on the Mac (Sharpe 0.347 against 0.351). The models differ (not byte-identical), but by about half as much as two seeds do.

> **Sealed holdout, 2026-11-26:** the same two designs were evaluated once on 2025 to 4 November 2026 ([results](holdout_results_20261126.md)).
> - **Point-in-time design:** it matched equal weight in 2025 (−8.8% each) and trailed it in 2026 (+23.7% against +29.7%), holding up to its 20% cash cap. Chained 1.128 against 1.183 for equal weight and 1.214 for the ETF. Its Sharpe shortfall against equal weight, −0.12 (−0.23 to −0.01), is significant. The first attempt failed at loading before any training and was rerun once with approval.
> - **Five-stock A2C at 300k:** it trailed equal weight of its five stocks (1.153 against 1.281) and the ETF, not significantly.
> - **Data:** stale prices are rare from 2025 on (0.2–1.8% of rows). The point-in-time membership is the 2024 H2 list held fixed.

> **Test years, 2026-11-25:** the two locked designs were evaluated once on 2023–2024 ([results](final_test_results_20261125.md)).
> - **Point-in-time design:** it beat point-in-time equal weight in both years (two years chained 0.827 against 0.799), not significantly. It lost to the SET50 ETF (0.984).
> - **Five-stock A2C at 300k:** it trailed equal weight of its five stocks (1.026 against 1.061), not significantly.
> - **2024 data:** Yahoo's 2024 prices contain stale rows on real trading days, disclosed in the report. Statements below that the test years are unread predate this run.

**Limits that apply to every answer:**
- **Data tier:** exploratory. Prices are Yahoo's, and 13% of point-in-time SET50 member-sessions have none.
- **Years:** validation years (2016–2022) unless a section says otherwise. The 2023–2024 test years and the sealed holdout were each read once (above); nothing below reads them again.
- **Statistics:** every Sharpe comparison is a paired stationary bootstrap on chained daily returns (1,699 sessions); "significant" means its 95% interval excludes zero.

| # | Question (abridged) | Short answer | Evidence |
| --- | --- | --- | --- |
| 1 | Can a continuous-action DRL agent learn profitable long-term allocation for Thai equities? | **Profitable, but no better than simple benchmarks.** | Strong for the negative part |
| 2 | Does multi-source data beat price-only and technical-only agents? | **No measurable gain.** Market-index context helped only on five stocks, borderline. On the point-in-time index, with the cash cap and five seeds, no input set (policy rate, macro proxies, ETF market inputs, or all) changed the result measurably; without the cap the policy rate and macro proxies had lowered it. | Moderate for the negative part, with the data available |
| 3 | Which data sources contribute most? | **None measurably on the point-in-time index;** technical inputs carry the agent. Market-index context helped only on five stocks, borderline; most planned sources have no historical data. | Weak, incomplete |
| 4 | Which DRL algorithm is most stable? | **On-policy (A2C, PPO) over off-policy (SAC, TD3, DDPG),** which overtrade and are significantly worse than the benchmarks; A2C is best, but not significantly better than PPO or the benchmarks. | Moderate (five stocks only) |
| 5 | Does the policy generalize across regimes, including COVID-19? | **Its behaviour is stable across regimes, but it shows no regime skill;** its crash-year gains came from cash that depended on the seed. | Moderate |
| 6 | Can BistKA HPC be used efficiently and reproducibly? | **Yes for training:** one 32-CPU job trained 210 models (1,776 task-hours) in 56 hours for about 2,465 SHr, and BistKA's models replay on the Mac to within 10⁻⁸. A retrain there reproduces the result but not the files. Tuning and sentiment extraction have not run there. | Moderate |

## RQ1. Can a continuous-action DRL agent learn profitable long-term portfolio allocation policies for Thai equities?

**Answer:** in absolute terms, yes; against simple benchmarks, no. No DRL design beat equal weight or the SET50 ETF, and what the best design learned is close to equal weight.

- **Five hand-picked stocks** ([walk-forward](walk_forward_reward_20261106.md); 7 folds × 5 seeds):
  - **DRL:** the long-term agent (plan reward, per-asset LSTM, weekly decisions) made 7.6% a year.
  - **Benchmarks:** the SET50 ETF made 7.1%, equal weight 10.1%, and the LSTM forecaster 13.8%.
  - **Significance:** no difference is significant.
  - **Survivorship:** these five stocks were chosen with hindsight, which flatters every strategy confined to them.
- **Point-in-time SET50**, the actual members of each session ([report](pit_walk_forward_20261107.md)):
  - **First design:** −1.1% a year, significantly worse than the ETF (Sharpe −0.51, 95% CI −0.87 to −0.26) and equal weight (−0.40, −0.58 to −0.28). It overtraded because it scored slot positions rather than stocks.
  - **Best design:** after four follow-ups (sticky slots, a shared-weight actor, a seed ensemble and a no-trade band), it made 5.5% a year, against 5.6% for equal weight and 7.1% for the ETF. Chained over 2016–2022, 1 baht became 1.41, as for equal weight; the ETF reached 1.55.
  - **Sharpe ratios:** 0.42 for the best design, 0.38 for equal weight and 0.50 for the ETF. Neither difference is significant: +0.04 (−0.05 to +0.11) and −0.08 (−0.36 to +0.15).
  - **Deflated Sharpe:** 0.22, counting the 19 designs tried (0.27 at 13, 0.23 at 17).
  - **Training with the band:** 5.1% a year, and 5.2% as a seed ensemble.
  - **Cash capped at 20%:** 4.4% a year, and 5.0% as a seed ensemble with the band ([report](pit_walk_forward_20261107.md#follow-up-cash-capped-at-20-2026-11-19)). It removed the cash-heavy models but not the gap to equal weight, which lies in trading costs and stock tilts that do not pay.
  - **Three times the training, five seeds (BistKA, 2026-11-25):** 3.2% a year, and 3.9% as a seed ensemble with the band ([report](bistka_300k_results_20261125.md)). 17 of the 35 models held 45–98% cash all year, so the return fell while the drawdown shrank (−7.0% against −11.3% a year on average; −12.0% for the 100k ensemble with five seeds, which made 5.2% a year). The ensemble's Sharpe ratio, 0.47, is level with the ETF's 0.50; its deflated Sharpe is 0.25 at 21 designs. On five stocks, PPO trained three times longer also turned to cash and made 4.4% a year against 7.6%, significantly worse. More training did not lift any agent above the benchmarks.
  - **The locked cash-capped design, five seeds and 300k (2026-11-26, validation years only):** with five seeds at 100k its ensemble with the band made 5.0% a year, as with three, and it stayed the best-scoring DRL design under the selection rule (−0.16, still below zero). At 300k the cap held average cash at 8.7% (41% without it), but 10 of 35 models averaged 19% cash or more, near the cap, and the ensemble made 4.9% a year (Sharpe 0.35 against 0.36 at 100k, not significant). 100k is enough for this design ([report](cash_cap_5seeds_300k_20261126.md)).
- **What it learned:** close to equal weight. Its daily correlation with equal weight is 0.998, with 43.9 effective holdings out of 44.5 tradable members.
- **Encoder (checked after the holdout, 2026-11-27):** replacing the locked design's LSTMs with an MLP over the same 30-session window made no measurable difference (single models 4.4% against 4.6% a year, Sharpe difference −0.006, −0.040 to +0.022; ensembles 5.1% against 5.0%) ([report](architecture_ablation_20261127.md)).
- **Other benchmarks on the same universe:** none is significantly better or worse than the agent.
  - **Heuristics:** momentum 5.3%, inverse volatility 5.0% and minimum variance 4.0%.
  - **Forecasters:** ridge 5.7%; LSTM 8.1%, through one rally in late 2020.

## RQ2. Does adding multi-source data improve out-of-sample performance compared with price-only and technical-indicator-only agents?

**Answer:** no measurable gain. Market-index context helped only on five stocks, borderline, and the daily macro proxies did not help there. On the point-in-time index without a cash cap, the official policy rate and the daily macro proxies lowered the return; with the locked design's 20% cash cap and five seeds, no input set changed the result measurably (2026-11-27, below), and neither did the sector (2026-11-28). A price-only agent, which sees only each stock's daily returns, did as well as every multi-source agent (2026-11-28). Fundamentals, news and investor flow have not been tested, because their historical data is missing.

- **Five stocks** ([source ablation](source_ablation_20261106.md); 7 folds × 3 seeds per arm, everything else fixed):

  | Inputs | Mean annual return | Sharpe difference (95% CI) |
  | --- | ---: | --- |
  | Technical only | 6.8% | — |
  | + SET50 market context | 8.2% | +0.10 vs technical only (−0.002 to +0.21; one-sided p = 0.03) |
  | + market + daily macro proxies | 7.1% | −0.08 vs + market (−0.20 to +0.05) |

  The market-context gain is borderline. Its interval touches zero, and three comparisons were made.
- **Price-only arm:** not run on the five stocks, where technical-only is the lowest arm. On the point-in-time SET50 it was run with the cash cap (2026-11-28, below).
- **Point-in-time SET50** ([report](pit_walk_forward_20261107.md#follow-up-bot-policy-rate-as-an-input-2026-11-11); the shared-weight agent, 7 folds × 3 seeds, only the inputs changed):

  | Inputs | Mean annual return | Seed ensemble + band | Ensemble Sharpe difference (95% CI) |
  | --- | ---: | ---: | --- |
  | Technical + SET50 market context | 4.6% | 5.5% | — |
  | + BOT policy rate (5 inputs) | 4.1% | 4.5% | −0.06 vs without it (−0.11 to −0.01) |
  | + daily macro proxies (8 inputs) | 3.9% | 4.3% | −0.08 vs without them (−0.20 to +0.06) |

  - **Policy rate:** the ensemble's shortfall is significant; the single models' is not (−0.02, −0.09 to +0.03).
  - **Why:** the trained models barely use the new inputs. So the shortfall reflects how training turned out with them, chiefly cash that depended on the seed, not information in the rate.
  - **Macro proxies** ([report](pit_walk_forward_20261107.md#follow-up-daily-macro-proxies-as-an-input-2026-11-11)): the shortfall is not significant (single models −0.05, −0.18 to +0.09). But the models read these inputs heavily, and removing their information raised the return of 15 of the 21 models, by 0.5 points a year on average.
  - **Best inputs:** on the point-in-time index, the technical and SET50 market inputs alone.
- **Point-in-time SET50 with the cash cap** ([report](source_ablation_20261127.md); checked after the holdout, 2026-11-27; the locked design, 7 folds × 5 seeds, only the inputs changed; BistKA job 4528 except the locked and sector rows):

  | Inputs | Single models | Seed ensemble + band | Single-model Sharpe difference vs technical only (95% CI) |
  | --- | ---: | ---: | --- |
  | Daily return only (no indicators, no market inputs; trained on the Mac, 2026-11-28) | 4.8% | 5.4% | +0.007 (−0.019 to +0.034) |
  | Technical only | 4.5% | 5.2% | — |
  | + SET50 market context (the locked design, trained on the Mac) | 4.6% | 5.0% | +0.001 (−0.028 to +0.028) |
  | + market + BOT policy rate | 4.6% | 5.2% | +0.009 (−0.027 to +0.048) |
  | + market + daily macro proxies | 4.4% | 5.0% | −0.011 (−0.051 to +0.033) |
  | + market + policy rate + macro proxies | 4.6% | 5.5% | +0.019 (−0.027 to +0.062) |
  | + market + sector (4 stock inputs; trained on the Mac, 2026-11-28) | 4.5% | 5.1% | −0.003 (−0.034 to +0.030) |

  - **No source helps:** none of these differences is significant, and the ensembles' differences against the technical-only ensemble are not significant either (all sources +0.038, −0.023 to +0.085).
  - **All sources, ensemble:** 5.5% a year (Sharpe 0.41), level with equal weight (5.6%, 0.38); its edge came from more cash (12 of its 35 models above 15%), mostly in 2020. It is the best of eight new designs; a lead to test on new years, not a finding.
  - **The policy-rate shortfall is gone under the cap,** which supports the reading that it came from cash that depended on the seed.
  - **Sector** ([report](sector_inputs_20261127.md)): against the locked design on the same platform, −0.005 for the single models (−0.033 to +0.027) and +0.010 for the ensembles (−0.020 to +0.043). Decision-7 puts its ensemble level with the locked one (−0.157 against −0.160 points a year).
  - **Price only** ([report](price_only_20261128.md)): the daily return alone did as well. Against the locked design on the same platform, the Sharpe difference is +0.006 for the single models (−0.013 to +0.028) and +0.016 for the ensembles (−0.007 to +0.046). It stays closer to equal weight, with less cash (3.3% against 5.1%) and less trading, but its ensemble still does not beat equal weight (−0.006, −0.035 to +0.033).

## RQ3. Which data sources contribute most to performance?

**Answer:** none measurably on the point-in-time index. The agent's information comes from the stocks' own price histories, and there the ten technical indicators add nothing over the daily returns (2026-11-28). Market-index context helped only on five stocks, borderline, and added nothing on the point-in-time index (2026-11-27). The sector, tested on 2026-11-28, is read by the models but adds nothing. Fundamentals, news sentiment and investor flow have not been tested, because the historical data is missing.

| Source | Status | Finding |
| --- | --- | --- |
| Technical indicators | Tested (every arm) | The base of every agent; on its own, 6.8% a year on five stocks. On the point-in-time index with the cash cap, an agent with the daily returns only did as well (+0.007 Sharpe over technical inputs for the single models, not significant; 2026-11-28) |
| Market index context (SET50 ETF) | Tested | +1.4 points a year and +0.10 Sharpe on five stocks, borderline. On the point-in-time index with the cash cap: +0.001 Sharpe over technical inputs alone (2026-11-27) |
| Daily macro proxies (USD/THB, oil, gold, US 10-year yield) | Tested on five stocks and the point-in-time index | **With the cash cap and five seeds (2026-11-27):** no measurable effect (−0.011 Sharpe for single models). **Five stocks:** no gain. Shuffling them moves the weights by 2.9%, against 1.9% for market context, and changes the return by 0.25 points or less. **Point-in-time:** 3.9% against 4.6% a year (seed ensemble 4.3% against 5.5%, not significant). The agent leans on them: shuffling moves its weights by 3.8%, against 2.1% for market context and 4.5% for the technical inputs. Removing their information raised the return of 15 of 21 models |
| Official BOT policy rate | Tested on the point-in-time index | With the cash cap and five seeds (2026-11-27): no measurable effect (+0.009 Sharpe for single models, ensemble 5.2% a year as with technical inputs alone). Without the cap, it lowered the return: 4.1% against 4.6% a year, and 4.5% against 5.5% for the seed ensemble with the band (Sharpe −0.06, significant). The agent barely reads it: shuffling it moves the weights by 0.7%, against 0.9% for market context and 3.5% for the technical inputs |
| Financial statements | Not tested | Five stocks and recent years only; no shares outstanding for P/E and P/BV |
| Thai news sentiment | Not tested | The scorer exists; there is no historical news for 2014–2024 |
| Sector context (dated SET sectors) | Tested on the point-in-time index with the cash cap (2026-11-28) | No measurable effect: −0.005 Sharpe for the single models and +0.010 for the ensembles against the locked design. The agent reads it: shuffling the four inputs moves the weights by 2.4%, against 1.5% for market context and 4.1% for the technical inputs. Holding them at their training mean changes the return by +0.06 points a year (19 of 35 models higher) |
| Investor flow | Not tested | The importer exists; there is no data |

## RQ4. Which DRL algorithm is most stable under high noise and limited compute: PPO, A2C, DDPG, TD3 or SAC?

**Answer:** the on-policy algorithms. With the same reward, model and inputs on the five starter stocks, SAC, TD3 and DDPG trade about 30 times their capital a year, lose about 6 points a year to costs, and are significantly worse than the benchmarks. A2C is the best and cheapest algorithm at 300k timesteps, but not significantly better than PPO, the ETF or equal weight. The point-in-time index was run with PPO only.

- **All five algorithms at 300k timesteps × 5 seeds** ([report](bistka_300k_results_20261125.md); BistKA job 4409, 7 folds each):

  | | A2C | PPO | SAC | TD3 | DDPG |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Mean annual return | **8.7%** | 4.4% | 0.3% | −0.9% | −1.4% |
  | Years beating the ETF | 3/7 | 2/7 | 0/7 | 1/7 | 0/7 |
  | Mean maximum drawdown | −13.1% | −9.8% | −15.8% | −17.5% | −16.5% |
  | Costs a year | 0.9 points | 1.3 | 6.1 | 6.6 | 6.3 |
  | Seed standard deviation | 5.9 points | 4.2 | 11.0 | 8.5 | 9.8 |
  | Sharpe (deflated, 14 designs) | 0.53 (0.35) | 0.39 (0.23) | 0.12 (0.08) | −0.03 (0.03) | −0.09 (0.03) |
  | Training per model on BistKA | 0.4 h | 1.1 h | 17.2 h | 10.7 h | 13.9 h |

  - **Benchmarks:** the SET50 ETF made 7.1% (Sharpe 0.50) and monthly equal weight 10.1% (0.58).
  - **Significant:** TD3 and DDPG are worse than the ETF, equal weight and PPO; SAC is worse than equal weight (Sharpe −0.46, −0.90 to −0.05).
  - **Not significant:** A2C against the ETF (+0.03, −0.36 to +0.37), equal weight (−0.05) and PPO (+0.14, −0.04 to +0.34).
  - **Training budget:** PPO fell from 7.6% a year at 100k timesteps to 4.4% at 300k (same five seeds; Sharpe −0.16, −0.33 to −0.01), as its average cash rose from 20% to 39%. A2C's change, 7.5% to 8.7% with five seeds on both sides (7.3% with three at 100k), is not significant.
  - **Stability:** no algorithm is stable across seeds at 300k. The off-policy seeds spread the most; one SAC seed lost 61% over the seven years.
- **PPO against A2C at 100k timesteps** ([algorithm comparison](algorithm_comparison_20261107.md); five stocks, 7 folds × 3 seeds, the same reward, model and inputs):

  | | PPO | A2C |
  | --- | ---: | ---: |
  | Mean annual return | 8.2% | 7.3% |
  | Years beating the ETF | 4/7 | 3/7 |
  | Mean maximum drawdown | −11.1% | −13.2% |
  | Seed standard deviation | 2.5 points | 2.3 points |
  | Time per run, locally | 10–20 minutes | about 3.7 minutes |

  - **Significance:** the Sharpe difference (A2C − PPO) is −0.11, with a 95% interval of −0.26 to +0.04.
  - **Compute:** A2C trains about four times faster, at similar seed stability.
- **SAC, TD3 and DDPG** needed about 1 GB of replay memory and hours per model, more than the 8 GB development Mac could run, so they ran only on BistKA (above).

## RQ5. Does the learned policy generalize across market regimes, including COVID-19 and large Thai market drawdowns?

**Answer:** its behaviour is stable across regimes, but there is no evidence that it adapts to them. Its gains in falling years came from cash holdings that differed from seed to seed.

- **Folds:** seven one-year validation folds, 2016–2022. They include the 2018 decline, the 2020 COVID-19 crash and the 2022 rate rises. Each fold is trained only on earlier years.
- **Five stocks:**
  - **Drawdown:** the agent had the lowest mean maximum drawdown of any strategy (−11.1%).
  - **2020:** it lost 9.8%, against 11.8% for the ETF and 12.8% for equal weight.
  - **2022:** it made 4.9%, against 7.8% for the ETF.
- **Point-in-time SET50:**
  - **Every year:** the best design stayed close to equal weight.
  - **Falling years:** where it lost less than equal weight, cash explained it.
    - **2020:** the ensemble with the band lost 3.8% against 8.5%, because the band left cash in place during the March crash.
    - **2018:** the agent trained with the band lost 7.2% against 10.0%, because one of its three models held 93% cash all year.
  - **Seeds:** in neither case was the cash holding consistent across seeds.
  - **Not market timing:** fifteen models across four designs held 36–94% cash on average over their year. Re-running them with an input group shuffled across sessions moved their cash by 5 points or less in twelve of them. The cash is mostly a fixed bias each seed learned, not a response to the market. Where it did follow the inputs, it mostly followed the macro proxies, and that cost return ([attribution](pit_walk_forward_20261107.md#follow-up-daily-macro-proxies-as-an-input-2026-11-11)).
  - **Without the cash:** with its cash capped at 20%, the seed ensemble with the band lost 7.9% in 2020, against 3.8% without the cap and 8.5% for equal weight. Outside 2020 the two ensembles are level, so the uncapped agent's 2020 cushion was cash it happened to hold ([report](pit_walk_forward_20261107.md#follow-up-cash-capped-at-20-2026-11-19)).
  - **2022:** every equal-weight-like strategy trailed the ETF (agent 0.2%, equal weight 0.7%, ETF 7.8%), as the largest members led.
  - **Longer training (300k timesteps, 5 seeds):** the ensemble with the band lost less than equal weight in 2018 (−4.9% against −10.0%) and 2020 (−3.6% against −8.5%) and trailed it in every other year. The reason is the same: 17 of 35 models held a nearly fixed 45–98% cash all year, whatever the market did ([report](bistka_300k_results_20261125.md)).

## RQ6. Can BistKA HPC be used efficiently for reproducible training, hyperparameter tuning and sentiment feature extraction?

**Answer:** for training, yes. In November one 32-CPU job trained the 210 models of the algorithm comparison and the point-in-time design at 300k timesteps, a second trained the 35 models of the locked cash-capped design, a third the 147 models of the source ablation, and the trained models replay on the development Mac to within 10⁻⁸ of their curves. Retraining on BistKA reproduces the result but not the files. Hyperparameter tuning and sentiment extraction have not run there.

- **Job 4409, November 2026** ([report](bistka_300k_results_20261125.md#compute-rq6); [HPC record](../hpc/hpc_sync_status.md)):
  - **Layout:** one job with a pool of 210 tasks, one worker per CPU (`slurm/walk_forward_pool.sbatch`), instead of a 210-element array. The project's QOS allows 32 CPUs at once across all its jobs.
  - **Time:** 1,776 task-hours of training in 56.0 hours of wall time, about 32 times faster than one worker. No task failed.
  - **Cost:** about 2,465 SHr (44 SHr an hour, of which 12.8 came from a 128 GB memory request; the job used at most 49 GB). About 7,500 of the project's 10,000 SHr remain.
  - **Per model:** each CPU is one hyperthread, so two workers share a physical core. A point-in-time model trained at about 11 timesteps a second, against about 25 on one core of the Mac; SAC took 17 hours per model.
  - **Estimate versus measurement:** the job was estimated at 12–13 hours from Mac timings and took 56. Its progress was measured while it ran, without touching it (`hpc/buffer_rate.py`, `hpc/rollout_period.py`).
  - **Reproducibility across machines:** the 35 point-in-time models, trained with torch 2.10 and pandas 3.0 on BistKA, replayed on the Mac (torch 2.15, pandas 2.3) to within 6 × 10⁻⁹ of their saved validation curves.
- **Job 4518, November 2026** ([report](cash_cap_5seeds_300k_20261126.md); [HPC record](../hpc/hpc_sync_status.md)): the 35 cash-capped models at 300k in 11.2 hours for 450 SHr (32 CPUs, 80 GB; about 7,064 SHr remain). With 32 models at once each took 8.9 hours; the last three, with the other CPUs idle, took 2.3 hours. The models replayed on the Mac to within 1.0 × 10⁻⁸.
- **Job 4528, November 2026** ([report](source_ablation_20261127.md#the-retrain-on-bistka); [HPC record](../hpc/hpc_sync_status.md)): the 140 models of the source ablation and 7 retrained models of the locked design in 14.0 hours for 561 SHr (about 6,503 SHr remain). With 32 at once a point-in-time model took about 3 hours (about 36 minutes on the Mac with three workers).
  - **Same seed, another machine:** the locked design retrained with seed 42 made 4.8% a year on BistKA against 5.0% on the Mac (Sharpe 0.347 against 0.351). The models differ from the first update, as expected for floating-point training on different hardware and libraries, but the gap in a fold's final value (0.6 points on average) is about half the gap between two seeds on the Mac (1.1 points).
  - **Same machine:** a model retrained on the Mac after its process exited with an error gave byte-identical curves and holdings ([report](architecture_ablation_20261127.md#compute-and-one-task-rerun)).
  - **Inference:** the Mac replays BistKA's models to within 4 × 10⁻⁸.

- **Calibration on BistKA, July 2026** (`reports/bistka_calibration_execution_20260715.md` in the project folder; `reports/` is not tracked by git):
  - **Job 3530:** 7 min 46 s on 18 CPUs.
  - **Reproducibility:** two independent 549-step training trajectories matched exactly.
  - **Throughput:** vectorized environments lowered it, from 347.9 samples a second with one environment to 83.6 with sixteen, while memory rose 14.6 times.
  - **Consequence:** one environment per process. Parallelism comes from arrays of independent runs (folds, seeds, algorithms).
- **Before job 4409:** every November walk-forward ran locally ([compute note](local_compute_accounting_20261110.md)).
  - **Total:** 259 models, 169.5 task-hours on the 8 GB Mac.
  - **Memory:** it limits the Mac to three point-in-time workers, so a 21-model run takes 3.5–4.3 hours overnight and 9–10 hours during the day.
- **Not done on BistKA:**
  - **Hyperparameter tuning:** no tuning has run; all reported designs use fixed settings.
  - **Sentiment:** sentiment extraction (WangchanBERTa on `gpu4500-normal`) has no historical news to score.

## What would change these answers

- **RQ1, RQ5:** both single evaluations are done (test years and holdout, above); a new claim needs years after 4 November 2026. Licensed SETSMART data would remove the survivorship gap.
- **RQ2, RQ3:** historical news, investor flow and point-in-time fundamentals, which first need data.
- **RQ4:** the algorithms on the point-in-time index, and off-policy settings that limit trading (a turnover penalty or a no-trade band), which this comparison did not try.
- **RQ6:** hyperparameter tuning or sentiment extraction run on BistKA.
