# Answers to the Research Questions (draft)

Date: 2026-11-11
Status: **draft** for the M8 item "answer each research question explicitly, including negative results". It answers the six questions of the implementation plan (`implementation_plan.md`, section 2.2) from the evidence in the repository today. Each answer links the report it comes from.

**Limits that apply to every answer:**
- **Data tier:** exploratory. Prices are Yahoo's, and 13% of point-in-time SET50 member-sessions have none.
- **Years:** validation years only (2016–2022). The 2023–2024 test years and the sealed 2025+ holdout are unread.
- **Statistics:** every Sharpe comparison is a paired stationary bootstrap on chained daily returns (1,699 sessions); "significant" means its 95% interval excludes zero.

| # | Question (abridged) | Short answer | Evidence |
| --- | --- | --- | --- |
| 1 | Can a continuous-action DRL agent learn profitable long-term allocation for Thai equities? | **Profitable, but no better than simple benchmarks.** | Strong for the negative part |
| 2 | Does multi-source data beat price-only and technical-only agents? | **Only market-index context helped, and only borderline;** macro proxies did not. | Weak; point-in-time arms pending |
| 3 | Which data sources contribute most? | **Market-index context, among the sources tested;** most planned sources have no historical data. | Weak, incomplete |
| 4 | Which DRL algorithm is most stable? | **PPO over A2C, not significantly;** the off-policy algorithms are not run. | Partial |
| 5 | Does the policy generalize across regimes, including COVID-19? | **Its behaviour is stable across regimes, but it shows no regime skill;** its crash-year gains came from cash that depended on the seed. | Moderate |
| 6 | Can BistKA HPC be used efficiently and reproducibly? | **Reproducibility and the parallel layout were established on BistKA;** efficiency at scale is not yet measured. | Partial |

## RQ1. Can a continuous-action DRL agent learn profitable long-term portfolio allocation policies for Thai equities?

**Answer:** in absolute terms, yes; against simple benchmarks, no. No DRL design beat equal weight or the SET50 ETF, and what the best design learned is close to equal weight.

- **Five hand-picked stocks** ([walk-forward](walk_forward_reward_20261106.md); 7 folds × 5 seeds):
  - **DRL:** the long-term agent (plan reward, per-asset LSTM, weekly decisions) made 7.7% a year.
  - **Benchmarks:** the SET50 ETF made 7.1%, equal weight 10.1%, and the LSTM forecaster 13.8%.
  - **Significance:** no difference is significant.
  - **Survivorship:** these five stocks were chosen with hindsight, which flatters every strategy confined to them.
- **Point-in-time SET50**, the actual members of each session ([report](pit_walk_forward_20261107.md)):
  - **First design:** −1.1% a year, significantly worse than the ETF (Sharpe −0.51, 95% CI −0.87 to −0.26) and equal weight (−0.40, −0.58 to −0.28). It overtraded because it scored slot positions rather than stocks.
  - **Best design:** after four follow-ups (sticky slots, a shared-weight actor, a seed ensemble and a no-trade band), it made 5.5% a year, against 5.6% for equal weight and 7.1% for the ETF. Chained over 2016–2022, 1 baht became 1.41, as for equal weight; the ETF reached 1.55.
  - **Sharpe ratios:** 0.42 for the best design, 0.38 for equal weight and 0.50 for the ETF. Neither difference is significant: +0.04 (−0.05 to +0.11) and −0.08 (−0.36 to +0.15).
  - **Deflated Sharpe:** 0.27, counting the 13 designs tried.
  - **Training with the band:** 5.1% a year, and 5.2% as a seed ensemble.
- **What it learned:** close to equal weight. Its daily correlation with equal weight is 0.998, with 43.9 effective holdings out of 44.5 tradable members.
- **Other benchmarks on the same universe:** none is significantly better or worse than the agent.
  - **Heuristics:** momentum 5.3%, inverse volatility 5.0% and minimum variance 4.0%.
  - **Forecasters:** ridge 5.7%; LSTM 8.1%, through one rally in late 2020.

## RQ2. Does adding multi-source data improve out-of-sample performance compared with price-only and technical-indicator-only agents?

**Answer:** only weakly, and only for market-index context. The daily macro proxies did not help. The other sources have not been tested, and the point-in-time arms are still running.

- **Five stocks** ([source ablation](source_ablation_20261106.md); 7 folds × 3 seeds per arm, everything else fixed):

  | Inputs | Mean annual return | Sharpe difference (95% CI) |
  | --- | ---: | --- |
  | Technical only | 6.8% | — |
  | + SET50 market context | 8.2% | +0.10 vs technical only (−0.002 to +0.21; one-sided p = 0.03) |
  | + market + daily macro proxies | 7.1% | −0.08 vs + market (−0.20 to +0.05) |

  The market-context gain is borderline. Its interval touches zero, and three comparisons were made.
- **Not run:** a price-only arm. Technical-only is the lowest arm.
- **Point-in-time SET50, in progress:**
  - **BOT policy rate:** the shared-weight agent plus five official policy-rate inputs. It started on 2026-11-11 at 02:38 (`config/pit_walk_forward_policy_rate.yaml`).
  - **Macro proxies:** the same agent plus the eight daily macro-proxy inputs. It is queued to start right after (`config/pit_walk_forward_macro.yaml`).
  - **Comparison:** both arms will be compared with the shared-weight run, which differs only in its inputs.

## RQ3. Which data sources contribute most to performance?

**Answer:** among the sources tested, market-index context; technical indicators carry the agent. Fundamentals, news sentiment, sector context and investor flow have not been tested, because the historical data is missing.

| Source | Status | Finding |
| --- | --- | --- |
| Technical indicators | Tested (every arm) | The base of every agent; on its own, 6.8% a year on five stocks |
| Market index context (SET50 ETF) | Tested | +1.4 points a year and +0.10 Sharpe on five stocks, borderline |
| Daily macro proxies (USD/THB, oil, gold, US 10-year yield) | Tested on five stocks; point-in-time arm queued | No gain. The agent reads them: shuffling them moves its weights by 2.9%, against 1.9% for market context. Shuffling any group changes the return by 0.25 points or less |
| Official BOT policy rate | Point-in-time arm running | Pending |
| Financial statements | Not tested | Five stocks and recent years only; no shares outstanding for P/E and P/BV |
| Thai news sentiment | Not tested | The scorer exists; there is no historical news for 2014–2024 |
| Sector context | Not tested | Implemented for the V3 universe; not switched on in the point-in-time runs |
| Investor flow | Not tested | The importer exists; there is no data |

## RQ4. Which DRL algorithm is most stable under high noise and limited compute: PPO, A2C, DDPG, TD3 or SAC?

**Answer:** PPO is the better on-policy choice, but not significantly. The off-policy algorithms have not been run.

- **PPO against A2C** ([algorithm comparison](algorithm_comparison_20261107.md); five stocks, 7 folds × 3 seeds, the same reward, model and inputs):

  | | PPO | A2C |
  | --- | ---: | ---: |
  | Mean annual return | 8.2% | 7.3% |
  | Years beating the ETF | 4/7 | 3/7 |
  | Mean maximum drawdown | −11.1% | −13.2% |
  | Seed standard deviation | 2.5 points | 2.3 points |
  | Time per run, locally | 10–20 minutes | about 3.7 minutes |

  - **Significance:** the Sharpe difference (A2C − PPO) is −0.11, with a 95% interval of −0.26 to +0.04.
  - **Compute:** A2C trains about four times faster, at similar seed stability.
- **SAC, TD3 and DDPG:** not run. Each needs about 1 GB of replay memory and an hour per model, which exceeded the 8 GB development Mac. They are set up for a BistKA array (`slurm/walk_forward_reward_array.sbatch`).

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
  - **2022:** every equal-weight-like strategy trailed the ETF (agent 0.2%, equal weight 0.7%, ETF 7.8%), as the largest members led.

## RQ6. Can BistKA HPC be used efficiently for reproducible training, hyperparameter tuning and sentiment feature extraction?

**Answer:** partly. BistKA runs established reproducibility and the right way to parallelize. The efficiency of the actual experiments at scale has not been measured, because no HPC job has been submitted since July. Hyperparameter tuning and sentiment extraction have not run there.

- **Calibration on BistKA, July 2026** (`reports/bistka_calibration_execution_20260715.md` in the project folder; `reports/` is not tracked by git):
  - **Job 3530:** 7 min 46 s on 18 CPUs.
  - **Reproducibility:** two independent 549-step training trajectories matched exactly.
  - **Throughput:** vectorized environments lowered it, from 347.9 samples a second with one environment to 83.6 with sixteen, while memory rose 14.6 times.
  - **Consequence:** one environment per process. Parallelism comes from arrays of independent runs (folds, seeds, algorithms).
- **Since November:** every walk-forward ran locally ([compute note](local_compute_accounting_20261110.md)).
  - **Total:** 217 models, 146.1 task-hours on the 8 GB Mac.
  - **Memory:** it limits the Mac to three point-in-time workers, so a 21-model run takes 3.5 hours overnight and 9–10 hours during the day.
  - **On BistKA:** a SLURM array for these runs is ready. The projection is 30–60 minutes per 21-model run, but it is not measured: submitting needs the operator's approval.
- **Not done on BistKA:**
  - **Hyperparameter tuning:** no tuning has run; all reported designs use fixed settings.
  - **Sentiment:** sentiment extraction (WangchanBERTa on `gpu4500-normal`) has no historical news to score.

## What would change these answers

- **RQ1, RQ5:** the single evaluation of the final configuration on the 2023–2024 test years and the sealed 2025+ holdout, after the design is locked. Licensed SETSMART data would remove the survivorship gap.
- **RQ2, RQ3:** the point-in-time policy-rate and macro arms (running and queued), then historical news, investor flow and point-in-time fundamentals.
- **RQ4:** the SAC, TD3 and DDPG array on BistKA.
- **RQ6:** that array, with its service hours, queue time and measured speed-up.
