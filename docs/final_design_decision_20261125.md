# Choosing the Final Design for the Single Test Evaluation (M6–M8)

Date: 2026-11-25
Status: **approved on 2026-11-25** as recommended, both locks together, by Nathawat Youngyim in chat. Only that name is recorded in the locks; the co-author and the advisor have not signed them. Both test stages ran once that evening through the gate ([results](final_test_results_20261125.md)); the sealed 2025+ holdout is still unread.
Evidence tier: exploratory (Yahoo prices, validation years 2016–2022).

**Tools added for this step:**
- [`scripts/rank_final_candidates.py`](../scripts/rank_final_candidates.py) ranks designs by the decision-7 default score. It reads validation curves only and refuses any year after 2022.
- [`config/final_candidates_20261125.yaml`](../config/final_candidates_20261125.yaml) lists every walk-forward design with complete validation curves, plus the forecast and heuristic baselines.
- [`scripts/run_final_test_evaluation.py`](../scripts/run_final_test_evaluation.py) is the run-once gate for the test years.
- [`config/final_test_protocol_pit.yaml`](../config/final_test_protocol_pit.yaml) and [`config/final_test_protocol_five_stock.yaml`](../config/final_test_protocol_five_stock.yaml) are the two locks. Both are drafts, and the gate refuses them as they stand.

> **Five seeds at 100k (2026-11-25, evening):** the two 100k designs this choice depends on had only 3 seeds, against 5 at 300k. Seeds 45 and 46 of the point-in-time shared-weight design and of five-stock A2C were trained locally on validation years only (`slurm/pools/local_20261125.tasks`, 28 new models in 2 hours 40 minutes). A rerun of a 2026-11-10 model matched it exactly on every session. The 5-seed 100k ensemble with the band is a new design in the ranking below. **The ranking's choices did not change.** The 5-seed 100k ensemble scored −0.78, just above its 3-seed version (−0.83). Five-stock A2C at 100k rose from −2.55 to −1.30.

## What the authors have to decide

| # | Decision | Recommended | Main alternative |
| --- | --- | --- | --- |
| 1 | Universe of the headline test | **Point-in-time SET50**: membership as it was on each date, so there is no hindsight in the stock list | The five starter stocks, which were chosen with hindsight; the bias carries into 2023–2024 |
| 2 | How the design is picked | **The decision-7 default rule** (below), applied to every design tried, without choosing by eye | Highest Sharpe ratio against the ETF |
| 3 | What the test evaluates | **The rule's top DRL design in each universe**, beside equal weight and the ETF. The five-stock test is secondary, locked at the same time | The point-in-time design only |
| 4 | How the test is read | Stated now (see "Stage 1") | — |
| 5 | The holdout | **Stage 2, after stage 1:** retrain through 2024 and evaluate once on 2025-01-02 to 2026-11-05. For the point-in-time design this needs the 2025–2026 SET50 lists | The December 2024 membership held fixed |
| 6 | Who approves | Both authors and the advisor, by name and date in the lock | — |

Decision 7 itself (the primary metric) is still open in the author decision packet. Approving this file's rule approves its default for this use.

## What the pre-agreed rule says

The default for decision 7 ([formulas](evaluation_metric_formulas_20261110.md), [packet](v3_author_decision_packet_20260904.md)) ranks designs in four steps:
1. **Fold score:** per fold, the annualized excess net log return over active equal weight on the same sessions, $\frac{252}{N}\left(\ln\frac{V_N}{V_0} - \ln\frac{E_N}{E_0}\right)$.
2. **Design score:** the mean over seeds, then the **median of the seven fold values**.
3. **Ties:** broken by the worst fold, then excess drawdown, turnover and the smaller budget.
4. **Fallback:** *if no design scores above zero, it selects active equal weight.*

Validation years 2016–2022, scores in percentage points a year (`python -m scripts.rank_final_candidates --candidates config/final_candidates_20261125.yaml --base ~/drl_runs`):

**Point-in-time SET50** (reference: point-in-time equal weight, 5.6% a year; ETF 7.1%)

| Rank | Design | Kind | Score (median) | Mean excess | Worst year | Years ahead of equal weight | Median vs ETF | Mean annual return | Turnover a year |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Minimum variance, monthly | baseline | +1.15 | −1.87 | −17.09 | 4/7 | −4.69 | 4.0% | 3.5× |
| 2 | Momentum, top ten monthly | baseline | +0.03 | −1.05 | −20.97 | 4/7 | −7.15 | 5.3% | 6.7× |
| **3** | **Shared actor, cash capped at 20%, seed ensemble + band (3 seeds)** | **DRL** | **−0.01** | −0.56 | −2.43 | 3/7 | −2.67 | 5.0% | 1.7× |
| 4 | Shared actor + macro proxies, ensemble + band | DRL | −0.43 | −1.06 | −4.37 | 2/7 | −4.37 | 4.3% | 2.2× |
| 5 | Inverse volatility, monthly | baseline | −0.76 | −0.56 | −2.42 | 3/7 | −1.09 | 5.0% | 2.7× |
| 6 | Shared actor, 100k, seed ensemble + band (5 seeds) | DRL | −0.78 | −0.34 | −1.78 | 2/7 | −2.29 | 5.2% | 1.6× |
| 7 | Shared actor, 100k, seed ensemble + band (3 seeds) | DRL | −0.83 | **+0.07** | −1.81 | 2/7 | −2.66 | **5.5%** | 1.7× |
| 8 | Shared actor, cash capped at 20% | DRL | −0.98 | −1.13 | −2.64 | 1/7 | −3.66 | 4.4% | 4.3× |
| 9 | Shared actor trained with the band, ensemble | DRL | −1.11 | −0.22 | −1.64 | 2/7 | −1.68 | 5.2% | 1.5× |
| 10–16 | Band at evaluation only, ensemble without the band, policy rate (ensemble and single), trained with the band, single 100k models (5 seeds, −1.45), sticky slots | DRL | −1.20 to −1.73 | | | | | | |
| 17 | Shared actor, 300k, ensemble + band (5 seeds) | DRL | −1.78 | −1.28 | −6.39 | 2/7 | −5.41 | 3.9% | 0.9× |
| 18–21 | Macro proxies (single), LSTM forecaster (baseline: mean +2.10, median −2.47), 300k single models, slots by traded value | | −1.88 to −6.01 | | | | | | |

**Five starter stocks** (reference: five-stock monthly equal weight, 10.1% a year; ETF 7.1%)

| Rank | Design | Kind | Score (median) | Mean excess | Worst year | Years ahead of equal weight | Median vs ETF | Mean annual return | Turnover a year |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LSTM forecast, top two monthly | baseline | +3.18 | +2.79 | −7.55 | 5/7 | +8.48 | 13.8% | 13.7× |
| 2 | Ridge forecast, top two monthly | baseline | +2.74 | +2.61 | −3.96 | 5/7 | +4.06 | 12.8% | 14.1× |
| **3** | **A2C, 300k** | **DRL** | **−0.16** | −1.41 | −12.17 | 3/7 | −1.60 | 8.7% | 4.4× |
| 4 | PPO, 100k | DRL | −0.72 | −1.93 | −7.63 | 3/7 | −0.65 | 7.6% | 4.0× |
| 5 | A2C, 100k (5 seeds) | DRL | −1.30 | −2.19 | −6.49 | 2/7 | −0.45 | 7.5% | 3.4× |
| 6 | PPO, technical inputs only | DRL | −1.52 | −2.65 | −8.33 | 2/7 | −0.21 | 6.8% | 4.3× |
| … | PPO + macro, PPO 300k, PPO log reward, DDPG, TD3, SAC | DRL | −3.15 to −12.73 | | | | | | |

**What the rule gives:**
- **No DRL design scores above zero** in either universe, so the rule as written selects **active equal weight** in both. That is consistent with every report so far: no DRL design beat equal weight on validation.
- **The top DRL designs sit within a fraction of a point of zero.** They are the cash-capped point-in-time ensemble with the band (−0.01) and five-stock A2C at 300k (−0.16).
- **The order among the leaders is not meaningful.** Their fold values spread by several points from year to year, and neither leader's Sharpe ratio differs significantly from equal weight's or the ETF's. The rule still gives a fixed choice made before the test, which is its purpose.
- **Some baselines score above zero, and they are not DRL.** Minimum variance and momentum on the point-in-time index, and both forecasters on the five stocks, are scored here only for comparison and are never selected.

## Recommendation

1. **Follow the rule, and state its fallback openly.** The deployable choice from validation is active equal weight. The test then asks the question the title raises: does the best DRL design, picked by a fixed rule, beat equal weight and the ETF on years it never saw?
2. **Headline test:** the point-in-time rule's top DRL design. That is the cash-capped shared-weight actor (100k timesteps, seeds 42–44, seed ensemble with the 0.5-point band). The 5-seed update did not change it. Lock: `config/final_test_protocol_pit.yaml`.
3. **Secondary test, locked at the same time:** the five-stock rule's top DRL design, A2C at 300k with seeds 42–46, reported with the survivorship caveat. Lock: `config/final_test_protocol_five_stock.yaml`.
4. **Do not change a design after seeing either result.** Both locks are approved together, before either runs.

**Cost:**
- **Point-in-time:** 6 models at about 1–1.5 hours each on the Mac, about 3 hours with three workers.
- **Five stocks:** 10 A2C models at 300k, under an hour.

No HPC job is needed.

## Stage 1: the test years (2023–2024), run once

- **Training:** each test year is a walk-forward fold, as in validation. The 2023 models train on everything through December 2022, and the 2024 models on everything through December 2023, with the locked config, seeds and budget. The only change to the config is `splits.validation_end`, moved to 2024-12-31. The price files end in 2024, so nothing from the holdout can enter.
- **Reported, all pre-stated:**
  - each year's return and the chained 2023–2024 curve;
  - maximum drawdown, turnover and cost drag;
  - the decision-7 score over the two years;
  - Sharpe differences against equal weight and against the ETF, by paired stationary bootstrap on the chained daily returns (about 485 sessions), with 95% intervals;
  - the deflated Sharpe with every design tried counted (22 point-in-time, 14 five-stock).
- **How it will be read:**
  - **DRL adds value on unseen years** only if the 95% interval of the Sharpe difference against equal weight is above zero.
  - **The ETF** is reported the same way.
  - **Low power:** with two years the test has little power, so an interval that includes zero is the expected outcome. It will be reported as "no evidence of added value", not as evidence of no value.
- **The gate** ([`scripts/run_final_test_evaluation.py`](../scripts/run_final_test_evaluation.py)) refuses to run in any of these cases:
  - the lock is not `APPROVED`, or has no names and date;
  - the lock is not committed, or the code and configs have uncommitted changes;
  - the config's SHA-256 has changed;
  - the test end reaches 2025;
  - the output folder already exists, or the ledger (`results/FINAL_TEST_LEDGER.jsonl`) already has the stage.

  It writes start and end receipts with the commit and the lock's hash. `--dry-run` prints the checks and the plan without reading data.

## Stage 2: the sealed holdout (after stage 1)

- **Data:** `data/holdout_sealed/` holds Yahoo prices from 2025-01-02 to 2026-11-05, sealed on 2026-11-06 with SHA-256 manifests. The rule: evaluate exactly once, after the protocol is locked.
- **Plan:** retrain the locked design on everything through 2024, then evaluate once on the whole holdout.
- **Built:** `src/data/sealed_holdout.py` (9 tests). `verify_seal` checks the manifest against its recorded hash and every file against the manifest, hashing bytes without parsing them. `open_sealed` reads the files only after that, and only for an `APPROVED` holdout lock with names and a date.
- **Built 2026-11-26, not run** ([holdout stage](holdout_stage_20261126.md)): `scripts/run_holdout_evaluation.py` and two draft locks (`config/holdout_protocol_pit.yaml`, `config/holdout_protocol_five_stock.yaml`). The two items below were the open parts; the draft holds the last 2024 list for the membership, and the authors choose.
- **Was not built (2026-11-25):**
  - **Wiring:** a holdout run that feeds the opened files to the runners as two walk-forward folds: 2025 trained through 2024, and 2026 to 5 November, trained through 2025.
  - **2025–2026 SET50 membership for the point-in-time design.** The official archive ends in 2024. The options are the published SET50 lists for 2025 and 2026 (a download, which needs approval), or the December 2024 membership held fixed: no hindsight, but it ignores index revisions.

## How to approve and run

1. **Approve:** edit both locks. Change the design if the authors choose differently, set `status: APPROVED`, `approved_by` and `approved_on`, and commit them.
2. **Check:** dry-run both locks.
3. **Run:** start both locks, each once.

```bash
python -m scripts.run_final_test_evaluation --lock config/final_test_protocol_pit.yaml --dry-run
python -m scripts.run_final_test_evaluation --lock config/final_test_protocol_five_stock.yaml --dry-run
python -m scripts.run_final_test_evaluation --lock config/final_test_protocol_pit.yaml --workers 3
python -m scripts.run_final_test_evaluation --lock config/final_test_protocol_five_stock.yaml --workers 3
```
