# Point-in-Time SET50 Walk-Forward (Milestone M3)

Date: 2026-11-07; sticky-slot follow-up 2026-11-09
Evidence tier: **exploratory**. The index membership is official, but the prices are Yahoo's, and 13% of SET50 member-sessions (mostly delisted or merged names) have no prices.
Outputs (ignored by git): `results/pit_walk_forward_20261107/` (first run), `results/pit_walk_forward_sticky_20261109/` (sticky slots) and `results/pit_walk_forward_combined_20261109/` (both runs together) · Config: `config/pit_walk_forward_experiment.yaml` · Figures: [first run](figures/pit_walk_forward_chained_equity_20261107.png), [both runs](figures/pit_walk_forward_sticky_chained_equity_20261109.png)

**Summary.**
- **Result:** on the actual SET50 membership of each session, the DRL design that roughly matched the market on five hand-picked stocks lost money: −1.1% a year, against 7.1% for the SET50 ETF and 5.6% for equal weight. Unlike the five-stock comparisons, the Sharpe differences are statistically significant.
- **Survivorship:** choosing today's index members with hindsight inflates an equal-weight backtest by 6.6 points a year, also a significant difference.
- **Why the agent lost:** most of the shortfall is trading costs. The policy scores slot positions, so it reacts to the weekly reshuffling of slots and turns over 28 times its capital a year.
- **Fix (run 2026-11-09):** a sticky slot order, which keeps each member in its slot, cut turnover to a third and raised the agent's return to 3.7% a year. It beat the first run in all seven years, and the improvement is significant. The agent still trailed equal weight and the ETF, again mostly through trading costs. Both shortfalls are statistically significant, but only just.

## Question

Do the five-stock results survive when the agent trades the real index? Every earlier experiment used five stocks chosen in 2026 with hindsight (ADVANC, AOT, CPALL, KBANK, PTT). Here the investable set on each session is the SET50 membership in force at that time, from the official 2012–2024 archive ([report](point_in_time_universe_2012_2024_20261107.md)).

## Design

Everything except the universe copies the best five-stock design from the [seven-year walk-forward](walk_forward_reward_20261106.md):

- PPO, the plan reward (log return + differential Sharpe − increase in drawdown), weekly decisions;
- the shared per-asset LSTM model, and technical plus SET50 ETF market inputs;
- self-financing accounting at 0.20% per side, random two-year training episodes, and 100k timesteps;
- seven expanding folds: train on 2012 through December of year Y−1, validate on year Y (2016–2022). The 2023–2024 test years and the sealed 2025+ holdouts are never read.

New for the point-in-time universe (`src/envs/point_in_time_env.py`, `src/data/point_in_time_panel.py`):

- **Slots.** The agent sees 50 slots. On each session the tradable members fill them, ordered by mean traded value over the previous 60 sessions, and empty slots are masked out of the softmax. Between 39 and 48 members are tradable on any session.
- **Information lag.** A decision at session t uses features up to t−1 and the membership list in force at t−1.
- **Tradable members.** A member is tradable if it has prices at t and t+1, at least 60 sessions of price history, and complete features over the 30-session window.
- **Forced exits.** A holding that leaves the index, or whose prices stop, is sold at the close of t with costs, on any session.
- **Renamed members.** They keep one price series, for example BGH→BDMS, so a rename does not force a sale.

## Baselines and the survivorship effect

All baselines use the same sessions, costs and accounting as the agent. Only the stock universe changes.

| Universe (monthly equal weight unless noted) | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | **Mean** | Years beating ETF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hindsight: the 47 priced names of the 2024 H2 SET50, held throughout | 32.8% | 25.4% | −4.0% | 9.0% | −2.0% | 13.3% | 11.0% | **12.2%** | 7/7 |
| Five starter stocks | 34.7% | 35.3% | −8.0% | 5.7% | −12.8% | 12.3% | 3.9% | **10.1%** | 4/7 |
| SET50 ETF buy-and-hold (TDEX) | 24.5% | 18.2% | −6.5% | 5.7% | −11.8% | 12.1% | 7.8% | **7.1%** | — |
| **Point-in-time SET50** | 22.2% | 16.5% | −10.0% | 6.5% | −8.5% | 11.9% | 0.7% | **5.6%** | 2/7 |
| Point-in-time ridge forecast, top 10 monthly | 1.2% | 16.2% | −6.4% | 16.8% | 7.9% | 4.1% | 0.3% | **5.7%** | 3/7 |

**Two checks on the setup:**

- The five-stock equal weight gives the same 10.1% as the earlier five-stock walk-forward. The new environment's accounting reproduces the old one.
- The ETF tracks the whole index, including the members Yahoo cannot price. It is the only fully survivorship-free series here.

**What the table shows:**

- **Picking today's names with hindsight adds about 6.6 points a year** to an equal-weight backtest (12.2% against 5.6%), and it beats the market in every year. The five starter stocks sit between the two.
- **The ridge forecaster's edge disappears.** In the five-stock study it made 12.8% a year; on the point-in-time universe it makes 5.7%, level with equal weight. Its earlier edge came from the survivor universe, not from forecasting skill.
- **Equal weight trails the index.** Point-in-time equal weight (5.6%) is below the cap-weighted ETF (7.1%): over 2016–2022 the smaller SET50 members lagged the largest. Missing names push both ways. Delisted losers that Yahoo lacks flatter this figure, and take-overs at a premium (GLOW, BIGC, HEMRAJ, ROBINS) depress it.

## DRL results

21 models: 7 folds × 3 seeds × 100k timesteps. All were trained locally; no HPC job was submitted.

| Strategy | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | **Mean** | Years beating ETF | Mean max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SET50 ETF buy-and-hold (TDEX) | 24.5% | 18.2% | −6.5% | 5.7% | −11.8% | 12.1% | 7.8% | **7.1%** | — | −11.2% |
| Point-in-time SET50 equal weight | 22.2% | 16.5% | −10.0% | 6.5% | −8.5% | 11.9% | 0.7% | **5.6%** | 2/7 | −12.7% |
| **DRL, point-in-time SET50** (± s.d. over 3 seeds) | 15.7% ± 5.4 | 5.8% ± 3.4 | −15.0% ± 3.6 | 1.9% ± 5.2 | −15.1% ± 0.3 | 3.4% ± 5.2 | −4.5% ± 1.0 | **−1.1%** | 0/7 | −14.8% |
| DRL, five starter stocks (same design, 5 seeds) | 25.1% | 29.0% | −7.2% | 5.0% | −9.8% | 6.4% | 4.9% | **7.7%** | 3/7 | −11.1% |

**The agent trailed both the ETF and equal weight in every year.**
- **Return:** it made −1.1% a year, against 5.6% for equal weight over the same members and 7.1% for the ETF. Chained over the seven years, 1 baht became 0.89 (0.80–0.96 across seeds), against 1.41 in equal weight and 1.55 in the ETF ([figure](figures/pit_walk_forward_chained_equity_20261107.png)).
- **Drawdown:** its mean maximum drawdown (−14.8%) was deeper than the ETF's (−11.2%) and equal weight's (−12.7%).
- **2020 crash:** it lost 15.1%, against 11.8% for the ETF. The five-stock agent held 20% cash on average, but this agent stayed almost fully invested (3.7% cash on average).
- **Contrast:** with the same reward, model and folds, the five-stock agent made 7.7% a year with the lowest drawdown of any strategy.

**This time the difference is significant.** Sharpe ratios were measured on the 1,699 chained daily returns with a stationary bootstrap.

| Comparison | Sharpe difference | 95% interval |
| --- | ---: | ---: |
| DRL − ETF | −0.51 | −0.87 to −0.26 |
| DRL − point-in-time equal weight | −0.40 | −0.58 to −0.28 |
| Hindsight universe − point-in-time equal weight | +0.36 | +0.22 to +0.55 |
| Hindsight universe − ETF | +0.24 | +0.04 to +0.49 |
| Point-in-time equal weight − ETF | −0.12 | −0.37 to +0.11 |
| Point-in-time ridge top 10 − point-in-time equal weight | −0.02 | −0.41 to +0.34 |

- **DRL:** the agent's own Sharpe ratio was −0.02 (−0.74 to +0.70), and its deflated Sharpe ratio, counting the six designs screened earlier, was 0.09. Both comparisons that involve it exclude zero: it is significantly worse than both references.
- **Survivorship:** the advantage of the hindsight universe is also significant, so the survivorship effect is larger than noise.
- **Baselines:** none of the differences among the point-in-time baselines and the ETF is significant.

### Where the return went

`scripts/walk_forward_agent_diagnostics.py` splits each year's return into the return before costs and the cost drag. It also measures how the agent spread its weights on rebalance sessions. Means over the seven years:

| Strategy | Return before costs | Cost drag | Return after costs | Turnover a year | Average cash |
| --- | ---: | ---: | ---: | ---: | ---: |
| **DRL, point-in-time SET50** | 4.6% | 5.8% | −1.1% | 28.2× | 3.7% |
| Equal weight, point-in-time SET50 | 6.0% | 0.4% | 5.6% | 1.9× | 0.7% |
| DRL, five starter stocks | 8.5% | 0.8% | 7.7% | 3.9× | 20.3% |
| Equal weight, five starter stocks | 10.4% | 0.3% | 10.1% | 1.4× | 0.0% |

Turnover is the sum of absolute weight changes over the year, including the first purchase.

- **Trading costs explain about 80% of the gap to equal weight.**
  - **Before costs**, the agent was only 1.4 points a year behind equal weight over the same members. **After costs**, it was 6.7 points behind, because it paid 5.3 points more in costs.
  - **Turnover:** it turned over 28 times its capital a year, or 0.57 per weekly rebalance. In other words, it sold and bought back about 28% of the portfolio every week. The five-stock agent turned over 3.9 times.
- **It spread its weights; it did not concentrate them.** On rebalance sessions it held about 23 effective names (the inverse of the sum of squared weights) out of 44 tradable members. Its ten largest positions held 53% of capital. Its total-variation distance from equal weight was 0.36. Its daily returns correlated 0.95 with equal weight. The weights moved around a broadly diversified portfolio rather than expressing a few views.
- **Slot reshuffles drove the turnover.**
  - **Reshuffles:** slots follow the trailing traded-value order, so 48% of members changed slot between consecutive weekly decisions.
  - **Share of turnover:** those members accounted for 79% of rebalance turnover. Their average weight change was 2.1%, against 0.5% for members that kept their slot.
  - **Cause:** the policy network (the per-asset LSTM, then a dense head over all 50 slots) scores slot positions, not stocks. A change in the order therefore changes weights even when nothing about the stocks has changed. The five-stock agent never had this problem because its stocks never changed position.

## Follow-up: sticky slots (2026-11-09)

**The change.** `universe.slot_order: sticky` keeps a member in its slot while it stays tradable. Joiners take their previous slot if it is free, otherwise the lowest free slot. Nothing else changed: the variant `pit_plan_weekly_lstm_sticky` uses the same reward, model, folds, seeds and budget (21 models, 7 folds × 3 seeds × 100k timesteps), all trained locally.

**What it tests:** whether removing the reshuffles brings the agent's costs down and its return to at least equal weight.

| Strategy | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | **Mean** | Years beating ETF | Mean max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SET50 ETF buy-and-hold (TDEX) | 24.5% | 18.2% | −6.5% | 5.7% | −11.8% | 12.1% | 7.8% | **7.1%** | — | −11.2% |
| Point-in-time SET50 equal weight | 22.2% | 16.5% | −10.0% | 6.5% | −8.5% | 11.9% | 0.7% | **5.6%** | 2/7 | −12.7% |
| **DRL, sticky slots** (± s.d. over 3 seeds) | 20.4% ± 6.3 | 15.5% ± 5.8 | −9.3% ± 3.3 | 4.4% ± 4.1 | −10.0% ± 1.7 | 8.0% ± 3.5 | −3.3% ± 2.3 | **3.7%** | 1/7 | −14.2% |
| DRL, traded-value slots (first run) | 15.7% ± 5.4 | 5.8% ± 3.4 | −15.0% ± 3.6 | 1.9% ± 5.2 | −15.1% ± 0.3 | 3.4% ± 5.2 | −4.5% ± 1.0 | **−1.1%** | 0/7 | −14.8% |

**Better than the first run, still behind both references.**
- **Return:** 3.7% a year against −1.1%, and higher in every year. Chained over the seven years, 1 baht became 1.24 (1.12–1.37 across seeds), against 0.89 in the first run, 1.41 in equal weight and 1.55 in the ETF ([figure](figures/pit_walk_forward_sticky_chained_equity_20261109.png)).
- **Against the references:** it beat equal weight only in 2018 and the ETF only in 2020. Even its best seed (5.2% a year) fell short of equal weight.
- **Drawdown:** its mean maximum drawdown (−14.2%) remained deeper than the ETF's (−11.2%) and equal weight's (−12.7%).

**Significance.** Sharpe ratios on the same 1,699 chained daily returns, with the same stationary bootstrap:

| Comparison | Sharpe difference | 95% interval |
| --- | ---: | ---: |
| DRL sticky − DRL first run | +0.29 | +0.16 to +0.47 |
| DRL sticky − point-in-time equal weight | −0.11 | −0.24 to −0.004 |
| DRL sticky − ETF | −0.23 | −0.52 to −0.01 |

- **DRL:** the sticky agent's Sharpe ratio was 0.27 (−0.45 to +1.03). Its deflated Sharpe ratio was 0.25, counting seven designs: the six screened earlier, plus the sticky order, which was chosen after seeing the first run's validation results. Counting seven designs, the first run's value would be 0.08 rather than 0.09.
- **Reading:** the improvement over the first run is significant. The shortfalls against equal weight and the ETF are also significant, but only just, because their intervals end at −0.004 and −0.01.

**Where the return went.** The same diagnostics as above, as means over the seven years:

| Strategy | Return before costs | Cost drag | Return after costs | Turnover a year | Average cash |
| --- | ---: | ---: | ---: | ---: | ---: |
| **DRL, sticky slots** | 5.6% | 2.0% | 3.7% | 9.4× | 2.7% |
| DRL, traded-value slots (first run) | 4.6% | 5.8% | −1.1% | 28.2× | 3.7% |
| Equal weight, point-in-time SET50 | 6.0% | 0.4% | 5.6% | 1.9× | 0.7% |

- **The slot reshuffles are gone.** Counting joins and exits, only 0.4% of members changed slot between weekly decisions (48% before). They caused 2% of rebalance turnover (79% before).
- **Turnover fell to a third:** from 28.2 to 9.4 times capital a year, or from 0.57 to 0.17 per weekly decision. The cost drag fell from 5.8 to 2.0 points a year.
- **Return before costs rose too:** from 4.6% to 5.6%, now only 0.4 points a year behind equal weight.
- **Costs are still most of the gap.** After costs, the agent was 1.9 points a year behind equal weight, and 1.6 of those points were extra costs. After the initial purchase it still turned over 8.4 times its capital a year, against 0.9 for equal weight. 98% of its rebalance turnover now came from members that kept their slot, so the remaining trading is the policy's own week-to-week reweighting.
- **Slightly more concentrated:**
  - **Holdings:** about 19 effective names (23 before).
  - **Top ten positions:** 59% of capital (53% before).
  - **Distance from equal weight:** a total-variation distance of 0.40 (0.36 before).
  - **Correlation:** its daily returns still correlated 0.94 with equal weight.

**What it means:**
- **Diagnosis confirmed:** the slot reshuffles were the first run's main defect, and removing them recovered 4.8 points a year.
- **No outperformance:** the design still does not beat equal weight or the market on the actual index.
- **Remaining problem:** turnover is the problem again, but now it comes from the policy's own reweighting, not from the slot order. Next steps:
  - **Policy head (implemented 2026-11-09, not yet run):** `SharedAssetPolicy` (`src/models/shared_asset_policy.py`) scores each stock with shared weights, an M5 item, which removes the dependence on slot order by construction. Its variant is `pit_plan_weekly_lstm_shared`.
  - **Reweighting (not tried):** reducing the policy's own reweighting is a separate problem.

## Run it

Local: on the 8-core development Mac (4 performance and 4 efficiency cores), seven workers took about 65 minutes per 100k-timestep run.

**Memory limits the worker count there.** Each worker needs about 750 MB.
- **First run:** seven workers plus the usual desktop apps exhausted the Mac's 8 GB of memory. 10.7 of 11.3 GB of swap was in use, and training slowed to 10–50% of a core per worker. Pausing three workers (`kill -STOP`, later `kill -CONT`) brought the other four back to about 95% of a core.
- **Sticky run:** with more desktop apps open, even four workers thrashed. Three ran at about 95% of a core each.

**Use three workers on that machine, or four with few other apps open.**

**Sticky run time (three workers):** a run took 29–37 minutes while the Mac was awake and on AC power. All 21 runs took 8 h 42 min of wall time, of which the Mac was asleep for about 3.8 hours. Low Power Mode on battery also slows training.

```bash
RUNNER=scripts.run_pit_walk_forward CONFIG=config/pit_walk_forward_experiment.yaml VARIANTS=pit_plan_weekly_lstm SEEDS="42 43 44" \
  PYTHON=python scripts/run_walk_forward_reward.sh results/pit_walk_forward 7 100000
python -m scripts.pit_universe_baselines --config config/pit_walk_forward_experiment.yaml --run-dir results/pit_walk_forward \
  --years 2016 2017 2018 2019 2020 2021 2022
python -m scripts.aggregate_walk_forward_reward --run-dir results/pit_walk_forward --n-trials 6
python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_walk_forward --variant pit_plan_weekly_lstm \
  --pit-config config/pit_walk_forward_experiment.yaml
```

The sticky-slot follow-up trains into its own folder. Its baselines come from the first run, so it is aggregated together with it:

```bash
RUNNER=scripts.run_pit_walk_forward CONFIG=config/pit_walk_forward_experiment.yaml VARIANTS=pit_plan_weekly_lstm_sticky SEEDS="42 43 44" \
  PYTHON=python scripts/run_walk_forward_reward.sh results/pit_walk_forward_sticky 3 100000
python -m scripts.aggregate_walk_forward_reward --run-dir results/pit_walk_forward_sticky --extra-run-dirs results/pit_walk_forward \
  --n-trials 7 --output-dir results/pit_walk_forward_combined --references pit_plan_weekly_lstm \
  --labels pit_plan_weekly_lstm="DRL, traded-value slots" pit_plan_weekly_lstm_sticky="DRL, sticky slots"
python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_walk_forward_sticky --variant pit_plan_weekly_lstm_sticky \
  --pit-config config/pit_walk_forward_experiment.yaml
```

The shared-weight actor runs the same way with `VARIANTS=pit_plan_weekly_lstm_shared`. It standardizes its own features, so it uses no `VecNormalize`. To check that a trained model ignores the slot order, roll it out under both orders:

```bash
python -m scripts.check_slot_order_invariance --config config/pit_walk_forward_experiment.yaml --fold-year 2019 \
  --run-dirs results/<run>/fold_2019/<variant>_seed42/runs/<variant>_seed42
```

**On BistKA**, with more seeds and longer training:

1. **Copy the inputs that git does not track:**
   - `data/raw/extended/prices_set100_union_2012_2024.csv.gz`
   - `data/raw/extended/index_set50_etf_2012_2024.csv`
   - the folder `data/raw/extended/set100_2012_2024_additions/` (prices, price map, receipts)
2. **Submit the array job.** Each task needs one CPU and under 1 GB of memory.

   ```bash
   export RUNNER=scripts.run_pit_walk_forward CONFIG=config/pit_walk_forward_experiment.yaml VARIANTS=pit_plan_weekly_lstm
   export YEARS="2016 2017 2018 2019 2020 2021 2022" SEEDS="42 43 44 45 46" TIMESTEPS=300000 RUN_NAME=pit_walk_forward_bistka
   sbatch --export=ALL --array=0-34%16 slurm/walk_forward_reward_array.sbatch
   ```

3. **Add the baselines and aggregate**, with the two commands above pointed at `results/pit_walk_forward_bistka`.

For the sticky variant, set `VARIANTS=pit_plan_weekly_lstm_sticky`. The array size is the same.
