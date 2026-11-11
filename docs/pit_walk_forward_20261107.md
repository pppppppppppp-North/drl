# Point-in-Time SET50 Walk-Forward (Milestone M3)

Date: 2026-11-07; sticky-slot follow-up 2026-11-09; shared-weight actor, seed ensemble and no-trade band 2026-11-10; training with the band, LSTM forecast benchmark, BOT policy rate and macro proxies 2026-11-11
Evidence tier: **exploratory**. The index membership is official, but the prices are Yahoo's, and 13% of SET50 member-sessions (mostly delisted or merged names) have no prices.
Outputs (ignored by git): `results/pit_walk_forward_20261107/` (first run), `results/pit_walk_forward_sticky_20261109/` (sticky slots), `results/pit_walk_forward_shared_20261110/` (shared-weight actor), `results/pit_walk_forward_combined_20261109/` (first two runs together), `results/pit_walk_forward_combined_20261110/` (all three), `results/pit_seed_ensemble_20261110/` (seed ensemble and no-trade band, no training), `results/pit_seed_ensemble_combined_20261110/` (with the shared-weight run), `results/pit_heuristic_baselines_20261110/` (momentum, inverse volatility and minimum variance), `results/pit_band_sensitivity_20261110/` (band widths), `results/pit_walk_forward_band_20261110/` (trained with the band), `results/pit_band_ensemble_20261111/` (its seed ensemble), `results/pit_walk_forward_combined_band_20261111/` (both, with the shared-weight and ensemble runs), `results/pit_lstm_forecast_20261111/` (LSTM forecast benchmark), `results/pit_lstm_forecast_combined_20261111/` (with the other benchmarks), `results/pit_walk_forward_policy_20261111/` (BOT policy rate as an input, with the input attribution), `results/pit_policy_ensemble_20261111/` (its seed ensemble with the band), `results/pit_walk_forward_combined_policy_20261111/` (both, with the shared-weight run and its ensemble) and `results/pit_attribution_shared_20261111/` and `results/pit_attribution_band_20261111/` (input attribution of the shared-weight and band-trained models), `results/pit_walk_forward_macro_20261111/` (daily macro proxies as an input, with the input attribution), `results/pit_macro_ensemble_20261111/` (its seed ensemble with the band), `results/pit_walk_forward_combined_macro_20261111/` (both, with the shared-weight run and its ensemble) and `results/pit_source_arms_combined_20261111/` (the three input sets' ensembles) · Configs: `config/pit_walk_forward_experiment.yaml`, `config/pit_walk_forward_policy_rate.yaml`, `config/pit_walk_forward_macro.yaml` · Figures: [first run](figures/pit_walk_forward_chained_equity_20261107.png), [first two runs](figures/pit_walk_forward_sticky_chained_equity_20261109.png), [all three runs](figures/pit_walk_forward_shared_chained_equity_20261110.png), [seed ensemble with no-trade band](figures/pit_seed_ensemble_chained_equity_20261110.png), [training with the band](figures/pit_band_training_chained_equity_20261111.png), [LSTM forecast benchmark](figures/pit_lstm_forecast_chained_equity_20261111.png), [BOT policy rate](figures/pit_policy_rate_chained_equity_20261111.png), [macro proxies](figures/pit_macro_proxies_chained_equity_20261111.png), [all three input sets](figures/pit_source_arms_chained_equity_20261111.png)

**Summary.**
- **Result:** on the actual SET50 membership of each session, the DRL design that roughly matched the market on five hand-picked stocks lost money: −1.1% a year, against 7.1% for the SET50 ETF and 5.6% for equal weight. Unlike the five-stock comparisons, the Sharpe differences are statistically significant.
- **Survivorship:** choosing today's index members with hindsight inflates an equal-weight backtest by 6.6 points a year, also a significant difference.
- **Why the agent lost:** most of the shortfall is trading costs. The policy scores slot positions, so it reacts to the weekly reshuffling of slots and turns over 28 times its capital a year.
- **Fix (run 2026-11-09):** a sticky slot order, which keeps each member in its slot, cut turnover to a third and raised the agent's return to 3.7% a year. It beat the first run in all seven years, and the improvement is significant. The agent still trailed equal weight and the ETF, again mostly through trading costs. Both shortfalls are statistically significant, but only just.
- **Shared-weight actor (run 2026-11-10):** a policy head that scores every stock with the same weights no longer depends on the slot order at all. It halved turnover again and raised the return to 4.6% a year, with a portfolio close to equal weight. It still trailed equal weight (5.6%) and the ETF (7.1%), but neither shortfall is statistically significant.
- **Seed ensemble and no-trade band (2026-11-10, no new training):** averaging the three seeds' portfolios and skipping trades smaller than 0.5 percentage points brought the shared-weight agent to 5.5% a year, level with equal weight (5.6%), while trading less than equal weight. Its Sharpe ratio was slightly above equal weight's and below the ETF's; neither difference is significant. Part of the gain came from cash it happened to hold through the 2020 crash.
- **Heuristic benchmarks (2026-11-10):** momentum (5.3% a year), inverse-volatility weighting (5.0%) and long-only minimum variance (4.0%) did no better than equal weight or the ETF on the point-in-time SET50. No difference between the active strategies is significant.
- **Training with the band (run 2026-11-11):** trained with the same 0.5-point band, the agent made 5.1% a year (its seed ensemble 5.2%), against 4.6% without the band. It halved its trading but doubled its average cash, to 19%: five of its 21 models held more than a third of the portfolio in cash over their year. It did no better than adding the band after training (5.5%) or equal weight (5.6%), and no difference is significant.
- **LSTM forecast benchmark (2026-11-11):** the five-stock study's supervised LSTM forecaster, scaled to the index and buying its ten best forecasts monthly, made 8.1% a year, the highest mean return here. Almost all of its lead came from the rally of late 2020; its seeds disagree (−14.9% to +31.2% in 2020), and it trades 16 times its capital a year. Its Sharpe ratio (0.46) is below the ETF's (0.50), and no difference is significant.
- **BOT policy rate (run 2026-11-11):** five official policy-rate inputs lowered the agent's return, to 4.1% a year from 4.6%. Its seed ensemble with the band fell to 4.5% from 5.5%, a significant Sharpe shortfall (−0.06). The trained models barely read the new inputs: the shortfall came from cash that depended on the seed, as in the earlier arms.
- **Daily macro proxies (run 2026-11-11):** eight USD/THB, oil, gold and US-yield inputs lowered the return too, to 3.9% a year (seed ensemble with the band 4.3%); neither shortfall is significant. Unlike the policy rate, the agent read them heavily, and removing their information raised the return of 15 of its 21 models. On the point-in-time index, the technical and market inputs alone remain the best.

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
  - **Policy head (run 2026-11-10):** `SharedAssetPolicy` (`src/models/shared_asset_policy.py`) scores each stock with shared weights, an M5 item, which removes the dependence on slot order by construction. See the next section.
  - **Reweighting (not tried):** reducing the policy's own reweighting is a separate problem.

## Follow-up: shared-weight actor (2026-11-10)

**The change.** The variant `pit_plan_weekly_lstm_shared` keeps the sticky slots and replaces the dense policy head with `SharedAssetPolicy` (`src/models/shared_asset_policy.py`, `training.policy.actor: shared_asset`):
- **Scoring:** one network scores every stock from its own LSTM embedding and the mean embedding of the members in slots. Cash and the value estimate use only that mean.
- **Noise and empty slots:** one noise level is shared by all stocks, and empty slots are left out.
- **Normalization:** features are standardized with training-window statistics shared by all stocks. `VecNormalize` is not used, because it keeps separate statistics for every slot.

The reward, encoder, folds, seeds and budget are unchanged: 21 models (7 folds × 3 seeds × 100k timesteps), all trained locally.

**What it tests:** whether a policy that cannot depend on the slot order trades less and earns more than the sticky agent.

| Strategy | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | **Mean** | Years beating ETF | Mean max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SET50 ETF buy-and-hold (TDEX) | 24.5% | 18.2% | −6.5% | 5.7% | −11.8% | 12.1% | 7.8% | **7.1%** | — | −11.2% |
| Point-in-time SET50 equal weight | 22.2% | 16.5% | −10.0% | 6.5% | −8.5% | 11.9% | 0.7% | **5.6%** | 2/7 | −12.7% |
| **DRL, shared-weight actor** (± s.d. over 3 seeds) | 20.5% ± 0.7 | 14.5% ± 1.8 | −10.2% ± 0.6 | 4.7% ± 3.0 | −6.8% ± 2.0 | 9.8% ± 1.9 | −0.4% ± 0.4 | **4.6%** | 1/7 | −11.7% |
| DRL, sticky slots | 20.4% ± 6.3 | 15.5% ± 5.8 | −9.3% ± 3.3 | 4.4% ± 4.1 | −10.0% ± 1.7 | 8.0% ± 3.5 | −3.3% ± 2.3 | **3.7%** | 1/7 | −14.2% |
| DRL, traded-value slots (first run) | 15.7% ± 5.4 | 5.8% ± 3.4 | −15.0% ± 3.6 | 1.9% ± 5.2 | −15.1% ± 0.3 | 3.4% ± 5.2 | −4.5% ± 1.0 | **−1.1%** | 0/7 | −14.8% |

**The best DRL result so far, but still not ahead of either reference.**
- **Return:** 4.6% a year, against 3.7% for the sticky agent and −1.1% for the first run. It beat the sticky agent in five of the seven years and the first run in all seven. Chained over the seven years, 1 baht became 1.32 (1.24–1.41 across seeds), against 1.24 for the sticky agent, 1.41 in equal weight and 1.55 in the ETF ([figure](figures/pit_walk_forward_shared_chained_equity_20261110.png)).
- **Against the references:** it beat equal weight and the ETF only in 2020. Its best seed (5.5% a year) ended level with equal weight: 1 baht became 1.41 in both.
- **Drawdown:** its mean maximum drawdown (−11.7%) was shallower than equal weight's (−12.7%) and close to the ETF's (−11.2%). In 2020 its maximum drawdown was 32.8%, against 32.6% for the ETF and 37.0% for equal weight.
- **Seeds agree more closely:** the seed s.d. of yearly returns averaged 1.5 points, against 3.9 for the sticky agent.

**Significance.** Sharpe ratios on the same 1,699 chained daily returns, with the same stationary bootstrap:

| Comparison | Sharpe difference | 95% interval |
| --- | ---: | ---: |
| DRL shared − DRL sticky | +0.08 | −0.06 to +0.23 |
| DRL shared − DRL first run | +0.36 | +0.24 to +0.54 |
| DRL shared − point-in-time equal weight | −0.03 | −0.11 to +0.04 |
| DRL shared − ETF | −0.15 | −0.42 to +0.08 |

- **DRL:** the shared agent's Sharpe ratio was 0.35 (−0.38 to +1.13). Its deflated Sharpe ratio was 0.29, counting eight designs: the six screened earlier, the sticky order and this head, both chosen after seeing earlier validation results. Counting eight designs, the sticky agent's value would be 0.23 rather than 0.25, and the first run's 0.07.
- **Reading:** only the improvement over the first run is significant. The gain over the sticky agent is not. Unlike the sticky agent's, the shortfalls against equal weight and the ETF are not significant either: both intervals include zero. That does not make the agent as good as equal weight; the point estimates are still below both references.

**Where the return went.** The same diagnostics as above, as means over the seven years:

| Strategy | Return before costs | Cost drag | Return after costs | Turnover a year | Average cash |
| --- | ---: | ---: | ---: | ---: | ---: |
| **DRL, shared-weight actor** | 5.5% | 0.9% | 4.6% | 4.3× | 10.7% |
| DRL, sticky slots | 5.6% | 2.0% | 3.7% | 9.4× | 2.7% |
| DRL, traded-value slots (first run) | 4.6% | 5.8% | −1.1% | 28.2× | 3.7% |
| Equal weight, point-in-time SET50 | 6.0% | 0.4% | 5.6% | 1.9× | 0.7% |

- **The slot order no longer matters.** `scripts/check_slot_order_invariance.py` rolled every trained model out on its validation year twice, once per slot order. The two orders seat members differently on every session.
  - **Shared head:** each of the 21 shared models ended at the same value under both orders, to within 3×10⁻⁸.
  - **Dense head:** the 21 dense sticky models, given the same test, ended a median 3.6 points of annual return apart, and up to 24.7 points.
- **Turnover halved again:** from 9.4 to 4.3 times capital a year, or from 0.17 to 0.07 per weekly decision. The cost drag fell from 2.0 to 0.9 points a year. After the initial purchase it still turned over 3.4 times its capital a year, against 0.9 for equal weight.
- **It learned a portfolio close to equal weight:**
  - **Holdings:** 43.7 effective names out of 44.5 tradable members on average (about 19 for the sticky agent).
  - **Top ten positions:** 23% of capital (59%).
  - **Distance from equal weight:** a total-variation distance of 0.13 (0.40).
  - **Correlation:** its daily returns correlated 0.995 with equal weight (0.94).
- **Cash depended on the seed; it was not market timing.**
  - **Spread:** average cash was 10.7%, but the median model held 3.7%. Four of the 21 models held 19–79% cash on average over their validation year (2017 and 2019 seed 44, 2020 seed 43, 2021 seed 42).
  - **Timing:** for most models the cash weight stayed within a few points all year. It did not rise in the 2020 crash: averaged over seeds, it fell from about 21% in February to 11% in March.
- **The 1.0-point gap to equal weight splits evenly:** 0.5 points of extra costs and 0.5 points of lower return before costs. At equal weight's yearly returns, the cash holdings would have cost about 0.4 points a year, so cash explains most of the second half.

**What it means:**
- **The head fixed what it was built to fix.** The policy no longer depends on the slot order, it trades half as much as the sticky agent, and its seeds agree more closely.
- **What it learned is close to equal weight, with somewhat higher costs and a seed-dependent cash holding.** The return is still below equal weight and the ETF, though no longer significantly. Across the three runs there is no evidence that the agent adds value over equal weight on the point-in-time index.
- **Next steps:** each adds a design to the deflated Sharpe count. Both were tried next (see the next section).
  - **Turnover:** a no-trade band, a turnover penalty or monthly decisions would target the remaining 3.4 times a year after entry.
  - **Cash:** averaging the three seeds' weights would damp the seed-specific cash holdings.

## Follow-up: seed ensemble and no-trade band (2026-11-10)

**What it tests:** whether the shared-weight agent's two remaining weaknesses, extra trading and seed-dependent cash, can be removed without training. Both changes were applied to the 21 trained shared-weight models (`scripts/evaluate_pit_seed_ensemble.py`):
- **Seed ensemble:** on every weekly decision, the three seeds' models score the same observation, and the portfolio holds the mean of their target weights, cash included.
- **No-trade band:** a holding is traded only when the policy's target differs from its current weight by more than 0.5 percentage points, about a fifth of an equal-weight position. (A 25% relative band at 44 holdings would be 0.57 points.) Otherwise the holding is left as it is, and cash takes up the difference (`environment.no_trade_band` in `src/envs/point_in_time_env.py`).
  - **Fixed in advance:** the width was set before any band result was seen.
  - **Evaluation only:** the models were trained without the band.
- **Check:** before the new evaluations, every unchanged model replayed its saved validation curve exactly (largest relative gap 2×10⁻¹⁶).

| Strategy | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | **Mean** | Years beating ETF | Mean max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SET50 ETF buy-and-hold (TDEX) | 24.5% | 18.2% | −6.5% | 5.7% | −11.8% | 12.1% | 7.8% | **7.1%** | — | −11.2% |
| Point-in-time SET50 equal weight | 22.2% | 16.5% | −10.0% | 6.5% | −8.5% | 11.9% | 0.7% | **5.6%** | 2/7 | −12.7% |
| **Seed ensemble + no-trade band** | 21.2% | 15.3% | −9.7% | 5.6% | −3.8% | 10.0% | 0.2% | **5.5%** | 1/7 | −11.3% |
| No-trade band at evaluation (± s.d. over 3 seeds) | 20.8% ± 1.1 | 15.0% ± 1.9 | −10.1% ± 0.8 | 5.1% ± 3.4 | −4.4% ± 4.3 | 10.5% ± 2.3 | 0.0% ± 0.3 | **5.3%** | 1/7 | −11.5% |
| Seed ensemble | 20.7% | 14.6% | −9.9% | 4.8% | −6.5% | 10.0% | −0.1% | **4.8%** | 1/7 | −11.7% |
| DRL, shared-weight actor (± s.d. over 3 seeds) | 20.5% ± 0.7 | 14.5% ± 1.8 | −10.2% ± 0.6 | 4.7% ± 3.0 | −6.8% ± 2.0 | 9.8% ± 1.9 | −0.4% ± 0.4 | **4.6%** | 1/7 | −11.7% |

**Level with equal weight, mostly through the band:**
- **Return:** with both changes the agent made 5.5% a year, against 5.6% for equal weight. Chained over the seven years, 1 baht became 1.41 in both ([figure](figures/pit_seed_ensemble_chained_equity_20261110.png)). It beat equal weight in 2018 and 2020, and the ETF only in 2020.
- **Each change on its own:** the band raised the return from 4.6% to 5.3% a year; the ensemble raised it to 4.8%.
- **2020 carries part of the band's gain.** Outside 2020, the band added 0.4 points a year to the ensemble, about what it saved in costs. In 2020 it added 2.7 points. In March the ensemble without the band cut its cash from 21% to 11%, buying into the crash. With the band, most of those purchases were smaller than the band, and cash fell only to 17%. The agent did not choose to hold that cash; the band left it in place.

**Significance.** Sharpe ratios on the same 1,699 chained daily returns, with the same stationary bootstrap:

| Comparison | Sharpe difference | 95% interval |
| --- | ---: | ---: |
| Seed ensemble + band − shared actor | +0.07 | +0.03 to +0.10 |
| Band at evaluation − shared actor | +0.05 | +0.03 to +0.06 |
| Seed ensemble − shared actor | +0.01 | +0.01 to +0.02 |
| Seed ensemble + band − point-in-time equal weight | +0.04 | −0.05 to +0.11 |
| Seed ensemble + band − ETF | −0.08 | −0.36 to +0.15 |

- **DRL:** the ensemble with the band had a Sharpe ratio of 0.42 (−0.33 to +1.18), against 0.38 for equal weight, 0.50 for the ETF and 0.35 for the shared actor.
- **Deflated Sharpe:** 0.27, counting 13 designs, all chosen after seeing earlier validation results:
  - the eight above;
  - the three here;
  - the run that trains with the band (below) and its seed ensemble.

  At 13 designs the shared actor's value is 0.22 (0.29 at eight).
- **Reading:** all three changes improve significantly on the shared agent. The curves are highly correlated, so even small differences are measurable. Against equal weight, the ensemble with the band has a slightly higher Sharpe ratio and a slightly lower return; neither difference is significant. It stays below the ETF, also not significantly.

**Where the return went.** The same diagnostics as above, as means over the seven years:

| Strategy | Return before costs | Cost drag | Return after costs | Turnover a year | after the initial purchase | Average cash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Seed ensemble + no-trade band** | 5.9% | 0.3% | 5.5% | 1.6× | 0.7× | 10.5% |
| No-trade band at evaluation | 5.7% | 0.5% | 5.3% | 2.2× | 1.4× | 11.0% |
| Seed ensemble | 5.5% | 0.7% | 4.8% | 3.3× | 2.4× | 10.6% |
| DRL, shared-weight actor | 5.5% | 0.9% | 4.6% | 4.3× | 3.4× | 10.7% |
| Equal weight, point-in-time SET50 | 6.0% | 0.4% | 5.6% | 1.9× | 0.9× | 0.7% |

- **It now trades less than equal weight.** After the initial purchase, the ensemble with the band turned over 0.7 times its capital a year, against 0.9 for equal weight. Its cost drag of 0.3 points was below equal weight's 0.4.
- **Averaging the seeds smooths the trading.** The ensemble alone cut turnover from 4.3 to 3.3 times a year, because the seeds' week-to-week changes partly cancel.
- **The ensemble did not remove the cash.**
  - **Average:** the ensemble holds the seeds' mean cash, so the average stays at about 10.6%.
  - **Spread:** no year now depends on one seed. The highest yearly average fell from 79% (2019, seed 44) to 29%.
- **Still close to equal weight:** daily correlation 0.998 with equal weight, and 43.9 effective holdings out of 44.5 tradable members.
- **The remaining 0.07-point gap:** 0.13 points lower return before costs, partly offset by 0.06 points lower costs.

**Band width, a sensitivity check.** The same ensemble with other band widths (`results/pit_band_sensitivity_20261110/`). None of them was used to choose the band. Counted as designs, they would raise the total from 13 to 16.

| Band (points) | Mean annual return | 2020 | Mean without 2020 | Turnover after entry | Cost drag | Average cash | Sharpe | Sharpe vs equal weight (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (ensemble only) | 4.8% | −6.5% | 6.7% | 2.4× | 0.7% | 10.6% | 0.36 | −0.02 (−0.10 to +0.05) |
| 0.25 | 5.3% | −5.4% | 7.0% | 1.2× | 0.4% | 10.7% | 0.39 | +0.01 (−0.07 to +0.08) |
| **0.5 (chosen in advance)** | 5.5% | −3.8% | 7.1% | 0.7× | 0.3% | 10.5% | 0.42 | +0.04 (−0.05 to +0.11) |
| 1.0 | 5.8% | −5.3% | 7.7% | 0.4× | 0.3% | 9.9% | 0.44 | +0.06 (−0.05 to +0.16) |
| 2.0 | 9.3% | +19.0% | 7.7% | 0.5× | 0.3% | 16.9% | 0.69 | +0.31 (−0.05 to +0.67) |
| Equal weight | 5.6% | −8.5% | 8.0% | 0.9× | 0.4% | 0.7% | 0.38 | — |

- **Up to 1 point:** a wider band cuts trading further and adds a little return. Outside 2020, none of these widths beats equal weight (7.0–7.7% against 8.0%), and no Sharpe difference is significant.
- **At 2 points the band stops working as a band.** It is about the size of a whole position: the ensemble's targets are about 2%, so most stocks are never bought.
  - **2019:** the portfolio held 72% cash.
  - **2020:** it held 98% cash in January and bought during the March crash. That made 2020 its best year (+19%), by accident.
- **The chosen 0.5 points** lies in the range where the band works as intended, and the conclusion does not depend on it.

**What it means:**
- **Engineering, not learning, closed most of the gap.** The two weaknesses of the shared-weight run were trading and seed dependence. An execution rule and averaging over seeds removed most of the gap to equal weight without new training.
- **Indistinguishable from equal weight, with less trading.** The agent now has a slightly higher Sharpe ratio and a slightly lower return than equal weight.
- **Still no evidence that it adds value.** What it learned is close to equal weight, and part of the remaining gain came from cash the band left in place during the 2020 crash.
- **Next:** training with the band (variant `pit_plan_weekly_lstm_shared_band`, 21 models) ran overnight on 2026-11-10/11. It is reported [below](#follow-up-training-with-the-no-trade-band-2026-11-11).

## Follow-up: heuristic benchmarks (2026-11-10)

**What it adds:** three standard benchmarks that the plan lists but that had not been run (M6). `scripts/pit_heuristic_baselines.py` runs them on the same sessions, members, costs and accounting. Like equal weight, they rebalance every 21 sessions, and they use closing prices up to the session before each decision. The parameters are textbook choices fixed before any result was seen:
- **Momentum:** equal weight in the ten members with the highest return from 12 months to 1 month before the decision.
- **Inverse volatility:** every member, weighted by the inverse of the volatility of its last 60 daily returns.
- **Minimum variance:** the long-only minimum-variance portfolio from 252 days of returns. The covariance is shrunk halfway toward its diagonal, and each weight is capped at 10%.

| Strategy | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | **Mean** | Years beating ETF | Mean max DD | Turnover a year | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SET50 ETF buy-and-hold (TDEX) | 24.5% | 18.2% | −6.5% | 5.7% | −11.8% | 12.1% | 7.8% | **7.1%** | — | −11.2% | 1.0× | 0.50 |
| Point-in-time SET50 equal weight | 22.2% | 16.5% | −10.0% | 6.5% | −8.5% | 11.9% | 0.7% | **5.6%** | 2/7 | −12.7% | 1.9× | 0.38 |
| DRL, seed ensemble + no-trade band | 21.2% | 15.3% | −9.7% | 5.6% | −3.8% | 10.0% | 0.2% | **5.5%** | 1/7 | −11.3% | 1.6× | 0.42 |
| Momentum, top ten | 11.0% | 10.0% | −26.5% | 27.9% | −4.6% | 18.5% | 0.7% | **5.3%** | 3/7 | −17.3% | 6.4× | 0.30 |
| Inverse volatility | 19.5% | 17.0% | −8.3% | 5.0% | −10.6% | 11.1% | 1.5% | **5.0%** | 1/7 | −12.2% | 2.6× | 0.36 |
| Minimum variance | 19.0% | 9.9% | −3.3% | 7.7% | −22.4% | 14.4% | 2.5% | **4.0%** | 3/7 | −10.9% | 3.3× | 0.30 |

The ETF's turnover is its initial purchase. Sharpe ratios are on the 1,699 chained daily returns.

- **None of the heuristics beat equal weight or the ETF on average.**
  - **Momentum** came closest (5.3%), with the deepest drawdowns (−26.5% in 2018) and the most trading (6.4 times capital a year).
  - **Minimum variance** had the shallowest mean drawdown (−10.9%) but the lowest return, and the largest 2020 loss of any strategy here (−22.4%).
- **Significance:** against equal weight, the Sharpe differences are −0.08 for momentum (95% CI −0.52 to +0.34), −0.02 for inverse volatility (−0.09 to +0.06) and −0.08 for minimum variance (−0.36 to +0.27). Against the ETF they are −0.20, −0.14 and −0.20; every interval includes zero.
- **Where the agent stands:** the seed ensemble with the band has the highest Sharpe ratio of the active strategies (0.42), ahead of the three heuristics (0.30–0.36) and equal weight (0.38), and below the ETF (0.50). None of these differences is significant. Seven validation years separate only large differences, except between strategies as closely correlated as the agent and equal weight.

**What it means:** the agent's failure to beat the references is not unusual. On the point-in-time SET50 over 2016–2022, none of the standard heuristic strategies beat equal weight or the ETF either. The ETF remains the strongest reference: it holds the whole index, including the members without Yahoo prices, by market value.

## Follow-up: training with the no-trade band (2026-11-11)

**What it tests:** whether the agent does better when it learns with the band than when the band is added after training.
- **Variant:** `pit_plan_weekly_lstm_shared_band` is the shared-weight agent with the same 0.5-point band in its environment, in training and in validation.
  - **Unchanged:** seeds, timesteps and everything else: 7 folds × 3 seeds × 100k timesteps (`results/pit_walk_forward_band_20261110/`).
  - **Seed ensemble:** evaluated afterwards, with the band and without further training (`results/pit_band_ensemble_20261111/`).
- **Designs 12 and 13** of the deflated Sharpe count. Both were planned before any result of this run was seen.
- **Check:** every trained model replayed its saved validation curve exactly (largest relative gap 2×10⁻¹⁶).

| Strategy | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | **Mean** | Years beating ETF | Mean max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SET50 ETF buy-and-hold (TDEX) | 24.5% | 18.2% | −6.5% | 5.7% | −11.8% | 12.1% | 7.8% | **7.1%** | — | −11.2% |
| Point-in-time SET50 equal weight | 22.2% | 16.5% | −10.0% | 6.5% | −8.5% | 11.9% | 0.7% | **5.6%** | 2/7 | −12.7% |
| Seed ensemble, band at evaluation (previous section) | 21.2% | 15.3% | −9.7% | 5.6% | −3.8% | 10.0% | 0.2% | **5.5%** | 1/7 | −11.3% |
| **Seed ensemble, trained with the band** | 20.2% | 15.2% | −7.1% | 5.4% | −7.6% | 10.3% | 0.3% | **5.2%** | 1/7 | −10.8% |
| DRL, trained with the band (± s.d. over 3 seeds) | 20.5% ± 0.8 | 14.8% ± 1.5 | −7.2% ± 5.0 | 4.8% ± 3.8 | −7.1% ± 1.6 | 9.9% ± 3.5 | 0.3% ± 0.3 | **5.1%** | 1/7 | −10.8% |
| DRL, shared-weight actor, no band (± s.d. over 3 seeds) | 20.5% ± 0.7 | 14.5% ± 1.8 | −10.2% ± 0.6 | 4.7% ± 3.0 | −6.8% ± 2.0 | 9.8% ± 1.9 | −0.4% ± 0.4 | **4.6%** | 1/7 | −11.7% |

**Better than without the band, not better than adding it afterwards:**
- **Return:** trained with the band, the agent made 5.1% a year, 0.5 points more than without it. Its seed ensemble made 5.2%.
  - **Against the references:** both stay below the ensemble with the band added after training (5.5%) and equal weight (5.6%).
  - **Chained** over the seven years, 1 baht became 1.39 for the ensemble trained with the band. It became 1.41 for both the ensemble with the band at evaluation and equal weight, and 1.55 for the ETF ([figure](figures/pit_band_training_chained_equity_20261111.png)).
- **One model's cash in 2018 carries most of the gain.**
  - **2018:** the seed-42 model held 93% cash all year and lost 1.4%, while equal weight lost 10.0%. That year adds 0.4 of the 0.5 points.
  - **Other years:** outside 2018, training with the band added 0.1 points a year (7.2% against 7.1%).

**It traded half as much, but held twice as much cash:**
- **Trading:** turnover fell from 4.3 to 2.0 times capital a year; after the initial purchase, from 3.4 to 1.2.
- **Cash:** average cash rose from 10.7% to 19.2%. Five of the 21 models held more than a third of the portfolio in cash over their year:
  - 2018, seed 42: 93%;
  - 2019, seeds 43 and 44: 37% and 94%;
  - 2021, seed 42: 44%;
  - 2022, seed 42: 57%.

  Without the band, two models did: 79% (2019, seed 44) and 49% (2020, seed 43).
- **Seed spread:** the seed standard deviation of annual returns rose from 1.5 to 2.4 points.
- **A plausible mechanism, not tested:**
  - **Drift without signal:** with the band, small changes in the targets change no trades, so training gets no signal against a cash weight that grows slowly.
  - **Hard to re-enter:** once the portfolio is mostly cash, a stock is bought only when its target exceeds 0.5 points, so the agent must cut its cash target in a large step to re-enter. At 93% cash, an even split of the rest gives each of about 44 stocks 0.16 points.
- **The cash does not follow the market (attribution, 2026-11-11).** The five cash-heavy models were re-run with each input group shuffled across sessions or held at its training mean (`scripts/pit_feature_group_attribution.py`, `results/pit_attribution_band_20261111/`).
  - **Shuffled inputs:** their cash target moved by 3 points or less. The target is the model's own cash weight before the band; the executed cash is higher, because the band blocks small purchases. In 2018, seed 42 targeted 82% and held 93%.
  - **Inputs held at their training mean:** the technical inputs lowered the cash by 9–17 points in three models. In the other two (86% in 2019, seed 44; 59% in 2022, seed 42), no group moved it by more than 2 points.
  - **Reading:** the cash is a near-constant output of each trained model, not a reaction to market conditions. That fits the drift explanation but does not prove it.

**Significance.** Sharpe ratios on the same 1,699 chained daily returns, with the same stationary bootstrap:

| Comparison | Sharpe difference | 95% interval |
| --- | ---: | ---: |
| Trained with the band − shared actor | +0.04 | −0.07 to +0.17 |
| Seed ensemble, trained with the band − shared actor | +0.05 | −0.06 to +0.18 |
| Seed ensemble, trained with the band − band at evaluation | −0.02 | −0.12 to +0.12 |
| Seed ensemble, trained with the band − point-in-time equal weight | +0.01 | −0.06 to +0.12 |
| Seed ensemble, trained with the band − ETF | −0.10 | −0.36 to +0.14 |

- **Sharpe ratios:** 0.39 for both the ensemble and the mean of the seeds trained with the band. For comparison: 0.42 for the ensemble with the band at evaluation, 0.38 for equal weight, 0.35 for the shared actor and 0.50 for the ETF.
- **No significant gain over the shared actor.** The band at evaluation improved on the shared actor significantly (+0.07, +0.03 to +0.10). Training with the band did not: the models that sat in cash made its returns differ from the shared actor's in a few years, so the interval is wide.
- **Deflated Sharpe:** 0.25 for both, counting 13 designs (0.27 for the ensemble with the band at evaluation).

**Where the return went.** The same diagnostics as above, as means over the seven years:

| Strategy | Return before costs | Cost drag | Return after costs | Turnover a year | after the initial purchase | Average cash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed ensemble, band at evaluation | 5.9% | 0.3% | 5.5% | 1.6× | 0.7× | 10.5% |
| **Seed ensemble, trained with the band** | 5.6% | 0.3% | 5.2% | 1.5× | 0.7× | 18.0% |
| DRL, trained with the band | 5.6% | 0.4% | 5.1% | 2.0× | 1.2× | 19.2% |
| DRL, shared-weight actor | 5.5% | 0.9% | 4.6% | 4.3× | 3.4× | 10.7% |
| Equal weight, point-in-time SET50 | 6.0% | 0.4% | 5.6% | 1.9× | 0.9× | 0.7% |

- **Costs:** training with the band cut the cost drag from 0.9 to 0.4 points, the level of equal weight. The ensemble cut it to 0.3.
- **Return before costs:** 5.6%, 0.3 points below the ensemble with the band at evaluation and 0.4 below equal weight.
- **Still close to equal weight:** the ensemble's daily correlation with equal weight was 0.997, with 43.8 effective holdings out of 44.5 tradable members (40.6 for a single model).

**What it means:**
- **Learning with the band did not beat imposing it afterwards.**
  - **As intended:** training with the band cut trading.
  - **Side effect:** the agent also learned to hold more cash, in a few models almost all of it.
  - **Result:** its return (5.1–5.2%) is below that of the band added after training (5.5%) and of equal weight (5.6%). None of these differences is significant.
- **Cash is now the agent's main weakness.** In both band designs, the gain over the shared actor came largely from cash held through one falling year: 2020 for the band at evaluation, 2018 here. In neither case was the cash consistent across seeds.
- **Best design so far:** the seed ensemble with the band at evaluation remains the best point-in-time DRL design on the validation years. It was fixed before this run, and the deflated Sharpe ratio counts all 13 designs.
- **Next:** penalizing or capping cash in training would target the remaining weakness. It would add a design to the deflated Sharpe count and has not been started.

## Follow-up: LSTM forecast benchmark (2026-11-11)

**What it adds:** the supervised sequence benchmark of the five-stock study, scaled to the point-in-time SET50. The plan lists it among the benchmarks (M6); until now only its ridge version had run on this universe. `scripts/pit_lstm_forecast_baseline.py` (`results/pit_lstm_forecast_20261111/`):
- **Forecaster:** for each fold, a bidirectional LSTM is fitted on the training years only.
  - **Input:** a member's last 30 completed sessions of the agent's 16 inputs (11 technical, 5 SET50 ETF).
  - **Target:** the return over the next 21 sessions.
  - **Samples:** members on sessions where they were tradable, every fifth session: 6,000 to 16,800 per fold. No target extends past the training end.
  - **Settings:** those of the five-stock study (the `LSTMForecaster` defaults), with early stopping on the last 15% of the training sessions.
- **Allocation:** every 21 sessions, equal weight in the ten highest positive forecasts among the tradable members, like the ridge benchmark. It uses the same sessions, costs and accounting as the agent.
- **Fixed in advance:** the sampling stride, the ten holdings and the three seeds (42–44) were set before any result was seen. All 21 fits took 3.5 minutes.

| Strategy | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | **Mean** | Years beating ETF | Mean max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SET50 ETF buy-and-hold (TDEX) | 24.5% | 18.2% | −6.5% | 5.7% | −11.8% | 12.1% | 7.8% | **7.1%** | — | −11.2% |
| **LSTM forecast, top ten monthly (± s.d. over 3 seeds)** | 19.4% ± 7.4 | 20.4% ± 1.0 | −13.6% ± 4.9 | 13.4% ± 1.8 | 10.8% ± 23.5 | 7.7% ± 6.3 | −1.7% ± 0.8 | **8.1%** | 3/7 | −14.7% |
| Ridge forecast, top ten monthly | 1.2% | 16.2% | −6.4% | 16.8% | 7.9% | 4.1% | 0.3% | **5.7%** | 3/7 | −17.6% |
| Point-in-time SET50 equal weight | 22.2% | 16.5% | −10.0% | 6.5% | −8.5% | 11.9% | 0.7% | **5.6%** | 2/7 | −12.7% |
| DRL, seed ensemble + no-trade band | 21.2% | 15.3% | −9.7% | 5.6% | −3.8% | 10.0% | 0.2% | **5.5%** | 1/7 | −11.3% |

**The highest mean return on the point-in-time index, from one rally:**
- **Return:** 8.1% a year, above the ETF (7.1%) and equal weight (5.6%). Chained, 1 baht became 1.62 (the mean of the seeds, which range from 1.52 to 1.76), against 1.55 for the ETF and 1.41 for equal weight ([figure](figures/pit_lstm_forecast_chained_equity_20261111.png)).
- **Late 2020 carries the lead.**
  - **The rally:** in November and December 2020, all three seeds gained 42–50%, against 23% for equal weight. On 10 November, after the first vaccine trial results, two seeds gained 9.8% and 8.4% in a day, against 6.0% for equal weight.
  - **Amplified, not different:** its largest daily moves were the market's, only larger.
  - **Outside 2020** it made 7.6% a year, against 8.0% for equal weight and 10.3% for the ETF.
- **Seeds disagree.** In 2020 they made +31.2%, +16.1% and −14.9%. The seed standard deviation of annual returns is 6.5 points, against 1.5 for the DRL agent.
- **Trading:** it replaces most of its ten holdings every month.
  - **Turnover:** 15.8 times capital a year, with a cost drag of 3.5 points.
  - **Before costs:** it made 11.5% a year, against 6.0% for equal weight.
- **Forecast skill is weak.** The information coefficient is the rank correlation between its forecasts and the realized 21-session returns on the validation decisions.
  - **Mean:** 0.03, positive in 13 of the 21 fits.
  - **By fold:** from −0.04 (2021) to 0.10 (2020).

**Significance.** Sharpe ratios on the same 1,699 chained daily returns, with the same stationary bootstrap:

| Comparison | Sharpe difference | 95% interval |
| --- | ---: | ---: |
| LSTM forecast − point-in-time equal weight | +0.08 | −0.23 to +0.36 |
| LSTM forecast − ETF | −0.03 | −0.42 to +0.31 |
| LSTM forecast − DRL seed ensemble + band | +0.05 | −0.25 to +0.33 |
| LSTM forecast − ridge forecast | +0.10 | −0.38 to +0.62 |

- **Sharpe ratio:** 0.46 (−0.27 to 1.25), between the DRL ensemble with the band (0.42) and the ETF (0.50).
- **No difference is significant.** The lead of a concentrated, high-turnover portfolio in one rally is well within what seven years can produce by chance.

**What it means:**
- **Supervised forecasting did not reliably beat the references either.** The LSTM forecaster has the highest mean return on the point-in-time index. But its lead came from one rally, its seeds disagree, and its Sharpe ratio is below the ETF's; no difference is significant.
- **Against the DRL agent:** the agent's return is lower, but steadier across seeds, and it trades a tenth as much.
- **Against the five-stock study:** there the LSTM forecaster made 13.8% a year. On the actual index it makes 8.1%, so, like the ridge forecaster, part of its earlier result came from the hand-picked universe.

## Follow-up: BOT policy rate as an input (2026-11-11)

**What it tests:** whether an official macro source improves the point-in-time agent (RQ2, RQ3). The Bank of Thailand policy rate is the first source beyond prices and the SET50 ETF tested on this universe.
- **Variant:** `pit_plan_weekly_lstm_shared_policy` is the shared-weight agent with five more market-level inputs from the BOT decision history (`src/features/policy_rate_context.py`):
  - the rate in effect;
  - its change over the past year;
  - the size of the last change;
  - the time since that change;
  - the share of MPC votes against the latest decision.
- **Timing:** each decision is available from the session after its announcement. Decisions after the validation end are dropped before the merge.
- **Unchanged:** data, splits, seeds, timesteps and everything else. `config/pit_walk_forward_policy_rate.yaml` is generated from the main config: 7 folds × 3 seeds × 100k timesteps (`results/pit_walk_forward_policy_20261111/`).
- **Seed ensemble with the band:** evaluated afterwards without training, as for the shared run (`results/pit_policy_ensemble_20261111/`).
- **Designs 14 and 15** of the deflated Sharpe count. The macro-proxy arm (designs 16 and 17) was configured before any result of this run was seen.
- **Check:** every trained model replayed its saved validation curve exactly (largest relative gap 2×10⁻¹⁶).

| Strategy | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | **Mean** | Years beating ETF | Mean max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SET50 ETF buy-and-hold (TDEX) | 24.5% | 18.2% | −6.5% | 5.7% | −11.8% | 12.1% | 7.8% | **7.1%** | — | −11.2% |
| Point-in-time SET50 equal weight | 22.2% | 16.5% | −10.0% | 6.5% | −8.5% | 11.9% | 0.7% | **5.6%** | 2/7 | −12.7% |
| Seed ensemble + band, without the policy rate | 21.2% | 15.3% | −9.7% | 5.6% | −3.8% | 10.0% | 0.2% | **5.5%** | 1/7 | −11.3% |
| **Seed ensemble + band, with the policy rate** | 20.3% | 12.4% | −9.3% | 5.2% | −7.1% | 9.6% | 0.5% | **4.5%** | 1/7 | −11.3% |
| DRL with the policy rate (± s.d. over 3 seeds) | 20.3% ± 0.5 | 10.8% ± 4.5 | −10.1% ± 0.5 | 5.0% ± 2.6 | −6.4% ± 3.5 | 8.9% ± 3.0 | 0.0% ± 0.1 | **4.1%** | 1/7 | −11.4% |
| DRL, shared-weight actor, without it (± s.d. over 3 seeds) | 20.5% ± 0.7 | 14.5% ± 1.8 | −10.2% ± 0.6 | 4.7% ± 3.0 | −6.8% ± 2.0 | 9.8% ± 1.9 | −0.4% ± 0.4 | **4.6%** | 1/7 | −11.7% |

**Lower, not higher:**
- **Return:** with the policy rate, the agent made 4.1% a year, 0.5 points less than without it. Its seed ensemble with the band made 4.5%, a full point less than the same ensemble without the policy rate (5.5%).
- **Chained** over the seven years, 1 baht became 1.32 for the ensemble with the policy rate. It became 1.41 without it and for equal weight, and 1.55 for the ETF ([figure](figures/pit_policy_rate_chained_equity_20261111.png)).
- **Two years carry the loss.**
  - **2017:** the single models made 10.8% against 14.5%. The seed-43 model held 54% cash in a year when equal weight made 16.5%, and made 6.3%.
  - **2020:** the ensemble lost 7.1% against 3.8% without the policy rate. Both held about a fifth of the portfolio in cash through the year. The gap opened between January and April, through the crash and most of all in the April rebound, and never closed.
  - **Other years:** the ensembles differ by less than a point in each of the other five years.

**Cash again, and more of it:**
- **Models in cash:** four of the 21 models held more than a third of the portfolio in cash over their year:
  - 2017, seed 43: 54%;
  - 2019, seed 44: 45%;
  - 2020, seed 42: 49%;
  - 2021, seed 43: 42%.

  Without the policy rate, two did: 79% (2019, seed 44) and 49% (2020, seed 43).
- **Average cash:** 12.2%, against 10.7%.
- **Seed spread:** the seed standard deviation of annual returns rose from 1.5 to 2.1 points.

**Significance.** Sharpe ratios on the same 1,699 chained daily returns, with the same stationary bootstrap:

| Comparison | Sharpe difference | 95% interval |
| --- | ---: | ---: |
| With the policy rate − shared actor | −0.02 | −0.09 to +0.03 |
| Seed ensemble + band: with − without the policy rate | −0.06 | −0.11 to −0.01 |
| Seed ensemble + band with the policy rate − point-in-time equal weight | −0.03 | −0.09 to +0.03 |
| Seed ensemble + band with the policy rate − ETF | −0.14 | −0.41 to +0.07 |

- **Sharpe ratios:** 0.33 for the mean of the seeds with the policy rate and 0.35 for its ensemble with the band. For comparison: 0.35 for the shared actor, 0.42 for its ensemble with the band, 0.38 for equal weight and 0.50 for the ETF.
- **The ensemble's shortfall is significant; the single models' is not.** The two ensembles hold nearly the same portfolio, so even a small, steady gap is measured precisely. The gap comes mostly from 2017 and 2020.
- **Deflated Sharpe,** counting 15 designs: 0.20 for the ensemble with the policy rate and 0.18 for the mean of its seeds. The best design, the ensemble with the band and without the policy rate, falls from 0.27 to 0.25 under the larger count.

**Where the return went.** The same diagnostics as above, as means over the seven years:

| Strategy | Return before costs | Cost drag | Return after costs | Turnover a year | after the initial purchase | Average cash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed ensemble + band, without the policy rate | 5.9% | 0.3% | 5.5% | 1.6× | 0.7× | 10.5% |
| **Seed ensemble + band, with the policy rate** | 4.8% | 0.3% | 4.5% | 1.4× | 0.6× | 12.3% |
| DRL with the policy rate | 4.9% | 0.8% | 4.1% | 3.8× | 2.9× | 12.2% |
| DRL, shared-weight actor | 5.5% | 0.9% | 4.6% | 4.3× | 3.4× | 10.7% |
| Equal weight, point-in-time SET50 | 6.0% | 0.4% | 5.6% | 1.9× | 0.9× | 0.7% |

- **The shortfall is before costs.** With the policy rate, the agent traded slightly less but earned 0.6 points a year less before costs (1.1 for the ensembles).
- **Still close to equal weight:** the ensemble's daily correlation with equal weight was 0.998, with 44.0 effective holdings out of 44.5 tradable members.

**Out of range in validation.** The rate changes rarely. The training data hold seven cuts (2012–2015) and then no change for 3.6 years, until the December 2018 hike; the validation years bring nine changes. So in four of the seven validation years, at least one input spent much of the year outside the range of its training years:

| Fold | Input | Validation sessions outside the training range | Largest distance from the training mean (training s.d.) |
| --- | --- | ---: | ---: |
| 2016 | Time since the last change | 68% | 5.6 |
| 2017 | Time since the last change | 100% | 5.2 |
| 2018 | Time since the last change | 97% | 3.9 |
| 2018 | Size of the last change (the first hike in the data) | 3% | 50 |
| 2020 | Rate in effect | 90% | 2.6 |
| 2020 | Change over the past year | 62% | 4.2 |
| 2022 | Change over the past year | 26% | 2.9 |

In every training session of the folds up to 2018, the last change was a 25-basis-point cut. So that input's training standard deviation is set to 1, and the first hike sits 50 units away.

**Does the agent read the policy rate?** `scripts/pit_feature_group_attribution.py` re-ran each of the 21 models on its validation year with one input group changed and the model unchanged (`attribution_*` files in the run folder). Two kinds of change:
- **Shuffled:** the group's values are permuted across sessions, five times, as in the five-stock attribution.
- **Held at the training mean:** the actor standardizes this to zero, so the model gets no information from the group, and the out-of-range values disappear.

The portfolio shift is the half-L1 distance between the stock weights with and without the change, averaged over decisions. All changes are means over the 21 models:

| Input group | Inputs | Portfolio shift, shuffled | Portfolio shift, at the training mean | Change in annual return, at the training mean (mean absolute) | Change in cash, at the training mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Technical | 11 | 3.5% | 3.5% | 0.7 points | −2.5 points |
| SET50 market context | 5 | 0.9% | 1.0% | 0.2 points | +0.6 points |
| **BOT policy rate** | 5 | 0.7% | 2.0% | 0.4 points | +0.9 points |

- **It reads the policy rate, but only a little:** shuffling it moved the portfolio by 0.7%, less than market context (0.9%) and a fifth as much as the technical inputs (3.5%). Holding it at the training mean moved the portfolio by 2.0%.
- **It does not explain the cash.** In the four models that held more than a third in cash, holding the policy rate at its training mean changed the cash by 5 points or less, in either direction. The 2017 seed-43 model went from 54% to 51% cash, and its return from 6.3% to 7.6%, against 16.5% for equal weight. In three of those four models, holding the *technical* inputs at their mean cut the cash by 11–18 points.
- **Largest single effect:** the 2017 seed-44 model, which held 25% cash. Held at the training mean, the policy rate raised its cash by 20 points and cut its return by 2.7. Here the out-of-range inputs *reduced* its cash.
- **For comparison,** the shared-weight models, trained without the policy rate, moved more with their inputs. Shuffling moved their portfolio by 4.9% for the technical inputs and 1.6% for market context (`results/pit_attribution_shared_20261111/`). Their two cash-heavy models look the same:
  - **2019, seed 44:** it held 79% cash, and no input group held at its mean changed that by more than 2 points.
  - **2020, seed 43:** holding the technical inputs at their mean cut its 49% cash by 13 points.

**What it means:**
- **The policy rate did not help the agent.** With it, the agent made 4.1% a year against 4.6%, and its best form, the seed ensemble with the band, 4.5% against 5.5%. The ensemble's shortfall is statistically significant.
- **The inputs themselves explain little of the shortfall.** The trained models barely use the policy rate. Holding it at its training mean changes their returns by 0.4 points on average and leaves the cash-heavy models' cash nearly unchanged. The shortfall comes from what training produced with the extra inputs, chiefly cash that depends on the seed. That weakness appears in every arm, so the significant gap describes these 21 models rather than showing that the policy rate harms.
- **Out-of-range inputs remain a risk for any slow macro series.** The rate's few changes leave most of its states unseen in training, and an agent trained on 4–10 years cannot learn how markets respond to them.
- **For RQ2 and RQ3:** the first official macro source tested on the point-in-time index is a negative result, under this design and training budget. The macro-proxy arm (designs 16 and 17) follows in the next section.

## Follow-up: daily macro proxies as an input (2026-11-11)

**What it tests:** whether the daily macro proxies of the five-stock ablation help on the point-in-time index (RQ2, RQ3). It is the second source arm, run like the policy-rate arm.
- **Variant:** `pit_plan_weekly_lstm_shared_macro` is the shared-weight agent with eight more market-level inputs from Yahoo market prices: the 5- and 20-session returns of USD/THB, Brent crude and gold, and the one-session change and 20-session return of the US 10-year Treasury yield.
- **Timing:** each value is the last close on or before the session, carried forward over holidays. Prices after the validation end are dropped before the merge (`add_macro_proxies` in `scripts/run_pit_walk_forward.py`; `tests/test_pit_macro_proxies.py`).
- **Unchanged:** everything else. `config/pit_walk_forward_macro.yaml` is generated from the main config: 7 folds × 3 seeds × 100k timesteps (`results/pit_walk_forward_macro_20261111/`), with the seed ensemble and band evaluated afterwards (`results/pit_macro_ensemble_20261111/`).
- **Designs 16 and 17** of the deflated Sharpe count, configured before any result of the policy-rate arm was seen.
- **Check:** every trained model replayed its saved validation curve exactly (largest relative gap 2×10⁻¹⁶).

| Strategy | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | **Mean** | Years beating ETF | Mean max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SET50 ETF buy-and-hold (TDEX) | 24.5% | 18.2% | −6.5% | 5.7% | −11.8% | 12.1% | 7.8% | **7.1%** | — | −11.2% |
| Point-in-time SET50 equal weight | 22.2% | 16.5% | −10.0% | 6.5% | −8.5% | 11.9% | 0.7% | **5.6%** | 2/7 | −12.7% |
| Seed ensemble + band, without the macro proxies | 21.2% | 15.3% | −9.7% | 5.6% | −3.8% | 10.0% | 0.2% | **5.5%** | 1/7 | −11.3% |
| **Seed ensemble + band, with the macro proxies** | 18.4% | 13.4% | −7.2% | 6.3% | −8.1% | 7.3% | 0.3% | **4.3%** | 2/7 | −11.5% |
| DRL with the macro proxies (± s.d. over 3 seeds) | 18.6% ± 0.6 | 13.5% ± 3.8 | −7.6% ± 2.2 | 4.6% ± 1.6 | −8.0% ± 0.4 | 7.0% ± 4.0 | −0.7% ± 0.9 | **3.9%** | 1/7 | −11.6% |
| DRL, shared-weight actor, without them (± s.d. over 3 seeds) | 20.5% ± 0.7 | 14.5% ± 1.8 | −10.2% ± 0.6 | 4.7% ± 3.0 | −6.8% ± 2.0 | 9.8% ± 1.9 | −0.4% ± 0.4 | **4.6%** | 1/7 | −11.7% |

**Lower again:**
- **Return:** with the macro proxies, the agent made 3.9% a year, 0.7 points less than without them. Its seed ensemble with the band made 4.3%, against 5.5%.
- **Chained** over the seven years, 1 baht became 1.31 for the ensemble with the macro proxies. It became 1.41 without them and for equal weight, and 1.55 for the ETF ([figure](figures/pit_macro_proxies_chained_equity_20261111.png)). With the policy rate it became 1.32 ([all three input sets](figures/pit_source_arms_chained_equity_20261111.png)).
- **By year:** it lost less in 2018 (−7.6% against −10.2%) but trailed in the rising years 2016 and 2021 by 1.9 and 2.8 points.
- **Cash and trading both rose:**
  - **Cash:** average cash was 15.2%, against 10.7%. Four of the 21 models held more than a third of the portfolio in cash over their year: 44% (2017, seed 43), 52% (2018, seed 43), 36% (2019, seed 42) and 72% (2021, seed 44).
  - **Trading:** turnover rose from 4.3 to 5.4 times capital a year. One model (2021, seed 43) turned over 14.6 times and paid 3.2 points in costs.
  - **Seed spread:** the seed standard deviation of annual returns rose from 1.5 to 1.9 points.

**Significance.** Sharpe ratios on the same 1,699 chained daily returns, with the same stationary bootstrap:

| Comparison | Sharpe difference | 95% interval |
| --- | ---: | ---: |
| With the macro proxies − shared actor | −0.05 | −0.18 to +0.09 |
| Seed ensemble + band: with − without the macro proxies | −0.08 | −0.20 to +0.06 |
| Seed ensemble + band: macro proxies − policy rate | −0.02 | −0.13 to +0.11 |
| Seed ensemble + band with the macro proxies − point-in-time equal weight | −0.05 | −0.13 to +0.04 |
| Seed ensemble + band with the macro proxies − ETF | −0.17 | −0.43 to +0.07 |

- **Sharpe ratios:** 0.30 for the mean of the seeds with the macro proxies and 0.33 for its ensemble with the band. For comparison: 0.35 for the shared actor, 0.42 for its ensemble with the band, 0.35 for the ensemble with the policy rate, 0.38 for equal weight and 0.50 for the ETF.
- **No difference is significant.** The macro ensemble's shortfall is larger than the policy-rate ensemble's (−0.08 against −0.06), but its returns track the ensemble without extra inputs less closely, so its interval is wider. The tracking error is 2.8% a year, against 1.2% for the policy-rate ensemble.
- **Deflated Sharpe,** counting 17 designs: 0.17 for the ensemble with the macro proxies and 0.15 for the mean of its seeds. The best design, the ensemble with the band and only technical and market inputs, falls to 0.23 (0.27 at 13 designs, 0.25 at 15); the policy-rate ensemble falls to 0.19.

**Where the return went.** The same diagnostics as above, as means over the seven years:

| Strategy | Return before costs | Cost drag | Return after costs | Turnover a year | after the initial purchase | Average cash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed ensemble + band, without the macro proxies | 5.9% | 0.3% | 5.5% | 1.6× | 0.7× | 10.5% |
| **Seed ensemble + band, with the macro proxies** | 4.8% | 0.4% | 4.3% | 2.1× | 1.2× | 14.7% |
| DRL with the macro proxies | 5.0% | 1.1% | 3.9% | 5.4× | 4.5× | 15.2% |
| DRL, shared-weight actor | 5.5% | 0.9% | 4.6% | 4.3× | 3.4× | 10.7% |
| Equal weight, point-in-time SET50 | 6.0% | 0.4% | 5.6% | 1.9× | 0.9× | 0.7% |

- **Mostly before costs:** the agent with the macro proxies earned 0.5 points a year less before costs and paid 0.2 points more in costs. Its ensemble earned 1.1 points less before costs.
- **Still close to equal weight,** a little less so: the ensemble's daily correlation with equal weight was 0.996 (0.998 without the macro proxies), with 43.6 effective holdings out of 44.5 tradable members.

**The agent uses the macro proxies, and they cost it return.** The same input attribution as for the policy rate (`attribution_*` files in the run folder; all 21 models replayed exactly):

| Input group | Inputs | Portfolio shift, shuffled | Portfolio shift, at the training mean | Change in annual return, at the training mean (mean) | Change in cash, at the training mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Technical | 11 | 4.5% | 3.9% | +0.2 points | −0.4 points |
| SET50 market context | 5 | 2.1% | 2.2% | 0.0 points | +1.4 points |
| **Macro proxies** | 8 | 3.8% | 3.9% | +0.5 points | −3.5 points |

- **Read heavily:** shuffling the macro proxies moved the portfolio by 3.8%, almost as much as the technical inputs (4.5%). The policy rate moved it by 0.7%.
- **Their information hurt:** shuffled or held at their training mean, the macro proxies left the agent better off on average, by 0.5 points a year. The return rose in 15 of the 21 models either way.
- **They drove some of the cash.** Holding the macro proxies at their training mean cut the cash of three of the four cash-heavy models:
  - 2017, seed 43: from 43% to 14%, and its return rose from 9.1% to 13.5% (equal weight made 16.5%);
  - 2018, seed 43: by 17 points;
  - 2019, seed 42: by 13 points.

  The fourth (72% cash in 2021) did not respond to any input group.
- **Against the five-stock study:** there, shuffling the macro proxies moved the weights by 2.9% and changed the return by 0.25 points or less. On the point-in-time index the agent leans on them more, and to its cost.

**Cash across the four designs.** Fifteen models in the four shared-weight designs held 36–94% cash on average over their year: two without extra inputs, four with the policy rate, five trained with the band and four with the macro proxies.
- **Mostly fixed:** shuffling an input group across sessions moved the cash by 5 points or less in 12 of them.
- **The exceptions:** the 2020 shared-weight model (technical inputs, 10 points) and two macro-proxy models (macro inputs, 25 and 6 points).
- **Reading:** the cash is mostly a fixed bias that each seed learned. Where it did follow the inputs, it followed the macro proxies, and that cost return.

**What it means:**
- **Neither macro source helped the agent.** With the macro proxies it made 3.9% a year against 4.6%, and its seed ensemble with the band 4.3% against 5.5%. With the policy rate: 4.1% and 4.5%.
- **Two different failures:**
  - **Policy rate:** the models barely used it. The shortfall came from how training turned out with the extra inputs.
  - **Macro proxies:** the models used them heavily, and removing their information made most models better. They added noise that the agent learned to treat as signal, which moved it into cash and raised its trading.
- **For RQ2 and RQ3:** on the point-in-time index, the best inputs are the technical and SET50 market inputs alone. Neither macro source adds value under this design and training budget; the five-stock ablation reached the same conclusion for the macro proxies. The other planned sources (news sentiment, investor flow, point-in-time fundamentals) still have no historical data.

## Run it

Local: on the 8-core development Mac (4 performance and 4 efficiency cores), seven workers took about 65 minutes per 100k-timestep run.

**Memory limits the worker count there.** Each worker needs about 750 MB.
- **First run:** seven workers plus the usual desktop apps exhausted the Mac's 8 GB of memory. 10.7 of 11.3 GB of swap was in use, and training slowed to 10–50% of a core per worker. Pausing three workers (`kill -STOP`, later `kill -CONT`) brought the other four back to about 95% of a core.
- **Sticky run:** with more desktop apps open, even four workers thrashed. Three ran at about 95% of a core each.

**Use three workers on that machine, or four with few other apps open.**

**Sticky run time (three workers):** a run took 29–37 minutes while the Mac was awake and on AC power. All 21 runs took 8 h 42 min of wall time, of which the Mac was asleep for about 3.8 hours. Low Power Mode on battery also slows training.

**Shared-weight run time (three workers):** all 21 runs took 9 h 56 min of wall time. Batches of three took between 50 minutes and about 3 hours: the Mac ran on battery in Low Power Mode for part of the morning, and other heavy work shared the CPU in the afternoon.

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

The shared-weight actor runs the same way. It standardizes its own features, so it uses no `VecNormalize`. It is aggregated with both earlier runs, counting eight designs, and its significance is also tested against both earlier DRL runs:

```bash
RUNNER=scripts.run_pit_walk_forward CONFIG=config/pit_walk_forward_experiment.yaml VARIANTS=pit_plan_weekly_lstm_shared SEEDS="42 43 44" \
  PYTHON=python scripts/run_walk_forward_reward.sh results/pit_walk_forward_shared 3 100000
python -m scripts.aggregate_walk_forward_reward --run-dir results/pit_walk_forward_shared \
  --extra-run-dirs results/pit_walk_forward_sticky results/pit_walk_forward --n-trials 8 \
  --output-dir results/pit_walk_forward_combined_shared --references pit_plan_weekly_lstm_sticky pit_plan_weekly_lstm \
  --labels pit_plan_weekly_lstm="DRL, traded-value slots" pit_plan_weekly_lstm_sticky="DRL, sticky slots" \
  pit_plan_weekly_lstm_shared="DRL, shared-weight actor"
python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_walk_forward_shared --variant pit_plan_weekly_lstm_shared \
  --pit-config config/pit_walk_forward_experiment.yaml
```

The seed ensemble and the no-trade band reuse the trained shared-weight models, so they need no training. They take about a minute. Before evaluating, the script replays every unchanged model and stops if a replay differs from its saved curve:

```bash
python -m scripts.evaluate_pit_seed_ensemble --config config/pit_walk_forward_experiment.yaml \
  --source-run results/pit_walk_forward_shared --variant pit_plan_weekly_lstm_shared \
  --output-dir results/pit_seed_ensemble --band 0.005
python -m scripts.aggregate_walk_forward_reward --run-dir results/pit_seed_ensemble \
  --extra-run-dirs results/pit_walk_forward_shared --n-trials 13 --output-dir results/pit_seed_ensemble_combined \
  --references pit_plan_weekly_lstm_shared --labels pit_plan_weekly_lstm_shared="DRL, shared-weight actor" \
  pit_plan_weekly_lstm_shared_ensemble="Seed ensemble" pit_plan_weekly_lstm_shared_ensemble_band="Seed ensemble, no-trade band" \
  pit_plan_weekly_lstm_shared_evalband="No-trade band at evaluation"
for v in ensemble ensemble_band evalband; do
  python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_seed_ensemble \
    --variant pit_plan_weekly_lstm_shared_$v --prefix diagnostics_$v
done
```

The band-width sensitivity names each width's policy with a suffix, so that the runs can be aggregated together:

```bash
for pair in "0.0025 _b0025" "0.01 _b0100" "0.02 _b0200"; do
  set -- ${=pair}   # zsh; in bash: set -- $pair
  python -m scripts.evaluate_pit_seed_ensemble --config config/pit_walk_forward_experiment.yaml \
    --source-run results/pit_walk_forward_shared --variant pit_plan_weekly_lstm_shared \
    --output-dir results/pit_band_sensitivity/band$2 --band $1 --modes ensemble_band --suffix $2
  python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_band_sensitivity/band$2 \
    --variant pit_plan_weekly_lstm_shared_ensemble_band$2 --prefix diagnostics
done
python -m scripts.aggregate_walk_forward_reward --run-dir results/pit_band_sensitivity/band_b0025 \
  --extra-run-dirs results/pit_band_sensitivity/band_b0100 results/pit_band_sensitivity/band_b0200 results/pit_seed_ensemble \
  --n-trials 13 --output-dir results/pit_band_sensitivity/combined
```

The heuristic benchmarks need no training and take under a minute:

```bash
python -m scripts.pit_heuristic_baselines --config config/pit_walk_forward_experiment.yaml \
  --run-dir results/pit_heuristic_baselines --years 2016 2017 2018 2019 2020 2021 2022
python -m scripts.aggregate_walk_forward_reward --run-dir results/pit_heuristic_baselines \
  --extra-run-dirs results/pit_seed_ensemble results/pit_walk_forward_shared --n-trials 13 \
  --output-dir results/pit_heuristic_baselines/combined --references pit_plan_weekly_lstm_shared_ensemble_band
```

Training with the band runs like the shared-weight actor. Its seed ensemble is evaluated in its own environment, band included, so only the ensemble mode applies. Overnight on AC power, with nothing else running, three workers trained the 21 models in 3 h 30 min, about 29 minutes per model:

```bash
RUNNER=scripts.run_pit_walk_forward CONFIG=config/pit_walk_forward_experiment.yaml VARIANTS=pit_plan_weekly_lstm_shared_band SEEDS="42 43 44" \
  PYTHON=python scripts/run_walk_forward_reward.sh results/pit_walk_forward_band 3 100000
python -m scripts.evaluate_pit_seed_ensemble --config config/pit_walk_forward_experiment.yaml \
  --source-run results/pit_walk_forward_band --variant pit_plan_weekly_lstm_shared_band \
  --output-dir results/pit_band_ensemble --modes ensemble
python -m scripts.aggregate_walk_forward_reward --run-dir results/pit_walk_forward_band \
  --extra-run-dirs results/pit_band_ensemble results/pit_seed_ensemble results/pit_walk_forward_shared --n-trials 13 \
  --variants pit_plan_weekly_lstm_shared pit_plan_weekly_lstm_shared_band --output-dir results/pit_walk_forward_combined_band \
  --references pit_plan_weekly_lstm_shared pit_plan_weekly_lstm_shared_ensemble_band pit_plan_weekly_lstm_shared_band \
  pit_plan_weekly_lstm_shared_band_ensemble \
  --labels pit_plan_weekly_lstm_shared="DRL, shared-weight actor" pit_plan_weekly_lstm_shared_ensemble_band="Seed ensemble, band at evaluation" \
  pit_plan_weekly_lstm_shared_band="DRL, trained with the band" pit_plan_weekly_lstm_shared_band_ensemble="Seed ensemble, trained with the band"
python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_walk_forward_band --variant pit_plan_weekly_lstm_shared_band \
  --pit-config config/pit_walk_forward_experiment.yaml
python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_band_ensemble \
  --variant pit_plan_weekly_lstm_shared_band_ensemble --prefix diagnostics_ensemble
```

The LSTM forecast benchmark needs no DRL training; its 21 fits take a few minutes. It is aggregated with the other benchmarks and the DRL ensemble with the band:

```bash
python -m scripts.pit_lstm_forecast_baseline --config config/pit_walk_forward_experiment.yaml \
  --run-dir results/pit_lstm_forecast --years 2016 2017 2018 2019 2020 2021 2022
python -m scripts.aggregate_walk_forward_reward --run-dir results/pit_lstm_forecast \
  --extra-run-dirs results/pit_heuristic_baselines results/pit_seed_ensemble results/pit_walk_forward \
  --variants pit_lstm_forecast_top10_monthly --n-trials 13 --output-dir results/pit_lstm_forecast_combined \
  --drop hindsight_set50_equal_weight_monthly starter5_equal_weight_monthly pit_plan_weekly_lstm_shared_ensemble \
  --references pit_lstm_forecast_top10_monthly pit_plan_weekly_lstm_shared_ensemble_band pit_ridge_forecast_top10_monthly \
  --labels pit_lstm_forecast_top10_monthly="LSTM forecast, top ten monthly" \
  pit_plan_weekly_lstm_shared_ensemble_band="DRL, seed ensemble + no-trade band"
python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_lstm_forecast \
  --variant pit_lstm_forecast_top10_monthly --rebalance-every 21 --prefix diagnostics
```

The policy-rate arm trains like the shared-weight actor, from its own config. Its seed ensemble with the band is evaluated as for the shared run, and both are aggregated with the shared run and its ensemble. The input attribution then re-runs each saved model with one input group shuffled or held at its training mean; it takes a few minutes for 21 models:

```bash
RUNNER=scripts.run_pit_walk_forward CONFIG=config/pit_walk_forward_policy_rate.yaml VARIANTS=pit_plan_weekly_lstm_shared_policy SEEDS="42 43 44" \
  PYTHON=python scripts/run_walk_forward_reward.sh results/pit_walk_forward_policy 3 100000
python -m scripts.evaluate_pit_seed_ensemble --config config/pit_walk_forward_policy_rate.yaml \
  --source-run results/pit_walk_forward_policy --variant pit_plan_weekly_lstm_shared_policy \
  --output-dir results/pit_policy_ensemble --band 0.005 --modes ensemble_band
python -m scripts.aggregate_walk_forward_reward --run-dir results/pit_walk_forward_policy \
  --extra-run-dirs results/pit_policy_ensemble results/pit_seed_ensemble results/pit_walk_forward_shared --n-trials 15 \
  --variants pit_plan_weekly_lstm_shared pit_plan_weekly_lstm_shared_policy --drop pit_plan_weekly_lstm_shared_ensemble \
  --output-dir results/pit_walk_forward_combined_policy \
  --references pit_plan_weekly_lstm_shared pit_plan_weekly_lstm_shared_ensemble_band pit_plan_weekly_lstm_shared_policy \
  pit_plan_weekly_lstm_shared_policy_ensemble_band \
  --labels pit_plan_weekly_lstm_shared="DRL, shared-weight actor" pit_plan_weekly_lstm_shared_ensemble_band="Seed ensemble + band" \
  pit_plan_weekly_lstm_shared_policy="DRL + BOT policy rate" \
  pit_plan_weekly_lstm_shared_policy_ensemble_band="DRL + BOT policy rate: seed ensemble + band"
python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_walk_forward_policy --variant pit_plan_weekly_lstm_shared_policy \
  --pit-config config/pit_walk_forward_policy_rate.yaml
python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_policy_ensemble \
  --variant pit_plan_weekly_lstm_shared_policy_ensemble_band --prefix diagnostics_ensemble_band
python -m scripts.pit_feature_group_attribution --config config/pit_walk_forward_policy_rate.yaml \
  --run-dir results/pit_walk_forward_policy --variant pit_plan_weekly_lstm_shared_policy
python -m scripts.pit_feature_group_attribution --config config/pit_walk_forward_experiment.yaml \
  --run-dir results/pit_walk_forward_shared --variant pit_plan_weekly_lstm_shared --output-dir results/pit_attribution_shared
```

The macro-proxy arm runs the same way from its own config (three workers, 06:16 to 10:31 in the morning: 4 h 15 min for the 21 models). The last aggregation puts the three input sets' seed ensembles with the band side by side; `--variants` with no names leaves out the single-seed models:

```bash
RUNNER=scripts.run_pit_walk_forward CONFIG=config/pit_walk_forward_macro.yaml VARIANTS=pit_plan_weekly_lstm_shared_macro SEEDS="42 43 44" \
  PYTHON=python scripts/run_walk_forward_reward.sh results/pit_walk_forward_macro 3 100000
python -m scripts.evaluate_pit_seed_ensemble --config config/pit_walk_forward_macro.yaml \
  --source-run results/pit_walk_forward_macro --variant pit_plan_weekly_lstm_shared_macro \
  --output-dir results/pit_macro_ensemble --band 0.005 --modes ensemble_band
python -m scripts.aggregate_walk_forward_reward --run-dir results/pit_walk_forward_macro \
  --extra-run-dirs results/pit_macro_ensemble results/pit_seed_ensemble results/pit_walk_forward_shared --n-trials 17 \
  --variants pit_plan_weekly_lstm_shared pit_plan_weekly_lstm_shared_macro --drop pit_plan_weekly_lstm_shared_ensemble \
  --output-dir results/pit_walk_forward_combined_macro \
  --references pit_plan_weekly_lstm_shared pit_plan_weekly_lstm_shared_ensemble_band pit_plan_weekly_lstm_shared_macro \
  pit_plan_weekly_lstm_shared_macro_ensemble_band \
  --labels pit_plan_weekly_lstm_shared="DRL, shared-weight actor" pit_plan_weekly_lstm_shared_ensemble_band="Seed ensemble + band" \
  pit_plan_weekly_lstm_shared_macro="DRL + macro proxies" \
  pit_plan_weekly_lstm_shared_macro_ensemble_band="DRL + macro proxies: seed ensemble + band"
python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_walk_forward_macro --variant pit_plan_weekly_lstm_shared_macro \
  --pit-config config/pit_walk_forward_macro.yaml
python -m scripts.walk_forward_agent_diagnostics --run-dir results/pit_macro_ensemble \
  --variant pit_plan_weekly_lstm_shared_macro_ensemble_band --prefix diagnostics_ensemble_band
python -m scripts.pit_feature_group_attribution --config config/pit_walk_forward_macro.yaml \
  --run-dir results/pit_walk_forward_macro --variant pit_plan_weekly_lstm_shared_macro
python -m scripts.aggregate_walk_forward_reward --run-dir results/pit_walk_forward_macro \
  --extra-run-dirs results/pit_macro_ensemble results/pit_walk_forward_policy results/pit_policy_ensemble \
  results/pit_seed_ensemble results/pit_walk_forward_shared \
  --n-trials 17 --variants --drop pit_plan_weekly_lstm_shared_ensemble --output-dir results/pit_source_arms_combined \
  --references pit_plan_weekly_lstm_shared_ensemble_band pit_plan_weekly_lstm_shared_policy_ensemble_band \
  pit_plan_weekly_lstm_shared_macro_ensemble_band \
  --labels pit_plan_weekly_lstm_shared_ensemble_band="Seed ensemble + band: technical + market inputs" \
  pit_plan_weekly_lstm_shared_policy_ensemble_band="Seed ensemble + band: + BOT policy rate" \
  pit_plan_weekly_lstm_shared_macro_ensemble_band="Seed ensemble + band: + macro proxies"
```

The band-trained models' attribution writes into its own folder:

```bash
python -m scripts.pit_feature_group_attribution --config config/pit_walk_forward_experiment.yaml \
  --run-dir results/pit_walk_forward_band --variant pit_plan_weekly_lstm_shared_band --output-dir results/pit_attribution_band
```

To check that trained models ignore the slot order, roll them out under both orders, one validation year at a time:

```bash
python -m scripts.check_slot_order_invariance --config config/pit_walk_forward_experiment.yaml --fold-year 2019 \
  --run-dirs results/pit_walk_forward_shared/fold_2019/pit_plan_weekly_lstm_shared_seed4{2,3,4}/runs/pit_plan_weekly_lstm_shared_seed4?
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

For the sticky or shared-weight variant, set `VARIANTS=pit_plan_weekly_lstm_sticky` or `VARIANTS=pit_plan_weekly_lstm_shared`. The array size is the same.
