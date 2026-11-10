# Point-in-Time SET50 Walk-Forward (Milestone M3)

Date: 2026-11-07; sticky-slot follow-up 2026-11-09; shared-weight actor, seed ensemble and no-trade band 2026-11-10; training with the band 2026-11-11
Evidence tier: **exploratory**. The index membership is official, but the prices are Yahoo's, and 13% of SET50 member-sessions (mostly delisted or merged names) have no prices.
Outputs (ignored by git): `results/pit_walk_forward_20261107/` (first run), `results/pit_walk_forward_sticky_20261109/` (sticky slots), `results/pit_walk_forward_shared_20261110/` (shared-weight actor), `results/pit_walk_forward_combined_20261109/` (first two runs together), `results/pit_walk_forward_combined_20261110/` (all three), `results/pit_seed_ensemble_20261110/` (seed ensemble and no-trade band, no training), `results/pit_seed_ensemble_combined_20261110/` (with the shared-weight run), `results/pit_heuristic_baselines_20261110/` (momentum, inverse volatility and minimum variance), `results/pit_band_sensitivity_20261110/` (band widths), `results/pit_walk_forward_band_20261110/` (trained with the band), `results/pit_band_ensemble_20261111/` (its seed ensemble) and `results/pit_walk_forward_combined_band_20261111/` (both, with the shared-weight and ensemble runs) · Config: `config/pit_walk_forward_experiment.yaml` · Figures: [first run](figures/pit_walk_forward_chained_equity_20261107.png), [first two runs](figures/pit_walk_forward_sticky_chained_equity_20261109.png), [all three runs](figures/pit_walk_forward_shared_chained_equity_20261110.png), [seed ensemble with no-trade band](figures/pit_seed_ensemble_chained_equity_20261110.png), [training with the band](figures/pit_band_training_chained_equity_20261111.png)

**Summary.**
- **Result:** on the actual SET50 membership of each session, the DRL design that roughly matched the market on five hand-picked stocks lost money: −1.1% a year, against 7.1% for the SET50 ETF and 5.6% for equal weight. Unlike the five-stock comparisons, the Sharpe differences are statistically significant.
- **Survivorship:** choosing today's index members with hindsight inflates an equal-weight backtest by 6.6 points a year, also a significant difference.
- **Why the agent lost:** most of the shortfall is trading costs. The policy scores slot positions, so it reacts to the weekly reshuffling of slots and turns over 28 times its capital a year.
- **Fix (run 2026-11-09):** a sticky slot order, which keeps each member in its slot, cut turnover to a third and raised the agent's return to 3.7% a year. It beat the first run in all seven years, and the improvement is significant. The agent still trailed equal weight and the ETF, again mostly through trading costs. Both shortfalls are statistically significant, but only just.
- **Shared-weight actor (run 2026-11-10):** a policy head that scores every stock with the same weights no longer depends on the slot order at all. It halved turnover again and raised the return to 4.6% a year, with a portfolio close to equal weight. It still trailed equal weight (5.6%) and the ETF (7.1%), but neither shortfall is statistically significant.
- **Seed ensemble and no-trade band (2026-11-10, no new training):** averaging the three seeds' portfolios and skipping trades smaller than 0.5 percentage points brought the shared-weight agent to 5.5% a year, level with equal weight (5.6%), while trading less than equal weight. Its Sharpe ratio was slightly above equal weight's and below the ETF's; neither difference is significant. Part of the gain came from cash it happened to hold through the 2020 crash.
- **Heuristic benchmarks (2026-11-10):** momentum (5.3% a year), inverse-volatility weighting (5.0%) and long-only minimum variance (4.0%) did no better than equal weight or the ETF on the point-in-time SET50. No difference between the active strategies is significant.
- **Training with the band (run 2026-11-11):** trained with the same 0.5-point band, the agent made 5.1% a year (its seed ensemble 5.2%), against 4.6% without the band. It halved its trading but doubled its average cash, to 19%: five of its 21 models held more than a third of the portfolio in cash over their year. It did no better than adding the band after training (5.5%) or equal weight (5.6%), and no difference is significant.

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
- **Next:** penalizing or capping cash in training would target the remaining weakness. It would be design 14 and has not been started.

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
