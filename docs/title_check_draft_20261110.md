# Title Check (draft)

Date: 2026-11-10; band-training result added 2026-11-11
Status: **draft** for the M8 item "every phrase in the title is supported by a result table". It maps each phrase to the evidence in the repository now and to what is still missing. It does not choose the final wording; that is the authors' decision.

Title: **Deep Reinforcement Learning for Long-Term Profit Optimization in Thai Stock Markets Using Multi-Source Data**

| Phrase | Supported by | Strength | Gap, and what would close it |
| --- | --- | --- | --- |
| Deep Reinforcement Learning | PPO with a per-asset LSTM encoder. On the point-in-time universe it uses a shared-weight actor. A2C is compared, and every design runs a 7-fold walk-forward with 3–5 seeds ([five stocks](walk_forward_reward_20261106.md), [algorithms](algorithm_comparison_20261107.md), [point-in-time SET50](pit_walk_forward_20261107.md)). | Supported as the method | **No DRL design beats equal weight on the point-in-time SET50:** the best made 5.5% a year against 5.6%, and the difference is not significant. **Algorithms:** SAC, TD3 and DDPG are not run; they need more memory than the Mac has and are set up for BistKA. **Budget:** 100k timesteps × 3 seeds, below the plan's ≥ 200k × ≥ 5. |
| Long-Term Profit Optimization | **Reward:** log return plus differential Sharpe minus drawdown increase, with weekly decisions. **Data and accounting:** total-return prices, self-financing accounting at 0.20% per side, 2012–2024 history. **Evaluation:** seven one-year walk-forward folds (2016–2022), including the 2020 crash. | Supported as an objective and an evaluation design | **No edge over simple strategies:** the agent optimizes this objective but does no better than simple strategies on the actual index. **Untouched data:** the 2023–2024 test years and the sealed 2025+ holdout are still unread; the final configuration still needs its single evaluation there. |
| Thai Stock Markets | **Membership:** official point-in-time SET50 lists for 2012–2024 (34 lists, 227 tickers). **Walk-forward:** on each session's actual members ([universe](point_in_time_universe_2012_2024_20261107.md)). | Supported for SET50, exploratory tier | **Prices:** they are Yahoo's, and 13% of member-sessions, mostly delisted or merged names, have none. **Not used:** SET100 and the licensed SETSMART package (Tier A). **Wording:** "Markets" (plural) claims more than one index; M3 suggests "…in the Thai Stock Market (SET50)…". |
| Multi-Source Data | **Inputs:** technical features from price and volume, SET50 ETF market context, Yahoo macro proxies, and fundamentals (five stocks only). **Ablation:** market context helped (+0.10 Sharpe, one-sided p = 0.03); macro did not ([ablation](source_ablation_20261106.md)). **Policy rate:** the BOT policy rate is ingested but not yet used in a run. | Partly supported | **Point-in-time runs** use two sources, technical and market. **No data yet:** sentiment and investor flow have code but no historical data. **Fundamentals** lack shares outstanding for P/E and P/BV. **To close it:** at least the policy rate and one more populated source on the point-in-time universe, with an ablation. |

## What the evidence supports today

A narrower claim than the title's:
- **Point-in-time result:** on the point-in-time SET50 over 2016–2022, a DRL agent with a long-term (log-wealth) reward reached equal-weight performance while trading less. With a seed ensemble and a no-trade band it made 5.5% a year, against 5.6%, with a Sharpe ratio of 0.42 against 0.38.
- **No edge:** it did not beat equal weight or the SET50 ETF (7.1%). Neither did the agent trained with the band (5.1% a year, 5.2% as a seed ensemble) or three standard heuristic strategies. A supervised LSTM forecaster made 8.1% a year, but only through the late-2020 rally, and not significantly more than equal weight or the ETF.
- **Five stocks:** the earlier advantage on five hand-picked stocks came from choosing those stocks with hindsight.

## Options for the authors (not decided)

1. **Keep the title** and state the negative result plainly in the abstract and the conclusion: the agent does not beat simple benchmarks on the actual index.
2. **Narrow the title to the evidence**, for example: "Deep Reinforcement Learning for Long-Term Portfolio Allocation on the Point-in-Time SET50 Using Price and Market Data".
3. **Close the multi-source gap first**, then decide. Add the BOT policy rate, plus investor flow or sentiment once the data arrives, to the point-in-time runs, with an ablation.

**Whichever is chosen, before submission:**
- **Holdout:** run the final configuration once on the 2023–2024 test years and the sealed 2025+ holdout.
- **Report:** report that result as it comes out.
