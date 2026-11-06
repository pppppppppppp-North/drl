# Long-Term Reward and Rebalancing Experiment (Milestone M1/M5)

Date: 2026-11-06
Evidence tier: **exploratory**. Yahoo Finance data, five starter tickers, validation window only.
Results: [`results/long_term_reward_experiment_20261106/`](../results/long_term_reward_experiment_20261106/)

## Question

Does the agent's objective match the project title's *long-term* profit? The V2 trainer used a one-day simple return, a turnover penalty about 50× the real cost, and a drawdown-level penalty. Its accounting also double-counted the cash leg and never let weights drift. This experiment changes one design element at a time and measures real, net-of-cost P&L.

## Data and protocol

| Item | Setting |
| --- | --- |
| Stocks | ADVANC, AOT, CPALL, KBANK, PTT (`.BK`), Yahoo `auto_adjust=True` (dividend- and split-adjusted) |
| Market series | SET50 ETF `TDEX.BK`. Yahoo now returns only one row for `^SET.BK`/`^SET50.BK`. |
| History | 2012-01-04 → 2024-12-30 (`data/raw/extended/`, SHA-256 receipts). 40 sessions where every stock had zero volume (exchange closed) were removed. |
| Training | 2012-01-04 → 2019-12-30 (1,953 sessions) |
| Validation | 2020-01-02 → 2022-12-30 (725 decisions, plus 30 earlier context sessions for the lookback). Covers the COVID crash, the recovery, and the 2022 rate hikes. |
| Not read | 2023–2024 test years, and the sealed 2025+ holdout (`data/holdout_sealed/2025_onward/`) |
| Features | 11 technical indicators per stock + 5 SET50-ETF market-context features, 30-session lookback |
| Costs | 0.15% commission + 0.05% slippage per unit of turnover, for every policy |
| Agent | PPO, 100,000 timesteps, γ = 0.99, `n_steps` 1024, batch 128, lr 3e-4, train-only observation normalization; seeds 42, 43, 44 |
| Evaluation | Every policy is replayed under correct **self-financing** accounting (security-only turnover, weights drift with returns), so all curves are real P&L |

### Variants

| Variant | Accounting (training) | Reward | Decisions |
| --- | --- | --- | --- |
| `v2_legacy` | legacy (cash-leg double count, no drift) | simple daily return − 0.05·turnover − 0.1·drawdown level | daily |
| `daily_simple_fixed` | self-financing | simple daily return | daily |
| `log_weekly` | self-financing | log return (episode sum = log wealth) | every 5 sessions |
| `plan_main_weekly` | self-financing | log return + 0.1·differential Sharpe − 0.5·drawdown increase | every 5 sessions |
| `log_monthly` | self-financing | log return | every 21 sessions |
| `plan_main_weekly_lstm` | self-financing | as `plan_main_weekly` | every 5 sessions, with the M5 shared per-asset LSTM fusion extractor |

The new variants also use softmax weights with an explicit cash logit, and they train on random 2-year episodes drawn from the training period.

## Results (validation 2020–2022, mean ± s.d. across 3 seeds)

| Policy | Cumulative return | Ann. return | Sharpe | Max drawdown | Avg turnover / session | Costs paid* | Avg cash | Beta vs SET50 ETF | Info. ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2_legacy` | −48.8% ± 7.6 | −20.9% | −0.87 | −50.7% | 0.519 | 75.3% | 5% | 0.95 | −1.48 |
| `daily_simple_fixed` | −41.9% ± 6.9 | −17.3% | −0.83 | −44.6% | 0.386 | 56.0% | 15% | 0.86 | −1.60 |
| `log_weekly` | −2.8% ± 5.8 | −1.0% | −0.04 | −28.6% | 0.110 | 16.0% | 18% | 0.86 | −0.33 |
| `plan_main_weekly` | −8.8% ± 17.2 | −3.4% | −0.18 | −28.0% | 0.081 | 11.7% | 18% | 0.79 | −0.64 |
| `log_monthly` | −5.2% ± 5.3 | −1.9% | −0.10 | −30.4% | 0.031 | 4.5% | 16% | 0.84 | −0.44 |
| **`plan_main_weekly_lstm`** | **+1.9% ± 1.1** | **+0.7%** | **0.04** | **−26.0%** | **0.011** | **1.6%** | 26% | 0.77 | −0.25 |
| SET50 ETF buy & hold (TDEX) | +9.0% | +3.1% | 0.16 | −32.6% | — | 0.2% | 0% | 1.00 | 0.00 |
| Ridge forecast, top-2 monthly | +4.9% | +1.7% | 0.06 | −33.2% | 0.050 | 7.3% | 3% | 1.04 | −0.07 |
| Equal weight, monthly | +2.7% | +0.9% | 0.04 | −31.6% | 0.003 | 0.5% | 0% | 0.98 | −0.20 |
| Equal weight, buy & hold | −2.4% | −0.8% | −0.04 | −31.2% | — | 0.2% | 0% | 0.95 | −0.37 |
| Best single stock (KBANK) | +4.9% | +1.7% | 0.04 | −51.8% | — | 0.2% | 0% | 1.29 | −0.05 |

\*Commission plus slippage, summed over sessions as a fraction of portfolio value. The Sharpe column uses `src/evaluation/metrics.py` (annualized compound return / volatility).

![Validation equity curves](../results/long_term_reward_experiment_20261106/validation_equity_curves.png)

### Statistical significance

Results are in `validation_significance.csv`: stationary block bootstrap with 1,000 resamples and mean block length 20, plus probabilistic and deflated Sharpe.

- **No policy's Sharpe is distinguishable from zero over this 3-year window.** That includes the SET50 ETF, whose 95% CI is −1.00 to 1.45. The LSTM seeds have CIs of roughly −1.05 to 1.26, and probabilistic Sharpe vs 0 is 0.58–0.59.
- None of the LSTM seeds beats equal-weight buy-and-hold with statistical confidence (paired p = 0.18–0.32).
- Three years of daily data cannot separate these strategies. This is why the walk-forward folds and the broader universe in M2/M3/M6 are needed before any performance claim.

## Findings

1. **The old objective lost money through overtrading, not through bad stock selection.** The V2-trained policy turned over the whole portfolio about 377 times in three years and paid about 75% of capital in costs. Fixing the accounting alone (`daily_simple_fixed`) barely helped: a daily-decision agent still churns.
2. **A long-horizon objective plus less frequent decisions cuts the loss from −49% to about −3%.** Costs fall by 79–94%. Max drawdown improves from −51% to −28/−30%, shallower than the SET50 ETF (−32.6%), both equal-weight baselines, and the ridge forecast.
3. **The M5 fusion extractor gave the only consistently positive DRL result.** All three seeds were positive (+0.7% to +2.9%), seed-to-seed spread was the smallest (s.d. 1.1 points), and turnover and drawdown were the lowest among DRL variants. The same reward with a flat MLP had a 17-point spread across seeds.
4. **DRL still trails the passive market.** The SET50 ETF returned +9.0% and the ridge forecast +4.9%. The LSTM agent kept about 26% in cash with beta 0.77, which protected it in 2020 but cost it the 2021–2022 recovery. This matches the project's earlier result, and the report should say so honestly.

## Limitations

- Five hand-picked current mega-caps carry survivorship bias. The Tier B 138-ticker download is the next step.
- The data is Yahoo (exploratory tier), not licensed SETSMART data. TDEX is a SET50 ETF, not the SET index.
- There is one validation window, 3 seeds, 100k timesteps, PPO only, and no hyperparameter search.
- Costs: the 0.20% per side charged here covers the published explicit cost of about 0.176% per side (`src/envs/thai_costs.py`). Market impact beyond about 0.025% is not modeled; mid-cap universes need more.
- The differential-Sharpe reward weight (0.1) and drawdown weight (0.5) were set by hand, not tuned.

## Recommendation for the next run (M6)

> **Follow-up done:** the seven-year walk-forward of the two carried-forward designs is in [`walk_forward_reward_20261106.md`](walk_forward_reward_20261106.md).

- Carry **`plan_main_weekly_lstm`** and **`log_weekly`** forward. Drop the daily-decision variants.
- Use ≥ 5 seeds and ≥ 500k timesteps on BistKA, with expanding walk-forward folds (2016–2022 validation years), the Tier B universe, and the multi-source features from M4.
- Before any test-period or holdout read, the authors should record the chosen reward in the protocol. This is the same human-approval step as the V3 decision packet.

## Reproduce

```bash
python -m scripts.download_extended_history --tickers PTT.BK AOT.BK CPALL.BK KBANK.BK ADVANC.BK \
  --index TDEX.BK --index-name set50_etf --start 2012-01-01 --dev-end 2024-12-31 --holdout-start 2025-01-01 \
  --dev-output data/raw/extended/prices_starter5_2012_2024.csv \
  --index-dev-output data/raw/extended/index_set50_etf_2012_2024.csv \
  --holdout-dir data/holdout_sealed/2025_onward    # refuses to overwrite an existing sealed holdout
python -m scripts.run_reward_variant_experiment --config config/long_term_reward_experiment.yaml \
  --output-dir results/long_term_reward_experiment_<date>
python -m scripts.plot_reward_experiment --run-dir results/long_term_reward_experiment_<date>
```

Each variant can run in its own process (`--variants <name> --seeds <s>`). Merge the outputs with `--aggregate-dirs <dir> ...`. On one CPU core, an MLP run takes about 3.5–5 minutes and an LSTM run about 9–18 minutes.
