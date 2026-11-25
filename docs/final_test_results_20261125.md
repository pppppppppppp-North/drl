# The Single Test Evaluation, 2023–2024 (M6–M8)

Date: 2026-11-25
Evidence tier: **exploratory** (Yahoo data). This is the one evaluation of the locked designs on the 2023–2024 test years. Nothing about either design was changed after the locks were approved, and neither stage can be run again through the gate. The sealed 2025+ holdout is still unread.

## Summary

- **Headline: point-in-time SET50.** The rule-picked DRL design was ahead of equal weight of the same stocks in both test years: −11.4% against −13.5% in 2023, and −6.7% against −7.6% in 2024. The difference is not significant. It lost to the SET50 ETF in both years. Over the two years it ended at 0.827, against 0.799 for equal weight and 0.984 for the ETF.
- **Secondary: five starter stocks.** A2C at 300k gained 2.6% over the two years (mean of 5 seeds), against 6.1% for equal weight of the same five stocks and −1.6% for the ETF. Neither difference is significant. The seeds' two-year results ranged from −5.7% to +15.2%.
- **For the title:** no DRL design made a significant gain over its equal-weight reference or the SET50 ETF on years it never saw, matching the validation results. The headline design lost money over 2023–2024 and trailed a passive index fund.
- **Data caveat:** Yahoo's 2024 prices contain stale rows on real trading days. This affects both test stages in 2024 (see the data-problem section). The locked results stand as recorded.

## What was run

Both designs were picked from the validation years (2016–2022) by the decision-7 default rule ([decision packet](final_design_decision_20261125.md)). Both locks were approved as recommended on 2026-11-25, together and before either stage ran. Only Nathawat Youngyim's name is recorded in the locks. The co-author (Kunach Kijsanayothin) and the advisor (Prompong Pakawanwong) have not signed them.

| | Point-in-time SET50 (headline) | Five starter stocks (secondary) |
| --- | --- | --- |
| Lock | [`config/final_test_protocol_pit.yaml`](../config/final_test_protocol_pit.yaml) | [`config/final_test_protocol_five_stock.yaml`](../config/final_test_protocol_five_stock.yaml) |
| Design | Shared-weight actor, cash capped at 20%, 100k timesteps, seeds 42–44, seed ensemble with the 0.5-point no-trade band | A2C, 300k timesteps, seeds 42–46 |
| Validation score (decision 7) | −0.01 (rank 1 of the point-in-time DRL designs) | −0.16 (rank 1 of the five-stock DRL designs) |
| Ran | 19:50–20:49, 6 models | 20:48–21:09, 10 models |
| Receipts | `results/final_test_point_in_time/receipt_start.json`, `receipt_end.json` | `results/final_test_five_stock/receipt_*.json` |

The gate ([`scripts/run_final_test_evaluation.py`](../scripts/run_final_test_evaluation.py)) checked the approved and committed lock, the unchanged configs and clean code at commit `bc223bb` before each stage. It then recorded each stage in `results/FINAL_TEST_LEDGER.jsonl`. Each test year is a walk-forward fold: the 2023 models trained only on data up to 2022, and the 2024 models only on data up to 2023. Costs are 0.20% per side, as in validation. The ensemble replay matched every seed's own run (largest gap 2.2 × 10⁻¹⁶).

## Point-in-time SET50 (headline)

| Policy | 2023 | 2024 | Two years chained | Sharpe (2 years) | Worst drawdown 2023 / 2024 | Turnover a year | Average cash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **DRL design** (ensemble + band) | **−11.4%** | **−6.7%** | **0.827** (−17.3%) | −0.80 | −15.6% / −14.4% | 1.3× | 12.6% / 7.6% |
| DRL single models (mean of 3 seeds) | −11.8% | −6.5% | 0.825 | −0.82 | | 2.3–5.0× | |
| Point-in-time equal weight, monthly | −13.5% | −7.6% | 0.799 (−20.1%) | −0.86 | −18.3% / −14.6% | 1.8× | 1.2% / 0.8% |
| SET50 ETF (TDEX) buy and hold | −9.1% | +8.2% | 0.984 (−1.6%) | −0.01 | −14.3% / −7.4% | — | 0% |

Single-seed returns: 2023 −12.2%, −11.6%, −11.5%; 2024 −7.4%, −5.6%, −6.6% (seeds 42, 43, 44). Figure: [chained test-year equity](figures/final_test_pit_chained_equity_20261125.png).

**Against equal weight (the rule's reference):**
- The DRL design was ahead in **both** test years: by 2.2 points in 2023 and 0.9 points in 2024. Every seed was ahead in both years.
- Its decision-7 score on the test years is **+1.89** (2023 +2.57, 2024 +1.21). On validation it was −0.01, so the test result sits within the year-to-year spread seen on validation, on the positive side.
- The Sharpe difference is **not significant**: +0.055 (95% interval −0.062 to +0.169; p = 0.20 that it is no better). Two test years are too few to separate the two.
- It held more cash than equal weight (12.6% against 1.2% in 2023, a falling year), traded less (1.3× against 1.8× a year) and paid less in costs (0.19% against 0.27% of capital a year). Its worst drawdown was smaller in 2023 and about the same in 2024.

**Against the SET50 ETF (the title's benchmark):**
- The DRL design **lost to the ETF in both years**: by 2.3 points in 2023 and by 14.9 points in 2024. Over the two years it ended at 0.827 against the ETF's 0.984.
- Its Sharpe ratio was lower by 0.79 (interval −1.66 to +0.04), short of significance at the 5% level. Equal weight's shortfall against the ETF, −0.84 (−1.75 to −0.04), is significant.
- The DRL design holds the same point-in-time SET50 members as equal weight, and its curve follows equal weight's closely. Most of the gap to the ETF is the gap between equal weight and the capitalisation-weighted ETF in these two years: 4.4 points in 2023 and 15.8 points in 2024.

**Deflated Sharpe:** with 22 point-in-time designs counted, the deflated Sharpe ratio is about 0.00 for every policy except the ETF (0.03). All three Sharpe ratios are at or below zero over 2023–2024.

**Reading for the title:**
- On years it never saw, the rule-picked DRL design improved on the equal-weight portfolio of the same stocks, but not significantly.
- It did not beat a passive SET50 ETF. In a falling market (2023) and in a year led by the largest stocks (2024), long-term profit after costs was negative and below the index.

## Five starter stocks (secondary)

The five stocks (ADVANC, AOT, CPALL, KBANK, PTT) were chosen with hindsight, so this result carries the survivorship caveat.

| Policy | 2023 | 2024 | Two years chained | Sharpe (2 years) | Worst drawdown 2023 / 2024 | Turnover a year | Average cash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A2C, 300k** (mean of 5 seeds) | **−1.9%** | **+4.4%** | **1.026** (seeds 0.943–1.152) | 0.19 | −10.2% / −9.4% | 4.7× / 3.6× | 24.7% / 13.0% |
| Equal weight, monthly | −3.6% | +10.1% | 1.061 | 0.33 | −11.3% / −7.8% | 1.3× | 0.4% / 0.5% |
| Equal weight, buy and hold | −3.1% | +10.3% | 1.068 | 0.37 | −11.2% / −7.7% | — | |
| Ridge forecast, top two monthly (baseline) | −12.9% | +29.8% | 1.131 | 0.52 | −16.9% / −7.5% | 16.0× / 8.1× | |
| SET50 ETF (TDEX) buy and hold | −9.1% | +8.2% | 0.984 | −0.01 | −14.3% / −7.4% | — | 0% |

- **Single seeds (42–46):** 2023 +7.6%, −4.8%, −8.4%, −3.2%, −0.6%; 2024 +7.0%, −0.9%, +7.0%, +1.3%, +7.8%. Seed 43 held 71% cash on average in 2023.
- **Single stocks over the two years:** ADVANC 1.579, KBANK 1.135, PTT 1.088, CPALL 0.826, AOT 0.770.
- **Figure:** [chained test-year equity](figures/final_test_five_stock_chained_equity_20261125.png).

**Against equal weight (the rule's reference):**
- A2C was 1.8 points ahead in 2023, with 3 of 5 seeds ahead. It was 5.7 points behind in 2024, with no seed ahead.
- Its decision-7 score on the test years is **−2.46** (2023 +1.73, 2024 −6.65), against −0.16 on validation.
- The Sharpe difference is −0.15 (95% interval −0.54 to +0.21; p = 0.76 that it is no better). It is not significant.

**Against the SET50 ETF:**
- A2C was 7.2 points ahead in 2023 and 3.8 points behind in 2024.
- Over the two years it ended at 1.026 against 0.984. The Sharpe difference is +0.20 (−0.73 to +1.04), not significant.
- Equal weight of the five stocks did better against the ETF over these two years (1.061) than A2C did.

**Deflated Sharpe** with 14 five-stock designs counted: 0.07 for A2C and 0.10 for equal weight.

**Reading:** the secondary test does not support a DRL advantage either. A2C trailed equal weight of its own five stocks, and its results depended heavily on the seed.
- **The ridge baseline:** it finished highest over the two years, but it lost most in 2023 and traded 8–16 times a year. It was not a candidate for selection.
- **Hindsight:** with only two years and five hand-picked stocks, none of these differences can be told apart from luck.

## A data problem in 2024, found after the run

While checking the results, I found that the point-in-time test had only 200 sessions in 2024, against 243 in 2023. The cause is in the Yahoo price files, not in the code:

- **Stale rows.** In 2024, 7,797 of 44,511 stock rows (17.5%) have zero volume and open = high = low = close = the previous close. In 2022 and 2023 the same pattern covers about 250 rows a year, almost all of them THAI, which was suspended from trading. The 2024 rows are real trading days (for example Tuesday 2024-03-12). Every SET50 member is affected, on at least 7 days and a median of 47. The ETF has 76 such days.
- **43 sessions dropped.** The point-in-time session calendar ([`src/data/point_in_time_panel.py`](../src/data/point_in_time_panel.py)) keeps a date only if at least half the quoted series traded. The rule was written to drop Yahoo's holiday rows. In 2024 it also dropped 43 real sessions between March and October. No earlier year since 2012 lost more than one session to this rule.
- **The five-stock data has the same defect:** 292 of 1,215 rows in 2024 are stale, and on 39 days all five stocks are. The five-stock test dropped exactly those 39 days, so it kept 204 of 243 sessions in 2024. On the other stale days the affected stocks show no price change.

**What it does not affect:**
- **Training.** The 2024 models trained on data up to 2023, which is clean.
- **The 2023 test year.**
- **Comparisons within a year.** Every policy is evaluated on the same sessions and the same prices.
- **Yearly returns, to a first approximation.** A stale price is followed by a real one, so the move is deferred rather than lost. The first and last sessions of 2024 are real. A rough cross-check from the raw closes (SET50 members held equally from each half-year's start) gives −5.8% for 2024, close to the −7.6% of monthly-rebalanced equal weight after costs.

**What it may affect:**
- **Both DRL designs' 2024 decisions.** On a stale day a model saw stale prices and zero volume. Its volume features were distorted (and, for the point-in-time design, its traded-value slot order), and trades on those days were valued at stale prices. The direction of this effect is unknown.
- **2024 risk measures.** Sharpe ratios and the decision-7 score treat each kept session as one day. In 2024 some multi-day moves count as one-day returns. The score is annualised over 199 sessions (point-in-time) or 203 (five-stock) instead of about 242.

**What follows:**
- The locked results above stay the results of record. They must not be replaced by a rerun.
- A clearly labelled **sensitivity check** with corrected 2024 prices would show whether the defect matters. That needs a new price download, which needs approval. It would also be a second look at the test years, so it needs its own approved lock, reported beside the locked result and never instead of it.
- The **holdout (2025–2026) came from the same source** and may have the same defect. Its seal checks bytes only. Checking its volume would mean reading it, so the check belongs in the approved holdout stage, before any model is evaluated on it.

## Files

- **Point-in-time:** `results/final_test_point_in_time/`
  - `test_config.yaml`, `tasks.txt` and the receipts;
  - `runs/` (6 models);
  - `ensemble/` (the ensemble with the band, and its replay check);
  - `summary/` (yearly and chained metrics, significance against equal weight and the ETF, the chained-equity figure).
- **Five-stock:** `results/final_test_five_stock/`, with the same layout and no ensemble. The summary also includes buy and hold of each stock and the ridge-forecast baseline.
- **Ledger:** `results/FINAL_TEST_LEDGER.jsonl`.
- **Logs:** `~/drl_runs/final_test_logs_20261125/` (local).
- **Figures:** redrawn with readable legend labels in `~/drl_runs/final_test_figures_20261125/` from the stage outputs (no new evaluation; the chained curves are identical to the stage summaries), and copied to `docs/figures/`.
- **Report chapter:** section "Single Evaluation on the Test Years" (`tab:test`, `fig:test`) in `docs/report_drafts/results_title_alignment_20261106.tex`.
