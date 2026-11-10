# Evaluation Metric Formulas (as implemented)

Date: 2026-11-10
Status: **reference for decision 7** of [`v3_author_decision_packet_20260904.md`](v3_author_decision_packet_20260904.md) (M7). Nothing here is approved. It records what the code computes today, so that the authors and the advisor can approve it as it stands or name amendments. The last section lists where it differs from the packet's recommended default.

Every number in the walk-forward reports (five-stock, algorithm comparison, source ablation, point-in-time SET50) comes from the formulas below.

## Notation

- $V_0, V_1, \dots, V_N$: portfolio value at the close of each validation session, net of all costs. $V_0$ is the starting capital, before the first decision.
- $r_t = V_t / V_{t-1} - 1$: simple daily return, $t = 1, \dots, N$.
- One fold is one validation calendar year: $N$ is about 243 sessions.
- $w_{i,t}$: weight of stock $i$ after the decision at session $t$; cash is $1 - \sum_i w_{i,t}$.

## Accounting (the environments)

`src/envs/long_term_env.py` and `src/envs/point_in_time_env.py`, self-financing:

- **Turnover** of a session: $\tau_t = \sum_i |w^{\text{target}}_{i,t} - w^{\text{drifted}}_{i,t}|$. It counts stocks only; the balancing cash leg is not counted. A purchase from cash of 50% of capital is turnover 0.5.
- **Costs:** $c = 0.15\%$ commission and $s = 0.05\%$ slippage per unit of turnover (0.20% per side).
- **Net growth:** $1 + r_t = (1 - (c + s)\,\tau_t)(1 + g_t)$, where $g_t = \sum_i w^{\text{target}}_{i,t}\, R_{i,t+1}$ is the return before costs.
- **Drift:** after the session, weights become $w_i (1 + R_i) / (1 + g)$; cash earns nothing.
- **No-trade band** (optional, `environment.no_trade_band = b`): a seated holding is traded to its target only when $|w^{\text{target}}_i - w^{\text{drifted}}_i| > b$.

## Per-curve metrics

`src/evaluation/metrics.py` (`equity_metrics`; a locked V2 file, unchanged):

| Metric | Formula |
| --- | --- |
| Cumulative return | $R = V_N / V_0 - 1$ |
| Return series used below | $r_1, \dots, r_N$ plus a zero for the first row (`pct_change().fillna(0)`), so $N + 1$ values |
| Annualized return | $(1 + R)^{252/(N+1)} - 1$ |
| Annualized volatility | population standard deviation (ddof = 0) of the $N + 1$ values $\times \sqrt{252}$ |
| Sharpe (per fold) | annualized return / annualized volatility; 0 if the volatility is 0; no risk-free rate |
| Sortino | annualized return / $\left(\sqrt{\text{mean}(\min(r_t, 0)^2)} \times \sqrt{252}\right)$ |
| Maximum drawdown | $\min_t \left(V_t / \max_{s \le t} V_s - 1\right)$, in $[-1, 0]$ |
| Calmar | annualized return / \|maximum drawdown\| |
| VaR 95% / CVaR 95% | 5% quantile of the daily returns / mean of the returns at or below it |
| Average turnover | mean of $\tau_t$ over the $N + 1$ rows (the first row's zero counts) |
| Average cash | mean cash weight over the rows |

`src/evaluation/benchmark_metrics.py` (`benchmark_relative_metrics`), with $r^p$ and $r^b$ aligned by date:

| Metric | Formula |
| --- | --- |
| Beta | $\text{Cov}(r^p, r^b) / \text{Var}(r^b)$, population moments |
| Alpha (annualized) | $252 \times (\overline{r^p} - \beta\, \overline{r^b})$ |
| Tracking error | standard deviation (ddof = 0) of $r^p - r^b$, $\times \sqrt{252}$ |
| Excess annualized return | $\prod(1 + r^p)^{252/n} - \prod(1 + r^b)^{252/n}$ |
| Information ratio | excess annualized return / tracking error |
| Up / down capture | $\overline{r^p} / \overline{r^b}$ on days with $r^b > 0$ / $r^b < 0$ |

## Walk-forward aggregates

`scripts/aggregate_walk_forward_reward.py`:

- **Annual return of a fold:** the fold's cumulative return $R$ (one calendar year).
- **Seeds:** the fold value of a variant is the mean over its seeds. The seed s.d. is the sample standard deviation (ddof = 1) over seeds, averaged over folds.
- **Mean annual return:** the arithmetic mean of the seven fold values. It is not a compound rate.
- **Excess vs ETF:** fold return minus the ETF's fold return. "Years beating" counts folds with positive excess; the summary also reports the median excess and the worst fold.
- **Mean maximum drawdown:** the mean over folds of the seed-mean maximum drawdown.
- **Chained value:** each fold's curve is divided by its first value and multiplied onto the level where the previous fold ended. Folds are joined without a return between the last session of one year and the first of the next.
- **Seed-mean chained curve:** the arithmetic mean over seeds of the chained values. It is the same as holding a third of the capital in each seed's strategy from the start of 2016, never rebalancing between them.

## Significance

`src/evaluation/significance.py`, applied to the daily returns of the seed-mean chained curves (1,699 sessions for 2016–2022), on the dates both policies share:

- **Sharpe (all Sharpe ratios quoted with intervals in the reports):** $\overline{r} / \text{sd}(r) \times \sqrt{252}$ with the sample standard deviation (ddof = 1) and a zero risk-free rate.
  - **This differs from the per-fold Sharpe above,** which divides the compound annualized return by the population volatility.
- **Stationary bootstrap** (Politis and Romano): geometric block lengths with a mean of 20 sessions, 2,000 resamples, seed 42.
  - **Intervals:** the 95% interval is the 2.5% and 97.5% quantiles of the resampled statistic.
  - **Differences:** both policies are resampled with the same indices (paired).
  - **One-sided p-value:** `p_value_not_better` is the share of resamples in which the difference is at most zero.
  - **Per-fold runner:** `scripts/run_pit_walk_forward.py` writes its per-fold significance table with 1,000 resamples.
- **Probabilistic Sharpe ratio** (Bailey and López de Prado, 2012). The terms:
  - $\widehat{SR} = \overline{r}/\text{sd}(r)$ per period;
  - $\hat\gamma_3$ and $\hat\gamma_4$ are the population skewness and (non-excess) kurtosis;
  - $n$ is the number of returns.

  $$\text{PSR}(SR^*) = \Phi\!\left(\frac{(\widehat{SR} - SR^*)\sqrt{n - 1}}{\sqrt{1 - \hat\gamma_3 \widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\widehat{SR}^2}}\right)$$

  `probabilistic_sharpe_vs_zero` uses $SR^* = 0$.
- **Deflated Sharpe ratio** (Bailey and López de Prado, 2014): the PSR at the expected maximum Sharpe of $K$ unskilled designs,

  $$SR^* = \sqrt{V}\left[(1 - \gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{K}\right) + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{K e}\right)\right],$$

  where $\gamma$ is the Euler–Mascheroni constant.
  - **Variance $V$:** taken as $1/(n - 1)$, the sampling variance of one per-period Sharpe estimate, because the cross-design variance is not supplied.
  - **Number of designs $K$:** the designs tried, passed as `--n-trials`. The point-in-time results of 2026-11-10 use $K = 13$.

## Agent diagnostics

`scripts/walk_forward_agent_diagnostics.py`, per fold and seed, then averaged as above:

- **Return before costs:** $\prod_t (1 + r_t)/(1 - p_t) - 1$, where $p_t$ is the cost fraction paid on session $t$.
- **Cost drag:** return before costs minus return after costs.
- **Turnover a year:** $\sum_t \tau_t$ over the fold. "After the initial purchase" leaves out the first decision.
- **Per decision:** turnover per rebalance is the mean of $\tau$ on the decisions after the first.
- **Concentration, on rebalance sessions:**
  - **Effective holdings:** $1/\sum_i \tilde w_i^2$, where $\tilde w$ are the stock weights rescaled to sum to one.
  - **Largest and top-ten weights:** shares of total capital.
  - **Distance from equal weight:** the total-variation distance $\tfrac12 \sum |w - u|$ between the weights, cash included, and $u$ = $1/n$ on each of the $n$ tradable members.
- **Relative to the reference:** correlation and tracking error of daily returns against monthly equal weight on the same sessions.

## Against the decision-7 recommended default

| # | Recommended default | As implemented | Gap |
| --- | --- | --- | --- |
| 1 | Net log return from the complete self-financing ledger | Value curves from the self-financing environments; simple returns | Simple, not log, returns |
| 2 | Annualized excess net log return vs active equal weight: $\frac{252}{N}(\ln V^{\text{cand}}_N/V^{\text{cand}}_0 - \ln V^{\text{EW}}_N/V^{\text{EW}}_0)$ | Excess simple annual return vs the SET50 ETF; equal weight is a reference in the significance tables and diagnostics | Different formula and reference |
| 3 | Seed mean per fold, then the median of the fold values as the primary score | Seed mean per fold; the summary reports the mean and the median of fold excesses; the reports lead with the mean | Primary statistic not fixed |
| 4 | Maximum drawdown in $[-1, 0]$, larger is better | Same | None |
| 5 | Mean absolute security turnover per transition, without the cash leg | Same, but the first row's zero enters the mean | Negligible |
| 6 | Sharpe = compound annualized return / population volatility of the $N$ returns | Per-fold metrics: same, over $N + 1$ values. Significance tables: arithmetic mean / sample s.d. | Two definitions in use |
| 7 | Tie order: worst-fold excess, excess drawdown, turnover, smaller budget, candidate ID | Not implemented; the walk-forward scripts report and do not rank | Not needed until candidates are ranked |
| 8 | If the best primary score is not positive, select active equal weight | Not automated; the reports conclude that no DRL design beat equal weight | Consistent in outcome |

**Before the review, decide in particular:**
- **Sharpe:** which of the two definitions in item 6 the report quotes;
- **Reference:** whether the primary score compares against equal weight (item 2) or the ETF;
- **Statistic:** mean or median of folds (item 3).

The code can follow either choice without retraining, because every curve is saved.
