# Improvement Milestones — Title Alignment

Prepared: 2026-11-05
Project title: **Deep Reinforcement Learning for Long-Term Profit Optimization in Thai Stock Markets Using Multi-Source Data**

This file lists what to improve so that the project's results match each part of its title. It uses the same legend as `PROGRESS_CHECKLIST.md` (`[x]` done, `[~]` partial, `[ ]` not started, `[!]` blocked). Weeks are counted from the start of the work (W1 = first week), not from fixed calendar dates. Adjust them to the real submission deadline.

> **Progress update 2026-11-06:** the code-side items in M1, M2, M5 and M7, plus parts of M4 and M6, are implemented and tested. 43 new tests pass, including a randomized check that the new environment with default options matches the V2 one exactly. All new behavior lives in new modules, so no file in the locked V2 release is changed. The first long-term reward experiment is reported in [`docs/long_term_reward_experiment_20261106.md`](docs/long_term_reward_experiment_20261106.md). Items still open need human decisions, licensed or external data, or BistKA access.

---

## 1. Where the project stands against its title

| Title phrase | What the project has now | What is missing | Milestone |
| --- | --- | --- | --- |
| **Deep Reinforcement Learning** | PPO/A2C with the SB3 `MlpPolicy`. Runs use 10k–20k timesteps, which is about 18–36 passes over the 549-step training episode. Ablations use 1,000 timesteps, which is under 2 episodes. DDPG/TD3/SAC have only been smoke-tested. `src/models/` is empty. | Training long enough to converge, repeated seeds, a real network architecture, and a DRL result that beats a simple baseline. On validation, PPO/A2C lose to buy-and-hold PTT and to equal weight. | M1, M5, M6 |
| **Long-Term** | The reward is one day's simple return minus a turnover penalty and a drawdown-level penalty. Data covers 2021–2024 only. The test window is 194 trading days (about 9 months). γ = 0.90–0.99, so the agent looks only about 10–100 days ahead. | The planned reward `α·log(1+R) + β·ΔSharpe − γ·DD − λ·Turnover` is not implemented, and there are no reward variants. History needs to be multi-year (the proposal targeted 10 years). The test needs to span multiple years. Dividends and total return are not handled. | M1, M2 |
| **Thai Stock Markets** | Five hand-picked current mega-caps (PTT, AOT, CPALL, KBANK, ADVANC). A 138-ticker point-in-time V3 panel exists but cannot train because licensed data is missing. The SETSMART request is **DRAFT — UNSENT**. | A broad SET50/SET100 universe that changes membership over time. The five-ticker choice carries survivorship bias. | M0, M3 |
| **Multi-Source Data** | Real inputs: OHLCV, the SET index, Yahoo macro proxies, and Yahoo fundamentals (13 periods, zero-filled before they start). Sentiment is a placeholder: `_hash_sentiment()` sums character codes and is 0 for all of 2021–2024. Official BOT macro uses revised values, not point-in-time ones. The sector map covers 5 tickers in 5 different sectors, so the sector-relative features carry no information. Investor-flow data is not implemented. | At least 4–5 *real, populated, point-in-time* sources inside the modeling window, and an ablation that shows what each source adds. | M4, M6 |

### Accounting issues in the environment behind every reported result

The V2 environment `src/envs/trading_env.py` produced every number in `docs/final_methodology_results.md`. It has problems that V3 already fixed:

- **Cash-leg double counting (line 145):** turnover adds `|Δcash|` on top of the security trades. A 50% buy from cash therefore counts as 100% turnover and pays twice the cost.
- **No weight drift (line 165):** after prices move, `self.weights = target_weights` resets the weights to the target. Turnover is then measured against stale weights.
- **Penalties are out of scale:** the reward charges `0.05 × turnover` on top of the 0.1% cost already taken out of the return. That makes the turnover penalty 50× the real cost. One full rebalance costs as much reward as about five days of 1% returns. The drawdown penalty applies to the *level* of drawdown on every step spent underwater. Together these push the agent toward holding cash and barely trading. This matches PPO's low drawdown but poor return.

These problems need fixing (M1) before any new compute is spent on the V2 trainer.

### Progress note

The last new scientific result is from **2026-05-31**. From June to September the work went into the gating, receipt, and checksum infrastructure (V2 control plane, V3 capsule). That infrastructure is rigorous and worth keeping. Even so, canonical completion is still **11/37 (29.7%)**, and none of research questions RQ1–RQ6 in the implementation plan has a final answer yet. The milestones below put producing results first.

---

## M0 — Clear the blockers and consolidate (W1)

Goal: remove the human and administrative blockers so that compute can start.

- `[ ]` Hold one review meeting with the advisor on the 7 decisions in `docs/v3_author_decision_packet_20260904.md` and sign its review table.
- `[ ]` Send the SETSMART academic data request (`requests/setsmart_v3_20260901/`) through the advisor or KVIS. It has been ready but unsent since 2026-09-01.
- `[ ]` Ask the BistKA admin for project-group/SGID ownership or about 1 GiB more personal-group (`s00658`) soft quota. This clears the 0.692 GiB screening hold.
- `[ ]` Decide what happens to the READY V2 150-trial screening. **Recommendation:** do not run it on the current V2 environment. It carries the accounting issues above and the one-day reward, so its results cannot support a "long-term" claim. Rebuild after M1 and run it then.
- `[ ]` Make `final_project_plan/` the single canonical folder. Move the earlier copies (`FINAL_PROJECT_*_TRAIL`, `Final_Project_DRL`, `FINAL_PROJECT_DRL_CURRENT_PROGRESS`, the 501 MB `final_project_complete.zip`) to an archive outside the working path.
- `[!]` Mark `final_project_plan/` as "Keep Downloaded" in Finder **and free disk space first**. On 2026-11-06 the disk was 96% full (about 9 GiB free), so macOS kept offloading project files to iCloud. About 155 of 297 code/test/config files were offloaded at once, and `README.md`/`PROGRESS_CHECKLIST.md` were offloaded while being edited. Tests, imports, and `git` hang on these files.
- `[!]` The project `.venv/` is empty (no packages). Recreate it outside iCloud, for example `python3.11 -m venv ~/venvs/drl_thai && ~/venvs/drl_thai/bin/pip install -r requirements.txt "pandas<3"`.
- `[!]` The July–September work (V2 control plane, V3 capsule) is **not committed at all**. Local `main` and GitHub (`pppppppppppp-North/drl`) both end at `514e68c` (2026-06-17), so that work exists only in this working tree. Commit and push once the files are local again. An empty branch `title-alignment-milestones`, pointing at `514e68c`, is ready for this.
- `[ ]` Time-box new gating/receipt infrastructure. Add more only where a result in M1–M6 needs it.

**Exit criteria:** decisions signed, data request sent, quota resolved, one canonical folder.

---

## M1 — Make the environment optimize *long-term* profit (W1–W2)

Goal: fix the accounting and give the agent a reward that matches the title.

- `[x]` Port the V3 accounting fixes into the trainer environment, or train directly on `src/v3/trading_env.py`. The fixes are security-only turnover, post-return weight drift, and costs charged once. Add regression tests for the 50%-from-cash case and the drift case. *Done as opt-in `environment.accounting: self_financing` in the new `src/envs/long_term_env.py` (`LongTermTradingEnv`), trained via `python -m src.train_long_term`. The manifest-locked V2 files (`src/envs/trading_env.py`, `src/train.py`) are unchanged. With default options the new environment is bit-identical to the V2 one; `tests/test_long_term_env.py` checks this on 20 random configurations.*
- `[x]` Add `environment.reward.type` with these variants (*implemented, along with `sharpe_weight`, `sharpe_eta`, `sharpe_clip`, and `scale`*):
  - `simple`: the current reward, kept for comparison
  - `log_return`: `log(1 + R_net)`. Summed over an episode, this equals log wealth, so it is the natural long-term objective.
  - `plan_main`: `α·log(1+R) + β·ΔSharpe − γ·ΔDrawdown − λ·Turnover` (implementation plan §5). Use the differential Sharpe ratio (Moody & Saffell) for ΔSharpe.
  - `sortino`: downside-deviation-penalized log return.
- `[x]` Penalize the *increase* in drawdown at each step, not the drawdown level (`reward.drawdown_mode: increment`).
- `[x]` Recalibrate λ so the turnover penalty is about the same size as the real cost, or remove it, since costs already reduce the net return. *The new variants use λ = 0 with costs charged once. The experiment shows overtrading, not too little penalty, was the failure: legacy-trained policies paid 56% of capital in commission plus 19% in slippage over 2020–2022.*
- `[x]` Add a rebalancing interval (`environment.rebalance_every`: 1, 5, or 21 sessions) so long-horizon holding is a policy the agent can choose. Hold positions between rebalances and let them drift.
- `[x]` Use total-return prices. *Verified: `src/data/ingest_ohlcv.py` and the new download use Yahoo `auto_adjust=True` (split- and dividend-adjusted). Dividends are also kept as an audit column.*
- `[x]` Use γ ≥ 0.99–0.995 and random episode start dates within the training split (`environment.train_random_start`, `train_episode_length`, training envs only).
- `[x]` Use a Thai cost model: broker commission + 7% VAT on commission + exchange fees + slippage, instead of a flat 0.1%. *`src/envs/thai_costs.py` encodes the published schedule: 0.157% internet commission (≤ THB 5M/day tier) + SET 0.005% / TSD 0.001% / SEC 0.001% fees, all + 7% VAT = **0.176% per side**. The experiments charge 0.15% + 0.05% slippage = 0.20% per side, which covers this with about 0.025% left for market impact.*
- `[x]` Also added: softmax portfolio actions with an explicit cash logit (`environment.action_transform: softmax`), and train-only observation normalization (`training.normalize_observations`) that is frozen for evaluation.

**Exit criteria:** reward-variant table on validation (5 seeds each); environment tests pass; the chosen reward is recorded in the protocol. *Status: the table exists with 3 seeds per variant ([report](docs/long_term_reward_experiment_20261106.md)). Recording the chosen reward in the protocol is still a human decision.*

---

## M2 — Extend the history to match "long-term" (W2)

Goal: enough years to train and test a long-horizon strategy.

- `[~]` Extend daily OHLCV, SET index, and macro proxies back to at least **2012–2014**. The proposal targeted 10 years. Check how far back Yahoo `.BK` coverage goes for each ticker. *Done for the five starter stocks and the market series. The data covers 2012-01-04 → 2024-12-30, 3,165 sessions per ticker, in `data/raw/extended/` with SHA-256 receipts. **Yahoo now returns only one row for `^SET.BK`, `^SET50.BK` and `^SET100.BK`**, so the market series is the investable SET50 ETF `TDEX.BK`. Macro proxies are not extended yet.*
- `[x]` Include the 2020 COVID crash. *The validation window 2020–2022 contains it.*
- `[x]` **Seal a genuinely untouched holdout now: 2025-01-01 → latest available.** *Sealed in `data/holdout_sealed/2025_onward/`: 2,230 stock rows plus 444 ETF rows from 2025-01-02 to 2026-11-05, with `HOLDOUT_MANIFEST.json` and its SHA-256. The script refuses to overwrite it. Nothing has been evaluated on it.*
- `[ ]` Rebuild walk-forward windows with **≥ 5 one-year out-of-sample test windows** instead of two 126-day windows.

**Exit criteria:** feature table for 2012/2014–2024, sealed 2025+ holdout with checksum, walk-forward schedule with multi-year test coverage.

---

## M3 — Broaden the Thai market universe (W2–W4)

Goal: move from 5 hand-picked stocks to a point-in-time SET50/SET100 universe.

- `[ ]` **Tier A (publication-grade):** load the licensed SETSMART package into the V3 path (138 tickers, trading status, corporate actions) when it arrives.
- `[~]` **Tier B (exploratory, start now in parallel):** *Data downloaded 2026-11-06: 126/138 tickers, 325,913 rows, 2012–2024, in `data/raw/extended/prices_set100_union_2012_2024.csv.gz` with a receipt and per-ticker coverage. 71 tickers have full history from 2012, 27 start between 2012 and 2016, and 28 start later. A separate sealed holdout is in `data/holdout_sealed/2025_onward_set100_union/`. **Yahoo has no data for the 12 delisted, merged, or renamed names** (BPP, BSRC, DTAC, ESSO, GULF, INTUCH, KEX, SABUY, STARK, STEC, TIDLOR, TMB). That gap is the survivorship bias Tier A must close. Next: train on the point-in-time mask.* Use the existing point-in-time membership mask (`data/processed/v3_development/set100_membership_session_mask_2021_2024.csv.gz`) with publicly downloadable OHLCV for SET50 members. Label every output *exploratory* so the project's rule against scraped data in the final package still holds. If Tier A is late, this tier still produces results.
- `[ ]` Extend the constituent history before 2021 to match M2, and after 2024 to cover the holdout.
- `[ ]` Keep delisted and exited names where data exists. Measure survivorship bias by comparing results on the fixed 5-ticker set with results on the point-in-time universe.
- `[ ]` If the final universe ends up as SET50 only, tighten the title wording (for example "…in the Thai Stock Market (SET50)…") so the claim matches the evidence.

**Exit criteria:** DRL trained and evaluated on a point-in-time universe of 30 or more tickers.

---

## M4 — Make "multi-source" real (W3–W5)

Goal: every source in the title is real, historical, point-in-time, and populated.

- `[~]` **Sentiment:** replace `_hash_sentiment()` with real model scoring (WangchanBERTa as planned, or an LLM-based event/risk scorer as in the Dec 2025 proposal). Run it on `gpu4500-normal`. *Code done in the new `src/sentiment/score_news.py` (the locked `extract_embeddings.py` is unchanged): `--scorer classifier` uses any sentiment-fine-tuned HF model and rejects models without sentiment labels. `--scorer embedding` gives WangchanBERTa mean-pooled embeddings with PCA fitted on training dates only. The hash survives only as an explicit `--scorer placeholder`. The GPU job is `slurm/sentiment_embeddings_gpu.sbatch`. Still needs a historical news file and a GPU run.*
- `[ ]` **Historical news source:** collect news for 2014–2024 from a source with history, for example SET company news and disclosures, Thai financial news sites, or GDELT. The Yahoo latest-news probe has no overlap with the modeling window.
- `[~]` **Investor flow:** this source is in the plan but has not been built. Ingest SET's daily trading value by investor type (foreign, local institution, proprietary, retail) and add net-flow features. It is Thai-specific and fits the title directly. *Code done: `src/data/ingest_investor_flow.py` accepts long or wide format and English or Thai group names, checks that daily nets sum to zero, and writes a receipt. `src/features/investor_flow_context.py` builds net share, 5/20-day, and past-only z-score features, available from the next session. There is a template at `data/reference/investor_flow_template.csv`. Still needs the historical export from SETSMART or the SET website.*
- `[ ]` **Official macro (point-in-time):** prefer series with known announcement dates and few revisions, such as the BOT policy rate (MPC dates), FX, and CPI. Lag them by their actual release date. Mark the revised BOT LEI history exploratory or drop it.
- `[~]` **Fundamentals:** replace zero-fill with a missing indicator. Add derived ratios (P/E, P/BV, ROE, D/E, dividend yield) and keep the reporting lag. *Added `fundamental_available`, `fundamental_metric_coverage`, and `fundamental_report_age_years` in the new `src/features/fundamental_flags.py` (the locked `fundamental_context.py` is unchanged). ROE and D/E already existed. Price-based ratios (P/E, P/BV, yield) still need shares outstanding.*
- `[ ]` **Sector:** once M3 provides the broader universe, switch on the V3 active-member sector-relative features, which are already implemented.
- `[ ]` Bring `src.data.source_readiness` to **5/5**, and add a coverage table (share of non-zero, non-imputed values) for each source.

**Exit criteria:** at least 5 genuinely populated sources in the training window; source readiness 5/5.

---

## M5 — A model architecture for multi-source fusion (W4–W6)

Goal: a network that can actually use multi-source input across many assets.

- `[x]` Implement a custom SB3 `BaseFeaturesExtractor` in `src/models/` (*`src/models/feature_extractors.py`, `asset_lstm`, configured via `training.policy`. It is permutation-equivariant across assets, and its output size grows by only `fusion_size` per added stock*):
  - a per-asset temporal encoder with weights shared across assets (LSTM or Bi-LSTM over the past lookback window, as the proposal promised, or a 1D-CNN/Transformer)
  - separate encoders for market-wide sources (index, macro, investor flow, sentiment aggregates)
  - a fusion layer feeding the policy and value heads
- `[ ]` The flattened MLP input is `30 × N × F`. That is about 9k dimensions for 5 tickers with fundamentals and grows linearly with N, so it will not scale to SET50/SET100.
- `[x]` Normalize features with train-only statistics (reuse `src/v3/fold_preprocessing.py`, or `VecNormalize` with frozen evaluation statistics). *`training.normalize_observations: true`; statistics are saved as `<algo>_vecnormalize.pkl`.*
- `[x]` Output portfolio weights through N+1 logits → softmax (cash included) instead of clipping Box(0,1) values.
- `[ ]` Optionally compare against `RecurrentPPO` (sb3-contrib).

**Exit criteria:** architecture ablation (MLP vs. fusion encoder) on validation with 5 or more seeds. *First version done with 3 seeds. Under the same reward, the fusion encoder gave +1.9% ± 1.1 and the MLP −8.8% ± 17.2 (2020–2022 validation). Repeat with ≥ 5 seeds on the broader universe.*

---

## M6 — Scaled experiments that answer RQ1–RQ6 (W5–W8, BistKA)

Goal: final evidence for each research question in the implementation plan.

- `[~]` Training budget of **≥ 200k–1M timesteps per run** (currently 10k–20k) and **≥ 5 seeds** (10 if credit allows). *The first local run used 100k timesteps × 3 seeds (about 4 minutes per MLP run on one CPU core). Scale up on BistKA.*
- `[ ]` **RQ1/RQ4:** PPO vs A2C vs DDPG vs TD3 vs SAC on the fixed environment (rebuilt screening).
- `[ ]` **RQ2/RQ3:** cumulative feature ablation (price → +technical → +index → +macro → +fundamentals → +sentiment → +flow → all) **plus** leave-one-source-out.
- `[~]` **RQ5:** multi-year walk-forward and regime tests, including the 2020 crash, 2022 rate hikes, and sideways periods. *There is now a single 3-year validation window (2020–2022) covering the crash and the rate hikes. Walk-forward is not done.*
- `[ ]` **RQ6:** report HPC usage in SHr, throughput, and speed-up from parallel arrays (the existing accounting scripts cover this).
- `[~]` Benchmarks on identical periods: SET index / SET50 buy-and-hold (with an investable ETF proxy), equal weight, mean-variance, momentum, and the **prediction-based LSTM/XGBoost → allocation baselines**. Both the plan and the proposal list these, and neither is implemented yet. *Added the SET50 ETF (TDEX) buy-and-hold benchmark, equal weight (buy-and-hold and monthly), and a first prediction-based baseline: `src/agents/predictive_baselines.py`, a ridge 21-day forecaster fitted on training dates only that buys its top 2 monthly. XGBoost/LSTM forecasters are still to add.*
- `[ ]` Evaluate the final chosen configuration **once** on the sealed 2025+ holdout.

**Exit criteria:** mean ± CI tables per research question; one holdout result.

---

## M7 — Evaluation rigor (parallel to M6)

- `[x]` Add **beta, tracking error, and information ratio vs SET** to the evaluation metrics. They are listed in plan §6.14 but missing. Sharpe, Sortino, Calmar, VaR, and CVaR already exist. *Added in the new `src/evaluation/benchmark_metrics.py`: `equity_metrics_with_benchmark()` and `benchmark_relative_metrics()`, plus alpha and up/down capture. The locked `metrics.py` is unchanged.*
- `[x]` Add block-bootstrap confidence intervals for the Sharpe and return difference versus each benchmark (*`src/evaluation/significance.py`, using a stationary bootstrap with paired differences*).
- `[x]` Apply the **Deflated Sharpe Ratio** or a similar multiple-testing correction. The 30-configuration × 6-fold × 3-seed grid has 540 cells, so the best cell will look good partly by chance. *Probabilistic and Deflated Sharpe are implemented, with a CLI: `python -m src.evaluation.significance --curves ... --n-trials N`.*
- `[ ]` Add policy attribution (permutation importance or SHAP on the actor) so "which source contributes most" has evidence beyond ablations.
- `[ ]` Write the final metric formulas into the evaluation policy. This is decision 7 in the decision packet.

---

## M8 — Final report, presentation, and reproducibility (W9–W10)

- `[ ]` Rewrite `docs/final_methodology_results.md` (last updated 2026-05-31) and `kvis_final_research_report.tex` with the M1–M7 results. Answer each research question explicitly, including negative results.
- `[ ]` Write a limitations section covering licensed versus exploratory data, survivorship bias, the cost model, and the revised macro data.
- `[ ]` Update the Beamer deck and posters. Add the team photo and the approved member paragraphs, which are still pending.
- `[ ]` Package reproducibility: one documented command per stage, a slim archive of configs, receipts, and metrics, with large checkpoints excluded.
- `[ ]` Final title check: every phrase in the title is supported by a result table.

---

## Suggested order and dependencies

```text
M0 ──► M1 ──► M2 ──┬──► M3 ──┐
                   └──► M4 ──┼──► M5 ──► M6 ──► M8
                             │          ▲
                             └── M7 ────┘
```

The **critical path** is M0 → M1 → M2 → M5 → M6 → M8. M3 Tier A (licensed data) and M4 news collection have the longest outside lead times, so start them in W1 alongside M0.

## Minimum viable alignment (if time is short)

If the deadline allows only part of the plan, these steps give the strongest title match for the effort:

1. Fix the accounting and use the `log_return` / `plan_main` reward with weekly or monthly rebalancing (M1).
2. Use 2014–2024 history with a sealed 2025+ holdout (M2).
3. Run the exploratory point-in-time SET50 universe (M3 Tier B).
4. Add real sentiment, investor flow, the policy rate, and fundamentals with missing flags (M4).
5. Run PPO and SAC × 5 seeds × ≥ 200k timesteps, with a source ablation against SET buy-and-hold and equal weight (M6).
6. Write it up honestly (M8).
