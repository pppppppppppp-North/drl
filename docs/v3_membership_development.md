# Isolated V3 Membership Development Capsule

Last updated: 2026-09-07

## Purpose

This capsule turns the official 2021–2024 SET constituent archive into a fixed-slot, point-in-time membership panel without changing the locked five-ticker V2 study. It also contains the isolated dated-sector feature path, a self-financing execution primitive, draft investability and evaluation policies, a deterministic 540-cell work plan, a gate-first launcher, exact fold dataset and preprocessing contracts, transition-result validation, and fail-closed licensed-market and development gates. It is designed for local structural validation before licensed broad-universe market data exists.

The [7 September execution readiness report](../reports/v3_execution_readiness_20260907.md) and [capsule receipt](../reports/data_quality_v3/v3_execution_capsule_readiness_20260907.json) describe the current structural state. All 540 cells remain blocked, zero V3 HPC jobs have been submitted, and canonical whole-project scientific completion remains **11/37 = 29.7%**.

Membership is not final investability. A security is a known index member when the official dated record says so; it becomes development-eligible only when a valid current bar and a complete strictly prior lookback also exist. Final investability additionally requires observable trading/suspension status, corporate actions, ticker continuity, and an approved handling policy.

## Safe rebuild

```bash
python -m scripts.build_v3_membership_readiness --project-root . --generated-at-utc 2026-07-31T00:00:00+00:00
python -m scripts.build_v3_market_package_gap --project-root . --generated-at-utc 2026-07-31T00:00:00+00:00
python -m scripts.build_v3_investability_policy_readiness --project-root . --generated-at-utc 2026-07-31T00:00:00+00:00
python -m scripts.build_v3_evaluation_policy_readiness --project-root . --generated-at-utc 2026-08-20T00:00:00+00:00
python -m scripts.build_v3_execution_capsule --project-root . \
  --plan-output reports/data_quality_v3/v3_execution_plan_20260907.csv \
  --fold-workload-output reports/data_quality_v3/v3_execution_plan_fold_workload_20260907.csv \
  --receipt-output reports/data_quality_v3/v3_execution_capsule_readiness_20260907.json \
  --generated-at-utc 2026-09-07T00:00:00+00:00
python -m src.v3.development_gate --project-root . --split structural
python -m pytest -q \
  tests/test_v3_membership_mask.py \
  tests/test_v3_eligibility_and_env.py \
  tests/test_v3_execution_ledger.py \
  tests/test_v3_investability_policy.py \
  tests/test_v3_sector_context.py \
  tests/test_v3_market_data_readiness.py \
  tests/test_v3_evaluation_policy.py \
  tests/test_v3_development_gate.py \
  tests/test_build_v3_membership_readiness.py \
  tests/test_build_v3_market_package_gap.py \
  tests/test_build_v3_investability_policy_readiness.py \
  tests/test_build_v3_evaluation_policy_readiness.py \
  tests/test_v3_execution_plan.py \
  tests/test_v3_trial_launcher.py \
  tests/test_v3_fold_dataset.py \
  tests/test_v3_fold_preprocessing.py \
  tests/test_v3_result_evidence.py \
  tests/test_build_v3_execution_capsule.py \
  tests/test_setsmart_v3_request_packet.py
```

The structural gate passes. The market-package, investability, evaluation-readiness, and execution-capsule commands intentionally produce blocked receipts for the current development state. Train requests fail while the protocol is unsigned, the licensed-market receipt is unbound, and both policies remain drafts; `test` and `all` are rejected. The dated execution outputs above preserve the 4 September capsule. These commands enumerate and check prospective work; they do not submit jobs or create a licensed fold dataset.

## Implemented components

| Component | Purpose |
| --- | --- |
| `src/v3/membership_mask.py` | Dense 138-slot SET50/SET100 membership and dated sector panel |
| `src/v3/eligibility.py` | Past-only development eligibility; never uses tomorrow's bar to decide today |
| `src/v3/sector_context.py` | Active-member-only dated sector returns with explicit contributor and availability evidence |
| `src/v3/execution_ledger.py` | Self-financing security/cash accounting, one-way traded notional, frozen holdings, and post-return drift |
| `src/v3/investability_policy.py` | Draft lifecycle validation and observable membership-exit path assessment |
| `src/v3/evaluation_policy.py` | Deterministic fold schedule, fixed candidate ledger, completeness checks, and validation-only ranking rule |
| `src/v3/execution_plan.py` | Exact blocked candidate/fold/seed registry and deterministic three-batch mapping |
| `src/v3/trial_launcher.py` | Locked gate before candidate/fold/seed binding or V3 backend import |
| `src/v3/fold_dataset.py` | Exact source grids, fold slices, context/decision/outcome geometry, and owned transition ledgers |
| `src/v3/fold_preprocessing.py` | Eligible-training-decision-only population z-scores and hashed fold-specific state |
| `src/v3/result_evidence.py` | Complete validation-transition identity, accounting, baseline, and row-hash checks; guarded aggregate conversion |
| `src/v3/trading_env.py` | Fixed-action mask-aware close-to-close environment |
| `src/v3/baselines.py` | Active-member equal-weight and mask-normalization helpers |
| `src/v3/market_data_readiness.py` | Licensed sessions/bars/status/actions/symbol/license/checksum intake gate |
| `src/v3/development_gate.py` | Draft/lock, market-receipt, policy, hash, split, training, and test-access firewall |
| `config/v3/development_protocol.yaml` | Deliberately unsigned, training-disabled protocol draft |
| `config/v3/investability_policy_draft.yaml` | Checksum-bound but unapproved execution and valuation policy |
| `config/v3/evaluation_policy_draft.yaml` | Checksum-bound but unapproved folds, seeds, selection rule, and fixed compute budget |
| `config/v3/market_package_manifest_template.yaml` | Five-file provider package and license/checksum contract |
| `scripts/build_v3_membership_readiness.py` | Deterministic panel, summary, receipt, sidecars, and report builder |
| `scripts/build_v3_market_package_gap.py` | Deterministic current-gap receipt, checks, coverage, chart, and technical report |
| `scripts/build_v3_investability_policy_readiness.py` | Deterministic exit inventory, policy receipt, chart, and technical report |
| `scripts/build_v3_evaluation_policy_readiness.py` | Deterministic schedule, candidate ledger, receipt, chart, and technical report |
| `scripts/build_v3_execution_capsule.py` | Deterministic execution plan, fold workload, blocked receipt, and SHA-256 sidecars |

## Conservative publication timing

Scheduled records are usable from their effective session when already published. A revision published on the same date is usable only from the next observed SET session because the PDFs do not provide intraday publication times.

| Revision publication | First safe V3 session |
| --- | --- |
| 2021-02-11 | 2021-02-15 |
| 2023-04-07 | 2023-04-10 |

This timing rule is specific to the isolated V3 close-to-close decision path. It does not rewrite the normalized source archive.

## Current materialized evidence

| Measure | Value |
| --- | ---: |
| Observed SET sessions | 965 |
| Fixed acquisition slots | 138 |
| Dense membership rows | 133,170 |
| SET100 members per session | 100 |
| Source records used | 10 |
| Future-record violations | 0 |
| HPC jobs submitted | 0 |

Generated artifacts:

- `data/processed/v3_development/set100_membership_session_mask_2021_2024.csv.gz`
- `reports/data_quality_v3/membership_session_summary.csv`
- `reports/data_quality_v3/v3_membership_mask_readiness_20260731.json`
- `reports/v3_membership_mask_readiness_20260731.md`

Every generated artifact has a SHA-256 sidecar. The builder also verifies that the locked V2 preregistration, campaign, and active storage-safe release manifest have not changed.

## Evaluation and selection readiness

The draft evaluation policy turns abundant future HPC capacity into repeated, comparable evidence rather than an adaptive search. It defines six provisional expanding half-year validation folds from 2022 H1 through 2024 H2. Each fold uses a 30-session context window for lagged features, assigns each transition to one role, fits preprocessing only on that fold's training decisions, and discards the train-to-validation boundary transition.

| Measure | Draft design |
| --- | ---: |
| Provisional validation folds | 6 |
| Fixed configurations | 30 |
| Fixed seeds per configuration/fold | 3 |
| Planned fits | 540 |
| Planned timesteps | 63,000,000 |
| Maximum future array-job submissions | 3 |
| Maximum concurrent jobs | 2 |
| HPC jobs submitted for this milestone | 0 |

The primary ranking rule compares annualized net log return with the active-member equal-weight baseline on validation data, averaging the three seeds inside a fold before aggregating across folds. The worst-fold result, drawdown, turnover, smaller training budget, and candidate ID form deterministic tie-breakers. If the best candidate's median excess result is not positive, the declared outcome is the equal-weight baseline rather than a DRL model.

Generated artifacts:

- `config/v3/evaluation_policy_draft.yaml`
- `reports/data_quality_v3/v3_evaluation_fold_schedule_20260820.csv`
- `reports/data_quality_v3/v3_evaluation_fold_summary_20260820.csv`
- `reports/data_quality_v3/v3_candidate_ledger_20260820.csv`
- `reports/data_quality_v3/v3_evaluation_policy_readiness_20260820.json`
- `reports/figures/v3_evaluation_fold_schedule_20260820.png`
- `reports/v3_evaluation_policy_readiness_20260820.md`

These artifacts contain a schedule and execution contract only. They contain no fitted model, performance metric, comparison, or scientific result. The dates are provisional because the current membership-derived sessions are not an authoritative licensed exchange calendar. Training, `test`, and `all` remain blocked, and the final holdout is absent.

## Exact execution plan and launcher boundary

The [7 September execution plan](../reports/data_quality_v3/v3_execution_plan_20260907.csv) crosses 30 candidates × six folds × seeds 42/43/44 into exactly **540 cells and 63,000,000 planned timesteps**. Candidates sort by their exact ID; folds use their registered order; seeds use the registered tuple. Each row binds the protocol, evaluation policy/readiness receipt, candidate ledger, complete fold schedule, fold summary, candidate identity, per-fold schedule identity, and its own plan-row contents by SHA-256.

| Prospective array batch | Folds | Cells | Planned timesteps |
| --- | --- | ---: | ---: |
| `array_batch_01` | 01–02 | 180 | 21,000,000 |
| `array_batch_02` | 03–04 | 180 | 21,000,000 |
| `array_batch_03` | 05–06 | 180 | 21,000,000 |

All rows are `blocked_not_authorized`, and every `hpc_submission_authorized` flag is false. Three future array submissions and at most two concurrent jobs are policy ceilings, not execution permission. The [fold workload](../reports/data_quality_v3/v3_execution_plan_fold_workload_20260907.csv) contains 90 cells and 10,500,000 planned timesteps per fold.

The launcher accepts only project root, protocol, exact candidate ID, fold ID, and seed. It calls the locked development gate before resolving executable cell metadata, loading market data, creating output directories, or importing a trainer. It rejects Boolean/unregistered seeds and exposes no `test`, `all`, data-path, schedule, or candidate-ledger override. After the gate, it rereads and hashes bound inputs at point of use and builds an immutable authorization object. Its authorization hash and the plan-row hash describe different payloads; they are not interchangeable. The planned registry cannot unlock a launch. `src.v3.trial_runner` remains intentionally absent pending a separately reviewed V3 release.

## Fold dataset geometry

The dataset materializer validates the complete canonical schedule and exact session × asset grids before slicing one fold. Features and eligibility must have the same date/ticker keys and fixed asset-slot order, with no missing, duplicate, orphan, or unauthorized rows. Submitted model-feature values must be finite. Each eligible owned transition also requires a finite positive current and next valuation price and a complete strictly prior feature lookback.

Each train or validation slice contains **30 context sessions + N owned decision sessions + one terminal outcome session**. The transition ledger must equal consecutive `dates[30:-1] → dates[31:]`; context can supply lagged observations but cannot create extra scored transitions. The train-to-validation boundary transition remains unowned, and validation decisions cannot fit preprocessing or update a model. Earlier dates can recur in later expanding training folds; validation ownership is unique.

| Fold | Validation period | Training decisions | Validation decisions | Train slice sessions | Validation slice sessions |
| --- | --- | ---: | ---: | ---: | ---: |
| 01 | 2022 H1 | 209 | 118 | 240 | 149 |
| 02 | 2022 H2 | 328 | 121 | 359 | 152 |
| 03 | 2023 H1 | 450 | 120 | 481 | 151 |
| 04 | 2023 H2 | 571 | 120 | 602 | 151 |
| 05 | 2024 H1 | 692 | 118 | 723 | 149 |
| 06 | 2024 H2 | 811 | 122 | 842 | 153 |

These counts come from the [provisional fold summary](../reports/data_quality_v3/v3_evaluation_fold_summary_20260820.csv): **719 validation transitions**, not returns or completed fits. For the planned 138-slot universe, each slice requires 138 rows per session. No licensed broad-universe fold dataset has been materialized. The source, slices, eligibility, schedule, and transition ledgers have separate canonical hashes when materialized.

## Train-only preprocessing

The 7 September validation added rejection of arithmetic overflow after scaling, before output hashing. Finite raw features can still overflow during subtraction or division. The current capsule binds this corrected implementation; the preserved 4 and 6 September receipts describe the earlier implementation and are historical snapshots.

The implemented draft method fits one population z-score state per fold using only eligible rows on the exact ordered training-decision dates. It pools those rows per model feature, uses population standard deviation (`ddof=0`), and uses unit scale for a constant feature. Context-only rows, terminal-only outcomes, validation rows, and arbitrary subsets of training dates cannot enter the fit.

State binds feature order, fit date/asset-slot/ticker keys and values, row count, parameters, schedule, and training transitions by hash. The same verified state transforms that fold's train and validation slices without refitting. Application rejects cross-fold state reuse, changed fitting observations, schedule/transition drift, reordered features, and invalid state hashes. Raw prices, volumes, dates, identifiers, portfolio values, returns, costs, turnover, and other protected accounting fields cannot be selected for scaling. This is a structural method awaiting policy approval, not a fitted production V3 preprocessor.

## Transition evidence and aggregate-metric stop

`result_evidence.py` requires exactly one row for every authorized validation transition and binds each row to candidate, fold/schedule, seed, dataset manifest, preprocessing state, and model hashes. It checks contiguous indices, exact decision/outcome dates, row hashes, finite nonnegative turnover/costs/slippage, `net_return = gross_return − transaction_cost_paid − slippage_paid`, and self-financing reconciliation error within `1e-10`. Candidate, active-equal-weight, and SET-index value chains must remain positive and satisfy `value_after = value_before × (1 + net_return)` without discontinuities.

Validating a transition ledger does not approve a model or authorize a run. The audit-summary helper can derive summaries from validated evidence, but selection-ledger conversion additionally requires the exact `result_ledger.metric_derivation` mapping: 252 periods per year, annualized compound return divided by population return volatility for Sharpe, mean security turnover per transition, and negative-underwater drawdown. That mapping is absent from the current draft policy, so aggregate conversion remains blocked. These definitions are pending recommendations in the author/advisor packet; they have not been self-approved or written into the policy. No real V3 transition performance, aggregate metric, or ranking has been generated.

## Provider request and human decisions

The [SETSMART request packet](../requests/setsmart_v3_20260901/README.md) remains **DRAFT — UNSENT**. It includes a checksum inventory, exact five-file/33-field schema, byte-identical 138-ticker acquisition union, license/data-protection checklist, non-data example manifest, and cover letter. Preparing the packet did not send a message, upload files, acquire a license, or provide the missing market data.

The [author/advisor decision packet](v3_author_decision_packet_20260904.md) contains seven unresolved choices: stale-valuation age; dividend credit/reinvestment timing; terminal delisting/successor handling; the order-information cutoff; fold geometry; candidate/seed/compute budget; and selection metrics/tie-breakers/formulas. All require recorded human review. A completed decision packet must be encoded in reviewed policy revisions and rebuilt receipts before any lock; it cannot itself enable training or authorize an HPC submission.

## Licensed-market package gap

The current five-ticker snapshot is now measured against a separate V3 intake contract:

| Measure | Current evidence |
| --- | ---: |
| Acquisition tickers with any bar | 5/138 (3.62%) |
| Active SET100 session keys with a bar | 4,825/96,500 (5.00%) |
| Active keys with explicit trading status | 0/96,500 (0.00%) |
| Covered active keys with at least one covered dated-sector peer | 0/4,825 (0.00%) |
| Authoritative licensed session-calendar coverage | 0/965 (0.00%) |
| Membership exits with certified liquidation/settlement paths | 0/62 (0.00%) |
| Stock-bar dates outside the provisional membership calendar | 3 |
| HPC jobs submitted | 0 |

The old five-ticker private-pilot readiness receipt is not accepted as V3 evidence. A candidate package must separately bind adjusted OHLCV, daily trading status, authoritative SET sessions, corporate actions, and stable-instrument symbol history. Status must continue beyond active membership until each of the 62 observed exits can be liquidated or terminally settled. The candidate manifest and every file are checked by path, SHA-256, bytes, and row count; the license must allow research use and publication of derived findings.

See `reports/v3_market_package_gap_20260731.md`, `reports/data_quality_v3/v3_market_package_gap_20260731.json`, `reports/data_quality_v3/v3_market_package_checks.csv`, and the reproducible companion `notebooks/v3_market_package_readiness.ipynb`.

## Investability lifecycle and accounting

The checksum-bound draft at `config/v3/investability_policy_draft.yaml` separates membership, order submission, execution, valuation, and instrument identity. It requires a membership exit to remain pending during suspension, halt, or no-trade status and forbids converting the holding to cash without a valid bar or explicit terminal settlement.

The self-financing ledger:

- measures turnover as absolute security trades, excluding the balancing cash leg;
- charges transaction cost and slippage once on actual traded notional;
- reduces purchases when necessary to fund costs without borrowing;
- freezes an untradeable forced exit without charging a fictitious trade cost;
- allows a mandatory executable exit to override a discretionary turnover cap;
- carries return-drifted security and cash weights into the next decision; and
- verifies portfolio wealth reconciliation every step.

The draft remains non-executable. Four choices still require both authors and the advisor: maximum stale-valuation age, exact dividend credit timing, terminal delisting recovery, and the final order cutoff around the closing auction. See `reports/v3_investability_policy_readiness_20260731.md`.

## Dated sector context

The V3 sector builder uses the membership panel's historical `sector_as_of` label. Its aggregation cohort is active index membership, and its contributor set is active membership plus an explicit finite observed-return flag. Inactive acquisition rows never contribute.

The output keeps active-member counts, contributor counts, return coverage, an inclusive active-sector return, a separately named leave-one-out peer return, and explicit availability flags. It does not create data-dependent one-hot columns or silently replace an unobserved return with a real-looking zero. The MINT and NEX sector changes beginning 3 July 2023 are retained.

## Environment behavior

The V3 environment:

- keeps all 138 action slots stable across dates;
- includes the current eligibility mask in the policy observation;
- zeros actions in inactive slots;
- leaves unused capital in cash;
- uses the structural development assumption that a newly ineligible holding is immediately fillable, while the separate final-policy path explicitly freezes an untradeable holding;
- charges transaction cost and slippage once on security notional rather than counting cash as a second trade;
- carries post-return drifted portfolio weights and a self-financing reconciliation error in step evidence;
- records the decision date, mask, eligible count, forced-exit turnover, and cap override;
- explicitly reindexes every feature and close matrix to the fixed caller asset-slot order;
- records the mask applied to the decision separately from the next observation's mask;
- retains a callable `close()` lifecycle in the minimal no-Gymnasium smoke runtime;
- rejects incomplete masks and non-finite active lookback features; and
- rejects an eligible asset without finite positive current and next valuation prices.

The environment cannot consume the membership-only panel directly. It requires the separately derived development-eligibility schema, preventing accidental treatment of index membership as complete tradability.

## Remaining stop conditions

Do not enable V3 training until all of these are complete:

1. licensed adjusted OHLCV for all 138 acquisition tickers;
2. an authoritative licensed SET session calendar;
3. observable trading and suspension status;
4. corporate actions, ticker changes, and delisting continuity;
5. a passing checksum-bound licensed-market package receipt;
6. author- and advisor-approved entry, exit, suspension, stale-valuation, and forced-liquidation rules;
7. author approval and locking of the checksum-bound V3 fold, preprocessing, seed, selection-metric, tie-breaker, and candidate-budget policy draft;
8. explicit approved Sharpe, turnover, and drawdown definitions in the aggregate metric contract;
9. an author-approved checksum-locked V3 protocol and separate release;
10. a separately implemented and reviewed V3 runner that enforces the dataset, preprocessing, lifecycle, and result-evidence contracts;
11. extended post-2024 constituent history; and
12. a genuinely untouched final holdout.

The 2021–2024 period is appropriate for V3 development but cannot be relabeled as a new final holdout because it has already been examined in V2.
