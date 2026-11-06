# Project Progress Checklist

Last updated: 2026-09-07

Use this file as the single running checklist for what has been completed and what remains. Update it whenever code, data, HPC jobs, results, or report artifacts change.

## Status Legend

- `[x]` Done and verified
- `[~]` Started or partially complete
- `[ ]` Not started
- `[!]` Blocked or needs attention

## V3 Execution Readiness Completion — 2026-09-07

- `[x]` Complete the [technical readiness report](reports/v3_execution_readiness_20260907.md), six-fold workload table, validation-transition chart, and gate-first diagram. Update the README, V3 development guide, and artifact manifest.
- `[x]` Reject nonfinite values produced by preprocessing subtraction/division before output hashing. Preserve the existing regression tests; update the capsule test to the current dated receipt.
- `[x]` Pass **53 focused tests** without exclusions and **524 broader regression tests with three documented historical V2/runtime checks deselected**. Preserve exact results and exclusions in the [QA record](reports/v3_execution_readiness_20260907_qa.md).
- `[x]` Rebuild and independently verify the [7 September capsule](reports/data_quality_v3/v3_execution_capsule_readiness_20260907.json): 24 descriptors, 540 row hashes, 63 million planned timesteps, three batches, and zero authorized cells. Preserve the 4 and 6 September historical receipts.
- `[x]` Pass syntax compilation and patch-whitespace checks. All three locked V2 hashes remain unchanged; unrelated tracked diff sections match the initial snapshot.
- `[!]` Training remains blocked on the licensed package, human policy decisions, locked protocol, and reviewed V3 backend. The SETSMART packet remains unsent. **Zero HPC jobs submitted; scientific completion remains 11/37 (29.7%).**

## V3 Execution Readiness Documentation And Verification — 2026-09-06

- `[x]` Rebuild the [`540-cell plan`](reports/data_quality_v3/v3_execution_plan_20260906.csv), [`six-fold workload`](reports/data_quality_v3/v3_execution_plan_fold_workload_20260906.csv), and [`blocked capsule receipt`](reports/data_quality_v3/v3_execution_capsule_readiness_20260906.json) through the existing builder with explicit dated outputs, preserving the 4 September evidence. All sidecars are regenerated; the plan remains 30 × 6 × 3 cells, 63,000,000 planned timesteps, and three prospective batches of 180 cells.
- `[x]` Bring the README, [`V3 development guide`](docs/v3_membership_development.md), and [`artifact manifest`](docs/final_artifacts_manifest.md) into line with the six implemented contracts: deterministic plan, gate-first launcher, exact fold data, train-only preprocessing, transition-result evidence, and capsule builder. Record 719 provisional validation transitions and the distinction between context, decisions, and terminal outcomes.
- `[!]` Keep the 12-condition training denial, absent reviewed V3 runner, missing approved metric-derivation contract, and seven pending human policy decisions explicit. The licensed five-file package/calendar and untouched post-2024 holdout are still absent; the SETSMART packet remains **DRAFT — UNSENT**.
- `[x]` Submit **zero V3 HPC jobs** and leave canonical scientific completion at **11/37 = 29.7%**. This update documents and verifies structural readiness; it completes no scientific-compute stage.

## V3 Execution, Fold-Data, And Result-Evidence Update — 2026-09-04

- `[x]` Materialize an exact [`540-cell execution plan`](reports/data_quality_v3/v3_execution_plan_20260904.csv): 30 registered candidates × six chronological folds × seeds 42/43/44. Every row binds the protocol, policy, readiness receipt, candidate ledger, complete fold schedule, per-fold schedule, and its own cell identity by SHA-256.
- `[x]` Map the work deterministically into **three prospective array batches of 180 cells**—two folds per batch and no more than two future batches concurrently. Reconcile the plan to **63,000,000 timesteps** and mark all 540 cells `blocked_not_authorized`; submit **zero HPC jobs**.
- `[x]` Add a dedicated [`gate-first V3 launcher`](src/v3/trial_launcher.py). Its command surface accepts only an exact candidate ID, fold ID, seed, project root, and protocol. It invokes the locked development gate before reading execution metadata or importing a backend, rejects Boolean/unregistered seeds and any `test`/`all` or artifact override, rehashes inputs at point of use, and keeps the separate V3 runner intentionally absent.
- `[x]` Add [`fold_dataset.py`](src/v3/fold_dataset.py) to join the canonical fold schedule to exact feature and eligibility grids, materialize context + owned decisions + terminal outcomes, emit transition ledgers, and reject missing, duplicate, orphan, cross-boundary, or unauthorized rows.
- `[x]` Add [`fold_preprocessing.py`](src/v3/fold_preprocessing.py) to fit one population z-score state per fold using eligible training-decision rows only. Bind feature order, fit keys, observations, parameters, schedule, and transitions by hash; reject validation/context fitting, cross-fold state reuse, protected price/accounting scaling, and source drift.
- `[x]` Add [`result_evidence.py`](src/v3/result_evidence.py) so future candidate metrics must reconcile from every authorized validation transition. It verifies portfolio and baseline value chains, gross return minus costs/slippage, self-financing error, dates, data/preprocessor/model identities, and row hashes before aggregation.
- `[x]` Keep aggregate metric conversion blocked until the authors explicitly register the exact Sharpe, turnover, and drawdown formulas. The [`author/advisor decision packet`](docs/v3_author_decision_packet_20260904.md) consolidates all seven unresolved policy decisions and recommends precise defaults without approving them.
- `[x]` Produce the checksum-bound [`execution-capsule receipt`](reports/data_quality_v3/v3_execution_capsule_readiness_20260904.json) and [`six-fold workload table`](reports/data_quality_v3/v3_execution_plan_fold_workload_20260904.csv). The receipt records structural readiness, the live 12-condition training denial, no performance metrics, and zero HPC submissions.
- `[x]` Prepare a checksum-inventoried, **DRAFT — UNSENT** [`SETSMART request packet`](requests/setsmart_v3_20260901/README.md) containing the exact 33-field five-file schema, byte-identical 138-ticker acquisition union, license/data-protection checklist, non-data example manifest, and cover letter. Nothing was sent or uploaded.
- `[x]` Pass **50 focused adversarial tests** covering the launcher, 540-cell plan, fold slicing, train-only preprocessing, transition-result reconciliation, artifact receipt, and request packet. Preserve all three locked V2 SHA-256 baselines exactly.
- `[!]` Keep training, `test`, `all`, model ranking, and scientific-performance claims blocked. Required next inputs remain the licensed five-file market package, authoritative SET calendar, seven recorded human policy decisions, locked policies/protocol, a reviewed V3 runner, and a genuinely untouched post-2024 holdout.
- `[x]` Keep canonical whole-project completion at **11/37 = 29.7%**. This milestone closes major methodology and execution-integrity gaps but completes no scientific-compute stage.

## V3 Evaluation And Selection Readiness Update — 2026-08-20

- `[x]` Replace the absent-policy gap with a checksum-bound but deliberately unapproved [`V3 evaluation policy draft`](config/v3/evaluation_policy_draft.yaml). It defines transition ownership, context-only rows, train-only preprocessing, seed handling, the primary selection metric, deterministic tie-breakers, and a fixed candidate budget.
- `[x]` Materialize six provisional expanding half-year validation folds from 2022 H1 through 2024 H2. Validation periods do not overlap; boundary transitions are unused; and context dates may supply lagged features but cannot be scored or used to fit preprocessing or update a model.
- `[x]` Freeze **30 configurations** across five DRL algorithms, two exact hyperparameter profiles, and three training budgets, then cross them with **six folds and three seeds** for **540 planned fits and 63,000,000 planned timesteps**. This is a preregistered compute envelope, not a completed experiment.
- `[x]` Produce the deterministic [`fold schedule`](reports/data_quality_v3/v3_evaluation_fold_schedule_20260820.csv), [`candidate ledger`](reports/data_quality_v3/v3_candidate_ledger_20260820.csv), checksum-bound [`readiness receipt`](reports/data_quality_v3/v3_evaluation_policy_readiness_20260820.json), inspected [`schedule visualization`](reports/figures/v3_evaluation_fold_schedule_20260820.png), and reader-facing [`technical report`](reports/v3_evaluation_policy_readiness_20260820.md).
- `[x]` Bind the draft policy and receipt into the V3 development gate. Recompute every embedded receipt descriptor transitively, reject policy/ledger/schedule/input/implementation drift, pass structural validation, and keep every executable request closed until the full policy is locked.
- `[x]` Bound eventual HPC use to at most **three array-job submissions**, with at most **two concurrent jobs**. Submit **zero** jobs for this milestone and make no remote change.
- `[x]` Pass **77 focused V3 tests** and a **471-test broader project regression** with the same three known historical-V2 checks explicitly deselected. Preserve all three locked V2 SHA-256 baselines exactly.
- `[x]` Preserve the evidence boundary: the schedule figure shows only temporal allocation, the receipt contains no return or risk result, and no algorithm has been ranked. Canonical whole-project completion remains **11/37 = 29.7%** because no scientific-compute stage completed.
- `[!]` Keep training, `test`, and `all` blocked. The six-fold schedule is provisional until rebuilt from an authoritative licensed SET session calendar; the licensed market package, author approval of the evaluation and investability policies, a locked protocol/release, and a genuinely untouched post-2024 final holdout are still required.

## V3 Investability Lifecycle And Exit-Path Update — 2026-07-31

- `[x]` Implement [`src/v3/execution_ledger.py`](src/v3/execution_ledger.py) as a self-financing security-and-cash primitive. Measure gross security trades without double-counting cash, fund costs without borrowing, preserve frozen holdings, allow mandatory exits to override discretionary turnover caps, and carry post-return drifted weights.
- `[x]` Correct two additional V3 accounting defects: a 50% cash-to-security purchase now has 50% turnover rather than 100%, and cross-sectional returns now change the next decision's portfolio weights instead of resetting them to the previous target.
- `[x]` Add the checksum-bound but deliberately unapproved [`investability policy draft`](config/v3/investability_policy_draft.yaml), covering information timing, entry/fill behavior, suspension, halt, no-trade, membership exit, stale valuation, delisting, corporate actions, symbol continuity, and accounting.
- `[x]` Identify **62 active-to-inactive membership transitions across 54 tickers and 9 effective-session batches**. The largest observed batch contains 11 exits on 3 January 2024.
- `[x]` Extend the licensed-market gate so every exit needs continuous status until a valid liquidation bar or explicit terminal settlement. Current certified exit-path coverage is **0/62**, because the licensed status/action/symbol package is absent—not because all 62 securities were necessarily untradeable.
- `[x]` Generate the checksum-bound [`exit inventory`](reports/data_quality_v3/v3_membership_exit_path_inventory_20260731.csv), [`readiness receipt`](reports/data_quality_v3/v3_investability_policy_readiness_20260731.json), inspected [`visualization`](reports/figures/v3_membership_exit_transitions_20260731.png), and reader-facing [`technical report`](reports/v3_investability_policy_readiness_20260731.md).
- `[x]` Make the V3 evidence graph acyclic and enforce exact cross-receipt binding: the market and investability builders reject a checksummed receipt when its panel, panel sidecar, or upstream receipt descriptor belongs to a different build.
- `[x]` Map the required provider fields to official SETSMART/SET tables and prepare an advisor/institution [`academic data-support request draft`](docs/setsmart_academic_data_request_draft.md). Do not scrape public quote pages into the final training package.
- `[x]` Add a separate evaluation/selection-policy stop condition so large HPC resources cannot unlock adaptive V3 training merely by satisfying the data and investability gates.
- `[x]` Pass all **60 focused V3 tests**, deterministic evidence rebuilds, structural development-gate validation, bytecode compilation, all 14 generated-artifact checksum checks, locked-V2 hash checks, and the patch-whitespace check.
- `[x]` Pass a **454-test broader project regression** after explicitly excluding three known non-V3 checks. The unfiltered result is 454 passed and 3 failed: two correctly flag that current local dependencies/environment code no longer equal historical active-V2 release evidence, while one V2 staging subprocess lacks PyYAML in the isolated QA runtime. Do not rewrite historical V2 evidence or submit HPC work merely to silence these checks.
- `[!]` Keep the investability policy unapproved until both authors and the advisor decide maximum stale-valuation age, dividend credit timing, terminal delisting recovery, and the final closing-auction order cutoff.
- `[x]` The evaluation-policy design gap recorded on 31 July is now addressed by the 20 August checksum-bound draft and readiness evidence. Keep it unapproved and non-executable until its remaining author decisions and upstream licensed-data conditions are resolved.
- `[x]` Submit zero HPC jobs and make no remote change. Canonical whole-project completion remains **11/37 = 29.7%** because this is structural methodology work, not a completed scientific-compute stage.

## V3 Market-Package And Sector-Context Update — 2026-07-31

- `[x]` Implement a separate fail-closed V3 market-package validator in [`src/v3/market_data_readiness.py`](src/v3/market_data_readiness.py); never reuse the five-ticker private-pilot PASS receipt as broad-universe evidence.
- `[x]` Define and validate five separately bound provider components: adjusted OHLCV bars, daily trading status, authoritative SET sessions, corporate actions, and stable-instrument symbol history.
- `[x]` Add [`config/v3/market_package_manifest_template.yaml`](config/v3/market_package_manifest_template.yaml), requiring project-relative paths, SHA-256, bytes, row counts, a manifest sidecar, completeness attestations, and a license that permits research and publication of derived findings.
- `[x]` Measure the current data gap exactly: 5/138 acquisition tickers (3.62%), 4,825/96,500 point-in-time active member/session keys with bars (5.00%), 0/96,500 with explicit trading status, and 0/4,825 covered active keys with a covered same-sector peer.
- `[x]` Identify three stock-bar dates outside the provisional observed-SET membership calendar—2021-04-02, 2023-12-25, and 2024-12-25—and require an authoritative licensed session package instead of treating missing vendor index rows as exchange holidays.
- `[x]` Implement active-member-only, date-aware sector context in [`src/v3/sector_context.py`](src/v3/sector_context.py). Preserve dated `sector_as_of`, exclude inactive acquisition rows, require explicit observed-return flags, and emit contributor counts, coverage, inclusive sector returns, separately named leave-one-out peer returns, and availability evidence.
- `[x]` Correct two adversarial V3 environment defects: reindex every feature/price matrix to the fixed caller asset-slot order, and separate the mask applied to a decision from the next observation's mask in step evidence.
- `[x]` Restore a callable environment `close()` lifecycle in the minimal no-Gymnasium fallback; the focused close/determinism regression now passes without changing the locked V2 preregistration, campaign, or deployed release manifest.
- `[x]` Add future-suffix invariance coverage so perturbing or deleting later market rows cannot change eligibility through an earlier cutoff.
- `[x]` Harden the executable V3 gate so a train/validation request also requires a bound passing licensed-market receipt and a locked investability handling policy; keep `test` and `all` forbidden.
- `[x]` Generate the checksum-bound [`V3 market-package gap receipt`](reports/data_quality_v3/v3_market_package_gap_20260731.json), full checks table, coverage table, inspected chart, and reader-facing [`technical report`](reports/v3_market_package_gap_20260731.md).
- `[x]` Add the inspectable companion notebook at [`notebooks/v3_market_package_readiness.ipynb`](notebooks/v3_market_package_readiness.ipynb); validate its JSON and execute all four code cells in order against the current project data.
- `[x]` Complete 174 relevant validation checks: 39 focused V3 tests and 135 adjacent/locked-regression tests passed. Four plotting-only tests were deselected because the local QA runtime does not contain the project-declared `matplotlib` dependency.
- `[x]` Submit zero HPC jobs and make no remote change. Keep canonical whole-project completion at **11/37 = 29.7%** because this is local structural readiness work, not a completed scientific-compute stage.
- `[!]` Keep V3 model input and training blocked until the five licensed market files, authoritative calendar, license/checksum bindings, execution policy, post-2024 constituent extension, and untouched holdout pass their gates.

## V3 Point-in-Time Membership Development Update — 2026-07-31

- `[x]` Create an isolated [`src/v3/`](src/v3/) development capsule instead of changing the shared V2 environment, configurations, campaign, preregistration, or deployed release.
- `[x]` Implement a dense point-in-time membership panel with a fixed 138-slot acquisition layout, dated sector labels, exact source provenance, and a conservative next-observed-session rule for same-day revisions.
- `[x]` Materialize [`set100_membership_session_mask_2021_2024.csv.gz`](data/processed/v3_development/set100_membership_session_mask_2021_2024.csv.gz): 133,170 rows across 965 observed SET sessions, exactly 100 active SET100 members on every session, 10 source records, and zero future-record violations.
- `[x]` Delay the 11 February 2021 and 7 April 2023 revisions until the next observed sessions, 15 February 2021 and 10 April 2023, because the source PDFs provide dates but no intraday publication times.
- `[x]` Keep membership distinct from investability. Implement development eligibility as point-in-time membership AND a valid current bar AND a complete strictly prior lookback; never use next-session prices to decide today's mask.
- `[x]` Implement the isolated mask-aware environment with a fixed action dimension, a visible current mask, inactive-action zeroing, mandatory exits to cash, one-time execution costs, turnover-cap override reporting, and fail-closed current/next valuation checks.
- `[x]` Add mask-aware active equal-weight support and retain dated `MINT.BK` and `NEX.BK` sector changes.
- `[x]` Add the structurally valid but deliberately unsigned [`V3 development protocol`](config/v3/development_protocol.yaml). Its gate rejects executable training until both authors approve and lock it, and rejects `test` and `all` unconditionally in the development capsule.
- `[x]` Bind the generated panel and 10-record summary to checksum sidecars and a [`V3 readiness receipt`](reports/data_quality_v3/v3_membership_mask_readiness_20260731.json). Preserve the technical [`readiness report`](reports/v3_membership_mask_readiness_20260731.md).
- `[x]` Verify that the locked V2 preregistration, research campaign, and active storage-safe release manifest still match their recorded SHA-256 baselines.
- `[x]` Pass 22 new focused V3 membership, sector, publication-boundary, eligibility, fixed-slot, all-cash, environment, cost-accounting, valuation, deterministic-build, receipt-binding, protocol-gate, and V2-immutability tests.
- `[x]` Pass 59 checks across the final validation runs: 22 new V3 cases, 27 adjacent constituent/source/intake cases, and 10 locked-V2 preregistration/campaign/deployment cases; compilation and patch-whitespace checks also pass.
- `[x]` Submit zero HPC jobs and make zero remote changes for this milestone. Canonical whole-project campaign completion remains **11/37 = 29.7%** because no scientific-compute stage or performance result changed.
- `[!]` Keep V3 training blocked until licensed 138-ticker adjusted OHLCV, observable trading/suspension status, corporate actions, an author-approved handling policy, a locked protocol, extended post-2024 constituent history, and a genuinely untouched holdout exist.

## Main Progress Update — 2026-07-31

- `[x]` Preserve ten official SET constituent PDFs covering 2021–2024, including the published 2021 H1 and 2023 H1 revisions, under [`data/external/original/set_constituent_history/`](data/external/original/set_constituent_history/).
- `[x]` Bind every original to its SET URL, retrieval timestamp, byte count, and SHA-256 in [`data/external/set_constituent_source_manifest.csv`](data/external/set_constituent_source_manifest.csv); the preserved package contains 2,723,035 bytes.
- `[x]` Implement the fail-closed legacy/modern PDF importer in [`src/data/ingest_set_constituent_history.py`](src/data/ingest_set_constituent_history.py). It rejects raw-file drift, incomplete ranks, duplicate symbols, blank companies or sectors, malformed `.BK` mappings, count drift, sector disagreement, and SET50/SET100 subset violations.
- `[x]` Normalize ten continuous effective intervals into [`data/reference/set50_set100_membership_2021_2024.csv`](data/reference/set50_set100_membership_2021_2024.csv): 1,000 rows, exactly 50 SET50 and 100 SET100 members per interval, 138 unique tickers, and 23 canonical sectors.
- `[x]` Capture both official mid-period changes without future-membership leakage: 2021 H1 SET50 `+OR −TTW` and SET100 `+OR −TKN`; 2023 H1 SET50 `+SAWAD −DTAC` and SET100 `+SIRI +SJWD −DTAC −STARK`.
- `[x]` Implement the validated as-of resolver in [`src/data/constituent_history.py`](src/data/constituent_history.py), using half-open intervals and requiring publication before availability.
- `[x]` Generate the 138-ticker acquisition-only union at [`data/reference/set50_or_set100_universe.csv`](data/reference/set50_or_set100_universe.csv). It is explicitly not a static model universe.
- `[x]` Generate the compatibility sector map and preserve the dated classifications. Verify two genuine changes: `MINT.BK` moves from Food & Beverage to Tourism & Leisure and `NEX.BK` moves from Electronic Components to Automotive beginning with 2023 H2.
- `[x]` Write the checksum-bound source/output receipt at [`reports/data_sources/set_constituent_archive_20260731.json`](reports/data_sources/set_constituent_archive_20260731.json) and its SHA-256 sidecar. The receipt marks point-in-time membership/sector lookup eligible and model input ineligible.
- `[x]` Harden [`src/data/source_readiness.py`](src/data/source_readiness.py) so the broader-universe and sector files pass only when their receipt, sidecar, path, row count, byte count, and SHA-256 all match.
- `[x]` Advance structural external-source readiness from **1/5 to 3/5**: broader membership, dated sector history, and structurally complete revised BOT macro history are present; historical news/sentiment and licensed fundamentals remain missing.
- `[x]` Add and pass 18 focused importer, interval, boundary, no-future-leakage, deterministic-build, receipt-binding, and drift-rejection tests.
- `[x]` Produce the reader-facing [`SET constituent archive report`](reports/set_constituent_archive_readiness_20260731.md), per-interval readiness table, and inspected turnover visualization. Visually cross-check representative 2021, revised 2023, and modern 2024 source pages.
- `[x]` Preserve the scientific boundary: the locked five-ticker V2 release and preregistration are unchanged. The isolated V3 membership and development-eligibility machinery now exists, but licensed market/trading-status data and an author-approved locked V3 protocol are still required for broad-universe training.
- `[x]` Submit zero HPC jobs for this milestone. Keep canonical whole-project campaign completion unchanged at **11/37 = 29.7%** and the active graph at 9 `VERIFIED`, 1 `READY`, and 25 `PENDING`.
- `[x]` Confirm on 2026-07-31 that the VPN reaches the BistKA SSH endpoint. The key-only read-only probe was rejected because no accepted public key is configured; stop without password authentication, remote writes, or job submission pending separate explicit approval.
- `[!]` Before eventual full-screening submission, build a new manifest-locked, one-shot submission capsule that binds the authorization receipt at the final `sbatch` boundary and prevents replay. Do not modify the active 248-file release in place.

## Main Progress Update — 2026-07-26

- `[x]` Implement the standalone fail-closed full-screening execution-window gate in [`scripts/prepare_screening_execution_window.py`](scripts/prepare_screening_execution_window.py). It recomputes storage admission, verifies four live timestamps, requires empty user and account queues, checks conservative service-hour coverage, enforces at least 1 GiB of personal-group soft-quota margin, validates the fresh remote and complete local locked release, recomputes campaign state read-only, checks the exact `0-4%2` SLURM resources, and never calls `sbatch`.
- `[x]` Add 51 focused execution-window test cases covering exact success, stale quota/write/preflight evidence, canonical smoke binding, scheduler checksum and semantic failures, frozen integer-only safety-policy floors (including NaN and Boolean rejection), same-stamp operational evidence and receipt paths, conservative credit, low margin, user- and account-level queue state, local and live release drift, full local/remote inventory, wrong array/resources, QOS flags/duplicate resources, partition state/time, optional module-list failures, gate-issued sealed receipt creation, fabricated dictionary/direct-constructor rejection, intermediate-symlink rejection, exclusive-write enforcement, expired-write rejection, and byte-for-byte no-mutation holds.
- `[x]` Implement [`scripts/render_authorized_screening.py`](scripts/render_authorized_screening.py), the supported render-only consumer. It verifies the receipt/sidecar and invariants, reruns the entire live gate at wall time, requires exact evidence identity, claims the receipt exactly once, binds the rendered attempt in a consumption record, and never executes the rendered SLURM command.
- `[x]` Add 23 focused consumer cases for revalidation, receipt/evidence stamp binding, strict and advancing TTL, live hold, Boolean-policy rejection, evidence drift, copied/distinct and concurrent receipt replay, stage-wide locking, preexisting-attempt rejection, symlinked output-directory/leaf rejection across the command, attempt, status-report, and event-log paths, immutable-attempt/command-export linkage, failure recording, and removal of public clock/provider overrides. Pass **74 authorization/render-focused cases and 124 authorization/quota/storage cases together**.
- `[x]` Reconnect to BistKA through the VPN and capture authoritative checksum-bound scheduler/release, quota/write-probe, and 248-file live-preflight evidence at 14:37 UTC without submitting a job.
- `[x]` Verify empty `s00658` and whole-account `25sfcs03` queues, `compute-long` up with a four-day limit, 9,978.72 project service hours, exact QOS limits, exact deployed bundle/manifest hashes, and a fresh 248/248-file remote preflight.
- `[x]` Preserve the fresh evidence under [`campaigns/large_scale_thai_drl_v2/operational/screening_capacity/`](campaigns/large_scale_thai_drl_v2/operational/screening_capacity/) so it does not alter the broad historical capacity-stage glob.
- `[!]` Hold the full screening: the final 6,806,358,016-byte group soft headroom exceeds the recomputed 6,475,756,231-byte requirement by only 330,601,785 bytes (0.308 GiB), below the new 1 GiB authorization minimum. At least 743,140,039 additional bytes (0.692 GiB) of safe headroom are needed.
- `[x]` Confirm the hold returned exit code `2`, created no authorization receipt, rendered no attempt, submitted zero jobs, and left the active graph at 9 `VERIFIED`, 1 `READY`, and 25 `PENDING`.
- `[x]` Re-run the hardened CLI near evidence expiry and confirm it independently added `execution_window_ttl_below_minimum` when only 70 seconds remained, still wrote no receipt, and remained fail-closed.
- `[x]` Close the final review bypasses: production cannot raise the 15-minute age/skew limits or lower the 120-second TTL/1 GiB margin/reserve floors, including through non-integer/NaN values; the CLI accepts only a same-stamp evidence triplet in the operational directory; only an unmodified gate-issued in-process capability can be written; output path components cannot be symlinks; and exclusive creation prevents receipt overwrite races.
- `[!]` Carry same-user time-of-check/time-of-use hardening into the next manifest-locked controller release: replace the controller's status, event-log, and command writers with no-follow, directory-bound publication before permitting eventual script execution. Static/pre-existing symlinks are already rejected by the supported consumer; do not drift the deployed release in place.
- `[!]` Keep direct use of `src.campaigns.control commands --stage screening` and direct execution of the generated `screening.sh` outside the supported workflow. The consumer now serializes rendering with an evidence-identity ledger and permanent stage lock, but mandatory TTL/receipt enforcement at the eventual `sbatch` boundary requires a new release, deployment, and remote preflight before it can replace the active 248-file release.
- `[x]` Record the exact evidence and decision in [`reports/screening_execution_window_hold_20260726.md`](reports/screening_execution_window_hold_20260726.md).
- `[x]` Keep canonical whole-project progress unchanged at **11/37 = 29.7%** because no scientific compute stage completed.

## Main Progress Update — 2026-07-25

- `[x]` Re-audit the active V2 campaign: the full validation screening remains the only `READY` scientific stage, with 150/150 trials still unsubmitted and no new HPC jobs submitted in this work session.
- `[!]` Confirm that `10.205.100.101` currently routes through the home interface instead of a VPN interface. Live BistKA queue, credit, quota, write-probe, and submission actions therefore remain unavailable until the VPN route is restored.
- `[x]` Implement an atomic full-screening fetch/import gate in [`scripts/fetch_bistka_screening_storage_safe_results.sh`](scripts/fetch_bistka_screening_storage_safe_results.sh).
- `[x]` Implement strict 150-trial validation in [`scripts/validate_screening_storage_safe_results.py`](scripts/validate_screening_storage_safe_results.py): five completed SLURM array elements, exact immutable attempt/release/preflight/config/data lineage, complete task summaries, all trial identities, finite validation objectives, 150 retention receipts, zero checkpoint paths, content fingerprinting, and an isolated aggregation dry run.
- `[x]` Add eight focused tests for the full-screening import path; all eight pass.
- `[x]` Locate and retrieve the official no-key Bank of Thailand `EC_EI_002_S2` full-history CSV. Preserve the 24,755-byte source vintage at [`data/external/original/EC_EI_002_S2_ENG_ALL_20260725.CSV`](data/external/original/EC_EI_002_S2_ENG_ALL_20260725.CSV), SHA-256 `03851a69b651320d45230df97a3eedf3d3eea04ba2b94384445c3a2a3cd79636`.
- `[x]` Implement a fail-closed transposed-export normalizer in [`src/data/ingest_bot_historical_export.py`](src/data/ingest_bot_historical_export.py), including source preservation, exact metric/month completeness checks, revision flags, source URLs, retrieval timestamp, checksums, and a scientific-eligibility receipt.
- `[x]` Normalize October 2020–December 2024 to [`data/external/bot_historical_macro.csv`](data/external/bot_historical_macro.csv): 459 rows, 51 monthly periods, nine indicators, zero missing numeric observations.
- `[x]` Advance structural external-source readiness from 0/5 to 1/5: official macro history is present; broader universe, full sector mapping, historical news, and licensed fundamentals remain absent.
- `[x]` Build [`data/processed/features_real_ohlcv_official_macro_revised_exploratory.csv`](data/processed/features_real_ohlcv_official_macro_revised_exploratory.csv): 4,840 rows, 968 dates, five tickers, five populated BOT feature columns, 48 joined release dates, and zero joins to a future scheduled release date.
- `[!]` Keep the BOT table outside causal or publication-grade backtesting. All 459 retained observations are marked revised, so the current full-history download is not a point-in-time vintage. Vintage snapshots or another point-in-time source are still required before official-macro performance claims or the preregistered protocol gate can pass.
- `[x]` Add a standalone fail-closed revised-history builder and validator. The builder requires the visibly named `--acknowledge-revised-history` flag, binds the source receipt, writes only explicitly exploratory outputs, and leaves the legacy `data/raw/bot_official_macro.csv` model-input path untouched.
- `[x]` Validate the exploratory merge with a checksum-bound receipt at [`reports/data_sources/bot_macro_exploratory_feature_validation_20260725.json`](reports/data_sources/bot_macro_exploratory_feature_validation_20260725.json).
- `[x]` Pass 24 focused screening/data tests together: eight screening-gate tests, seven new BOT normalizer/builder/integration tests, and nine existing intake/source-readiness regression tests.
- `[x]` Reconcile the active release after detecting plotting-script drift: preserve the dynamic job-label implementation as [`scripts/plot_vector_calibration_dynamic_job.py`](scripts/plot_vector_calibration_dynamic_job.py), restore the manifest-locked path exactly, and reverify the active graph at 9 `VERIFIED`, 1 `READY`, and 25 `PENDING`.
- `[x]` Record this work and its evidence boundary in [`reports/main_progress_update_20260725.md`](reports/main_progress_update_20260725.md).

## Large-Scale Campaign Status — 2026-07-17

- **Whole-project campaign completion:** **29.7%** by the canonical verified-stage denominator (`11/37`); one additional stage is technically ready and 25 are pending. The active consolidated graph is **9 VERIFIED, 1 READY, 25 PENDING** across 35 stages. Its 9/35 native fraction is structural—not a loss of progress—because `calibration_lineage` consolidates three former gates into one.
- `[x]` Resolve the observed-data campaign to 28 dependency-gated stages, 498 planned model runs, and 29.44 million requested timesteps.
- `[x]` Apply and verify corrected exact-release deployment `6d66448341c780c5b90a2d2d54cf2b4d4bebd972a18885aad727a05ff8236aaa` on BistKA.
- `[x]` Bind calibration and screening to the exact corrected `data/processed/features_real_ohlcv_market.csv` (2,183,530 bytes; SHA-256 `b877b08ed7a022db47739423291df0232ce377869cb7e0de6162ab82291ac6cc`).
- `[x]` Submit exactly five calibration-related jobs in the historical V1 pass: `3525`, `3526`, `3528`, `3529`, and `3530`; stop all further V1 submissions after reaching that pass's five-job ceiling.
- `[x]` Quarantine rejected result artifacts from diagnostic job `3528` and stale-data job `3529` under [`reports/rejected_calibration/20260715_quota_and_data_shadow/`](reports/rejected_calibration/20260715_quota_and_data_shadow/); retain scheduler logs for all five jobs under [`logs/`](logs/) and exclude every rejected artifact from accepted evidence.
- `[x]` Accept combined recovery job `3530` only after its attempt-, release-, preflight-, data-, and output-bound receipts passed locally.
- `[x]` Verify the vector benchmark and select `num_envs: 1`; median throughput for 1/2/4/8/16 environments is 347.9200/265.9007/217.0773/143.4171/83.6104 samples/second.
- `[x]` Record maximum process-tree RSS for 1/2/4/8/16 environments as 602.484/1,641.066/2,661.281/4,690.129/8,768.617 MiB.
- `[x]` Verify fixed-seed determinism across two independent processes for 549 steps.
- `[x]` Materialize `num_envs: 1`, deploy the calibrated screening overlay, and verify its exact-release BistKA preflight.
- `[x]` Preserve the accepted V1 campaign boundary at 10 `VERIFIED`, 1 `READY` (`screening`), and 17 `PENDING`; V1 screening remained unsubmitted after the historical five-job ceiling was reached.
- `[x]` Freeze the accepted V1 state as a 20-artifact historical snapshot with SHA-256 `883bebea341c9173acb60a52e19d9a88fb68ca842124be2c4e67190e02cf1c7d`.
- `[x]` Implement the strict V2 campaign as 37 dependency-gated stages with six concrete deployment/preflight pairs, 503 planned runs, 29,450,240 requested timesteps, and 23 bounded SLURM array elements.
- `[x]` Implement five cumulative profiles—screening, budget, nested, protocol, and stress—that inherit the prior exact manifest and add newly locked evidence.
- `[x]` Require every aggregate stage to inherit its `run_source` release, preflight, and effective runtime configurations and fail closed on a missing or mismatched binding.
- `[x]` Bind validation decisions and protocol-lock selection inputs to exact evidence from currently `VERIFIED` aggregate gates; reject newer unverified glob matches.
- `[x]` Build, deploy, and verify the current V2 base release `4b7d1d4b887bf08675bc664683b06f0fb103369016f7233e6caeb594e41ee382`.
- `[x]` Capture live BistKA V2 deployment capacity in [`reports/bistka_capacity_v2_20260716.txt`](reports/bistka_capacity_v2_20260716.txt): project storage 36.31 GiB/300 GiB, project credit 9,984.12 SHr remaining, user credit 496.25 SHr remaining, and an empty queue.
- `[x]` Atomically apply V2 base release `4b7d1d4b887bf08675bc664683b06f0fb103369016f7233e6caeb594e41ee382` to BistKA.
- `[x]` Verify the returned [`reports/bistka_calibration_v2_preflight.json`](reports/bistka_calibration_v2_preflight.json): `bistka-master`, exact release and observed-market binding, execution eligible, and no validation issues.
- `[x]` Verify the deployed bundle marker, both remote manifest hashes, receipt hash, cleanup of release temporary/backup files, and an empty post-apply queue.
- `[x]` Preserve the checksum-bound [`V1-to-V2 calibration migration receipt`](campaigns/large_scale_thai_drl_v2/imports/archive/v1_calibration_evidence_before_dependency_lock_20260716.json), SHA-256 `405ce3758b763fd94e82ac3a45f6d20d3fac803ce16d786c370b4b3a364a299c`, as archived historical compatibility evidence for accepted source job `3530`.
- `[x]` Run fresh release-native V2 vector calibration in job `3540`; verify its attempt-, release-, preflight-, data-, runtime-, and output-bound receipts and retain measured `num_envs: 1`.
- `[x]` Run fresh release-native V2 independent-process determinism in job `3541`; verify deterministic reproduction and exact provenance.
- `[x]` Restrict calibration compatibility normalization to the non-computational `project.name` field; verify both configurations normalize to SHA-256 `7a413f1f023b8ca942092f516bbc6e2463336b02cc39b89c55e67708bfba92e7`.
- `[x]` Materialize the V2 locked screening configuration with SHA-256 `d7b6ab71ede33ae4c4106fded3078b9a4eb3c56af345e67960061feea1897c1e`.
- `[x]` Build and apply the current cumulative V2 screening release `ef89ff5a738c5b25460ae40ad7789239134d761ce511a36665f857e01e3d74b4`, manifest SHA-256 `d9a1fd96af46cfe022d12b38d2de97642296a4a1f66928bbbbf3669af4dcfd86`.
- `[x]` Verify [`reports/bistka_screening_v2_preflight.json`](reports/bistka_screening_v2_preflight.json), SHA-256 `99ff875694ff47b6a02541718b6af93699de1f789ef35effaefe24d2e9655951`, against that exact cumulative release.
- `[x]` Preserve failed screening-smoke job `3542` under [`campaigns/large_scale_thai_drl_v2/incidents/screening_smoke_3542/`](campaigns/large_scale_thai_drl_v2/incidents/screening_smoke_3542/); its exact failure was a checkpoint write rejected after the personal Unix group `s00658` exceeded its soft quota and grace period.
- `[x]` Recover without deleting research evidence: clean the reproducible package cache, relocate all historical `final_archive/*.tar.gz` files to project-owned storage with SHA-256 verification, and retain their documented paths as verified symlinks.
- `[x]` Complete and strictly verify the five-algorithm screening smoke in retry job `3543`; all five manifests are complete and bound to the exact attempt, release, preflight, configuration, and observed feature data.
- `[x]` Verify smoke-based full-screening runtime projections for PPO, A2C, DDPG, TD3, and SAC are all below the 72-hour limit.
- `[!]` Do not submit the original full-screening path: the measured 6.45 GiB personal-group soft-quota headroom is below its approximately 7.69 GiB projected output before logs and metadata.
- `[x]` Record the fresh read-only quota evidence in [`reports/screening_capacity_post_smoke_20260717.txt`](reports/screening_capacity_post_smoke_20260717.txt) and the checksum-backed operational hold in [`reports/screening_capacity_decision_20260717.json`](reports/screening_capacity_decision_20260717.json).
- `[x]` Implement a storage-safe screening runner that suppresses redundant checkpoint writes before the job `3542` failure point, retains final models and scientific evidence, emits checksum-bound per-trial retention receipts, and rejects unsafe reuse.
- `[x]` Implement a fail-closed capacity gate with strict group/project quota parsing, exact sidecars, a direct-child 1 MiB `fsync` write probe, byte/inode checks, and a 15-minute evidence-freshness limit.
- `[x]` Capture checksum-bound BistKA quota/write evidence and compute a conservative point-in-time candidate pass: 3.854 GiB retained including receipt/task-summary budgets, 3.840 GiB of checkpoint writes avoided, 49.88% net reduction, 6.031 GiB safety-adjusted requirement, and 0.420 GiB margin beneath measured group soft headroom.
- `[x]` Implement a dedicated five-algorithm storage-safe smoke adapter and SLURM entrypoint that require valid retention receipts, reject checkpoint inventory paths, emit a separate checksum-bound gate receipt, and restore all temporary hooks on failure.
- `[x]` Build the deterministic 218-file storage-safe candidate release `16ca55f9264a54e3bc86ecf51d2e4b3ab0c4e85ddb80f22e7dd8585a466a3dc2`; validate its manifest and bundle twice with identical identities and perform a local dry run.
- `[x]` Preserve the prior control plane and active release in a rollback archive with SHA-256 `5dd485c41ee5e287c893081f760f1f496fb458e1e1097e691fb64e254e822aee`; accept pre-cutover calibration without recomputation through lineage receipt SHA-256 `25cda3f51aeb200353064e495fd58484ade711f36ca575c9633ae231e069c807`.
- `[x]` Build the deterministic 248-file final cutover release `56358f2a4996f654426814d88e622b7f217ac89bfc02f8bbdacc27e66873b55e`, manifest SHA-256 `439b590bc4a85f03a6d98723ee3d28e40acdc00d95bdfbf48d310f07d867e25c`, and bundle SHA-256 `a5135665dd14fdfcd0a5dcdbfab70a7e6f88fd78fc75ff946df9851c01a4072e`.
- `[x]` Apply the final release atomically with rollback available and verify [`reports/bistka_screening_storage_safe_cutover_v2_preflight.json`](reports/bistka_screening_storage_safe_cutover_v2_preflight.json), SHA-256 `1ccac9827e2f018927d711c0fb0918119c79f8721bd6d82d3d08ed026ab59f0c`: exact release, 248/248 files, observed data, runtime packages, and no issues.
- `[x]` Capture a fresh pre-smoke capacity pass with 448,098,472 bytes (0.417 GiB) modeled margin and submit exactly one new cutover-window job, storage-safe smoke `3550`.
- `[x]` Verify job `3550` as `COMPLETED` with five complete manifests, five checksum-bound retention receipts, zero checkpoint paths, and a passing dedicated storage-safe gate.
- `[x]` Confirm the replacement smoke used 137,777,145 bytes versus 275,209,516 bytes for job `3543`, an observed reduction of approximately 49.94% despite seven additional evidence files.
- `[x]` Confirm all job-`3550` full-task projections remain below 72 hours: PPO 3.7285 h, A2C 0.2556 h, DDPG 20.4114 h, TD3 21.1114 h, and SAC 21.5986 h.
- `[x]` Recapture post-smoke capacity: all eight quota checks and the write probe passed, but modeled margin narrowed to 331,455,656 bytes (0.309 GiB) and the 15-minute receipt is now historical.
- `[x]` Reach the active consolidated V2 state of 9 `VERIFIED`, 1 `READY` (`screening`), and 25 `PENDING` across 35 stages while retaining **11/37 (29.7%)** as the canonical whole-project progress denominator.
- `[x]` Record five V2 submissions to date—`3540`, `3541`, `3542`, `3543`, and `3550`—with only job `3550` submitted in the final cutover window. The full-screening array remains unsubmitted.
- `[x]` Pass the expanded local regression suite with 293 tests on 2026-07-17; compilation, shell syntax, active/candidate/final manifest and bundle validation, lineage, rollback, import, checksum, and whitespace checks also pass.
- `[!]` Keep feature ablation blocked until observed macro and fundamental tables pass readiness, or formally amend the preregistered feature scope before test access.
- `[x]` Record the accepted/rejected execution history and evidence links in [`reports/bistka_calibration_execution_20260715.md`](reports/bistka_calibration_execution_20260715.md).
- `[x]` Record the V2 implementation, evidence, limitations, and bounded submission plan in [`reports/v2_control_plane_implementation_20260715.md`](reports/v2_control_plane_implementation_20260715.md).
- `[x]` Record the checksum-bound compatibility proof, migrated evidence, exact releases, limitations, and zero-job outcome in [`reports/v1_to_v2_calibration_migration_20260716.md`](reports/v1_to_v2_calibration_migration_20260716.md).
- `[x]` Record the screening-smoke execution, quota incident, lossless recovery, observed runtime projections, storage projection, and submission decision in [`reports/screening_smoke_execution_20260716.md`](reports/screening_smoke_execution_20260716.md).
- `[x]` Record the storage-safe implementation, measured capacity, deterministic candidate release, limitations, and controlled cutover in [`reports/screening_storage_safe_release_candidate_20260717.md`](reports/screening_storage_safe_release_candidate_20260717.md).
- `[x]` Record the applied cutover, exact preflight, job `3550`, post-smoke capacity, progress accounting, limitations, and next execution window in [`reports/screening_storage_safe_cutover_execution_20260717.md`](reports/screening_storage_safe_cutover_execution_20260717.md).

## Communication And Process Reporting — 2026-07-20

- `[x]` Produce a seven-slide, self-guided English HPC Showcase presentation explaining the project and the significance of HPC for a general audience, with no speaker required.
- `[x]` Produce a matching Thai-English presentation and visually verify that Thai typography renders correctly in both the editable PPTX and display PDF.
- `[x]` Preserve editable slide copy in [`reports/hpc_showcase_hpc_presentation_draft_20260720.md`](reports/hpc_showcase_hpc_presentation_draft_20260720.md).
- `[x]` Produce five focused, four-page PDF reports covering the research gap, storage-safe cutover, HPC job/capacity audit, reproducibility/provenance, and next-phase execution plan.
- `[x]` Render and visually inspect all 14 PPTX slides, 14 presentation-PDF pages, and 20 technical-report pages; verify presentation overflow and technical-report text/page bounds.
- `[x]` Keep public and technical reporting aligned with the accepted evidence boundary: 11/37 canonical completion (29.7%), job `3550` completed, 49.94% observed smoke-output reduction, and full screening unsubmitted.
- `[x]` Rebuild the HPC Showcase communication as one-page English and Thai-English posters using page 2 of the supplied 1:2 school poster template; preserve its project-summary, HPC/software, four evidence windows, central team-photo, and two team-perspective regions.
- `[x]` Export both template-based posters as editable PPTX files and visually faithful 1134 x 2268 pt PDFs; render both at 3024 x 6048 pixels and confirm zero slide overflow.
- `[~]` Public-poster content is ready, but the central team photo and the two drafted member-perspective paragraphs still require member-provided/approved inputs before final printing.

## Current Snapshot

- `[x]` Local project scaffold exists.
- `[x]` BistKA HPC project directory created: `/lustrefs/project/25sfcs03/drl_thai_stock`.
- `[x]` BistKA conda environment created: `/lustrefs/project/25sfcs03/envs/drl_env`.
- `[x]` Synthetic pilot pipeline runs end-to-end.
- `[x]` Trading environment, baselines, PPO training, metrics, action logging, and comparison summaries are implemented.
- `[x]` Latest verified HPC PPO technical run: `results/run_2850`.
- `[x]` Latest verified HPC tests: `53 passed` on 2026-05-31 after adding ticker-universe CSV support.
- `[x]` Buy-and-hold and moving-average crossover baselines are implemented and verified on HPC.
- `[x]` Mean-variance baseline is implemented and verified on HPC.
- `[x]` Result plotting utility for equity curves, drawdowns, turnover, and action weights is implemented and verified on HPC.
- `[x]` Latest local syntax check: `python3 -m compileall -q src tests` passed on 2026-05-30.
- `[x]` Latest local syntax check after Optuna trial-runner edits: `python3 -m compileall -q src tests` passed on 2026-05-30.
- `[x]` Latest local syntax check after walk-forward/report edits: `python3 -m compileall -q src tests` passed on 2026-05-30.
- `[x]` Latest local syntax check after file-separation helper script: `python3 -m compileall -q src tests scripts` passed on 2026-05-30.
- `[x]` Latest local syntax check after walk-forward runner edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-30.
- `[x]` Latest local syntax check after walk-forward aggregation edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-30.
- `[x]` Latest local syntax check after ablation tooling edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-30.
- `[x]` Latest local syntax check after regime-test edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-31.
- `[x]` Latest local syntax check after SET index feature edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-31.
- `[x]` Latest local syntax check after market-context config activation: `python3 -m compileall -q src tests scripts` passed on 2026-05-31.
- `[x]` Latest local syntax check after sentiment ingestion/merge edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-31.
- `[x]` Latest local syntax check after official macro ingestion/merge edits: `python3 -m compileall -q src tests scripts` passed on 2026-05-31.
- `[x]` Latest local syntax/notebook validation after data-quality notebook addition: `python3 -m compileall -q src tests scripts` and `python3 -m json.tool notebooks/data_quality_overview.ipynb` passed on 2026-05-31.
- `[x]` Latest local syntax check after historical sentiment CSV importer addition: `python3 -m compileall -q src tests scripts` passed on 2026-05-31.
- `[x]` Latest HPC validation after sector-index probe addition: `python -m compileall -q src tests scripts` and full `python -m pytest -q` passed with 43 tests on 2026-05-31.
- `[x]` Latest HPC validation after official macro CSV importer addition: `python -m compileall -q src tests scripts` and full `python -m pytest -q` passed with 46 tests on 2026-05-31.
- `[x]` Latest HPC validation after historical sentiment CSV importer addition: focused importer tests passed with 3 tests, CLI smoke test passed, and full `python -m pytest -q` passed with 49 tests on 2026-05-31.
- `[x]` Latest HPC validation after ticker-universe CSV support: focused ticker-universe tests passed with 4 tests, config resolver smoke test passed, and full `python -m pytest -q` passed with 53 tests on 2026-05-31.
- `[x]` Latest local full test validation: `.venv/bin/python -m pytest -q` passed with 53 tests on 2026-06-09 after creating a Python 3.11 local environment.
- `[x]` SET market-index ingestion and market-context feature merge implemented and verified on HPC.
- `[x]` SET index raw data collected on HPC: `data/raw/prices_market_indices.csv`.
- `[x]` Real OHLCV + SET market-context feature table built on HPC: `data/processed/features_real_ohlcv_market.parquet`.
- `[x]` Latest SET market-context pilot completed on HPC: `results/real_ohlcv_market_pilot_20260531`.
- `[x]` Pilot sector mapping added for the five starter tickers: `data/reference/sector_mapping_thai_pilot.csv`.
- `[~]` SET sector-index Yahoo candidate probe added and run on HPC: `data/raw/sector_index_yahoo_probe.csv`; only banking symbol `^BANK` was usable.
- `[x]` Real OHLCV + SET + sector-context feature table built on HPC: `data/processed/features_real_ohlcv_sector.parquet`.
- `[x]` Latest sector-context pilot completed on HPC: `results/real_ohlcv_sector_pilot_20260531`.
- `[x]` Yahoo Finance macro proxy ingestion added and verified on HPC: `data/raw/prices_macro_yahoo.csv`.
- `[x]` Real OHLCV + SET + sector + macro-proxy feature table built on HPC: `data/processed/features_real_ohlcv_macro.parquet`.
- `[x]` Latest macro-proxy pilot completed on HPC: `results/real_ohlcv_macro_pilot_20260531`.
- `[x]` Yahoo Finance annual and quarterly fundamentals ingestion added and verified on HPC: `data/raw/fundamentals_yahoo_quarterly.csv`.
- `[x]` Real OHLCV + SET + sector + macro-proxy + fundamentals feature table built on HPC: `data/processed/features_real_ohlcv_fundamentals.parquet`.
- `[x]` Latest fundamentals pilot completed on HPC: `results/real_ohlcv_fundamentals_pilot_20260531`.
- `[x]` Yahoo Finance latest-news probe added and run on HPC: `data/raw/news_yahoo_latest.csv`.
- `[x]` Daily sentiment extraction and no-lookahead sentiment merge scaffold implemented and verified on HPC.
- `[x]` Historical sentiment/news CSV importer added and verified on HPC: `src/data/ingest_sentiment_csv.py`.
- `[x]` Real OHLCV + SET + sector + macro-proxy + fundamentals + sentiment scaffold feature table built on HPC: `data/processed/features_real_ohlcv_sentiment.parquet`.
- `[!]` Yahoo latest-news probe returned only 11 rows dated 2025-06-24 to 2026-03-31, so it does not overlap the 2021-2024 modeling window and cannot support a valid historical sentiment pilot. A historical CSV import path is now implemented and verified on HPC.
- `[x]` Bank of Thailand official macro source probe and release-date merge scaffold implemented and verified on HPC.
- `[x]` BOT Leading Economic Indicator latest-table probe collected historically on HPC at `data/raw/bot_official_macro.csv`; it is not the current local exploratory input.
- `[x]` Real OHLCV + SET + sector + macro-proxy + fundamentals + official macro scaffold feature table built on HPC: `data/processed/features_real_ohlcv_official_macro.parquet`.
- `[x]` Record that the original 54-row web-table probe dated 2025-11-01 to 2026-04-01 did not overlap the modeling window. The later official full-history export closes structural coverage only; its revised values still do not satisfy point-in-time eligibility.
- `[x]` Official macro historical CSV importer added and verified on HPC: `src/data/ingest_official_macro_csv.py`.
- `[x]` Regime/stress-test config and runner implemented and verified on HPC.
- `[x]` Latest regime/stress test completed on HPC: `results/regime_tests_real_ohlcv_20260531`.
- `[x]` Real-OHLCV ablation configs, SLURM array script, and aggregation utility implemented and verified on HPC.
- `[x]` Latest ablation pilot completed on HPC: `results/ablation_pilot_2871`.
- `[x]` Latest algorithm comparison job completed on HPC: `results/algo_compare_2853`.
- `[x]` Train/validation/test split wiring added for training and baseline evaluation, verified on HPC.
- `[x]` Latest split-aware PPO/A2C validation comparison completed on HPC: `results/algo_compare_2853`.
- `[x]` Leakage check utilities added for date/ticker duplicates, split boundaries, and future release dates; verified on HPC.
- `[x]` Real Thai OHLCV starter ingestion added via Yahoo Finance/yfinance schema and verified on HPC.
- `[x]` Ticker-universe CSV resolver added and verified on HPC for real OHLCV, fundamentals, and latest-news ingestion: `src/data/ticker_universe.py`.
- `[x]` Starter ticker-universe CSV and example config added and verified on HPC: `data/reference/thai_starter_universe.csv`, `config/real_ohlcv_universe_csv.yaml`.
- `[x]` External-data handoff templates added for broad ticker universe, sector mapping, official macro, historical sentiment/news, and fundamentals: `data/reference/*_template.csv`.
- `[x]` External-source readiness checker added: `python -m src.data.source_readiness --output reports/source_readiness.csv`.
- `[x]` External-data handoff guide added: `docs/external_data_handoff.md`.
- `[x]` Real OHLCV feature table built on HPC: `data/processed/features_real_ohlcv.parquet`.
- `[x]` Real OHLCV feature table passed leakage and split-boundary checks on HPC.
- `[x]` Synthetic-regeneration guard added and verified so non-synthetic configs cannot overwrite real feature tables.
- `[x]` Latest real-OHLCV PPO/A2C validation comparison completed on HPC: `results/real_ohlcv_compare_2855`.
- `[x]` Real-data Optuna trial runner implemented, uploaded, and verified through SLURM sanity job `2856`.
- `[x]` Optuna trial aggregation utility implemented and verified on HPC.
- `[x]` Latest real-OHLCV Optuna search completed on HPC: `results/optuna_search_fixed_2863`.
- `[x]` Best Optuna PPO config prepared and run on HPC: `config/best_ppo_real_ohlcv.yaml`, job `2868`.
- `[x]` Final methodology/results writeup drafted and uploaded: `docs/final_methodology_results.md`.
- `[x]` Real OHLCV data-quality report generated on HPC: `reports/data_quality/real_ohlcv_data_quality.md`.
- `[x]` Walk-forward window config and summary writer implemented and verified on HPC: `results/walk_forward_summary_check/walk_forward_summary.csv`.
- `[x]` Walk-forward per-window training runner and SLURM array script implemented and verified on HPC.
- `[x]` Walk-forward aggregation utility implemented and verified on HPC.
- `[x]` Latest walk-forward pilot completed on HPC: `results/walk_forward_pilot_2869`.
- `[x]` File-separation snapshot helper added: `scripts/file_separation_snapshot.py`.
- `[x]` Final artifact manifest uploaded: `docs/final_artifacts_manifest.md`.
- `[x]` Final methodology/results report converted to LaTeX locally: `final_methodology_results.tex`.
- `[x]` Comprehensive introduction and literature review written in LaTeX and compiled locally: `comprehensive_introduction_literature_review.pdf` has 11 pages.
- `[x]` Latest local documentation/source-readiness artifacts synced to BistKA on 2026-06-09 by pulling the pushed GitHub `main` archive into `/lustrefs/project/25sfcs03/drl_thai_stock`.
- `[x]` HPC source-readiness check ran on 2026-06-09 and wrote `reports/source_readiness.csv`; it correctly reports the five expected missing external source files.
- `[x]` BistKA sync status and verified GitHub-archive refresh method documented: `hpc/hpc_sync_status.md`.
- `[x]` Full external-data config and rebuild-order guide added: `config/real_ohlcv_full_external.yaml`, `docs/full_external_rebuild.md`.
- `[x]` Full external-data config and rebuild guide synced to BistKA on 2026-06-09; HPC source-readiness still correctly reports the five missing provider files.
- `[x]` External data provider request packet and email template added: `docs/external_data_provider_request.md`.
- `[x]` External data provider request packet synced to BistKA on 2026-06-09.
- `[x]` External data intake checklist and provider manifest template added: `docs/external_data_intake_checklist.md`, `data/reference/external_data_manifest_template.csv`.
- `[x]` External data intake checklist and provider manifest template synced to BistKA on 2026-06-10.
- `[x]` External data intake validation CLI and focused tests added: `src/data/intake_validation.py`, `tests/test_intake_validation.py`.
- `[x]` External data intake validation CLI synced to BistKA on 2026-06-10; it correctly reports missing provider files in `reports/external_intake_validation.csv`.
- `[x]` External data intake validation CLI extended with explicit path and date-window arguments.
- `[x]` External data intake validation CLI extended with Markdown report output.
- `[x]` External data intake Markdown report synced and verified on BistKA on 2026-06-10; it correctly reports missing provider files in `reports/external_intake_validation.md`.
- `[x]` Comprehensive Beamer presentation drafted locally and synced to HPC: `final_project_beamer_100_pages.tex`.
- `[x]` Comprehensive Beamer presentation compiled locally: `final_project_beamer_100_pages.pdf`.
- `[x]` Comprehensive Beamer presentation refreshed on 2026-06-28 with explicit introduction, literature review, methodology, discussion, conclusion, and reference sections; local PDF verification shows 109 pages.
- `[x]` Final pilot archive created on HPC: `final_archive/20260530_final_project_pilot.tar.gz`.
- `[x]` Updated final pilot archive with walk-forward, ablation, and regime/stress-test artifacts: `final_archive/20260531_final_project_pilot.tar.gz`.
- `[x]` Updated final pilot archive with SET market-context artifacts: `final_archive/20260531_final_project_pilot_with_set_context.tar.gz`.
- `[x]` Updated final pilot archive with sector-context artifacts: `final_archive/20260531_final_project_pilot_with_sector_context.tar.gz`.
- `[x]` Updated final pilot archive with macro-proxy artifacts: `final_archive/20260531_final_project_pilot_with_macro_context.tar.gz`.
- `[x]` Updated final pilot archive with fundamentals artifacts: `final_archive/20260531_final_project_pilot_with_fundamentals.tar.gz`.
- `[x]` Updated final pilot archive with sentiment-scaffold artifacts: `final_archive/20260531_final_project_pilot_with_sentiment_scaffold.tar.gz`.
- `[x]` Updated final pilot archive with official-macro-scaffold artifacts: `final_archive/20260531_final_project_pilot_with_official_macro_scaffold.tar.gz`.
- `[x]` Updated final pilot archive with sector-index source probe artifacts: `final_archive/20260531_final_project_pilot_with_sector_index_probe.tar.gz`.
- `[x]` Updated final pilot archive with official macro CSV importer artifacts: `final_archive/20260531_final_project_pilot_with_official_macro_csv_import.tar.gz`.
- `[x]` Updated final pilot archive with historical sentiment CSV importer artifacts: `final_archive/20260531_final_project_pilot_with_sentiment_csv_import.tar.gz`.
- `[x]` Updated final pilot archive with ticker-universe CSV support: `final_archive/20260531_final_project_pilot_with_ticker_universe_csv.tar.gz`.
- `[x]` Latest file-separation snapshot on HPC: `_file_separation/20260601_100216`.
- `[!]` Real-OHLCV comparison job `2854` is invalid because the old training entrypoint regenerated synthetic data into real config paths; use corrected job `2855`.
- `[!]` Real OHLCV starter data, configurable ticker-universe loading, SET index context, pilot sector mapping, daily macro proxies, Yahoo statement fundamentals, sentiment merge/import scaffolding, and a structurally complete revised BOT history exist, but broad sector indices, point-in-time official macro vintages, licensed fundamentals, and historical sentiment source data are still incomplete.

## Phase 1: Foundation And Repository Setup

- `[x]` Create repository-style folder structure.
- `[x]` Add `README.md`.
- `[x]` Add `requirements.txt`.
- `[x]` Add `environment.yml`.
- `[x]` Add base configs: `config/default.yaml`, `config/debug.yaml`, `config/ppo_technical.yaml`.
- `[x]` Add source package folders under `src/`.
- `[x]` Add tests folder.
- `[x]` Add HPC notes and SLURM templates.
- `[x]` Upload project to BistKA.
- `[x]` Separate uploaded files from pre-existing/generated files on HPC with `_file_separation/latest`.

## Phase 2: Data Manifest And Pilot Data

- `[x]` Add `data/data_manifest.csv`.
- `[x]` Add `docs/data_sources.md`.
- `[x]` Implement deterministic synthetic price generator.
- `[x]` Generate synthetic pilot raw prices on HPC.
- `[x]` Generate synthetic pilot feature table on HPC.
- `[x]` Add validation so stale pilot feature files are regenerated when config ticker/date/column coverage changes.
- `[~]` Collect real SET50/SET100 OHLCV data; starter Yahoo Finance download completed for 5 `.BK` tickers, and the ingesters can now load a broader ticker list from CSV once a curated/licensed universe file is available.
- `[~]` Collect SET index and sector index data; SET index collected with Yahoo symbol `^SET.BK`, and the sector-index candidate probe found only banking symbol `^BANK`, so a complete sector-index source is still pending.
- `[x]` Collect daily macro proxy data from Yahoo Finance.
- `[~]` Collect official real macro data with release-date alignment; the official 2000–2026 BOT full-history export is now preserved and the 2020–2024 window is structurally complete, but it is a current revised vintage rather than point-in-time history.
- `[~]` Collect real news/sentiment source data; Yahoo latest-news probe works but has no 2021-2024 overlap, and a historical CSV import path is implemented locally.
- `[x]` Create data-quality notebook: `notebooks/data_quality_overview.ipynb`.
- `[x]` Add missing value and coverage reports for real data.

## Phase 3: Feature Engineering

- `[x]` Implement technical indicators.
- `[x]` Add returns: 1-day, 5-day, 20-day.
- `[x]` Add rolling volatility.
- `[x]` Add moving-average ratios.
- `[x]` Add RSI.
- `[x]` Add MACD.
- `[x]` Add Bollinger z-score.
- `[x]` Add volume ratio.
- `[x]` Include synthetic market return, macro change, and sentiment score placeholders.
- `[x]` Add ATR.
- `[~]` Add sector-relative features; the isolated V3 active-member/date-aware implementation and tests are complete, but licensed broad-universe bars and trading-status coverage are still required before materializing model-ready sector features.
- `[x]` Add daily macro proxy features.
- `[~]` Add official release-aligned macro features; an explicitly exploratory revised-history table is built and checksum-validated with no future scheduled-release joins, but historical vintage snapshots are still required for causal modeling.
- `[x]` Add Yahoo annual/quarterly statement fundamentals with reporting lag.
- `[~]` Add real sentiment embeddings or scores; daily extraction, historical CSV import, and merge scaffolding are implemented, but a historical licensed/news source is still needed.
- `[x]` Add train/validation/test split utilities.
- `[x]` Wire train/validation/test splits into training and baseline evaluation entrypoints.
- `[x]` Add leakage checks for real mixed-frequency data.

## Phase 4: Trading Environment And Baselines

- `[x]` Implement Gymnasium-compatible `ThaiStockTradingEnv`.
- `[x]` Continuous long-only action vector.
- `[x]` Portfolio accounting.
- `[x]` Transaction cost and turnover handling.
- `[x]` Reward with return, turnover penalty, and drawdown penalty.
- `[x]` Stable observation shape.
- `[x]` Info dict includes portfolio value, return, turnover, drawdown, weights, and cash weight.
- `[x]` Add environment tests.
- `[x]` Implement equal-weight baseline.
- `[x]` Implement random baseline.
- `[x]` Implement momentum baseline.
- `[x]` Implement buy-and-hold per ticker baseline.
- `[x]` Implement moving-average crossover baseline.
- `[x]` Implement mean-variance baseline.

## Phase 5: PPO Technical Agent

- `[x]` Integrate Stable-Baselines3 PPO.
- `[x]` Run debug PPO job on BistKA.
- `[x]` Run technical PPO job on BistKA.
- `[x]` Save PPO model.
- `[x]` Save PPO equity curve.
- `[x]` Save PPO metrics.
- `[x]` Save PPO action weights.
- `[x]` Save PPO action summary.
- `[x]` Save TensorBoard logs.
- `[x]` Compare PPO against baselines for latest verified run.
- `[x]` Latest verified run: `/lustrefs/project/25sfcs03/drl_thai_stock/results/run_2850`.
- `[x]` Add checkpoint callbacks during training.
- `[x]` Add action distribution plots via `src/evaluation/plot_results.py`.
- `[x]` Add drawdown plots via `src/evaluation/plot_results.py`.
- `[x]` Add turnover plots via `src/evaluation/plot_results.py`.

## Phase 6: Algorithm Comparison

- `[x]` Refactor training script to support multiple Stable-Baselines3 algorithms.
- `[x]` Add support paths for PPO, A2C, DDPG, TD3, and SAC.
- `[x]` Update comparison utility to aggregate multiple model metric files.
- `[x]` Add `slurm/algorithm_compare.sbatch`.
- `[x]` Upload and run `slurm/algorithm_compare.sbatch` on HPC.
- `[x]` Verify A2C run on HPC.
- `[x]` Compare PPO vs A2C vs baselines.
- `[x]` Run split-aware PPO vs A2C vs baselines validation comparison on HPC.
- `[x]` Decide whether DDPG, TD3, and SAC are worth running on the current environment: defer until real data and out-of-sample splits are active.

## Phase 7: Hyperparameter Search

- `[x]` Add initial Optuna/job-array SLURM template.
- `[x]` Convert Optuna trial runner to tune real PPO/A2C hyperparameters.
- `[x]` Add search space for learning rate, gamma, GAE lambda, clip range, entropy coefficient, rollout length, and batch size.
- `[x]` Add `config/optuna_search.yaml` for real-OHLCV PPO/A2C search.
- `[x]` Run small sanity search: SLURM array job `2856`, output `results/optuna_sanity_2856`.
- `[x]` Run HPC job array: replacement job `2863`, output `results/optuna_search_fixed_2863`.
- `[x]` Aggregate hyperparameter search results: `results/optuna_search_fixed_2863/trial_results.csv`.
- `[x]` Promote best Optuna PPO parameters into a dedicated config and run longer validation job.

## Phase 8: Multi-Source Features And Ablations

- `[~]` Synthetic placeholders exist for macro and sentiment.
- `[~]` Build real sentiment extraction input dataset; Yahoo latest-news probe produced `data/raw/news_yahoo_latest.csv` with no historical overlap, and `src/data/ingest_sentiment_csv.py` can normalize a user-provided historical export.
- `[~]` Run WangchanBERTa embedding or sentiment extraction on GPU; deterministic sentiment scorer is verified as a pipeline placeholder, GPU transformer extraction still pending.
- `[~]` Merge real sentiment features by date/ticker; no-lookahead merge scaffold is verified, but merged 2021-2024 sentiment values are zero until a historical news/sentiment CSV is provided.
- `[x]` Merge daily macro proxy features by no-lookahead forward fill.
- `[~]` Merge official real macro features by release date; the revised full-history BOT table now produces five populated exploratory features with no future scheduled-date joins, but point-in-time vintage values are still required before causal modeling.
- `[x]` Merge Yahoo annual/quarterly statement fundamentals with reporting lag.
- `[~]` Create feature-group configs: returns-only, technical, technical+context, index, sector, macro-proxy, fundamentals, sentiment-scaffold, and official-macro-scaffold configs are added and verified; licensed fundamentals, point-in-time official macro vintages, historical sentiment, and an all-real-feature release still depend on missing external data.
- `[x]` Run ablation pilot experiments: SLURM array job `2871`.
- `[x]` Write pilot ablation metrics: `results/ablation_pilot_2871/aggregate/ablation_metrics.csv`.

## Phase 9: Walk-Forward And Regime Testing

- `[x]` Implement walk-forward window config.
- `[x]` Add training/evaluation split support to environment or data loader.
- `[x]` Run one job per walk-forward window; pilot SLURM array job `2869` completed for windows 0 and 1.
- `[x]` Aggregate out-of-sample equity curves; output `results/walk_forward_pilot_2869/aggregate/walk_forward_test_equity_curves.csv`.
- `[x]` Compute pilot walk-forward metrics; output `results/walk_forward_pilot_2869/aggregate/walk_forward_policy_summary.csv`.
- `[x]` Define COVID/crisis period slices; acute 2020 COVID crash is marked unavailable because the starter real-OHLCV file begins in 2021.
- `[x]` Run regime/stress tests: `results/regime_tests_real_ohlcv_20260531`.

## Phase 10: Reporting And Final Artifacts

- `[x]` Main implementation plan exists: `implementation_plan.md`, `.tex`, `.pdf`.
- `[x]` Literature review exists: `literature_review/literature_review.md`, `.tex`, `.pdf`.
- `[x]` HPC guide exists.
- `[x]` SLURM guide exists.
- `[x]` Add final result table from synthetic algorithm comparison pilot.
- `[x]` Add equity curve figure generator.
- `[x]` Add drawdown figure generator.
- `[x]` Add action distribution figure generator.
- `[x]` Add ablation bar chart: `results/ablation_pilot_2871/aggregate/ablation_ppo_sharpe.png`.
- `[x]` Add final methodology/results writeup.
- `[x]` Add final methodology/results LaTeX report: `final_methodology_results.tex`.
- `[x]` Archive final config, logs, metrics, and model summaries.

## Latest Verified Results

Latest verified real-OHLCV split-aware algorithm comparison run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_compare_2855
```

Job `2855` completed on BistKA with an empty error log and produced PPO/A2C models, checkpoints, baseline comparisons, comparison summaries, action summaries, and figures.

Real OHLCV split used by `real_ohlcv_compare_2855`:

| split | start_date | end_date | rows | unique_dates | tickers |
| --- | --- | --- | ---: | ---: | ---: |
| train | 2021-01-04 | 2023-05-29 | 2900 | 580 | 5 |
| validation | 2023-05-30 | 2024-03-12 | 970 | 194 | 5 |
| test | 2024-03-13 | 2024-12-30 | 970 | 194 | 5 |

Main real-OHLCV validation comparison from `real_ohlcv_compare_2855`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 | 1043523.041075 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 | 1009998.557967 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 | 969426.916197 |
| a2c | -0.038061 | -0.057884 | -0.434028 | -0.120042 | 961938.712210 |
| ppo | -0.035900 | -0.054629 | -0.439532 | -0.066219 | 964100.020195 |
| buy_hold_KBANK.BK | -0.063331 | -0.095643 | -0.517619 | -0.125000 | 936669.309842 |
| buy_hold_AOT.BK | -0.077056 | -0.115925 | -0.569733 | -0.201365 | 922944.160998 |
| buy_hold_CPALL.BK | -0.093831 | -0.140495 | -0.658999 | -0.224335 | 906169.402777 |
| mean_variance | -0.099682 | -0.149008 | -1.097376 | -0.166737 | 900318.191064 |
| ma_crossover | -0.130706 | -0.193648 | -1.581191 | -0.178892 | 869294.229253 |
| random | -0.150932 | -0.222296 | -1.802937 | -0.191240 | 849068.332576 |
| momentum | -0.186758 | -0.272144 | -1.815650 | -0.216477 | 813242.480998 |

Invalidated run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_compare_2854
```

Run `2854` should not be used because the pre-fix training entrypoint regenerated synthetic data into the real config paths.

Latest verified real-OHLCV Optuna sanity run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/optuna_sanity_2856
```

Job `2856` ran two short 1000-timestep array tasks through SLURM:

| trial_id | algorithm | objective_sharpe | cumulative_return | max_drawdown |
| ---: | --- | ---: | ---: | ---: |
| 0 | ppo | -1.397044 | -0.003183 | -0.004519 |
| 1 | a2c | -2.325976 | -0.002303 | -0.002564 |

Invalidated Optuna search run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/optuna_search_2858
```

Job `2858` completed, but should not be used as a hyperparameter search because the pre-fix sampler produced identical sampled parameters for repeated PPO/A2C trial numbers.

Latest verified real-OHLCV Optuna search run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/optuna_search_fixed_2863
```

Job `2863` completed five 4000-timestep search tasks and wrote `trial_results.csv`.

| trial_id | algorithm | objective_sharpe | cumulative_return | max_drawdown | learning_rate | gamma | n_steps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | ppo | 0.279954 | 0.020408 | -0.068810 | 0.000389 | 0.98 | 256 |
| 0 | ppo | -0.305640 | -0.002649 | -0.008809 | 0.000017 | 0.90 | 512 |
| 1 | a2c | -0.519583 | -0.042686 | -0.127149 | 0.001210 | 0.99 | 256 |
| 2 | ppo | -1.079920 | -0.088712 | -0.115319 | 0.002160 | 0.95 | 256 |
| 3 | a2c | -1.708120 | -0.003138 | -0.004297 | 0.000029 | 0.90 | 128 |

Longer best-parameter PPO validation run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/best_ppo_real_ohlcv_2868
```

Job `2868` used `config/best_ppo_real_ohlcv.yaml` for 20000 timesteps. The error log was empty, but validation performance did not improve over the short Optuna trial:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| ppo | -0.030269 | -0.046132 | -0.403028 | -0.099395 | 969730.884751 |

Latest verified real-OHLCV walk-forward pilot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/walk_forward_pilot_2869
```

SLURM array job `2869` completed one short 1000-timestep PPO training/evaluation task per walk-forward window. Both error logs were empty:

- `logs/walk_forward_2869_0.err`
- `logs/walk_forward_2869_1.err`

Aggregated outputs:

- `results/walk_forward_pilot_2869/aggregate/walk_forward_all_metrics.csv`
- `results/walk_forward_pilot_2869/aggregate/walk_forward_policy_summary.csv`
- `results/walk_forward_pilot_2869/aggregate/walk_forward_test_equity_curves.csv`

Pilot test-window policy summary:

| policy | windows | mean_cumulative_return | mean_annualized_return | mean_sharpe | mean_max_drawdown | mean_final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| buy_hold_ADVANC.BK | 2 | 0.083927 | 0.281661 | 1.467928 | -0.064073 | 1083926.536029 |
| buy_hold_KBANK.BK | 2 | 0.064203 | 0.202352 | 1.157021 | -0.096110 | 1064202.767969 |
| buy_hold_PTT.BK | 2 | 0.040690 | 0.111308 | 0.672495 | -0.081828 | 1040690.202797 |
| equal_weight | 2 | 0.021133 | 0.066667 | 0.633491 | -0.068208 | 1021133.458705 |
| ppo | 2 | 0.004260 | 0.011669 | 0.057644 | -0.026071 | 1004259.896894 |

Latest verified real-OHLCV ablation pilot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/ablation_pilot_2871
```

SLURM array job `2871` completed three short 1000-timestep PPO ablation tasks:

- `config/ablation_returns_real_ohlcv.yaml`
- `config/ablation_technical_real_ohlcv.yaml`
- `config/ablation_technical_context_real_ohlcv.yaml`

All three error logs were empty:

- `logs/ablation_2871_0.err`
- `logs/ablation_2871_1.err`
- `logs/ablation_2871_2.err`

Aggregated outputs:

- `results/ablation_pilot_2871/aggregate/ablation_metrics.csv`
- `results/ablation_pilot_2871/aggregate/ablation_best_policy_summary.csv`
- `results/ablation_pilot_2871/aggregate/ablation_ppo_summary.csv`
- `results/ablation_pilot_2871/aggregate/ablation_ppo_sharpe.png`

PPO ablation summary:

| feature_group | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| ablation_returns_real_ohlcv | -0.010482 | -0.016061 | -0.453371 | -0.040127 | 989517.774541 |
| ablation_technical_context_real_ohlcv | -0.055886 | -0.084574 | -0.823513 | -0.099160 | 944114.266259 |
| ablation_technical_real_ohlcv | -0.078913 | -0.118657 | -1.611158 | -0.109727 | 921086.783720 |

The best policy in each pilot ablation comparison was still `buy_hold_PTT.BK` with validation Sharpe `0.405480`, so these short ablation runs are implementation checks rather than evidence that PPO is competitive yet.

Latest verified real-OHLCV regime/stress test:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/regime_tests_real_ohlcv_20260531
```

The stress-test runner evaluated baselines plus the saved PPO/A2C models from `results/real_ohlcv_compare_2855` across named slices and two auto-selected 63-trading-date high-volatility windows. It also records that the acute 2020 COVID crash cannot be evaluated from the current starter data because the OHLCV file begins on 2021-01-04.

Outputs:

- `results/regime_tests_real_ohlcv_20260531/regime_slices.csv`
- `results/regime_tests_real_ohlcv_20260531/regime_metrics.csv`
- `results/regime_tests_real_ohlcv_20260531/regime_equity_curves.csv`
- `results/regime_tests_real_ohlcv_20260531/regime_summary.md`

Best policy by evaluated regime:

| regime | policy | cumulative_return | annualized_return | sharpe | max_drawdown |
| --- | --- | ---: | ---: | ---: | ---: |
| covid_recovery_2021 | buy_hold_ADVANC.BK | 0.377423 | 0.465851 | 2.375753 | -0.061453 |
| high_volatility_1_2021-01-28_2021-05-06 | buy_hold_ADVANC.BK | -0.016542 | -0.119601 | -0.735410 | -0.055866 |
| high_volatility_2_2021-06-01_2021-09-01 | a2c | 0.093690 | 0.981571 | 7.220885 | -0.027634 |
| inflation_hike_drawdown_2022 | buy_hold_AOT.BK | 0.160483 | 0.194535 | 1.366575 | -0.047458 |
| test_period | buy_hold_ADVANC.BK | 0.483925 | 0.833957 | 3.681822 | -0.074324 |
| validation_period | buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 |

Latest verified SET market-context feature pilot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_market_pilot_20260531
```

The market-context pipeline uses Yahoo Finance symbol `^SET.BK` to collect the SET index, then merges SET-derived features into the five-stock OHLCV table. The obvious Yahoo Finance variants probed for SET50/SET100 did not return usable rows in the HPC yfinance check, so sector/SET50/SET100 source work remains open.

Data artifacts:

- `data/raw/prices_market_indices.csv`: 965 SET index rows, 2021-01-04 to 2024-12-30, no missing values.
- `data/processed/features_real_ohlcv_market.parquet`: 4,840 stock rows, 26 columns.
- `reports/data_quality_market/real_ohlcv_data_quality.md`: complete five-ticker panel coverage and zero missing values in the active market-context table.

Added market-context columns:

- `set_return_1d`
- `set_return_5d`
- `set_return_20d`
- `set_volatility_20d`
- `set_ma_ratio_20`

Short 1000-timestep validation comparison using `config/real_ohlcv_market.yaml`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 | 1043523.041075 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 | 1009998.557967 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 | 969426.916197 |
| ppo | -0.054013 | -0.081783 | -0.952524 | -0.104711 | 945986.972584 |

Latest verified sector-context feature pilot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_sector_pilot_20260531
```

The sector-context pipeline uses `data/reference/sector_mapping_thai_pilot.csv`, a static five-ticker mapping from SET public listed-company and constituent classifications. It is a pilot fallback because the Yahoo SET50/SET100 and sector-index candidates probed from HPC did not return reliable history.

Sector data artifacts:

- `data/reference/sector_mapping_thai_pilot.csv`: ADVANC.BK, AOT.BK, CPALL.BK, KBANK.BK, and PTT.BK sector labels.
- `data/processed/features_real_ohlcv_sector.parquet`: 4,840 stock rows, 38 columns.
- `reports/data_quality_sector/real_ohlcv_data_quality.md`: complete five-ticker panel coverage and zero missing values.

Latest reproducible sector-index source probe:

```text
python -m src.data.probe_sector_indices --config config/real_ohlcv_sector.yaml
```

The probe wrote `data/raw/sector_index_yahoo_probe.csv` on HPC and tested 139 Yahoo Finance sector-code candidates. It found one usable sector index, banking symbol `^BANK`, with 1,004 rows from 2021-01-04 to 2024-12-30. The other 138 candidates failed, so `data/raw/prices_sector_indices_yahoo.csv` is not broad enough to replace the static pilot mapping. A tracked summary is in `docs/sector_index_probe.md`.

Short 1000-timestep validation comparison using `config/real_ohlcv_sector.yaml`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 | 1043523.041075 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 | 1009998.557967 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 | 969426.916197 |
| ppo | -0.031097 | -0.047383 | -0.626349 | -0.068484 | 968902.974943 |

The sector-context PPO pilot improves over the SET-market-context PPO pilot on cumulative return, Sharpe, and max drawdown, but it still does not beat the PTT buy-and-hold baseline.

Latest verified macro-proxy feature pilot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_macro_pilot_20260531
```

The macro-proxy pipeline uses Yahoo Finance daily series for USD/THB, Brent crude, WTI crude, gold, and the U.S. 10-year yield. These are liquid market proxies, not official macroeconomic release-date series. The merge uses forward fill from each macro proxy's latest available observation to each Thai stock trading date, avoiding use of future observations.

Macro proxy artifacts:

- `data/raw/prices_macro_yahoo.csv`: 5,059 rows across `usdthb`, `brent`, `wti`, `gold`, and `us10y`.
- `data/processed/features_real_ohlcv_macro.parquet`: 4,840 stock rows, 68 columns.
- `reports/data_quality_macro/real_ohlcv_data_quality.md`: complete five-ticker panel coverage and zero missing values.

Short 1000-timestep validation comparison using `config/real_ohlcv_macro.yaml`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 | 1043523.041075 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 | 1009998.557967 |
| ppo | -0.022360 | -0.034152 | -0.344318 | -0.085930 | 977639.781729 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 | 969426.916197 |

The macro-proxy PPO pilot improves over the sector-context PPO pilot and beats equal weight on this short validation run, but it still does not beat the PTT or ADVANC buy-and-hold baselines.

Latest verified fundamentals feature pilot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/real_ohlcv_fundamentals_pilot_20260531
```

The fundamentals pipeline uses Yahoo Finance annual and quarterly financial statements for the five starter tickers. Quarterly statements are preferred when available for a duplicate ticker/period/metric, and annual statements provide older fallback coverage. Features are merged only after a 60-day reporting lag.

Fundamentals artifacts:

- `data/raw/fundamentals_yahoo_quarterly.csv`: 8,701 statement rows across 5 tickers and 13 period ends.
- `data/processed/features_real_ohlcv_fundamentals.parquet`: 4,840 stock rows, 82 columns.
- `reports/data_quality_fundamentals/real_ohlcv_data_quality.md`: complete five-ticker panel coverage; numeric fundamental features are zero-filled before the first lagged report, while `period_end`/`effective_date` are missing for 27.7% of rows before coverage begins.

Short 1000-timestep validation comparison using `config/real_ohlcv_fundamentals.yaml`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_PTT.BK | 0.043523 | 0.067653 | 0.405480 | -0.103448 | 1043523.041075 |
| buy_hold_ADVANC.BK | 0.009999 | 0.015405 | 0.097452 | -0.098919 | 1009998.557967 |
| equal_weight | -0.030573 | -0.046591 | -0.416554 | -0.103240 | 969426.916197 |
| ppo | -0.025996 | -0.039666 | -0.416967 | -0.067777 | 974003.500341 |

The fundamentals PPO pilot has lower drawdown than equal weight and similar Sharpe, but it still does not beat the PTT or ADVANC buy-and-hold baselines. It also underperforms the macro-proxy-only pilot, so fundamentals need a wider licensed history before strong conclusions.

Latest real-OHLCV data-quality report:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/reports/data_quality/real_ohlcv_data_quality.md
```

The report shows 968 unique trading dates per ticker, 0 missing panel dates for each of the five `.BK` tickers, and 0 missing values across the active feature columns.

Final pilot archives:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260530_final_project_pilot.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_set_context.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_sector_context.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_macro_context.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_fundamentals.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_sentiment_scaffold.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_official_macro_scaffold.tar.gz
```

```text
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_sector_index_probe.tar.gz
```

```text
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_official_macro_csv_import.tar.gz
```

Latest archive size: 236 MB. It contains the final writeups, configs, source code, tests, real OHLCV data, SET index data, sector-index Yahoo probe output, usable banking sector-index probe data, pilot sector mapping, macro proxy data, BOT official macro source probe data, the official macro historical CSV importer, Yahoo statement fundamentals, Yahoo latest-news probe data, market-context, sector-context, macro-proxy, fundamentals, sentiment-scaffold, and official-macro-scaffold features, data-quality reports, comparison summaries, Optuna summary, walk-forward aggregates, ablation aggregates, regime/stress-test outputs, selected figures, and relevant run logs.

Latest HPC file-separation snapshot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/_file_separation/20260531_160247
```

`uploaded_current/` contains the files uploaded in the latest pass, while `preexisting_reference/generated_file_index.txt` records generated and previously present data/result artifacts.

Latest verified split-aware algorithm comparison run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/algo_compare_2853
```

Job `2853` completed on BistKA with an empty error log and produced:

- `ppo_model.zip`, `ppo_equity_curve.csv`, `ppo_actions.csv`, `ppo_metrics.csv`
- `a2c_model.zip`, `a2c_equity_curve.csv`, `a2c_actions.csv`, `a2c_metrics.csv`
- checkpoint files under `checkpoints/ppo/` and `checkpoints/a2c/` at 2k-step intervals
- `baselines/baseline_equity_curves.csv`, `baselines/baselines_metrics.csv`
- `comparison_metrics.csv`, `comparison_summary.md`, `action_summary.csv`, `ppo_action_summary.csv`
- `split_summary.csv`
- figures under `figures/`: `equity_curves.png`, `drawdowns.png`, `turnover.png`, `action_mean_weights.png`

Chronological split used by `algo_compare_2853`:

| split | start_date | end_date | rows | unique_dates | tickers |
| --- | --- | --- | ---: | ---: | ---: |
| train | 2021-01-01 | 2023-05-25 | 3125 | 625 | 5 |
| validation | 2023-05-26 | 2024-03-13 | 1045 | 209 | 5 |
| test | 2024-03-14 | 2024-12-31 | 1045 | 209 | 5 |

Main validation comparison from `algo_compare_2853`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold_KBANK.BK | 0.580296 | 0.904531 | 3.218974 | -0.143165 | 1580295.743752 |
| ma_crossover | 0.257937 | 0.381344 | 1.880703 | -0.091600 | 1257936.712797 |
| equal_weight | 0.152812 | 0.221644 | 1.397638 | -0.095300 | 1152811.724538 |
| a2c | 0.167278 | 0.243282 | 1.372647 | -0.103622 | 1167278.283851 |
| mean_variance | 0.122426 | 0.176558 | 0.860456 | -0.109790 | 1122426.235382 |
| buy_hold_PTT.BK | 0.078909 | 0.112851 | 0.524486 | -0.151637 | 1078909.487165 |
| random | 0.053055 | 0.075491 | 0.457392 | -0.110405 | 1053054.552875 |
| buy_hold_ADVANC.BK | 0.090772 | 0.130115 | 0.428325 | -0.159049 | 1090772.121175 |
| buy_hold_CPALL.BK | 0.015603 | 0.022035 | 0.084316 | -0.224856 | 1015602.576445 |
| buy_hold_AOT.BK | 0.000795 | 0.001119 | 0.005067 | -0.182746 | 1000794.673700 |
| ppo | -0.012888 | -0.018096 | -0.108208 | -0.100581 | 987112.489101 |
| momentum | -0.131968 | -0.180650 | -0.900621 | -0.152156 | 868032.024449 |

Previous full-period synthetic comparison:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/algo_compare_2852
```

Previous verified PPO-only run:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/results/run_2850
```

Run `2850` used the 5-ticker synthetic technical config:

- `ADVANC.BK`
- `AOT.BK`
- `CPALL.BK`
- `KBANK.BK`
- `PTT.BK`

Feature period:

```text
2021-01-01 to 2024-12-31
```

Main comparison from `run_2850`:

| policy | cumulative_return | annualized_return | sharpe | max_drawdown | final_portfolio_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| equal_weight | 1.359177 | 0.238027 | 1.438354 | -0.145711 | 2359177.198539 |
| ppo | 0.529157 | 0.111438 | 0.577816 | -0.200052 | 1529156.907643 |
| random | 0.277048 | 0.062725 | 0.358815 | -0.215310 | 1277047.942745 |
| momentum | -0.015674 | -0.003922 | -0.018431 | -0.336980 | 984326.057598 |

## Current Blockers

- `[!]` The live V2 five-algorithm screening stage is `READY` on the verified storage-safe release. A fresh 26 July execution window passed queue, credit, write, runtime, and release checks but left only 0.308 GiB of modeled personal-group soft-quota margin, below the required 1 GiB buffer. The full 150-trial screening remains unsubmitted.
- `[!]` New project outputs inherit personal Unix group `s00658`; its soft-quota grace caused job `3542` to fail even though the project-ID quota remained healthy. Project-group ownership/SGID support or a personal-group quota increase remains the durable fix.
- `[!]` The planned feature-ablation scope requires readiness-certified observed macro and fundamental tables, or a formal preregistration amendment before test access.
- `[!]` Full SET50/SET100 conclusions are still blocked by licensed historical OHLCV and trading status for the 138-ticker acquisition union, a final investability certification and locked V3 protocol, an untouched post-2024 holdout, historical news/sentiment, licensed fundamentals, and point-in-time macro vintages. Official point-in-time constituent/sector history and the isolated V3 membership-mask mechanics are complete for 2021–2024; the revised BOT macro history remains structural/exploratory only.

## Immediate Next Actions

1. Obtain at least 743,140,039 bytes (0.692 GiB) of additional personal-group soft headroom through safe reproducible-cache cleanup, a quota increase, or project-group/SGID ownership support; do not delete accepted research evidence.
2. Open another screening window and immediately recapture quota, credit, partitions, queue, deployed-release identity, and the checksum-bound direct-child write probe.
3. Run `scripts.prepare_screening_execution_window`; require a sealed, checksum-bound `authorized: true` receipt with at least 1 GiB of group soft-quota margin and at least 120 seconds of remaining TTL.
4. Only then render and inspect the single five-element, `%2`-concurrency full-screening array through the storage-safe campaign spec; submit it only with explicit operator intent, and do not pre-submit aggregation.
5. Fetch and verify all 150 validation-only manifests before ranking algorithms or rendering the next stage.
6. Acquire licensed or explicitly authorized OHLCV for all 138 tickers in `data/reference/set50_or_set100_universe.csv`; do not use that acquisition union as a static membership mask.
7. Complete the isolated V3 capsule by adding licensed trading-status/corporate-action rules, author-approving and checksum-locking the protocol, and certifying rebuilt feature-table hashes; the membership panel and mask-aware environment mechanics are already implemented and verified locally.
8. Export or license historical sentiment/news, fundamentals, and point-in-time macro vintages listed in `docs/data_sources.md`; rerun source and intake validation before feature ablation.
