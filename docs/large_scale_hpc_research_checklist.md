# Large-Scale HPC Research Checklist

Project: **Deep Reinforcement Learning for Thai Stock Portfolio Allocation**

Purpose: use abundant HPC resources to increase independent scientific evidence across time, random seeds, algorithms, features, and execution conditions. Do not use extra compute only to repeat the same historical trajectory for more timesteps.

**26 July 2026 execution-window update:** fresh VPN-authenticated BistKA evidence verified an empty queue, healthy credit, exact QOS, `compute-long`, a successful direct-child write probe, and the exact deployed storage-safe release. The new fail-closed gate nevertheless returned `HOLD`: 6.339 GiB of personal-group soft headroom minus the recomputed 6.031 GiB requirement left only 0.308 GiB, below the 1 GiB policy buffer. Zero jobs were submitted, screening remains 0/150, and canonical progress remains 11/37 = 29.7%.

**20 August 2026 V3 evaluation-readiness update:** a draft, checksum-bound execution design now fixes six provisional expanding half-year folds, 30 configurations, and three seeds, yielding 540 planned fits and 63 million planned timesteps. This future work is capped at three array-job submissions with at most two concurrent jobs. Zero jobs were submitted, no performance evidence was generated, training/`test`/`all` remain blocked, and canonical progress remains 11/37 = 29.7%.

Accepted V1 historical boundary: **28 dependency-gated stages**, **498 planned model runs**, **29.44 million requested timesteps**, and **10 VERIFIED, 1 READY, 17 PENDING**. Five jobs were submitted (`3525`, `3526`, `3528`, `3529`, and accepted recovery job `3530`), and that historical scheduler total remains unchanged. V2 retains **503 planned model runs**, **29,450,240 requested timesteps**, and **23 bounded SLURM array elements**. For longitudinal reporting, canonical whole-project completion remains **11/37 = 29.7%**. The active cutover graph is **9 VERIFIED, 1 READY (`screening`), 25 PENDING** across 35 stages; its 9/35 native fraction is structural because `calibration_lineage` consolidates three former gates into one, not because completed work was lost. The storage-safe release is now applied and remotely preflighted, and job `3550` verified five algorithms with five retention receipts and zero checkpoint paths. V2 has five submissions to date (`3540`, `3541`, `3542`, `3543`, and `3550`), with only `3550` submitted in the final cutover window. The July-17 post-smoke capacity pass is historical and expired with 0.309 GiB margin; the authoritative July-26 execution window separately reverified live readiness but held at 0.308 GiB margin under the stronger 1 GiB policy. The full 150-trial screening remains unsubmitted. Checked execution items below mean their returned evidence passed the stated local or remote gate; rejected or incident evidence is preserved but does not count as accepted results.

## Status Legend

- `[x]` Implemented or already verified
- `[~]` Partially implemented
- `[ ]` Not started
- `[!]` Blocked or requires a decision/data source

## Non-Negotiable Research Rules

- [x] Freeze the research question before opening final test results.
- [x] Declare primary baselines before testing: equal-weight portfolio and SET index buy-and-hold.
- [x] Use validation periods for algorithm selection and hyperparameter tuning only.
- [x] Use chronological test periods only for final research conclusions.
- [x] Never tune a model, feature, cost, or reward after viewing its final test result.
- [x] Keep every failed, invalidated, and completed run in the experiment manifest.
- [x] Lock the rule to report negative findings when DRL does not reliably outperform the baselines.
- [x] Do not claim whole-SET generalization from the current five-stock universe.
- [x] Do not claim sentiment or official-macro effects until overlapping historical source data exists.
- [x] Record that the frozen Yahoo snapshot is private-research-only and not eligible for raw redistribution or publication-grade claims.

## V3 Evaluation Readiness — 2026-08-20

- [x] Freeze six provisional expanding half-year validation folds from 2022 H1 through 2024 H2, with non-overlapping validation periods, context-only feature history, train-only preprocessing, and unused train-to-validation boundary transitions.
- [x] Freeze exactly 30 configurations across PPO, A2C, DDPG, TD3, and SAC, then pair each fold/configuration with the same three seeds for 540 planned fits and 63,000,000 planned timesteps.
- [x] Predeclare validation-only selection against the active-member equal-weight baseline, deterministic tie-breakers, and a negative-result rule that selects no DRL model when the best median excess result is non-positive.
- [x] Preserve the policy, schedule, candidate ledger, readiness receipt, schedule-only chart, and technical report under [`config/v3/evaluation_policy_draft.yaml`](../config/v3/evaluation_policy_draft.yaml), [`reports/data_quality_v3/`](../reports/data_quality_v3/), [`reports/figures/v3_evaluation_fold_schedule_20260820.png`](../reports/figures/v3_evaluation_fold_schedule_20260820.png), and [`reports/v3_evaluation_policy_readiness_20260820.md`](../reports/v3_evaluation_policy_readiness_20260820.md).
- [x] Limit eventual execution to at most three array-job submissions and at most two concurrent jobs; submit zero jobs for this readiness milestone.
- [!] Do not execute or adaptively expand the 30-configuration grid. Training, `test`, and `all` remain blocked until authoritative licensed market/calendar data pass, the evaluation and investability drafts receive author approval and are locked into a release, and a genuinely untouched post-2024 final holdout exists.
- [x] Keep whole-project progress at **11/37 = 29.7%** because this milestone contains no fitted models, performance evidence, or completed scientific-compute stage.

## Phase 0: Current Foundation

- [x] Five-stock real Thai OHLCV pilot pipeline is implemented.
- [x] Chronological train/validation/test splits are implemented.
- [x] Walk-forward window generation is implemented.
- [x] PPO and A2C training paths are implemented.
- [x] Equal-weight and per-stock buy-and-hold baselines are implemented.
- [x] SET index benchmark is implemented with aligned dates and initial transaction cost.
- [x] Transaction costs, turnover, drawdown, and continuous portfolio weights are implemented.
- [x] Repeated-seed robustness task runner is implemented.
- [x] Test-only robustness aggregation is implemented.
- [x] Bootstrap confidence intervals and paired baseline comparisons are implemented.
- [x] Seed-window heatmap and confidence-interval figure generation are implemented.
- [ ] Run the current 20-run robustness experiment as the first reference experiment.
- [ ] Archive the first reference experiment before expanding the design.

## Phase 1: Confirm Actual BistKA Capacity

- [x] Run `mycredit` and record the available project credit.
- [x] Run `myquota` and record home/project storage limits and current usage.
- [x] Run `sinfo` and record available CPU and GPU partitions.
- [x] Run `squeue -u "$USER"` and record the current queue state.
- [x] Run `ml av Python`, `ml av Miniforge3`, and `ml spider cuda` to confirm live module names.
- [x] Confirm the active SLURM account from the live `mycredit` association.
- [x] Confirm maximum submitted jobs, maximum running jobs, CPU-core limit, GPU limit, and walltime limit.
- [x] Save the capacity snapshot in the experiment log with the date.
- [x] Preserve the 2026-07-16 V2 deployment snapshot in [`reports/bistka_capacity_v2_20260716.txt`](../reports/bistka_capacity_v2_20260716.txt): 36.31 GiB/300 GiB project storage, 9,984.12 project SHr and 496.25 user SHr remaining, idle partitions, and an empty queue.
- [x] Diagnose the later storage failure as the personal Unix group `s00658` exceeding its soft quota and grace period; the project-ID quota remained healthy.
- [x] Recover writable capacity without deleting research evidence by cleaning the reproducible package cache and relocating all historical `final_archive/*.tar.gz` files to project-owned storage with checksum verification and path-preserving symlinks.
- [!] Treat the recovered approximately 6.45 GiB personal-group soft-quota headroom as insufficient for the approximately 7.69 GiB projected full-screening output before logs and safety margin.
- [x] Do not increase SLURM concurrency until the live limits are verified.

## Phase 1B: Lock and Deploy the Exact Campaign

- [x] Add a machine-readable, checksum-locked scientific preregistration.
- [x] Lock the exact question, five-ticker universe, observed period, baselines, primary paired outcomes, split firewall, multiplicity rule, negative-result rule, and claim exclusions.
- [x] Bind the preregistration to the current observed-market readiness receipt and SHA-256 sidecar.
- [x] Add a deterministic, content-addressed initial BistKA deployment bundle with a complete file inventory and SHA-256 sidecars.
- [x] Bind the initial release to the observed data, readiness evidence, preregistration, code, configuration, SLURM scripts, dependency declarations, remote project root, Python executable, and required packages.
- [x] Make deployment staging a dry run by default and require an explicit `--apply` for remote writes.
- [x] Back up every manifest-listed remote file that would be replaced before extracting an applied bundle.
- [x] Add a BistKA-only preflight that verifies hostname, project root, exact release/file hashes, observed-data eligibility, Python runtime, and required packages.
- [x] Require the passing initial deployment preflight before either calibration stage can become ready.
- [x] Add a content-addressed calibrated-screening overlay containing the measured locked screening configuration and decision evidence.
- [x] Require a second exact-release BistKA preflight for the calibrated-screening overlay before screening can become ready.
- [x] Preserve the accepted V1 149-file initial bundle and [`preflight receipt`](../reports/bistka_calibration_preflight.json) as historical job-3530 evidence.
- [x] Inspect the V1 staging dry run and initial release inventory without changing the remote project.
- [x] Apply the corrected V1 149-file initial release to BistKA with explicit operator intent.
- [x] Run the V1 initial exact-release preflight on BistKA and verify its returned [`receipt`](../reports/bistka_calibration_preflight.json) locally.
- [x] Bind the runtime configuration to the exact corrected `data/processed/features_real_ohlcv_market.csv` rather than the rejected stale Parquet shadow.
- [x] Build, inspect, apply, and remotely preflight the calibrated-screening overlay; verify its returned [`receipt`](../reports/bistka_screening_preflight.json) locally.
- [x] Preserve rejected diagnostic/stale-data result artifacts under [`reports/rejected_calibration/20260715_quota_and_data_shadow/`](../reports/rejected_calibration/20260715_quota_and_data_shadow/) and exclude them from accepted-result globs.
- [x] Build, apply, and verify the current V2 base release `4b7d1d4b887bf08675bc664683b06f0fb103369016f7233e6caeb594e41ee382` through the exact [`BistKA preflight`](../reports/bistka_calibration_v2_preflight.json).
- [x] Create and preserve the checksum-bound [`V1-to-V2 calibration migration receipt`](../campaigns/large_scale_thai_drl_v2/imports/archive/v1_calibration_evidence_before_dependency_lock_20260716.json), SHA-256 `405ce3758b763fd94e82ac3a45f6d20d3fac803ce16d786c370b4b3a364a299c`, as archived compatibility evidence from accepted job `3530`.
- [x] Permit compatibility normalization only for `project.name`; verify the normalized V1 and V2 calibration configurations both hash to `7a413f1f023b8ca942092f516bbc6e2463336b02cc39b89c55e67708bfba92e7`.
- [x] Materialize the V2 locked screening configuration with SHA-256 `d7b6ab71ede33ae4c4106fded3078b9a4eb3c56af345e67960061feea1897c1e`.
- [x] Run fresh V2-native vector calibration in job `3540` and fresh independent-process determinism in job `3541`, with strict attempt/release/preflight/data/output binding.
- [x] Build and apply cumulative screening release `ef89ff5a738c5b25460ae40ad7789239134d761ce511a36665f857e01e3d74b4`, manifest SHA-256 `d9a1fd96af46cfe022d12b38d2de97642296a4a1f66928bbbbf3669af4dcfd86`.
- [x] Verify the exact [`screening preflight`](../reports/bistka_screening_v2_preflight.json), SHA-256 `99ff875694ff47b6a02541718b6af93699de1f789ef35effaefe24d2e9655951`.
- [x] Close the pre-cutover V2 pass at exactly four jobs—`3540`, `3541`, failed smoke attempt `3542`, and successful smoke retry `3543`—before opening the separate one-job final cutover window.
- [x] Preserve the pre-cutover control plane and release in rollback archive SHA-256 `5dd485c41ee5e287c893081f760f1f496fb458e1e1097e691fb64e254e822aee` and accept calibration lineage without recomputation through receipt SHA-256 `25cda3f51aeb200353064e495fd58484ade711f36ca575c9633ae231e069c807`.
- [x] Build and apply final storage-safe release `56358f2a4996f654426814d88e622b7f217ac89bfc02f8bbdacc27e66873b55e`, then verify the exact [`cutover preflight`](../reports/bistka_screening_storage_safe_cutover_v2_preflight.json), SHA-256 `1ccac9827e2f018927d711c0fb0918119c79f8721bd6d82d3d08ed026ab59f0c`.
- [x] Submit only storage-safe smoke job `3550` in the final cutover window; no full-screening array was submitted.

## Phase 2: Improve Data Before Spending More Compute

- [ ] Attempt to extend the same five reliable Thai tickers to at least 2010–2025.
- [ ] Verify adjusted prices, corporate actions, missing dates, and ticker continuity.
- [ ] Add earlier SET index history covering the same dates.
- [ ] Generate a data-quality report for the extended period.
- [ ] Confirm that the extended data passes duplicate, missing-panel, and leakage checks.
- [ ] Increase walk-forward windows from 2 to at least 6 non-overlapping test windows.
- [ ] Include bull, bear, sideways, crisis, and high-volatility periods where supported by data.
- [x] Obtain and checksum-bind point-in-time SET50/SET100 membership for 2021–2024.
- [x] Implement the isolated 138-slot V3 membership panel, past-only development eligibility, mask-aware environment mechanics, and fail-closed draft protocol gate locally without submitting an HPC job.
- [x] Implement the isolated active-member/date-aware V3 sector context and the five-file licensed-market package gate; preserve the failed current-data receipt at `reports/data_quality_v3/v3_market_package_gap_20260731.json`.
- [x] Implement the isolated V3 evaluation-policy draft, six-fold schedule, fixed 30-configuration ledger, and bounded 540-fit/63-million-timestep plan locally without submitting an HPC job.
- [!] Keep broad-universe training blocked until a licensed market package passes with authoritative sessions/status/actions/symbol/license bindings, final investability and evaluation rules receive author approval and are locked, the V3 protocol/release is approved and locked, and an untouched post-2024 holdout exists.
- [!] Obtain historical Thai news/sentiment before running real sentiment experiments.
- [!] Obtain release-dated official macro history before running official macro experiments.
- [~] Obtain broader sector and licensed fundamental history before full multi-source claims. Dated constituent-level sector history is complete; licensed fundamentals remain missing.
- [x] Record the claim boundary if any data blockers remain.
- [x] Freeze and hash the 2021–2024 five-ticker observed pilot, SET-index series, processed feature panel, and source manifest.
- [x] Add a fail-closed data-readiness receipt covering schema, ticker contract, temporal coverage, balanced panel, OHLC validity, SET overlap, configured features, and leakage.
- [x] Reject generated-origin or drifted data from the real-market campaign gate.
- [x] Record the three missing same-day SET observations and constant macro/sentiment scaffolds as explicit warnings.
- [x] Disclose the 10 deterministically normalized OHLC rows, including per-ticker repair counts.

## Phase 3: Make Training Use the HPC Correctly

- [x] Replace the single training environment with vectorized environments.
- [x] Ensure vectorized environments use different training slices or bootstrapped trajectories.
- [x] Do not launch identical environment copies that generate highly correlated experience.
- [x] Wire `num_envs` to `DummyVecEnv` or `SubprocVecEnv` instead of leaving it as an unused configuration value.
- [x] Add a test confirming distinct complete training slices are created for the requested environment count.
- [ ] Complete a real four-environment PPO smoke run with the subprocess backend and clean shutdown on the exact deployed release.
- [x] Implement an observed-only vector benchmark for 1, 2, 4, 8, and 16 environments with three repeats per environment count.
- [x] Implement fresh-Python-subprocess isolation and sampled concurrent process-tree RSS for every benchmark trial.
- [x] Implement the selection rule: choose the smallest environment count whose median throughput reaches at least 95% of the maximum median throughput.
- [x] Benchmark 1, 2, 4, 8, and 16 environments for samples/second and memory usage in accepted job `3530`.
- [x] Record median throughput of 347.9200, 265.9007, 217.0773, 143.4171, and 83.6104 samples/second for 1, 2, 4, 8, and 16 environments.
- [x] Record maximum process-tree RSS of 602.484, 1,641.066, 2,661.281, 4,690.129, and 8,768.617 MiB for 1, 2, 4, 8, and 16 environments.
- [x] Select `num_envs: 1` from measured throughput rather than maximum allocation; spend additional HPC capacity on independent trials, seeds, windows, and algorithms.
- [x] Implement an observed-data fixed-seed determinism runner using two independent Python processes.
- [x] Verify deterministic environment reproduction on BistKA across two independent processes and 549 steps with matching trajectory and provenance hashes.
- [x] Make synthetic calibration data available only through an explicit unit-test mode that cannot satisfy observed campaign gates.
- [x] Bind each calibration receipt to the latest rendered attempt, exact deployment release, preflight receipt, BistKA hostname, SLURM job ID, code/config/data inputs, returned outputs, runtime versions, and elapsed time.
- [x] Reject stale attempts, changed inputs, mismatched output hashes, incomplete repeats, non-BistKA receipts, and incorrect job-numbered filenames.
- [x] Implement measured `num_envs` materialization into a locked screening configuration and checksum-backed decision receipt.
- [x] Run the repeated observed vector benchmark on BistKA and verify accepted job `3530` against its attempt-bound [`result`](../reports/vectorized_training_benchmark_3530.json), [`trial CSV`](../reports/vectorized_training_benchmark_3530.csv), and [`receipt`](../reports/vectorized_training_benchmark_3530.csv.receipt.json).
- [x] Run the independent-process determinism check within accepted job `3530` and verify its attempt-bound [`result`](../reports/determinism_3530.json) and [`receipt`](../reports/determinism_3530.json.receipt.json).
- [x] Materialize measured `num_envs: 1` into the checksum-locked [`screening configuration`](../campaigns/large_scale_thai_drl/derived/locked_large_scale_screening.yaml).
- [x] Import the accepted V1 calibration into V2 without rewriting its source provenance or submitting a rerun, then materialize the V2 [`screening configuration`](../campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml), SHA-256 `d7b6ab71ede33ae4c4106fded3078b9a4eb3c56af345e67960061feea1897c1e`.
- [x] Replace migration-only readiness with fresh V2-native evidence: verify vector benchmark job `3540` and determinism job `3541` against base release `4b7d1d4b887bf08675bc664683b06f0fb103369016f7233e6caeb594e41ee382`.
- [x] Include Python, PyTorch, Stable-Baselines3, Gymnasium, NumPy, and runtime-version fields in per-run manifests.
- [x] Route PPO/A2C/SAC/TD3/DDPG screening to CPU partitions.
- [x] Reserve the existing GPU path for WangchanBERTa training or larger neural policies.
- [x] Add checkpoint retention rules so thousands of runs do not exhaust storage.
- [x] Implement a screening-specific no-write callback that retains final models, metrics, manifests, actions, equity curves, metadata, and TensorBoard evidence while suppressing redundant checkpoint copies before they can trigger quota failure.
- [x] Require an atomic checksum-bound retention receipt, verify the final-model hash and callback activity, and reject a successful trial if any checkpoint was written or remains.

## Phase 4: Large-Scale Algorithm Screening

- [x] Define the screening algorithms: PPO, A2C, SAC, TD3, and DDPG.
- [x] Implement continuous-action training and algorithm-specific search spaces for every screening algorithm.
- [x] Enforce a shared screening configuration for identical data, features, costs, and evaluation dates.
- [x] Preserve job `3543` as the accepted checkpoint-writing baseline smoke on cumulative release `ef89ff5a738c5b25460ae40ad7789239134d761ce511a36665f857e01e3d74b4`.
- [x] Preserve failed job `3542` separately as a quota incident; do not let its partial checkpoint or manifest enter accepted-result globs.
- [x] Confirm all job-`3543` smoke-based full-screening runtime projections are below the 72-hour task limit: PPO 3.56 h, A2C 0.24 h, DDPG 20.54 h, TD3 21.57 h, and SAC 21.99 h.
- [x] Keep the legacy 150-trial checkpoint-writing path unsubmitted because its approximately 7.69 GiB projected output exceeded the approximately 6.45 GiB measured personal-group soft-quota headroom.
- [x] Capture the exact post-smoke group/project quota and empty queue in [`reports/screening_capacity_post_smoke_20260717.txt`](../reports/screening_capacity_post_smoke_20260717.txt), then issue the checksum-backed [`screening capacity decision`](../reports/screening_capacity_decision_20260717.json) with `eligible_for_submission: false`.
- [x] Recompute storage-safe output from the five verified smoke trials while explicitly budgeting per-trial retention evidence and per-algorithm task summaries: 3.854 GiB retained, 3.840 GiB of checkpoints avoided, and a 49.88% net reduction.
- [x] Capture checksum-bound group/project quotas plus a direct-child 1 MiB `fsync` probe, then pass the candidate at that instant with a 6.031 GiB safety-adjusted requirement and only 0.420 GiB group-soft-quota margin.
- [x] Implement the dedicated storage-safe smoke adapter/job, validate its receipt/checkpoint inventory contract locally, and emit a separate checksum-bound gate receipt that legacy smoke evidence cannot provide.
- [x] Build the deterministic storage-safe release candidate `16ca55f9264a54e3bc86ecf51d2e4b3ab0c4e85ddb80f22e7dd8585a466a3dc2` and complete a local deployment dry run without changing BistKA.
- [x] Cut over the campaign execution contract, preserve rollback evidence, build the deterministic 248-file final release, apply it atomically, and pass exact-release BistKA preflight.
- [x] Capture a fresh pre-smoke capacity pass with 0.417 GiB modeled margin, then run and strictly verify storage-safe five-algorithm job `3550` with five complete manifests, five retention receipts, zero checkpoint paths, and a passing dedicated gate.
- [x] Measure 137,777,145 bytes for safe smoke `3550` versus 275,209,516 bytes for checkpoint-writing smoke `3543`, an observed reduction of approximately 49.94% despite seven additional evidence files.
- [x] Confirm job-`3550` runtime projections remain below 72 hours: PPO 3.7285 h, A2C 0.2556 h, DDPG 20.4114 h, TD3 21.1114 h, and SAC 21.5986 h.
- [x] Implement and test a standalone execution-window gate that binds the canonical smoke, recomputes storage, verifies scheduler/quota/write/preflight freshness, requires empty user and account queues, checks conservative credit, complete live/local release identity and exact SLURM resources, enforces a 1 GiB group-soft margin, and never submits a job.
- [x] Recapture scheduler, credit, QOS, deployed-release, quota, write-probe, and a fresh 248-file remote preflight on 26 July; verify all non-margin checks and preserve exact sidecars.
- [x] Pass 51 focused gate cases, 23 one-time render-consumer cases, and 124 authorization/quota/storage cases together; verify that an expiring/advancing window, weakened/NaN/Boolean policy, mixed or renamed capture evidence, malformed QOS/partition evidence, drifted inventory, forged receipt/capability, symlink escape across command/attempt/status/event output directories or leaves, concurrent/distinct receipt replay, preexisting attempt, overwrite, or unsafe output path cannot authorize normal rendering.
- [x] Implement the supported consume-once renderer, which revalidates the entire gate on its own wall clock, binds receipt and evidence stamps, claims an evidence identity globally, acquires a stage-wide lock, verifies one immutable attempt plus command exports, records success or in-doubt failure, and executes no job.
- [!] Do not invoke the manifest-locked controller or generated screening command directly. Embedding mandatory receipt/TTL enforcement at the eventual submission boundary remains a next-release cutover item because editing it in place would invalidate the active 248-file release.
- [!] Keep full screening unsubmitted after the authoritative 26 July gate measured only 330,601,785 bytes (0.308 GiB) of margin, 743,140,039 bytes (0.692 GiB) short of the authorization policy.
- [x] Reject trials with non-finite metrics, impossible drawdowns, invalid weights, over-allocation, or broken nonnegative accounting.
- [ ] Run 30–100 validation-only hyperparameter trials per valid algorithm.
- [x] Search learning rate, discount factor, rollout/batch size, and entropy-related parameters where applicable.
- [x] Search algorithm-specific parameters such as PPO clip range, SAC entropy, and TD3 policy delay.
- [x] Store every sampled configuration and its validation objective.
- [x] Prevent test metrics from ranking screening trials; screening uses the validation split.
- [x] Implement validation-only algorithm ranking and cap selection at the top two algorithms.
- [ ] Produce the actual validation-only ranking from the complete screening campaign.

## Phase 5: Nested Walk-Forward Selection

- [x] Implement validation-only nested screening for five algorithms and each configured outer window.
- [x] Add a completeness gate and best-validation-trial selection per algorithm/window.
- [x] Prevent nested-screening jobs from evaluating outer test segments.
- [ ] Treat each outer walk-forward test window as unseen final data.
- [ ] Perform model and hyperparameter selection only inside each window's training/validation history.
- [ ] Lock the selected configuration before evaluating the outer test window.
- [ ] Repeat the selection procedure independently for every outer window.
- [ ] Prevent any later window from influencing an earlier window's configuration.
- [ ] Save the selected configuration and selection evidence for every window.
- [x] Verify through the runner contract that nested screening requests `validation` evaluation only.
- [x] Add split validation that rejects unsupported or empty evaluation split requests.

## Phase 6: Repeated-Seed Confirmation

- [ ] Increase confirmation seeds from 5 to 20–30 for each selected algorithm.
- [ ] Use exactly the same seed list for competing algorithms.
- [ ] Run every selected algorithm across every outer test window.
- [ ] Use a fixed training budget for fair comparison.
- [ ] Compare 20k, 100k, 250k, and 500k timesteps on validation data before fixing the final budget.
- [x] Implement a completeness-gated validation-only 20k/100k/250k/500k budget sweep with a declared plateau rule.
- [ ] Stop increasing timesteps when validation performance plateaus or becomes less stable.
- [ ] Record walltime, CPU/GPU hours, peak memory, and samples/second for each run.
- [ ] Confirm that missing or failed seeds are rerun or explicitly reported.
- [ ] Require complete seed/window coverage before final aggregation.

## Phase 7: Feature Ablation Study

- [!] Keep feature-ablation compute blocked until observed macro and fundamental tables pass readiness, or formally amend the preregistered feature scope before test access.
- [x] Implement a completeness-gated six-group, ten-seed, two-window ablation orchestrator.
- [x] Distribute six feature groups across five bounded BistKA tasks.
- [x] Implement test-only ablation summaries and bootstrap confidence-interval plots.
- [ ] Freeze the selected algorithm and training protocol before ablation testing.
- [ ] Run returns-only features.
- [ ] Run technical-indicator features.
- [ ] Run technical plus SET market context.
- [ ] Run technical plus sector context when valid sector data exists.
- [ ] Run technical plus macro proxies.
- [ ] Run technical plus release-dated official macro data when available.
- [ ] Run technical plus fundamentals when valid point-in-time data exists.
- [ ] Run technical plus historical sentiment when available.
- [ ] Run the complete all-real-feature configuration only after every source passes readiness checks.
- [ ] Use 10–20 seeds per feature group across all outer windows.
- [ ] Compare each feature group against the same technical-only control.
- [ ] Report feature groups that reduce performance or increase instability.

## Phase 8: Thai Sentiment Modeling (Only When Historical Text Exists)

- [ ] Validate licenses and allowed use for every historical text source.
- [ ] Align each headline/disclosure to publication time and ticker.
- [ ] Prevent future-return labels from leaking into DRL observations.
- [ ] Create train/validation/test text splits that follow market-time boundaries.
- [ ] Fine-tune WangchanBERTa using multiple seeds.
- [ ] Compare a frozen embedding model, fine-tuned classifier, and simple deterministic sentiment baseline.
- [ ] Cache embeddings or sentiment probabilities before RL training.
- [ ] Record NLP accuracy/F1 and class balance separately from trading performance.
- [ ] Run sentiment ablation against the identical technical-only DRL configuration.
- [ ] Verify that sentiment improves multiple windows rather than one isolated period.

## Phase 9: Execution and Cost Stress Tests

- [x] Implement test-only execution-scenario orchestration and model-readiness reporting.
- [x] Implement base, moderate, conservative, hostile, and constrained scenario templates.
- [x] Implement automatic fixed-seed model-registry construction from completed robustness evidence without using test performance for model selection.
- [x] Bind every registry entry to the locked campaign protocol, algorithm, walk-forward window, seed, feature schema, data, and artifact hashes.
- [x] Enforce evaluation of each registered model only on its matching outer walk-forward test window.
- [x] Implement exact base-scenario replay against the source robustness equity curve and fail closed on any date or portfolio-value mismatch.
- [ ] Test transaction costs of 0.1%, 0.2%, 0.5%, and 1.0%.
- [ ] Test at least three slippage assumptions.
- [ ] Add delayed execution or next-open execution testing.
- [x] Add maximum daily turnover limits to the environment.
- [x] Add maximum position-weight limits to the environment.
- [x] Add minimum cash requirements to the environment.
- [ ] Add liquidity or volume-participation constraints when volume data permits.
- [ ] Test missing observations and stale contextual features.
- [ ] Test doubled costs as a hostile execution scenario.
- [ ] Require final claims to survive at least one conservative cost/slippage setting.

## Phase 10: Regime and Failure Analysis

- [ ] Define regimes before evaluating final models.
- [ ] Evaluate bull-market windows.
- [ ] Evaluate bear-market windows.
- [ ] Evaluate sideways-market windows.
- [ ] Evaluate high-volatility windows.
- [ ] Evaluate crisis windows supported by the available history.
- [ ] Report the worst seed and worst temporal window for every selected algorithm.
- [ ] Inspect action weights, cash allocation, turnover, and drawdown during failures.
- [ ] Identify hold-only, overtrading, concentration, and unstable-policy failure modes.
- [ ] Include at least three concrete failure cases in the final discussion.

## Phase 11: Statistical Analysis

- [x] Implement mean, median, sample standard deviation, and 95% confidence-interval reporting.
- [x] Implement paired candidate-minus-baseline results for each matching seed/window.
- [x] Implement the proportion of runs beating equal weight.
- [x] Implement the proportion of runs beating the SET index.
- [x] Implement cumulative return, Sharpe, Sortino, maximum drawdown, Calmar, turnover, transaction-cost, and slippage reporting.
- [x] Treat a smaller negative maximum drawdown as an improvement.
- [x] Implement optional window-clustered bootstrap resampling for longer-history experiments.
- [x] Add paired randomization tests with Holm family-wise correction for algorithm and feature comparisons.
- [ ] Consider a deflated or probabilistic Sharpe analysis after large hyperparameter searches.
- [x] Encode the rule not to claim reliable outperformance when the excess-return interval crosses zero.
- [x] Implement separate reporting of paired economic differences and statistical uncertainty intervals.

## Phase 12: Experiment Tracking and Storage

- [x] Add BistKA capacity-snapshot and SLURM-accounting collection scripts.
- [x] Add a strict machine-readable quota capture and assessment path that verifies group/project byte and inode limits, exact GID/project identity, direct-child writeability, `fsync`, sidecars, and evidence freshness.
- [x] Assign a unique experiment-ID contract to every repeated-seed and screening run.
- [x] Include algorithm, window, seed, feature group, cost scenario, and SLURM identifiers in tracked manifests.
- [x] Include the full resolved configuration in every tracked manifest.
- [x] Include the Git commit hash and available data-file hashes in tracked provenance.
- [x] Include start time, end time, hostname, package versions, and SLURM identifiers in tracked provenance.
- [x] Route stdout and stderr to declared paths through the SLURM scripts.
- [x] Mark each tracked run as running, complete, or failed with an error reason.
- [x] Prevent robustness and screening aggregation when required runs are missing.
- [x] Add a cross-stage campaign control plane with a validated dependency DAG and persistent status report.
- [x] Expand the observed-data campaign to 28 dependency-gated stages, including initial deployment/preflight and calibrated-screening deployment/preflight.
- [x] Require exact nonzero manifest counts and unique experiment IDs before compute stages can verify.
- [x] Create immutable per-render attempt records that hash the campaign specification, job script, configuration, code, data, readiness evidence, deployment release, and preflight receipt.
- [x] Export only an explicit safe environment allowlist to SLURM rather than inheriting the caller's full environment.
- [x] Require exactly one attempt-bound calibration receipt for the latest rendered calibration command.
- [x] Add an alternative fail-closed checksum-bound calibration migration gate that preserves the V1 job, attempt, output, release, and preflight provenance while binding it to the exact V2 release and compatibility contract.
- [x] Implement deterministic content-addressed deployment archives and validation of source files, manifests, release IDs, bundles, and SHA-256 sidecars.
- [x] Keep remote deployment an explicit operator action; local campaign planning and command rendering perform no remote writes.
- [x] Implement materialization of validation-selected algorithms, budgets, and per-window parameters into derived configurations.
- [x] Add an immutable semantic protocol lock and propagate its hash to all post-lock manifests.
- [x] Add a checksum-backed model-registry gate that marks downstream stress evidence stale when a registered model or source artifact drifts.
- [x] Split compute and aggregation into separately verified stages to respect small BistKA submission limits.
- [x] Render at most one safe SLURM submission command at a time without executing it.
- [x] Make each aggregate stage inherit the exact verified release, preflight, and effective runtime configurations of its `run_source` compute stage and fail closed when a binding is missing or mismatched.
- [x] Keep validation and test outputs clearly separated.
- [ ] Maintain a compact final archive containing configs, metrics, manifests, selected models, and report figures.
- [x] Implement a compact archive builder with SHA-256 manifest verification and default cache/checkpoint exclusions.
- [~] Check quota and run the exact-output-path write probe before every large campaign; both the verified pre-smoke and post-smoke passes are historical after 15 minutes and must be regenerated for every new submission decision.
- [ ] Clean caches and redundant checkpoints only after final artifacts are verified.

## Current Execution State

- [x] Record exactly five submissions in the historical V1 pass: jobs `3525`, `3526`, `3528`, `3529`, and `3530`.
- [x] Diagnose jobs `3525`, `3526`, and `3528` as quota-affected failures without treating their partial artifacts as accepted evidence.
- [x] Reject completed job `3529` because it used the stale remote Parquet shadow rather than the frozen release-bound CSV.
- [x] Quarantine rejected result artifacts from jobs `3528` and `3529` under [`reports/rejected_calibration/20260715_quota_and_data_shadow/`](../reports/rejected_calibration/20260715_quota_and_data_shadow/), retain scheduler logs for all five jobs under [`logs/`](../logs/), and exclude rejected evidence from accepted globs.
- [x] Accept job `3530` only after both calibration receipts passed attempt, release, preflight, data, and output-hash gates.
- [x] Complete the corrected initial deployment and exact-release BistKA preflight.
- [x] Complete and verify vector-throughput calibration and independent-process determinism.
- [x] Materialize and deploy the historical V1 calibrated screening overlay, then pass its BistKA preflight.
- [x] Preserve the V1 campaign state at 10 `VERIFIED`, 1 `READY` (`screening`), and 17 `PENDING`; see the historical [`campaign status`](../campaigns/large_scale_thai_drl/reports/campaign_status.md).
- [x] Close the V1 execution pass after job `3530`; do not submit V1 screening or any further V1 job after reaching the five-job ceiling.
- [x] In the authorized V2 deployment window, recapture live quota/capacity, apply the base release, and verify the BistKA-hosted preflight without submitting a job.
- [x] Migrate accepted job `3530` calibration evidence into V2 through a checksum-bound receipt; verify `num_envs: 1`, deterministic reproduction, and the single-field `project.name` compatibility exception without submitting a job.
- [x] Materialize, build, apply, and preflight the cumulative V2 screening checkpoint with zero submitted jobs.
- [x] Submit and verify fresh V2 vector calibration job `3540` and determinism job `3541`.
- [x] Preserve failed screening-smoke job `3542` under [`campaigns/large_scale_thai_drl_v2/incidents/screening_smoke_3542/`](../campaigns/large_scale_thai_drl_v2/incidents/screening_smoke_3542/) after tracing its checkpoint-write failure to the expired personal-group soft-quota grace.
- [x] Clean only reproducible cache data and relocate historical final archives losslessly with SHA-256 checks before retrying.
- [x] Complete screening-smoke retry job `3543`; verify all five algorithm manifests and mark `screening_smoke` `VERIFIED`.
- [x] Preserve the pre-cutover canonical boundary of 11 `VERIFIED`, 1 `READY` (`screening`), and 25 `PENDING` across 37 gates for longitudinal completion reporting.
- [x] Implement and locally validate the checkpoint-free candidate, its checksum-bound capacity gate, dedicated safe smoke, deterministic 218-file release, and technical report without submitting another job.
- [x] Archive the pre-cutover control plane and active release, validate calibration lineage without recomputation, and build the deterministic 248-file final cutover release.
- [x] Apply storage-safe release `56358f2a4996f654426814d88e622b7f217ac89bfc02f8bbdacc27e66873b55e` and verify its exact-release BistKA preflight.
- [x] Submit exactly one job in the final cutover window, storage-safe smoke `3550`, and verify all five algorithm manifests, retention receipts, and the zero-checkpoint gate.
- [x] Confirm the active consolidated graph is 9 `VERIFIED`, 1 `READY` (`screening`), and 25 `PENDING` across 35 stages; retain **11/37 (29.7%)** as the canonical whole-project denominator because gate consolidation is structural.
- [x] Record five V2 submissions to date (`3540`, `3541`, `3542`, `3543`, and `3550`); failed `3542` remains quarantined and full screening remains unsubmitted.
- [!] Do not submit the 150-trial array from an expired capacity receipt; the post-smoke pass left only 0.309 GiB modeled personal-group margin.
- [x] Record the smoke execution and storage decision in [`reports/screening_smoke_execution_20260716.md`](../reports/screening_smoke_execution_20260716.md).
- [x] Record the candidate design, evidence, limitations, and controlled cutover in [`reports/screening_storage_safe_release_candidate_20260717.md`](../reports/screening_storage_safe_release_candidate_20260717.md).
- [x] Record the applied cutover, job `3550`, capacity evidence, stage-accounting distinction, and next safe window in [`reports/screening_storage_safe_cutover_execution_20260717.md`](../reports/screening_storage_safe_cutover_execution_20260717.md).

## Downstream Release Closure — Base And Screening Remotely Verified

- [x] Implement five cumulative profiles that include all prior exact-release files plus newly materialized configuration and decision evidence: screening, budget, nested, protocol, and stress.
- [x] Implement the cumulative checkpoint before `budget_sweep`.
- [x] Implement the cumulative checkpoint before `nested_screening`.
- [x] Implement a shared locked-protocol checkpoint before `robustness` and `feature_ablation`; both phases resolve the same exact release/preflight pair and feature ablation inherits it through `run_source`.
- [x] Implement the cumulative model-registry checkpoint before `execution_stress`.
- [x] Require aggregate stages to inherit and verify the release identity, preflight receipt, and runtime inputs recorded by their `run_source` stage.
- [x] Require strict validation decisions and protocol locks to consume only exact evidence from currently `VERIFIED` aggregate gates.
- [x] Require complete new-runtime feature expectations; require readiness-certified macro and fundamental hashes/bytes before the protocol release can build.
- [x] Build, stage, apply, and remotely verify current V2 base release `4b7d1d4b887bf08675bc664683b06f0fb103369016f7233e6caeb594e41ee382`.
- [x] Pass the expanded 293-test local regression suite on 2026-07-17, plus compilation, shell syntax, JSON/sidecar validation, active/candidate/final manifest and bundle validation, lineage, rollback, import, and frozen V1 snapshot verification.
- [x] Apply the current V2 base release and locally verify its returned [`BistKA-hosted preflight`](../reports/bistka_calibration_v2_preflight.json): release and manifest identities match and the observed-market runtime is execution eligible.
- [x] Preserve the checksum-bound V1-to-V2 calibration migration from accepted job `3530` as historical compatibility evidence; use fresh release-native jobs `3540` and `3541` for the current V2 calibration gates.
- [x] Build, apply, and preflight cumulative V2 screening release `ef89ff5a738c5b25460ae40ad7789239134d761ce511a36665f857e01e3d74b4`; bind manifest `d9a1fd96af46cfe022d12b38d2de97642296a4a1f66928bbbbf3669af4dcfd86` and preflight `99ff875694ff47b6a02541718b6af93699de1f789ef35effaefe24d2e9655951`.
- [x] Preserve job `3543` as the baseline smoke, then cut over to release `56358f2a4996f654426814d88e622b7f217ac89bfc02f8bbdacc27e66873b55e`, pass preflight, and verify storage-safe smoke job `3550`, leaving `screening` `READY`.
- [!] Keep protocol/feature-ablation construction blocked until `features_real_ohlcv_macro.csv` and `features_real_ohlcv_fundamentals.csv` are present and certified in a passing observed-market readiness receipt.

## Future Submission Windows — Maximum Four Jobs Each

- [x] Historical Window 0 used the checksum-bound migration of accepted job `3530` with zero new jobs; the current V2 pass later replaced migration-only readiness with fresh jobs `3540` and `3541`.
- [x] Window A completed the storage-safe cutover, exact preflight, and one-job five-algorithm smoke (`3550`) without submitting full screening.
- [!] Window A2: first gain at least 0.692 GiB of safe personal-group headroom, then recapture checksum-bound evidence through the trusted VPN/SSH channel and submit only the one bounded five-element screening array if the new gate issues a sealed authorization receipt.
- [ ] Window B: nested screening, nested aggregate, robustness, and robustness aggregate.
- [ ] Window C: feature ablation, feature aggregate, and execution stress.
- [x] Keep the controller render-only and emit no more than one `sbatch` command at a time.
- [x] Complete the dated 2026-07-16 base deployment, fresh calibration, cumulative screening release, quota recovery, and verified smoke checkpoint with four submitted jobs and no fifth.
- [x] Complete the dated 2026-07-17 final cutover window with exactly one new job (`3550`) and no full-screening submission.
- [x] Recapture quota, credit, partitions, queue, writeability, and deployed-release evidence for the 26 July window; continue repeating this at the start of every future window because each receipt expires after 15 minutes.
- [ ] Verify the cumulative release and returned BistKA preflight between phases before rendering the next compute command.

## Phase 13: Final Results and Reporting

- [ ] Generate the test-only policy summary table.
- [ ] Generate paired excess-return tables against both primary baselines.
- [ ] Generate confidence-interval plots.
- [ ] Generate seed-by-window heatmaps.
- [ ] Generate regime-performance tables.
- [ ] Generate cost/slippage sensitivity plots.
- [ ] Generate ablation plots with uncertainty intervals.
- [ ] State clearly whether DRL improves return, drawdown, both, or neither.
- [ ] State the exact market universe and historical period.
- [ ] State all unavailable data sources and resulting claim limitations.
- [ ] Remove all mock/simulated figures from the final evidence package.
- [ ] Verify every reported number against its source CSV.
- [ ] Archive the final configuration before writing the abstract.
- [ ] Write the abstract from locked test results only.

## Recommended Large-Resource Run Funnel

- [~] **V3 development screening:** 30 fixed configurations x 6 validation folds x 3 seeds = 540 planned fits; the design is ready for review, but execution remains blocked and has produced no performance evidence.
- [ ] **Screening:** 5 algorithms x 30–100 validation trials.
- [ ] **Confirmation:** top 2 algorithms x 20–30 seeds x 6–10 outer windows.
- [ ] **Ablation:** 7–8 feature groups x 10–20 seeds x all outer windows.
- [ ] **Stress testing:** final model x 5–10 execution scenarios x 10–20 seeds x all outer windows.
- [x] Enforce a staged campaign DAG; downstream commands remain unavailable until upstream evidence is verified.

## Final Completion Gate

- [ ] All required seed/window/model runs completed or have documented exclusions.
- [ ] All final conclusions use test-only data.
- [ ] Primary baseline comparisons are paired and predeclared.
- [ ] Confidence intervals and failure cases are reported.
- [ ] External-data limitations are explicit.
- [ ] Reproducibility archive has been verified on a clean environment.
- [ ] Final report contains no mock results or unsupported whole-market claims.
- [ ] Project conclusion answers: **where does DRL work, where does it fail, and is the result stable?**
