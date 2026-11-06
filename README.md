# Final Project Plan Deliverables

Project title: **Deep Reinforcement Learning for Long-Term Profit Optimization in Thai Stock Markets Using Multi-Source Data**

This folder contains the proposal-level implementation plan, literature review, extracted HPC usage notes, and SLURM templates.

It now also contains the first runnable implementation scaffold:

- deterministic pilot data generation,
- technical feature engineering,
- a Gymnasium-compatible long-only trading environment,
- baseline policies and metrics,
- smoke tests,
- BistKA-ready SLURM scripts targeting `/lustrefs/project/25sfcs03/drl_thai_stock`.

## Current Data Milestone

V3 readiness was verified on 7 September 2026: **53 focused tests passed; 524 broader tests passed with three documented historical V2/runtime exclusions**. Preprocessing now rejects arithmetic overflow before output hashing. The [readiness report](reports/v3_execution_readiness_20260907.md) and [QA record](reports/v3_execution_readiness_20260907_qa.md) document the rebuilt blocked capsule. All 540 planned cells remain unauthorized; zero HPC jobs were submitted and scientific completion remains 11/37 (29.7%).

The project now includes a checksum-bound official SET50/SET100 constituent and sector archive for 2021–2024:

- 10 preserved official PDFs, including two published revisions;
- 10 continuous effective intervals and 1,000 normalized membership rows;
- exactly 50 SET50 and 100 SET100 members per interval;
- 138 unique tickers in an acquisition-only union; and
- 23 canonical sectors with dated changes preserved.

Rebuild and validate it with:

```bash
python -m src.data.ingest_set_constituent_history --project-root .
python -m scripts.build_set_constituent_archive_report --project-root .
python -m src.data.source_readiness --output reports/source_readiness.csv
```

See `docs/set_constituent_history_intake.md` and `reports/set_constituent_archive_readiness_20260731.md`.

The next local milestone is also implemented in an isolated V3 development capsule:

- a fixed 138-slot, point-in-time SET100 panel across 965 observed sessions;
- conservative next-session activation for same-day revisions;
- dated membership and sector context;
- active-member-only dated sector aggregates with explicit coverage and availability;
- past-only development eligibility;
- a self-financing mask-aware environment with fixed asset-slot alignment, security-only turnover, post-return weight drift, valuation checks, and separate decision/next-observation mask evidence;
- a checksum-bound draft investability lifecycle with explicit suspension, exit, stale-valuation, delisting, corporate-action, and symbol-change rules;
- a five-file licensed-market package contract and fail-closed gap assessment;
- a deterministic draft evaluation/selection design with six provisional expanding half-year validation folds, 30 fixed configurations, and seeds 42/43/44;
- an exact 540-cell prospective execution plan totaling 63,000,000 timesteps, mapped into three batches of 180 cells with every cell `blocked_not_authorized`;
- a V3 launcher that checks the locked development gate before resolving a candidate/fold/seed or importing the intentionally absent V3 runner;
- exact fold dataset slices with 30 context sessions, owned decisions, terminal outcomes, and transition ledgers;
- fold-specific population z-score preprocessing fitted only on eligible training-decision rows, with protected raw price and accounting fields preserved;
- validation-transition evidence that checks identity, complete coverage, portfolio/baseline value chains, costs, and self-financing reconciliation before aggregate conversion; and
- a gate that rejects training until passing market evidence plus locked investability and evaluation policies are bound, while `test` and `all` remain forbidden.

Rebuild and validate this structural artifact with:

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
  tests/test_build_v3_evaluation_policy_readiness.py \
  tests/test_v3_development_gate.py \
  tests/test_v3_execution_plan.py \
  tests/test_v3_trial_launcher.py \
  tests/test_v3_fold_dataset.py \
  tests/test_v3_fold_preprocessing.py \
  tests/test_v3_result_evidence.py \
  tests/test_build_v3_execution_capsule.py \
  tests/test_setsmart_v3_request_packet.py
```

The market assessment intentionally fails for the current snapshot: 5/138 acquisition tickers, 4,825/96,500 active membership keys, no explicit trading status, no authoritative licensed session binding, no same-sector peer among the covered active rows, and 0/62 membership-exit paths certified. The evaluation design is structural readiness only: it caps a future run at three array-job submissions with at most two concurrent, but zero V3 jobs have been submitted and it contains no performance evidence. See the [V3 development guide](docs/v3_membership_development.md), [execution readiness report](reports/v3_execution_readiness_20260907.md), and [checksum-bound capsule receipt](reports/data_quality_v3/v3_execution_capsule_readiness_20260907.json). The membership, market, investability, and evaluation evidence remains preserved in the earlier dated reports.

The [SETSMART request packet](requests/setsmart_v3_20260901/README.md) is **DRAFT — UNSENT**. It contains the five-file/33-field schema, byte-identical 138-ticker acquisition union, license checklist, and cover letter. The [author/advisor decision packet](docs/v3_author_decision_packet_20260904.md) records seven pending decisions; its recommendations approve nothing. Aggregate metric conversion remains blocked until the exact Sharpe, turnover, and drawdown contract is approved and registered.

Training, `test`, `all`, model ranking, and performance claims remain blocked. Licensed OHLCV/trading status, an authoritative session calendar, corporate actions, stable symbol history, a passing checksum-bound package receipt, approved investability and evaluation policies, a locked protocol/release, a reviewed separate V3 runner, and a genuinely untouched post-2024 final holdout are still required. No V3 scientific-compute stage has completed; canonical whole-project completion remains **11/37 = 29.7%**.

## Quick Start

On a machine with the dependencies installed:

```bash
python -m src.data.synthetic --config config/debug.yaml
python -m src.features.build_features --config config/debug.yaml
python -m src.agents.baselines --config config/debug.yaml
python -m pytest -q
```

On BistKA, the current project copy is under:

```text
/lustrefs/project/25sfcs03/drl_thai_stock
```

The first smoke job is:

```bash
cd /lustrefs/project/25sfcs03/drl_thai_stock
sbatch slurm/smoke_cpu.sbatch
```

After the smoke job passes, a short PPO debug job is:

```bash
sbatch slurm/train_drl_debug.sbatch
```

For the first algorithm comparison run:

```bash
sbatch slurm/algorithm_compare.sbatch
```

To generate report-ready figures after a run finishes:

```bash
python -m src.evaluation.plot_results --run-dir results/run_2850
```

For the final repeated-seed, test-only robustness experiment on BistKA:

```bash
ARRAY_JOB_ID=$(sbatch --parsable slurm/robustness_array.sbatch)
RUN_DIR="results/robustness_real_ohlcv_market_${ARRAY_JOB_ID}"
sbatch --dependency="afterok:${ARRAY_JOB_ID}" --export="ALL,RUN_DIR=${RUN_DIR}" slurm/robustness_aggregate.sbatch
```

See `docs/robustness_experiment.md` for the locked design, smoke-run command, output files, and interpretation rules.

## Long-Term Objective, Fusion Model, and Rigor Extensions (2026-11-06)

These extensions live in new modules, so no file in the manifest-locked V2 release changes:

- `src/envs/long_term_env.py`: `LongTermTradingEnv`; with default options it is bit-identical to `ThaiStockTradingEnv`.
- `src/train_long_term.py`: the training entrypoint for the options below (`python -m src.train_long_term --config <config>`).
- `src/models/feature_extractors.py`: the `asset_lstm` fusion extractor.
- `src/evaluation/benchmark_metrics.py` and `src/evaluation/significance.py`: benchmark-relative metrics and Sharpe significance.
- `src/features/fundamental_flags.py`, `src/features/investor_flow_context.py`, `src/data/ingest_investor_flow.py`, `src/sentiment/score_news.py` (GPU job: `slurm/sentiment_embeddings_gpu.sbatch`): multi-source inputs.
- `src/agents/predictive_baselines.py`: ridge and LSTM (Bi-LSTM over the past window) forecast-then-allocate baselines.
- `src/data/ingest_bot_policy_rate.py` and `src/features/policy_rate_context.py`: the official BOT policy-rate decision history as a point-in-time macro source.
- `src/evaluation/attribution.py`: permutation attribution of data-source groups for trained agents.
- `src/envs/thai_costs.py`: the published Thai cost schedule (about 0.176% per side including VAT).

Config options read by `src.train_long_term` and the experiment scripts:

```yaml
environment:
  accounting: self_financing      # security-only turnover, weight drift (default: legacy)
  rebalance_every: 5              # decide weekly; hold and drift in between
  action_transform: softmax       # N+1 logits incl. cash
  train_random_start: true        # training envs only
  train_episode_length: 504
  reward:
    type: plan_main               # simple | log_return | plan_main | sortino
    sharpe_weight: 0.1            # differential Sharpe term
    drawdown_mode: increment      # penalize new drawdown only
    drawdown_penalty: 0.5
    turnover_penalty: 0.0
training:
  normalize_observations: true    # train-only VecNormalize, frozen for evaluation
  policy:
    features_extractor: asset_lstm
    market_feature_columns: [set50_etf_return_1d, set50_etf_volatility_20d]
```

```bash
# Extended 2012-2024 history plus a sealed 2025+ holdout (never overwritten)
python -m scripts.download_extended_history --tickers-csv data/reference/set50_or_set100_universe.csv \
  --dev-output data/raw/extended/prices.csv.gz --index-dev-output data/raw/extended/index.csv \
  --holdout-dir data/holdout_sealed/<name>
# Reward/rebalancing experiment and figure
python -m scripts.run_reward_variant_experiment --config config/long_term_reward_experiment.yaml --output-dir results/<run>
python -m scripts.plot_reward_experiment --run-dir results/<run>
# Bootstrap, probabilistic, and deflated Sharpe for any long equity-curve CSV
python -m src.evaluation.significance --curves results/<run>/validation_equity_curves.csv --reference equal_weight_buy_hold --n-trials 6 --output results/<run>/significance.csv
# Walk-forward (one validation year per fold, 2016-2022) with a worker cap; resumable
PYTHON=python scripts/run_walk_forward_reward.sh results/<wf-run> 5 100000
python -m scripts.evaluate_forecast_baselines --config config/long_term_reward_experiment.yaml --run-dir results/<wf-run> --years 2016 2017 2018 2019 2020 2021 2022
python -m scripts.aggregate_walk_forward_reward --run-dir results/<wf-run>
python -m scripts.feature_group_attribution --config config/long_term_reward_experiment.yaml --run-dir results/<wf-run> --variant plan_main_weekly_lstm
# Data-source ablation (technical / +market / +macro)
CONFIG=config/source_ablation_experiment.yaml VARIANTS="src_technical src_technical_market_macro" SEEDS="42 43 44" scripts/run_walk_forward_reward.sh results/<ablation-run> 5 100000
python -m scripts.aggregate_walk_forward_reward --run-dir results/<ablation-run> --extra-run-dirs results/<wf-run> --variants src_technical plan_main_weekly_lstm src_technical_market_macro --seeds 42 43 44 --output-dir results/<ablation-combined>
# Official BOT policy-rate history
python -m src.data.ingest_bot_policy_rate --input data/external/original/bot_policy_rate_mpc_table_asof_20261028.xlsx --output data/external/bot_policy_rate_history.csv --source-url <url> --retrieved-at <utc>
```

See [`docs/long_term_reward_experiment_20261106.md`](docs/long_term_reward_experiment_20261106.md) for the first results and [`IMPROVEMENT_MILESTONES.md`](IMPROVEMENT_MILESTONES.md) for what is left.

## Research Campaign Control Plane

The large campaign is now managed by a render-only, dependency-gated controller. It never submits jobs itself:

```bash
python -m src.campaigns.control plan --spec config/research_campaign.yaml
python -m src.campaigns.control status --spec config/research_campaign.yaml
python -m src.campaigns.control commands --spec config/research_campaign.yaml
python -m src.campaigns.control verify --spec config/research_campaign.yaml --stage <stage_id>
```

The controller requires exact manifests, explicit artifact verification, validation-decision materialization, and an immutable protocol lock before test-reading stages. See `docs/research_campaign_control_plane_report.md` for the architecture and current dry-run evidence.

## Files

- `implementation_plan.md` - main comprehensive two-month project proposal plan.
- `implementation_plan.tex` - LaTeX version of the main plan.
- `implementation_plan.pdf` - compiled PDF version of the main plan.
- `literature_review/literature_review.md` - standalone literature review in IEEE-style numbered citations.
- `literature_review/literature_review.tex` - LaTeX source for the literature review.
- `literature_review/literature_review.pdf` - compiled literature review PDF.
- `comprehensive_introduction_literature_review.tex` - expanded standalone introduction and literature review LaTeX source.
- `comprehensive_introduction_literature_review.pdf` - compiled expanded introduction and literature review PDF.
- `hpc/HPC.md` - extracted practical guide from `HPC.pdf` for this project.
- `docs/external_data_handoff.md` - exact handoff guide for the external files needed to move beyond the five-ticker pilot.
- `docs/external_data_provider_request.md` - provider-facing request packet and email template for the missing external files.
- `docs/external_data_intake_checklist.md` - intake checklist and manifest workflow for received provider files.
- `docs/full_external_rebuild.md` - rebuild order and validation notes for the full external-data feature table.
- `docs/robustness_experiment.md` - repeated-seed walk-forward design and BistKA execution guide.
- `docs/large_scale_hpc_research_checklist.md` - staged checklist for a large-compute research campaign.
- `docs/large_scale_hpc_execution.md` - commands for capacity capture, five-algorithm screening, large robustness, stress testing, and accounting.
- `hpc/local_vs_hpc_jobs.md` - detailed guide for deciding which project jobs run locally and which run on BistKA HPC.
- `hpc/job_routing_checklist.md` - short checklist for routing jobs to local, HPC CPU, or HPC GPU.
- `hpc/hpc_sync_status.md` - latest verified BistKA connection, sync method, and external-data boundary.
- `slurm/SLURM.md` - explanation of the provided SLURM scripts.
- `slurm/train_drl_cpu.sbatch` - CPU/vectorized RL training template.
- `slurm/train_sentiment_gpu.sbatch` - GPU WangchanBERTa fine-tuning or embedding extraction template.
- `slurm/optuna_array.sbatch` - job-array template for hyperparameter sweeps.
- `slurm/robustness_array.sbatch` - four-task algorithm/window array that runs repeated seeds.
- `slurm/robustness_aggregate.sbatch` - dependent test-only aggregation and uncertainty-report job.
- `slurm/large_scale_screening.sbatch` - five-algorithm validation screening campaign.
- `slurm/large_scale_robustness.sbatch` - 20-seed, 100,000-timestep confirmation campaign.
- `slurm/execution_stress.sbatch` - test-only cost, slippage, turnover, position, and cash stress tests.
- `slurm/large_scale_feature_ablation.sbatch` - six-group, ten-seed, two-window feature-ablation campaign.
- `slurm/nested_walk_forward_screening.sbatch` - validation-only tuning inside each outer walk-forward window.
- `slurm/jupyter_lab.sbatch` - JupyterLab-through-SLURM template.
- `sources/references.md` - source notes used when preparing the plan.

## Main Assumptions

- Audience level: high school research proposal, but written with enough rigor to execute the experiments.
- Time budget: two months of full-time work.
- Compute: local machine for development and BistKA HPC for long training, GPU NLP, and repeated experiments.
- Primary market: Thai equities, beginning with SET50/SET100 and expanding only if data quality is acceptable.
- Action space: continuous portfolio weights.
- Algorithms: PPO as the main algorithm, with A2C, DDPG, TD3, SAC, and simple baselines for comparison.
- Data: mixed frequency and multi-source data, including OHLCV, technical indicators, index data, macro data, financial statements, news/sentiment, sector data, and market flow where available.
