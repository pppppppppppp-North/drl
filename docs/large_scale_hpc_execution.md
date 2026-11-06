# Large-Scale HPC Execution Guide

This guide launches the code-backed parts of `large_scale_hpc_research_checklist.md`. Data-dependent items remain blocked until legitimate overlapping historical sources are available.

## Preferred Control-Plane Workflow

Use the dependency-gated controller before following any individual submission example below:

```bash
python -m src.campaigns.control plan --spec config/research_campaign_storage_safe.yaml
python -m src.campaigns.control status --spec config/research_campaign_storage_safe.yaml
python -m src.campaigns.control commands --spec config/research_campaign_storage_safe.yaml
```

The controller renders only the next verified stage and never runs `sbatch`. Submit compute and aggregation as separate stages; do not pre-submit the full DAG or an aggregate before its array has completed and passed exact-manifest verification.

> **Current screening hold (evidence through 2026-07-17):** the storage-safe contract, release `56358f2a4996f654426814d88e622b7f217ac89bfc02f8bbdacc27e66873b55e`, exact BistKA preflight, and five-algorithm smoke job `3550` are verified. Do not execute checkpoint-writing commands from an older copy of this guide. Also do not submit the full 150-trial screening from either saved capacity receipt: the post-smoke pass expired after 15 minutes and left only 0.309 GiB of modeled personal-group soft-quota margin. Open a new authorized window, capture fresh capacity, and prefer more headroom first.

## 1. Capture Live Capacity

Run on the BistKA login node before changing concurrency:

```bash
cd /lustrefs/project/25sfcs03/drl_thai_stock
bash scripts/capture_bistka_capacity.sh
```

Review the saved `reports/bistka_capacity_*.txt` file. Keep the committed `%2` concurrency limit unless the live account limits explicitly permit more.

Measure actual vectorization throughput on a compute node before choosing the worker count:

```bash
sbatch slurm/benchmark_vectorized_training.sbatch
```

The CSV records elapsed time, samples/second, and peak resident memory for 1, 2, 4, 8, and 16 environments. Use the fastest count that remains inside the measured memory and live CPU limits.

Before a campaign, verify startup, learning, and clean shutdown for every screening algorithm inside the active BistKA environment:

```bash
for ALGO in ppo a2c ddpg td3 sac; do
  python -m scripts.smoke_vectorized_training \
    --algorithm "$ALGO" --backend subproc --num-envs 4 --timesteps 32
done
```

## 2. Five-Algorithm Validation Screening

The screening campaign uses PPO, A2C, DDPG, TD3, and SAC. Each of five array tasks owns one algorithm and runs 30 validation-only trials sequentially.

Do not submit this stage from a handwritten `sbatch` line. The first five cutover gates are complete:

1. `[x]` Bind smoke to `slurm/screening_storage_safe_smoke.sbatch` and full screening to `slurm/large_scale_screening_storage_safe.sbatch`.
2. `[x]` Require `storage_safe_smoke_validation.json`, five retention receipt/sidecar pairs, and zero checkpoint artifact paths.
3. `[x]` Preserve rollback and calibration lineage, build the deterministic 248-file release, apply it atomically, and verify the exact remote preflight.
4. `[x]` Capture and assess a fresh quota snapshot and direct-child `fsync` probe inside the 15-minute window.
5. `[x]` Submit only storage-safe smoke job `3550`; fetch and verify all five manifests, retention receipts, and the dedicated safe gate.

The remaining full-screening gates are:

6. `[ ]` In a separate authorized window, confirm the active release marker and preflight, then recapture credit, queue, quota, ownership, and the direct-child write probe. Prefer materially more than the historical 0.309 GiB margin.
7. `[ ]` Use the controller to render and inspect the one five-element full-screening array, capped at `%2` concurrency. Submit only if the fresh decision passes.
8. `[ ]` Fetch and strictly verify all 150 validation-only manifests before rendering the aggregation stage. Never pre-submit aggregation by dependency alone.

The exact release, rollback, quota, job-`3550`, and progress evidence is indexed in [`reports/screening_storage_safe_cutover_execution_20260717.md`](../reports/screening_storage_safe_cutover_execution_20260717.md). Canonical project completion remains 11/37 (29.7%); the active consolidated graph is 9 VERIFIED, 1 READY, and 25 PENDING across 35 stages.

After the array is fully terminal, run the strict fetch from the local project copy. Use the immutable attempt ID created when the screening command was rendered:

```bash
ATTEMPT_ID="$(
  python -c 'import json; print(json.load(open("campaigns/large_scale_thai_drl_v2/attempts/screening_latest.json"))["attempt_id"])'
)"

bash scripts/fetch_bistka_screening_storage_safe_results.sh \
  "$SCREENING_JOB_ID" \
  "$ATTEMPT_ID"

python -m src.campaigns.control verify \
  --spec config/research_campaign_storage_safe.yaml \
  --stage screening
```

The fetch is atomic. It refuses an incomplete/failed array, unexpected account/partition/job name, mixed attempt or release lineage, a missing trial, non-finite validation objective, unsafe symlink, checkpoint path, or overwrite. It publishes results only after all 150 trials pass an isolated local aggregation dry run.

Aggregation fails if any of the 150 expected trials is missing. The top five validation trials per algorithm are written under `<screen_dir>/aggregate/`.
It also writes `screening_algorithm_ranking.csv` and `screening_selected_algorithms.json`, selecting at most two algorithms from validation evidence. Copy those algorithm names and their locked parameters into the budget-sweep configuration before continuing.

## 3. Validation-Only Training-Budget Sweep

The bounded four-task sweep compares 20k, 100k, 250k, and 500k timesteps with the same algorithms, seeds, windows, and validation objective:

```bash
BUDGET_JOB=$(sbatch --parsable slurm/training_budget_sweep.sbatch)
BUDGET_DIR="results/training_budget_sweep_${BUDGET_JOB}"

sbatch \
  --dependency="afterok:${BUDGET_JOB}" \
  --export="ALL,RUN_DIR=${BUDGET_DIR}" \
  slurm/training_budget_sweep_aggregate.sbatch
```

Before submission, update `budget_sweep.algorithms` and `budget_sweep.algorithm_parameters` from the locked screening output. Aggregation rejects missing, duplicate, non-complete, or non-validation runs. The selected budget is the smallest common budget within the declared tolerance of the best pooled validation mean.

## 4. Nested Walk-Forward Screening

After broad screening, run validation-only tuning independently inside each outer window. This campaign never evaluates the outer test segments.

```bash
NESTED_JOB=$(sbatch --parsable slurm/nested_walk_forward_screening.sbatch)
NESTED_DIR="results/nested_walk_forward_screening_${NESTED_JOB}"

sbatch \
  --dependency="afterok:${NESTED_JOB}" \
  --export="ALL,RUN_DIR=${NESTED_DIR}" \
  slurm/nested_walk_forward_screening_aggregate.sbatch
```

The selected validation configuration for each algorithm/window is written to `nested_selected_trials.csv`. Lock these choices before any outer-test confirmation.

## 5. Large Repeated-Seed Confirmation

The current large configuration uses PPO and A2C, 20 seeds, two available walk-forward windows, eight distinct vectorized training environments, and 100,000 timesteps. Replace the algorithms only after screening selection is frozen.

```bash
ROBUST_JOB=$(sbatch --parsable slurm/large_scale_robustness.sbatch)
ROBUST_DIR="results/large_scale_robustness_${ROBUST_JOB}"

sbatch \
  --dependency="afterok:${ROBUST_JOB}" \
  --export="ALL,RUN_DIR=${ROBUST_DIR},CONFIG_PATH=config/large_scale_robustness.yaml" \
  slurm/robustness_aggregate.sbatch
```

The confirmation grid contains 80 DRL runs: 2 algorithms x 20 seeds x 2 windows. Aggregation fails on missing candidate runs or non-complete manifests.
Paired randomization p-values and Holm-adjusted p-values are written alongside confidence intervals; claims should use the adjusted values when several policy/baseline comparisons are examined.

## 6. Execution Stress Test

Update `execution_stress.model_paths` to the locked selected models and set `require_models: true` for the final campaign. Then submit:

```bash
sbatch slurm/execution_stress.sbatch
```

The default scenarios cover base, moderate, conservative, hostile, and constrained execution. Outputs include test-only metrics, model readiness, equity curves, and a sensitivity plot.

## 7. Large Feature Ablation

The default ablation campaign covers returns-only, technical, technical-context, SET-market, macro-proxy, and pilot-fundamental configurations. Five bounded tasks execute 120 runs in total.

```bash
ABLATION_JOB=$(sbatch --parsable slurm/large_scale_feature_ablation.sbatch)
ABLATION_DIR="results/large_scale_feature_ablation_${ABLATION_JOB}"

sbatch \
  --dependency="afterok:${ABLATION_JOB}" \
  --export="ALL,RUN_DIR=${ABLATION_DIR}" \
  slurm/large_scale_feature_ablation_aggregate.sbatch
```

Do not add sentiment or official-macro groups until the corresponding historical sources pass readiness and leakage checks. Structural availability alone is insufficient for official macro: the accepted input must also carry verified `point_in_time_eligible=true` evidence, which the current revised BOT export does not.
The ablation aggregate also writes paired differences against the frozen technical control with Holm-adjusted p-values.

## 8. Collect SLURM Accounting

After each campaign completes:

```bash
bash scripts/collect_slurm_accounting.sh "$SCREEN_JOB"
bash scripts/collect_slurm_accounting.sh "$BUDGET_JOB"
bash scripts/collect_slurm_accounting.sh "$NESTED_JOB"
bash scripts/collect_slurm_accounting.sh "$ROBUST_JOB"
bash scripts/collect_slurm_accounting.sh "$ABLATION_JOB"
```

Record elapsed time, CPU allocation, memory, node, state, and exit code with the research artifacts.

## 9. Determinism and Checkpoint Retention

Run the fixed-seed trajectory verification on BistKA:

```bash
sbatch slurm/verify_determinism.sbatch
```

The storage-safe screening path writes no periodic checkpoint. It retains the final model, metrics, actions, equity curve, resolved configuration, manifest, metadata, TensorBoard evidence, and an atomic checksum-bound no-write receipt. Any unexpected checkpoint is removed only to recover capacity and causes the trial to fail. Other experiment families may retain bounded checkpoints under their own verified policies; do not apply the screening rule to them implicitly.

## 10. Build a Verified Reproducibility Archive

After final results and model selection are locked:

```bash
python -m scripts.build_reproducibility_archive \
  --project-root . \
  --result-dir "$ROBUST_DIR/aggregate" \
  --result-dir "$ABLATION_DIR/aggregate" \
  --selected-model models/selected/final_model.zip \
  --output "final_archive/reproducibility_${ROBUST_JOB}.tar.gz"
```

The archive excludes caches, TensorBoard events, and intermediate checkpoints by default. It contains a SHA-256 manifest and is re-opened and hash-verified before the command succeeds.

## Remaining External Blocks

- More than two broad outer windows requires longer trustworthy OHLCV and SET index history.
- Point-in-time universe expansion requires historical membership data.
- Sentiment experiments require licensed or otherwise legitimate historical Thai text.
- Official macro experiments require point-in-time, release-dated historical vintages. The current BOT full-history download is structurally complete but revised, so it is limited to source integration, descriptive context, and explicitly labeled sensitivity analysis.
- Full multi-source ablation must wait until every source passes readiness and leakage checks.
