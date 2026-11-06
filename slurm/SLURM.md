# SLURM Templates for the DRL Thai Stock Project

These scripts target the BistKA project workspace:

- project path: `/lustrefs/project/25sfcs03/drl_thai_stock`
- environment path: `/lustrefs/project/25sfcs03/envs`
- assumed SLURM account: `25sfcs03`

If `sbatch` reports an invalid account, check the live account with `mycredit` or `sacctmgr`/course instructions and update the `#SBATCH --account` line.

Prefer `python -m src.campaigns.control commands --spec config/research_campaign.yaml` to render the next safe command. The controller validates array shape, account, partition, CPU count, dependencies, artifacts, and protocol state, but deliberately never submits a job itself.

## Scripts

- `smoke_cpu.sbatch`: short CPU import/environment smoke test.
- `train_drl_debug.sbatch`: short compute-devel PPO run on synthetic pilot data.
- `train_drl_cpu.sbatch`: vectorized PPO/A2C training on CPU partitions.
- `train_sentiment_gpu.sbatch`: WangchanBERTa fine-tuning or embedding extraction on RTX PRO 4500.
- `optuna_array.sbatch`: hyperparameter sweep using `SLURM_ARRAY_TASK_ID`.
- `robustness_array.sbatch`: four algorithm/window tasks, capped at two concurrent tasks, with five sequential seeds per task.
- `robustness_aggregate.sbatch`: test-only robustness aggregation, confidence intervals, paired baseline comparisons, and plots.
- `large_scale_screening.sbatch`: bounded five-task validation screening across PPO, A2C, DDPG, TD3, and SAC.
- `large_scale_screening_aggregate.sbatch`: completeness-gated screening aggregation and top-trial selection.
- `training_budget_sweep.sbatch`: four bounded validation-only tasks comparing 20k, 100k, 250k, and 500k timesteps.
- `training_budget_sweep_aggregate.sbatch`: completeness-gated plateau selection of a common training budget.
- `large_scale_robustness.sbatch`: 20-seed, eight-environment, 100,000-timestep confirmation campaign.
- `execution_stress.sbatch`: locked-model testing across cost, slippage, turnover, position, and cash constraints.
- `large_scale_feature_ablation.sbatch`: six feature groups distributed across five bounded tasks.
- `large_scale_feature_ablation_aggregate.sbatch`: completeness-gated test-only ablation aggregation and confidence intervals.
- `nested_walk_forward_screening.sbatch`: five bounded algorithm tasks for validation-only tuning inside each outer window.
- `nested_walk_forward_screening_aggregate.sbatch`: completeness-gated selection of the best validation trial per algorithm/window.
- `benchmark_vectorized_training.sbatch`: measures throughput and peak memory for 1, 2, 4, 8, and 16 vector environments.
- `verify_determinism.sbatch`: records and verifies a fixed-seed trajectory fingerprint.
- `jupyter_lab.sbatch`: JupyterLab running inside SLURM, not on the login node.

## Recommended Workflow

1. Run tiny local tests.
2. Submit `train_drl_cpu.sbatch` to `compute-devel` with short walltime.
3. If it works, change to `compute-normal` and increase walltime.
4. Use `train_sentiment_gpu.sbatch` only after a small `gpu4500-devel` model-load test.
5. Use `optuna_array.sbatch` for many small independent experiments.
6. Check `myquota` and `mycredit` before and after large jobs.

## Repeated-Seed Robustness Run

The robustness array is deliberately four tasks rather than one task per seed. This stays within BistKA's small submitted-job limit while still using both available concurrent job slots.

```bash
ARRAY_JOB_ID=$(sbatch --parsable slurm/robustness_array.sbatch)
RUN_DIR="results/robustness_real_ohlcv_market_${ARRAY_JOB_ID}"

sbatch \
  --dependency="afterok:${ARRAY_JOB_ID}" \
  --export="ALL,RUN_DIR=${RUN_DIR}" \
  slurm/robustness_aggregate.sbatch
```

The aggregate job runs only after every array task succeeds. Its research tables are generated from test rows only. Full instructions are in `../docs/robustness_experiment.md`.

For a fuller local-versus-HPC decision guide, see:

- `../hpc/local_vs_hpc_jobs.md`
- `../hpc/job_routing_checklist.md`
