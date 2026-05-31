# SLURM Templates for the DRL Thai Stock Project

These scripts target the BistKA project workspace:

- project path: `/lustrefs/project/25sfcs03/drl_thai_stock`
- environment path: `/lustrefs/project/25sfcs03/envs`
- assumed SLURM account: `25sfcs03`

If `sbatch` reports an invalid account, check the live account with `mycredit` or `sacctmgr`/course instructions and update the `#SBATCH --account` line.

## Scripts

- `smoke_cpu.sbatch`: short CPU import/environment smoke test.
- `train_drl_debug.sbatch`: short compute-devel PPO run on synthetic pilot data.
- `train_drl_cpu.sbatch`: vectorized PPO/A2C training on CPU partitions.
- `train_sentiment_gpu.sbatch`: WangchanBERTa fine-tuning or embedding extraction on RTX PRO 4500.
- `optuna_array.sbatch`: hyperparameter sweep using `SLURM_ARRAY_TASK_ID`.
- `jupyter_lab.sbatch`: JupyterLab running inside SLURM, not on the login node.

## Recommended Workflow

1. Run tiny local tests.
2. Submit `train_drl_cpu.sbatch` to `compute-devel` with short walltime.
3. If it works, change to `compute-normal` and increase walltime.
4. Use `train_sentiment_gpu.sbatch` only after a small `gpu4500-devel` model-load test.
5. Use `optuna_array.sbatch` for many small independent experiments.
6. Check `myquota` and `mycredit` before and after large jobs.

For a fuller local-versus-HPC decision guide, see:

- `../hpc/local_vs_hpc_jobs.md`
- `../hpc/job_routing_checklist.md`
