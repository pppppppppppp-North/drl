# Extracted HPC Notes for This Project

Source: local `HPC.pdf`, BistKA Mini-HPC Cluster documentation.

This file summarizes only the HPC details needed for the project **Deep Reinforcement Learning for Long-Term Profit Optimization in Thai Stock Markets Using Multi-Source Data**.

## 1. Cluster Purpose

BistKA is the KVIS Mini-HPC Cluster. It is designed for advanced student and teacher computational work that exceeds normal desktop capability. This project should use BistKA for:

- long DRL training,
- vectorized environments,
- WangchanBERTa embedding extraction,
- GPU sentiment fine-tuning,
- hyperparameter sweeps,
- walk-forward validation jobs.

## 2. Login Node Rule

The master/login node is only for:

- login,
- file transfer,
- code editing or compilation,
- submitting SLURM jobs.

Do not run heavy Python training, Jupyter kernels, model inference, or data processing directly on the login node. All compute work must be inside SLURM jobs.

## 3. Important Commands

```bash
sbatch script.sbatch          # submit a batch job
sinfo                         # view partitions
squeue -u $USER               # view current jobs
myqueue                       # BistKA helper for current jobs
scancel <jobid>               # cancel a job
sinteractive                  # start an interactive SLURM session
myquota                       # check disk quota
mycredit                      # check job credit
ml av                         # list available modules
module load Miniforge3/25.3.0-3
```

## 4. Hardware Summary

The extracted PDF lists:

- master node with Intel Xeon Gold CPUs,
- storage node,
- compute node with AMD EPYC 7542, 64 cores / 128 threads, 512 GiB RAM,
- GPU nodes with AMD EPYC 9124, 16 cores / 32 threads, 96 GiB RAM,
- RTX PRO 4500 GPU partitions configured for GPU jobs.

The project should assume GPU memory is limited and should cache sentiment embeddings rather than running WangchanBERTa inside every RL environment step.

## 5. Partitions and Walltime

Use these practical choices:

| Workload | Partition | Walltime | Notes |
|---|---|---:|---|
| quick CPU debugging | `compute-devel` | up to 6 hours | high priority, short jobs |
| long CPU RL training | `compute-normal` | up to 2 days | vectorized PPO/A2C |
| very long CPU jobs | `compute-long` | up to 4 days | lower priority |
| quick GPU test | `gpu4500-devel` | up to 6 hours | model loading and small batches |
| GPU sentiment training | `gpu4500-normal` | up to 2 days | WangchanBERTa fine-tuning |
| long GPU job | `gpu4500-long` | up to 4 days | lower priority |

Maximum practical group limits from the PDF:

- 5 submitted jobs per group,
- 2 running jobs per group,
- 32 CPU cores per group,
- 2 GPUs per group.

## 6. Credit Policy

The PDF states:

- user credit starts at 500 SHr,
- project credit starts at 10,000 SHr,
- CPU cost is 1.0 SHr per core-hour,
- RAM cost is 0.1 SHr per GiB-hour,
- RTX PRO 4500 GPU cost is 7.0 SHr per GPU-hour,
- project credit expires on Feb 28 of the academic year unless extended.

Project implication:

- run small tests before large jobs,
- use job arrays carefully,
- keep GPU jobs for NLP and not for CPU-only RL,
- check `mycredit` before launching sweeps.

## 7. Storage Policy

The PDF states:

- user home quota: 50 GiB,
- project quota: 300 GiB,
- project folders are under `/lustrefs/project`,
- home folders are under `/lustrefs/home`,
- `myquota` reports block quota and inode usage,
- project storage expires on Feb 28 unless extended.

Project implication:

- store large datasets in `/lustrefs/project/<project_id>/data`,
- store only small configs and scripts in home,
- use Parquet instead of CSV for processed tables,
- save only best checkpoints,
- delete failed-run checkpoints,
- clear package caches after environment setup,
- download final results before storage expiry.

## 8. Recommended Project Layout on BistKA

```text
/lustrefs/project/<project_id>/drl_thai_stock/
  config/
  data/
    raw/
    processed/
    features/
  logs/
  models/
  reports/
  results/
  scripts/
  src/
  envs/
```

## 9. Miniforge Environment Setup

Create separate environments:

```bash
module load Miniforge3/25.3.0-3

mamba create -p /lustrefs/project/<project_id>/envs/drl_env python=3.12 -y
conda activate /lustrefs/project/<project_id>/envs/drl_env
mamba install -y numpy pandas scipy scikit-learn matplotlib seaborn pyarrow
pip install gymnasium stable-baselines3 optuna tensorboard yfinance ta
conda deactivate

mamba create -p /lustrefs/project/<project_id>/envs/llm_env python=3.12 -y
conda activate /lustrefs/project/<project_id>/envs/llm_env
mamba install -y pytorch pandas numpy scikit-learn pyarrow -c pytorch -c conda-forge
pip install transformers datasets accelerate pythainlp sentencepiece tensorboard
conda deactivate
```

Adjust packages based on available modules and CUDA compatibility.

## 10. Job Strategy

1. Use `compute-devel` for 5-minute to 1-hour tests.
2. Use `gpu4500-devel` to confirm WangchanBERTa loads.
3. Use `compute-normal` for PPO and baseline experiments.
4. Use `gpu4500-normal` for Thai sentiment embedding extraction or fine-tuning.
5. Use `--array` for Optuna trials or walk-forward windows.
6. Keep each job's output in a unique folder using `$SLURM_JOB_ID`.

## 11. Required Monitoring Routine

Before every large run:

```bash
myquota
mycredit
sinfo
squeue -u $USER
```

During training:

```bash
tail -f logs/<job_name>_<jobid>.out
squeue -u $USER
```

After training:

```bash
myquota
mycredit
```

## 12. Data Offload

Before Feb 28, download:

- `config/`,
- `src/`,
- `results/`,
- `reports/`,
- best `models/`,
- final processed feature tables,
- logs needed for reproducibility.

Example:

```bash
scp -r <username>@10.205.100.101:/lustrefs/project/<project_id>/drl_thai_stock/results ./results_backup
```

