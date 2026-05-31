# Job Routing Checklist

Project: **Deep Reinforcement Learning for Long-Term Profit Optimization in Thai Stock Markets Using Multi-Source Data**

Use this checklist before deciding whether a task should run locally or on BistKA HPC.

## Run Locally If

- The task takes seconds or a few minutes.
- The task needs human inspection or repeated editing.
- The task is documentation, plotting, or PDF compilation.
- The task uses a tiny sample dataset.
- The task is a unit test or smoke test that local hardware can run.
- The task is debugging environment logic, reward logic, feature shapes, or date alignment.
- The task downloads data from websites or APIs and needs manual verification.
- The task creates final charts from already-computed results.

Typical local commands:

```bash
python -m pytest
python -m src.train --config config/debug.yaml --episodes 1 --num-envs 1
python -m src.evaluate --run-dir results/debug_local
pdflatex implementation_plan.tex
```

## Run on HPC CPU If

- The task runs for hours.
- The task needs many CPU cores.
- The task trains PPO, A2C, DDPG, TD3, or SAC on the full dataset.
- The task runs multiple seeds.
- The task runs walk-forward folds.
- The task runs Optuna or grid-search trials.
- The task generates full SET50/SET100 feature tables.
- The task performs large evaluation, ablation, or stress-test batches.

Recommended partitions:

| Situation | Partition |
|---|---|
| Short CPU smoke test | `compute-devel` |
| Normal CPU training or preprocessing | `compute-normal` |
| Very long CPU job | `compute-long` |

Recommended templates:

- `../slurm/train_drl_cpu.sbatch`
- `../slurm/optuna_array.sbatch`

## Run on HPC GPU If

- The task loads WangchanBERTa or another transformer model.
- The task extracts embeddings for many Thai news articles.
- The task fine-tunes a Thai sentiment model.
- The task runs large batch NLP inference.
- The task needs CUDA.

Recommended partitions:

| Situation | Partition |
|---|---|
| GPU import/model-load smoke test | `gpu4500-devel` |
| Normal GPU sentiment work | `gpu4500-normal` |
| Very long GPU sentiment work | `gpu4500-long` |

Recommended template:

- `../slurm/train_sentiment_gpu.sbatch`

## Do Not Run on the Login Node

After SSH login to BistKA, do not directly run:

- full RL training,
- full feature generation,
- full sentiment embedding extraction,
- WangchanBERTa fine-tuning,
- Optuna sweeps,
- JupyterLab kernels,
- long `pandas` processing,
- long `nohup` or `tmux` jobs.

Use `sbatch` instead.

## Local-to-HPC Promotion Rule

A job is ready for HPC only after all of these are true:

- It works locally on tiny data.
- It has a config file.
- It writes outputs to a known directory.
- It can resume or fail without destroying previous results.
- It logs metrics.
- It saves the random seed.
- It saves enough information to reproduce the run.
- The expected runtime and resource needs are known approximately.

## HPC Submission Checklist

Before `sbatch`, verify:

- `#SBATCH --account=<project_account>` is correct.
- `#SBATCH --partition=...` matches CPU or GPU needs.
- `#SBATCH --time=...` is realistic.
- `#SBATCH --mem=...` is enough but not wasteful.
- GPU jobs include the correct `--gres` line.
- Paths point to BistKA storage, not local machine paths.
- Output folders include `$SLURM_JOB_ID` or a unique run name.
- `myquota` has enough space.
- `mycredit` has enough credit.
- `squeue -u $USER` does not already show too many queued/running jobs.

Useful commands:

```bash
myquota
mycredit
squeue -u $USER
sinfo
sbatch <script.sbatch>
```

## Project Job Mapping

| Project Phase | Main Tasks | Location |
|---|---|---|
| Phase 1: setup | docs, repo, tiny env tests | Local |
| Phase 1: HPC setup | conda env, import test, path test | HPC `compute-devel` |
| Phase 2: data prototype | small ticker/date data checks | Local |
| Phase 2: full feature build | full technical/macro/financial features | HPC CPU |
| Phase 3: sentiment prototype | inspect and clean small news sample | Local |
| Phase 3: sentiment full run | WangchanBERTa embeddings/fine-tuning | HPC GPU |
| Phase 4: environment build | reward/action/observation debugging | Local |
| Phase 5: training | PPO and baseline full runs | HPC CPU |
| Phase 6: tuning | Optuna and ablations | HPC CPU arrays |
| Phase 7: validation | walk-forward and stress tests | HPC CPU arrays |
| Phase 8: reporting | figures, tables, final documents | Local |

## Short Answer

Local: write code, debug tiny cases, inspect data, make plots, write reports.

HPC CPU: full RL training, feature generation, walk-forward validation, sweeps, ablations, stress tests.

HPC GPU: WangchanBERTa embedding extraction, sentiment fine-tuning, and large NLP inference.

BistKA login node: only login, transfer, light editing, environment setup, and `sbatch`.
