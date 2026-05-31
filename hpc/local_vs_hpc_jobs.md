# Local vs HPC Job Guide

Project: **Deep Reinforcement Learning for Long-Term Profit Optimization in Thai Stock Markets Using Multi-Source Data**

This guide states which parts of the project should run on a local machine and which parts should run on BistKA HPC. The goal is to use local compute for fast development and human inspection, while using HPC only when the workload is long, parallel, memory-heavy, GPU-heavy, or needs reproducible batch execution.

## Core Rule

Use the local machine for small, interactive, and exploratory work.

Use BistKA HPC for long-running training, repeated experiments, GPU sentiment processing, large feature generation, walk-forward evaluation, and hyperparameter sweeps.

Do not run heavy Python scripts, Jupyter kernels, model training, sentiment inference, or large data processing directly on the BistKA login/master node. The login node is only for login, file transfer, light code editing, environment setup, and `sbatch` submission.

## Quick Decision Table

| Task | Run Local? | Run HPC? | Recommended Location | Reason |
|---|---:|---:|---|---|
| Reading papers and writing proposal documents | Yes | No | Local | Human-driven writing does not need cluster resources. |
| Editing source code, configs, and Markdown | Yes | Light only | Local first | Faster feedback and safer iteration. |
| Git operations | Yes | Light only | Local first | Keep version control simple; HPC copy can be a run replica. |
| Installing and testing a small Python environment | Yes | Yes, once needed | Local first, HPC later | Local is easier for debugging; HPC needs its own environment for final jobs. |
| Unit tests on small toy data | Yes | Optional | Local | Fast enough locally. |
| Linting, formatting, and type checks | Yes | Optional | Local | Cheap and interactive. |
| One-episode RL smoke test | Yes | Optional | Local | Confirms environment logic before using HPC credits. |
| Tiny 5-minute SLURM smoke test | No | Yes | `compute-devel` | Confirms the code runs inside the cluster environment. |
| Full PPO training | No, unless tiny | Yes | `compute-normal` or `compute-long` | Long CPU workload with vectorized environments. |
| A2C baseline training | No, unless tiny | Yes | `compute-normal` | Similar to PPO, usually CPU-parallel. |
| DDPG, TD3, SAC baseline training | No, unless tiny | Yes | `compute-normal` | Repeated long training runs should be batched. |
| Hyperparameter tuning with Optuna | No | Yes | `compute-normal` array jobs | Many independent trials benefit from SLURM arrays. |
| Walk-forward validation across many windows | No | Yes | `compute-normal` array jobs | Independent windows can be split across jobs. |
| Stress testing and crisis-period evaluation | Optional | Yes | HPC if many runs | Use local only for one quick check. |
| Large OHLCV feature generation | Optional | Yes | Local for small sample, HPC for full universe | Full SET50/SET100 feature tables can be slow and memory-heavy. |
| Financial statement feature generation | Optional | Yes | Local for prototype, HPC for full run | Data joins and lag construction can become large. |
| News scraping or API data download | Yes | Usually no | Local, then transfer | Avoid long network-dependent jobs on HPC unless allowed and stable. |
| Thai text cleaning and tokenization | Optional | Yes | Local for sample, HPC for full corpus | Full corpus can be slow. |
| WangchanBERTa embedding extraction | No, except tiny test | Yes | `gpu4500-devel` test, then `gpu4500-normal` | GPU workload; embeddings should be cached. |
| WangchanBERTa fine-tuning | No | Yes | `gpu4500-normal` or `gpu4500-long` | Requires GPU and can take hours. |
| Running WangchanBERTa inside every RL step | No | No | Avoid | Cache embeddings before RL training. |
| Jupyter notebooks for explanation and plots | Yes | Optional | Local preferred; HPC through SLURM only | HPC Jupyter must run inside a scheduled job. |
| Final plotting from small CSV results | Yes | Optional | Local | Easier visual inspection. |
| Aggregating hundreds of experiment logs | Optional | Yes | HPC or local depending on size | Use HPC if logs/results are large. |
| Final PDF/LaTeX compilation | Yes | No | Local | Does not need HPC. |

## Local Jobs

Local work should be short, interactive, and easy to inspect. The local machine is the correct place to make mistakes, debug assumptions, and improve code before spending HPC credits.

### 1. Planning and Documentation

Run locally:

- editing `implementation_plan.md`,
- editing `literature_review.md`,
- editing `HPC.md`,
- preparing figures and tables for the proposal,
- compiling LaTeX/PDF documents,
- reading and annotating papers.

Why local:

- no cluster resources are needed,
- writing is interactive,
- PDF compilation and Markdown editing are cheap.

### 2. Code Development

Run locally:

- writing the trading environment,
- writing data loader interfaces,
- writing reward functions,
- writing unit tests,
- debugging the action-to-portfolio conversion,
- checking that observations have the expected shape,
- checking that no future data leaks into the state,
- checking that transaction dates line up correctly.

Local acceptance target before HPC:

- the code imports successfully,
- the environment resets and steps without crashing,
- one toy episode runs end-to-end,
- a tiny training run writes logs and a checkpoint,
- metrics can be computed from the saved result files.

### 3. Small Data Experiments

Run locally:

- one or two tickers,
- one short date range,
- synthetic toy OHLCV data,
- a few rows of news text,
- one walk-forward split only,
- one seed only,
- one algorithm only.

Examples:

```bash
python -m src.data.build_features --tickers PTT.BK AOT.BK --start 2022-01-01 --end 2022-03-31
python -m src.train --config config/debug.yaml --algo ppo --episodes 1 --num-envs 1
python -m src.evaluate --run-dir results/debug_run
```

### 4. Visualization and Interpretation

Run locally:

- equity curve plots,
- drawdown plots,
- action distribution plots,
- rolling Sharpe plots,
- reward component plots,
- thesis/proposal figures,
- final tables for the report.

Why local:

- plots require visual inspection,
- rerendering figures is quick,
- local files are easier to include in documents.

### 5. Data Download and Manual Verification

Usually run locally:

- downloading SET/BOT/market data,
- checking CSV/Excel schemas,
- manually inspecting missing values,
- checking ticker name changes,
- validating Thai news examples.

Then transfer clean raw data or processed samples to HPC only after the source format is understood.

Avoid launching large unreliable internet-download jobs on HPC unless the cluster policy and network route are known to support it.

## HPC Jobs

HPC work should be reproducible, scripted, and submitted through SLURM. Every HPC run should save its config, command, logs, metrics, and output directory.

### 1. Cluster Smoke Tests

Run on HPC:

- import test,
- environment activation test,
- tiny data path test,
- 1-episode training test,
- tiny embedding extraction test.

Use:

- `compute-devel` for CPU tests,
- `gpu4500-devel` for GPU model-load tests.

Purpose:

- verify that the conda environment works on BistKA,
- verify that project paths are correct,
- verify that `sbatch` scripts run before launching expensive jobs.

Example:

```bash
sbatch slurm/train_drl_cpu.sbatch
```

For a smoke test, temporarily reduce walltime, date range, tickers, episodes, and number of environments.

### 2. Full DRL Training

Run on HPC:

- PPO main experiments,
- A2C baseline experiments,
- DDPG baseline experiments,
- TD3 baseline experiments,
- SAC baseline experiments,
- multiple random seeds,
- full SET50 or SET100 universe,
- long historical periods,
- full walk-forward training windows.

Use:

- `compute-normal` for most training,
- `compute-long` only when a single job truly needs more than two days.

Why HPC:

- long runtime,
- repeated seeds,
- vectorized environments,
- reproducible batch logs,
- local machine remains free for writing and inspection.

Recommended script:

- `slurm/train_drl_cpu.sbatch`

Recommended resource style:

- CPU partition,
- 8 to 16 CPU cores,
- enough memory for feature tables and vectorized environments,
- unique output directory with `$SLURM_JOB_ID`.

### 3. Hyperparameter Search

Run on HPC:

- Optuna trials,
- learning rate sweeps,
- entropy coefficient sweeps,
- PPO clip ratio sweeps,
- batch size sweeps,
- reward-weight sweeps,
- number-of-environments sweeps.

Use:

- `compute-normal`,
- SLURM job arrays,
- one trial or small group of trials per array task.

Recommended script:

- `slurm/optuna_array.sbatch`

Why HPC:

- hyperparameter tuning is embarrassingly parallel,
- failed trials should not stop other trials,
- SLURM arrays make run tracking cleaner.

Important:

- keep the array size small enough for BistKA group limits,
- remember the expected limit of 5 submitted jobs and 2 running jobs per group,
- check `mycredit` before running large sweeps.

### 4. Walk-Forward Validation

Run on HPC:

- each walk-forward fold,
- each market regime split,
- each random seed,
- each algorithm comparison.

Use:

- `compute-normal`,
- job arrays if each fold can run independently.

Example division:

| Array Task | Work |
|---:|---|
| 0 | Fold 1, PPO, seed 42 |
| 1 | Fold 2, PPO, seed 42 |
| 2 | Fold 3, PPO, seed 42 |
| 3 | Fold 1, PPO, seed 123 |
| 4 | Fold 2, PPO, seed 123 |
| 5 | Fold 3, PPO, seed 123 |

After all folds finish, aggregate metrics locally or with a short HPC CPU job.

### 5. Multi-Source Feature Engineering at Full Scale

Run locally first:

- schema discovery,
- small feature prototype,
- manual inspection.

Run on HPC after prototype:

- full SET50/SET100 feature table construction,
- large rolling windows,
- technical indicators across all tickers,
- macro joins,
- financial statement joins,
- sector and market-flow joins,
- full missing-value reports,
- saving Parquet feature tables.

Use:

- `compute-devel` for a short test,
- `compute-normal` for the full feature build.

Why HPC:

- full feature generation may require substantial RAM,
- repeated rolling calculations can be slow,
- Parquet writes can be large.

### 6. Thai NLP and Sentiment Processing

Run locally:

- inspect raw news samples,
- clean 20 to 100 articles,
- verify Thai tokenization behavior,
- confirm label or sentiment schema,
- test one batch through WangchanBERTa if the local machine can handle it.

Run on HPC:

- full news corpus cleaning,
- WangchanBERTa embedding extraction,
- sentiment fine-tuning,
- batch inference over all articles,
- generating daily ticker-level sentiment features.

Use:

- `gpu4500-devel` for model-load and tiny batch tests,
- `gpu4500-normal` for full embedding extraction or fine-tuning,
- `gpu4500-long` only if the job cannot fit in two days.

Recommended script:

- `slurm/train_sentiment_gpu.sbatch`

Important:

- cache embeddings or daily sentiment features,
- do not call WangchanBERTa during each RL environment step,
- save output as Parquet or NumPy arrays,
- record the model name, checkpoint, tokenizer, max sequence length, and batch size.

### 7. Large Evaluation and Stress Tests

Run on HPC:

- many seeds,
- many market regimes,
- many algorithms,
- full crisis-period stress testing,
- full ablation matrix,
- repeated bootstrap-style resampling if used later.

Use:

- `compute-normal`,
- job arrays when independent.

Run locally:

- final chart rendering,
- final table formatting,
- small result sanity checks.

### 8. JupyterLab

Run locally by default.

Use HPC Jupyter only when:

- the notebook needs HPC-only data,
- the notebook needs the HPC conda environment,
- the notebook needs more memory than local,
- the notebook is a short analysis session.

On BistKA, JupyterLab must run inside a SLURM job. Do not start Jupyter on the login node.

Recommended script:

- `slurm/jupyter_lab.sbatch`

When finished, close the browser and cancel the SLURM job.

## Jobs That Should Not Be Run on HPC

Do not use HPC for:

- writing documents,
- compiling proposal PDFs,
- editing Markdown,
- formatting code,
- running one-line checks that local can do instantly,
- manual data inspection,
- web browsing,
- small plotting,
- small unit tests,
- interactive trial-and-error work that will repeatedly fail,
- anything that would keep a GPU idle while waiting for human input.

These jobs waste queue time, credits, or both.

## Jobs That Should Not Be Run on the BistKA Login Node

Do not run these directly after SSH login:

- `python -m src.train ...` for real training,
- `python -m src.sentiment.extract_embeddings ...` for real NLP work,
- long `pandas` feature generation,
- long Optuna sweeps,
- JupyterLab kernels,
- GPU model inference,
- large parallel data preprocessing,
- long `nohup` or `tmux` training sessions.

Instead, create or edit an `sbatch` script and submit it with `sbatch`.

## Recommended Project Workflow

### Stage 1: Local Prototype

Run locally:

1. Build a tiny dataset with two tickers and a short date range.
2. Validate data cleaning and missing-value handling.
3. Build the trading environment.
4. Run one random-policy episode.
5. Run one PPO debug episode.
6. Save debug metrics.
7. Plot a tiny equity curve.

Move to HPC only after the local prototype works.

### Stage 2: HPC Smoke Test

Run on HPC:

1. Transfer code and a tiny data sample.
2. Create or activate the BistKA conda environment.
3. Submit a short `compute-devel` job.
4. Confirm output files are written.
5. Confirm logs contain no import or path errors.
6. Submit a short `gpu4500-devel` job if NLP is needed.

### Stage 3: Full Feature Build

Run on HPC:

1. Build all OHLCV features.
2. Build macro and index features.
3. Build financial statement features.
4. Build sentiment features.
5. Save processed feature tables.
6. Write a data manifest.

Inspect locally:

1. Download summary reports.
2. Check missing values and date alignment.
3. Plot sample features.

### Stage 4: Full Training

Run on HPC:

1. Train PPO technical-only baseline.
2. Train PPO multi-source model.
3. Train A2C/DDPG/TD3/SAC baselines.
4. Run seeds.
5. Run walk-forward folds.
6. Save checkpoints and metrics.

Inspect locally:

1. Download metrics.
2. Plot equity and drawdown.
3. Compare model rankings.

### Stage 5: Tuning and Ablation

Run on HPC:

1. Optuna search.
2. Reward ablations.
3. Feature-source ablations.
4. Algorithm ablations.
5. Stress-period experiments.

Inspect locally:

1. Build comparison tables.
2. Identify the best model.
3. Prepare final figures.

### Stage 6: Final Reporting

Run locally:

1. Compile final Markdown/LaTeX.
2. Create final charts.
3. Write the experimental procedure.
4. Write limitations.
5. Package final results.

Only use HPC in this stage if a missing experiment must be rerun.

## Resource Selection Rules

Use `compute-devel` when:

- runtime is under 6 hours,
- the job is a smoke test,
- the job checks imports, paths, or a tiny dataset.

Use `compute-normal` when:

- runtime is under 2 days,
- the job is full CPU RL training,
- the job is full feature engineering,
- the job is a normal evaluation job.

Use `compute-long` when:

- runtime may exceed 2 days,
- the job is CPU-only,
- splitting the job into smaller jobs is not practical.

Use `gpu4500-devel` when:

- testing CUDA visibility,
- loading WangchanBERTa,
- running a tiny embedding batch,
- checking GPU package compatibility.

Use `gpu4500-normal` when:

- extracting full sentiment embeddings,
- fine-tuning WangchanBERTa,
- running large batch NLP inference.

Use `gpu4500-long` when:

- GPU work cannot finish within 2 days,
- the job has already passed a shorter GPU test.

## Practical Examples

### Local Debug Training

```bash
python -m src.train \
  --config config/debug.yaml \
  --algo ppo \
  --episodes 1 \
  --num-envs 1 \
  --seed 42 \
  --output-dir results/debug_local
```

### HPC CPU Training

Use `slurm/train_drl_cpu.sbatch` for real DRL training:

```bash
sbatch slurm/train_drl_cpu.sbatch
```

### HPC GPU Sentiment Embedding

Use `slurm/train_sentiment_gpu.sbatch` after a short GPU test:

```bash
sbatch slurm/train_sentiment_gpu.sbatch
```

### HPC Hyperparameter Sweep

Use `slurm/optuna_array.sbatch`:

```bash
sbatch slurm/optuna_array.sbatch
```

### HPC JupyterLab

Use `slurm/jupyter_lab.sbatch` only when local notebooks are insufficient:

```bash
sbatch slurm/jupyter_lab.sbatch
```

Then read the SLURM output file for the SSH tunnel command and browser URL.

## Before Submitting Any HPC Job

Check:

- the job script has the correct `--account`,
- the partition matches the workload,
- the walltime is realistic,
- the memory request is not excessive,
- the output directory includes `$SLURM_JOB_ID` or a unique run name,
- the config file is saved with the result,
- the command uses the intended seed,
- the dataset path points to BistKA storage,
- `myquota` shows enough storage,
- `mycredit` shows enough credit,
- `squeue -u $USER` shows capacity for another job.

Commands:

```bash
myquota
mycredit
sinfo
squeue -u $USER
```

## After Every HPC Job

Do:

- inspect the SLURM output file,
- copy important logs into the run directory,
- verify that metrics were written,
- verify that checkpoints are not excessive,
- delete failed-run temporary files,
- update the experiment tracker,
- download important summaries to local storage.

Minimum result files:

- `config.yaml`,
- `command.txt`,
- `metrics.csv`,
- `summary.json`,
- `slurm-<jobid>.out`,
- best model checkpoint if training succeeded.

## Summary

Local work is for thinking, editing, debugging, plotting, and tiny experiments.

HPC work is for full-scale training, full-scale feature generation, GPU NLP, sweeps, walk-forward validation, and stress testing.

The BistKA login node is never a training machine. Use it only to prepare and submit SLURM jobs.
