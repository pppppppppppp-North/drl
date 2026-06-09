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
- `docs/full_external_rebuild.md` - rebuild order and validation notes for the full external-data feature table.
- `hpc/local_vs_hpc_jobs.md` - detailed guide for deciding which project jobs run locally and which run on BistKA HPC.
- `hpc/job_routing_checklist.md` - short checklist for routing jobs to local, HPC CPU, or HPC GPU.
- `hpc/hpc_sync_status.md` - latest verified BistKA connection, sync method, and external-data boundary.
- `slurm/SLURM.md` - explanation of the provided SLURM scripts.
- `slurm/train_drl_cpu.sbatch` - CPU/vectorized RL training template.
- `slurm/train_sentiment_gpu.sbatch` - GPU WangchanBERTa fine-tuning or embedding extraction template.
- `slurm/optuna_array.sbatch` - job-array template for hyperparameter sweeps.
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
