#!/bin/bash
#SBATCH --job-name=drl_budget
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4-00:00:00
#SBATCH --partition=compute-long
#SBATCH --array=0-3%2
#SBATCH --account=25sfcs03
#SBATCH --output=logs/training_budget_%A_%a.out
#SBATCH --error=logs/training_budget_%A_%a.err

set -euo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
PYTHON="/lustrefs/project/25sfcs03/envs/drl_env/bin/python"
CONFIG_PATH="${CONFIG_PATH:-config/training_budget_sweep.yaml}"
RUN_PREFIX="${RUN_PREFIX:-training_budget_sweep}"
OUTPUT_DIR="results/${RUN_PREFIX}_${SLURM_ARRAY_JOB_ID}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
test -x "$PYTHON"
"$PYTHON" -m src.experiments.run_budget_sweep \
  --config "$CONFIG_PATH" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$OUTPUT_DIR"
