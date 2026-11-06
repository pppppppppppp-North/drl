#!/bin/bash
#SBATCH --job-name=drl_nested_screen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4-00:00:00
#SBATCH --partition=compute-long
#SBATCH --array=0-4%2
#SBATCH --account=25sfcs03
#SBATCH --output=logs/nested_screening_%A_%a.out
#SBATCH --error=logs/nested_screening_%A_%a.err

set -euo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
PYTHON="$ENV_DIR/bin/python"
CONFIG_PATH="${CONFIG_PATH:-config/nested_walk_forward_screening.yaml}"
OUTPUT_DIR="results/nested_walk_forward_screening_${SLURM_ARRAY_JOB_ID}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR" models
test -x "$PYTHON"
"$PYTHON" -m src.experiments.run_nested_screening \
  --config "$CONFIG_PATH" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$OUTPUT_DIR"
