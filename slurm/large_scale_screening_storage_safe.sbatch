#!/bin/bash
#SBATCH --job-name=drl_screen_safe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=4-00:00:00
#SBATCH --partition=compute-long
#SBATCH --array=0-4%2
#SBATCH --account=25sfcs03
#SBATCH --requeue
#SBATCH --output=logs/screening_safe_%A_%a.out
#SBATCH --error=logs/screening_safe_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
PYTHON="$ENV_DIR/bin/python"
LOCKED_CONFIG_PATH="campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH must bind the exact locked screening config}"
RUN_PREFIX="${RUN_PREFIX:-large_scale_screening}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${RUN_PREFIX}_${SLURM_ARRAY_JOB_ID}}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR" models
test -x "$PYTHON"
test "$CONFIG_PATH" = "$LOCKED_CONFIG_PATH"
test -f "$CONFIG_PATH"

"$PYTHON" -m src.experiments.run_screening_storage_safe \
  --config "$CONFIG_PATH" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$OUTPUT_DIR"
