#!/bin/bash
#SBATCH --job-name=drl_robustness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
#SBATCH --partition=compute-normal
#SBATCH --array=0-3%2
#SBATCH --account=25sfcs03
#SBATCH --output=logs/robustness_%A_%a.out
#SBATCH --error=logs/robustness_%A_%a.err

set -euo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
PYTHON="$ENV_DIR/bin/python"
CONFIG_PATH="${CONFIG_PATH:-config/robustness_real_ohlcv_market.yaml}"
RUN_PREFIX="${RUN_PREFIX:-robustness_real_ohlcv_market}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}"
OUTPUT_DIR="results/${RUN_PREFIX}_${SLURM_ARRAY_JOB_ID}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR" models
test -x "$PYTHON"

ARGS=(
  -m src.experiments.run_robustness
  --config "$CONFIG_PATH"
  --task-id "$SLURM_ARRAY_TASK_ID"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$TOTAL_TIMESTEPS" ]]; then
  ARGS+=(--total-timesteps "$TOTAL_TIMESTEPS")
fi

"$PYTHON" "${ARGS[@]}"
