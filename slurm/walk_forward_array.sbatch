#!/bin/bash
#SBATCH --job-name=drl_walk_forward
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --partition=compute-normal
#SBATCH --array=0-1
#SBATCH --account=25sfcs03
#SBATCH --output=logs/walk_forward_%A_%a.out
#SBATCH --error=logs/walk_forward_%A_%a.err

set -eo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
PYTHON="$ENV_DIR/bin/python"
CONFIG_PATH="${CONFIG_PATH:-config/real_ohlcv.yaml}"
RUN_PREFIX="${RUN_PREFIX:-walk_forward_real_ohlcv}"
ALGO="${ALGO:-ppo}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}"
SEED="${SEED:-42}"

cd "$PROJECT_DIR"
mkdir -p logs "results/${RUN_PREFIX}_${SLURM_ARRAY_JOB_ID}" models

test -x "$PYTHON"

set -u

ARGS=(
  -m src.experiments.run_walk_forward
  --window-id "$SLURM_ARRAY_TASK_ID"
  --config "$CONFIG_PATH"
  --algo "$ALGO"
  --seed "$SEED"
  --output-dir "results/${RUN_PREFIX}_${SLURM_ARRAY_JOB_ID}/window_${SLURM_ARRAY_TASK_ID}"
)

if [[ -n "$TOTAL_TIMESTEPS" ]]; then
  ARGS+=(--total-timesteps "$TOTAL_TIMESTEPS")
fi

"$PYTHON" "${ARGS[@]}"
