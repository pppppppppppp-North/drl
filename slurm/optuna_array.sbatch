#!/bin/bash
#SBATCH --job-name=drl_optuna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --partition=compute-normal
#SBATCH --array=0-19
#SBATCH --account=25sfcs03
#SBATCH --output=logs/optuna_%A_%a.out
#SBATCH --error=logs/optuna_%A_%a.err

set -eo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
PYTHON="$ENV_DIR/bin/python"
CONFIG_PATH="${CONFIG_PATH:-config/optuna_search.yaml}"
RUN_PREFIX="${RUN_PREFIX:-optuna_real_ohlcv}"
ALGORITHMS="${ALGORITHMS:-ppo,a2c}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}"
OBJECTIVE_METRIC="${OBJECTIVE_METRIC:-sharpe}"

cd "$PROJECT_DIR"
mkdir -p logs "results/${RUN_PREFIX}_${SLURM_ARRAY_JOB_ID}" models

test -x "$PYTHON"

set -u

ARGS=(
  -m src.experiments.run_trial
  --trial-id "$SLURM_ARRAY_TASK_ID"
  --config "$CONFIG_PATH"
  --algorithms "$ALGORITHMS"
  --objective-metric "$OBJECTIVE_METRIC"
  --output-dir "results/${RUN_PREFIX}_${SLURM_ARRAY_JOB_ID}/trial_${SLURM_ARRAY_TASK_ID}"
)

if [[ -n "$TOTAL_TIMESTEPS" ]]; then
  ARGS+=(--total-timesteps "$TOTAL_TIMESTEPS")
fi

"$PYTHON" "${ARGS[@]}"
