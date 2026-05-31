#!/bin/bash
#SBATCH --job-name=drl_ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --partition=compute-normal
#SBATCH --array=0-2
#SBATCH --account=25sfcs03
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err

set -eo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
PYTHON="$ENV_DIR/bin/python"
CONFIGS="${CONFIGS:-config/ablation_returns_real_ohlcv.yaml,config/ablation_technical_real_ohlcv.yaml,config/ablation_technical_context_real_ohlcv.yaml}"
RUN_PREFIX="${RUN_PREFIX:-ablation_real_ohlcv}"
ALGO="${ALGO:-ppo}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}"
SEED="${SEED:-42}"

cd "$PROJECT_DIR"
mkdir -p logs "results/${RUN_PREFIX}_${SLURM_ARRAY_JOB_ID}" models

test -x "$PYTHON"

set -u

IFS=',' read -r -a CONFIG_ARRAY <<< "$CONFIGS"
CONFIG_PATH="${CONFIG_ARRAY[$SLURM_ARRAY_TASK_ID]}"
GROUP_NAME="$(basename "$CONFIG_PATH" .yaml)"
OUTPUT_DIR="results/${RUN_PREFIX}_${SLURM_ARRAY_JOB_ID}/${GROUP_NAME}"

ARGS=(
  -m src.train
  --config "$CONFIG_PATH"
  --algo "$ALGO"
  --num-envs "$SLURM_CPUS_PER_TASK"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$TOTAL_TIMESTEPS" ]]; then
  ARGS+=(--total-timesteps "$TOTAL_TIMESTEPS")
fi

"$PYTHON" "${ARGS[@]}"

"$PYTHON" -m src.agents.baselines \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR"

"$PYTHON" -m src.evaluation.compare_results \
  --run-dir "$OUTPUT_DIR"

"$PYTHON" -m src.evaluation.plot_results \
  --run-dir "$OUTPUT_DIR"
