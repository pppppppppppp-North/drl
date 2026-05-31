#!/bin/bash
#SBATCH --job-name=drl_ppo_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
#SBATCH --partition=compute-normal
#SBATCH --account=25sfcs03
#SBATCH --output=logs/drl_ppo_cpu_%j.out
#SBATCH --error=logs/drl_ppo_cpu_%j.err

set -eo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
PYTHON="$ENV_DIR/bin/python"
CONFIG_PATH="${CONFIG_PATH:-config/ppo_technical.yaml}"
ALGO="${ALGO:-ppo}"
RUN_PREFIX="${RUN_PREFIX:-run}"
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-$SLURM_CPUS_PER_TASK}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${RUN_PREFIX}_${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
mkdir -p logs results models

test -x "$PYTHON"

set -u

"$PYTHON" -m src.train \
  --config "$CONFIG_PATH" \
  --algo "$ALGO" \
  --num-envs "$NUM_ENVS" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"
