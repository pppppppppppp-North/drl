#!/bin/bash
#SBATCH --job-name=drl_safe_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --partition=compute-devel
#SBATCH --account=25sfcs03
#SBATCH --output=logs/screening_safe_smoke_%j.out
#SBATCH --error=logs/screening_safe_smoke_%j.err

set -euo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
PYTHON="$ENV_DIR/bin/python"
LOCKED_CONFIG_PATH="campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH must bind the exact locked screening config}"
RUN_PREFIX="${RUN_PREFIX:-screening_storage_safe_smoke}"
OUTPUT_DIR="results/${RUN_PREFIX}_${SLURM_JOB_ID}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR" models
test -x "$PYTHON"
test "$CONFIG_PATH" = "$LOCKED_CONFIG_PATH"
test -f "$CONFIG_PATH"

"$PYTHON" -m src.experiments.run_screening_storage_safe_smoke \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --timesteps 2048
