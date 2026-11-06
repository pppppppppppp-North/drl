#!/bin/bash
#SBATCH --job-name=drl_robust_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --partition=compute-devel
#SBATCH --account=25sfcs03
#SBATCH --output=logs/robustness_aggregate_%j.out
#SBATCH --error=logs/robustness_aggregate_%j.err

set -euo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
PYTHON="$ENV_DIR/bin/python"
CONFIG_PATH="${CONFIG_PATH:-config/robustness_real_ohlcv_market.yaml}"
: "${RUN_DIR:?Set RUN_DIR to the completed robustness result directory}"

cd "$PROJECT_DIR"
mkdir -p logs "$RUN_DIR/aggregate"
test -x "$PYTHON"

"$PYTHON" -m src.experiments.aggregate_robustness \
  --config "$CONFIG_PATH" \
  --run-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/aggregate"
