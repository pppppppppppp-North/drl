#!/bin/bash
#SBATCH --job-name=drl_budget_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --partition=compute-devel
#SBATCH --account=25sfcs03
#SBATCH --output=logs/training_budget_aggregate_%j.out
#SBATCH --error=logs/training_budget_aggregate_%j.err

set -euo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
PYTHON="/lustrefs/project/25sfcs03/envs/drl_env/bin/python"
CONFIG_PATH="${CONFIG_PATH:-config/training_budget_sweep.yaml}"
: "${RUN_DIR:?Set RUN_DIR to the completed training-budget result directory}"

cd "$PROJECT_DIR"
test -x "$PYTHON"
"$PYTHON" -m src.experiments.aggregate_budget_sweep \
  --config "$CONFIG_PATH" \
  --run-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/aggregate"
