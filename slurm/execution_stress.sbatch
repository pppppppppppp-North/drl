#!/bin/bash
#SBATCH --job-name=drl_exec_stress
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=06:00:00
#SBATCH --partition=compute-devel
#SBATCH --account=25sfcs03
#SBATCH --output=logs/execution_stress_%j.out
#SBATCH --error=logs/execution_stress_%j.err

set -euo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
PYTHON="$ENV_DIR/bin/python"
CONFIG_PATH="${CONFIG_PATH:-config/execution_stress_real_ohlcv_market.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/execution_stress_${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
test -x "$PYTHON"
"$PYTHON" -m src.experiments.run_execution_stress --config "$CONFIG_PATH" --output-dir "$OUTPUT_DIR"
