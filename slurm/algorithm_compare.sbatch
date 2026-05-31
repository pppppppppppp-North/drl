#!/bin/bash
#SBATCH --job-name=drl_algo_cmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=1-00:00:00
#SBATCH --partition=compute-normal
#SBATCH --account=25sfcs03
#SBATCH --output=logs/drl_algo_cmp_%j.out
#SBATCH --error=logs/drl_algo_cmp_%j.err

set -eo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
PYTHON="$ENV_DIR/bin/python"
CONFIG_PATH="${CONFIG_PATH:-config/ppo_technical.yaml}"
RUN_PREFIX="${RUN_PREFIX:-algo_compare}"
OUTPUT_DIR="results/${RUN_PREFIX}_${SLURM_JOB_ID}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR" models

test -x "$PYTHON"

set -u

for algo in ppo a2c; do
  "$PYTHON" -m src.train \
    --config "$CONFIG_PATH" \
    --algo "$algo" \
    --num-envs "$SLURM_CPUS_PER_TASK" \
    --seed 42 \
    --output-dir "$OUTPUT_DIR"
done

"$PYTHON" -m src.agents.baselines \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR"

"$PYTHON" -m src.evaluation.compare_results \
  --run-dir "$OUTPUT_DIR"

"$PYTHON" -m src.evaluation.plot_results \
  --run-dir "$OUTPUT_DIR"
