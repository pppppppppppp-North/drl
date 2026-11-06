#!/bin/bash
#SBATCH --job-name=drl_vec_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --partition=compute-devel
#SBATCH --account=25sfcs03
#SBATCH --output=logs/vector_benchmark_%j.out
#SBATCH --error=logs/vector_benchmark_%j.err

set -euo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
PYTHON="/lustrefs/project/25sfcs03/envs/drl_env/bin/python"
OUTPUT="reports/vectorized_training_benchmark_${SLURM_JOB_ID}.csv"

cd "$PROJECT_DIR"
mkdir -p logs reports
test -x "$PYTHON"

"$PYTHON" -m scripts.benchmark_vectorized_training \
  --config "${CONFIG_PATH:?campaign controller must export CONFIG_PATH}" \
  --env-counts 1,2,4,8,16 \
  --timesteps 4096 \
  --repeats 3 \
  --backend subproc \
  --seed 42 \
  --output "$OUTPUT"
