#!/bin/bash
#SBATCH --job-name=drl_ppo_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH --partition=compute-devel
#SBATCH --account=25sfcs03
#SBATCH --output=logs/drl_ppo_dbg_%j.out
#SBATCH --error=logs/drl_ppo_dbg_%j.err

set -eo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
PYTHON="$ENV_DIR/bin/python"

cd "$PROJECT_DIR"
mkdir -p logs results/debug_ppo models/debug_ppo

test -x "$PYTHON"

set -u

"$PYTHON" -m src.train \
  --config config/debug.yaml \
  --algo ppo \
  --num-envs 1 \
  --seed 42 \
  --output-dir "results/debug_ppo/run_${SLURM_JOB_ID}"
