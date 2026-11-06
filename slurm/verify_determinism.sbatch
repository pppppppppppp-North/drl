#!/bin/bash
#SBATCH --job-name=drl_determinism
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:15:00
#SBATCH --partition=compute-devel
#SBATCH --account=25sfcs03
#SBATCH --output=logs/determinism_%j.out
#SBATCH --error=logs/determinism_%j.err

set -euo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
PYTHON="/lustrefs/project/25sfcs03/envs/drl_env/bin/python"

cd "$PROJECT_DIR"
mkdir -p logs reports
test -x "$PYTHON"
"$PYTHON" -m scripts.verify_determinism \
  --config "${CONFIG_PATH:?campaign controller must export CONFIG_PATH}" \
  --seed 42 \
  --output "reports/determinism_${SLURM_JOB_ID}.json"
