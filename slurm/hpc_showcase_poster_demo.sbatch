#!/bin/bash
#SBATCH --job-name=hpcshowcase_drl
#SBATCH --partition=compute-devel
#SBATCH --account=25sfcs03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:05:00
#SBATCH --output=logs/hpc_showcase_demo_%j.out
#SBATCH --error=logs/hpc_showcase_demo_%j.err

set -euo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
PYTHON="/lustrefs/project/25sfcs03/envs/drl_env/bin/python"
OUTPUT_DIR="${PROJECT_DIR}/output/poster_assets/hpc_job_${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

"${PYTHON}" scripts/build_hpc_showcase_equity_demo.py \
  --project-root "${PROJECT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --capture-seconds 90
