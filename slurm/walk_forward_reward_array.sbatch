#!/bin/bash
#SBATCH --job-name=drl_wf_reward
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --partition=compute-normal
#SBATCH --account=25sfcs03
#SBATCH --output=logs/wf_reward_%A_%a.out
#SBATCH --error=logs/wf_reward_%A_%a.err
#
# Exploratory walk-forward experiments on BistKA (milestones M3/M6): one array element trains and
# evaluates one validation year x variant x seed with scripts.run_reward_variant_experiment. This is
# NOT part of the gated V2 research campaign; submit only with explicit operator intent and
# record the job in PROGRESS_CHECKLIST.md. Validation years never pass the config's validation_end,
# so the 2023-2024 test years and the sealed holdouts are never read.
#
# Tasks are enumerated in the same order as scripts/run_walk_forward_reward.sh (variant, year, seed).
# Example (7 years x 2 variants x 5 seeds = 70 tasks, at most 16 at once):
#   export CONFIG=config/long_term_reward_experiment.yaml VARIANTS="plan_main_weekly_lstm log_weekly"
#   export YEARS="2016 2017 2018 2019 2020 2021 2022" SEEDS="42 43 44 45 46" TIMESTEPS=500000
#   export RUN_NAME=wf_reward_bistka
#   sbatch --export=ALL --array=0-69%16 slurm/walk_forward_reward_array.sbatch
# Then aggregate (on a login node it is light; or locally after fetching results/):
#   python -m scripts.aggregate_walk_forward_reward --run-dir results/$RUN_NAME

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lustrefs/project/25sfcs03/drl_thai_stock}"
ENV_DIR="${ENV_DIR:-/lustrefs/project/25sfcs03/envs/drl_env}"
PYTHON="${PYTHON:-$ENV_DIR/bin/python}"
CONFIG="${CONFIG:-config/long_term_reward_experiment.yaml}"
VARIANTS="${VARIANTS:-plan_main_weekly_lstm log_weekly}"
YEARS="${YEARS:-2016 2017 2018 2019 2020 2021 2022}"
SEEDS="${SEEDS:-42 43 44 45 46}"
TIMESTEPS="${TIMESTEPS:-100000}"
RUN_NAME="${RUN_NAME:-wf_reward_${SLURM_ARRAY_JOB_ID:-local}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?run as a SLURM array job (or set SLURM_ARRAY_TASK_ID for a local test)}"

cd "$PROJECT_DIR"
test -x "$PYTHON"
set -u

TASKS=()
for v in $VARIANTS; do for y in $YEARS; do for s in $SEEDS; do TASKS+=("$y $v $s"); done; done; done
if (( TASK_ID >= ${#TASKS[@]} )); then
  echo "array index $TASK_ID is beyond the ${#TASKS[@]} tasks; nothing to do"
  exit 0
fi
read -r YEAR VARIANT SEED <<< "${TASKS[$TASK_ID]}"
RUN_DIR="results/${RUN_NAME}"
OUT="${RUN_DIR}/fold_${YEAR}/${VARIANT}_seed${SEED}"
mkdir -p logs "${RUN_DIR}/logs"
if [[ -f "${OUT}/validation_equity_curves.csv" ]]; then
  echo "task ${TASK_ID} (${YEAR} ${VARIANT} ${SEED}) already finished; skipping"
  exit 0
fi
echo "task ${TASK_ID}/${#TASKS[@]}: year=${YEAR} variant=${VARIANT} seed=${SEED} timesteps=${TIMESTEPS} -> ${OUT}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/pycache_${USER:-drl}}"
"$PYTHON" -c "import torch; torch.set_num_threads(1); import runpy, sys; sys.argv=['x','--config','${CONFIG}','--output-dir','${OUT}','--variants','${VARIANT}','--seeds','${SEED}','--fold-year','${YEAR}','--timesteps','${TIMESTEPS}']; runpy.run_module('scripts.run_reward_variant_experiment', run_name='__main__')"
