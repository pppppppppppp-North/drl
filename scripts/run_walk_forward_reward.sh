#!/bin/zsh
# Walk-forward reward experiment (milestone M6 / RQ5): one process per fold x variant x seed,
# run through `xargs -P` so at most WORKERS processes run at once (shell job control is not
# available in detached, non-interactive shells). Validation years never exceed the config's
# validation_end (2022), so the 2023-2024 test years and the sealed 2025+ holdout are never read.
# Finished tasks are skipped, so the script can be re-run to resume after an interruption, and a
# per-task lock lets several launchers share one run directory without duplicate work.
#
# Usage: scripts/run_walk_forward_reward.sh <run-dir> [workers] [timesteps]
#   PYTHON=/path/to/python scripts/run_walk_forward_reward.sh results/walk_forward_reward_<date> 5 100000
set -euo pipefail

if [[ "${1:-}" == "--task" ]]; then
  y=$2 v=$3 s=$4
  out="$RUN_DIR/fold_${y}/${v}_seed${s}"
  if [[ -f "$out/validation_equity_curves.csv" ]]; then echo "skip $y $v $s"; exit 0; fi
  # Atomic per-task lock so concurrent launchers on the same run directory never train the same task.
  lock="$RUN_DIR/locks/fold${y}_${v}_seed${s}"
  mkdir -p "$RUN_DIR/locks"
  if ! mkdir "$lock" 2>/dev/null; then echo "busy $y $v $s (locked by another launcher; remove $lock if stale)"; exit 0; fi
  trap 'rmdir "$lock" 2>/dev/null' EXIT
  if "$PYTHON" -c "import torch; torch.set_num_threads(1); import runpy, sys; sys.argv=['x','--config','$CONFIG','--output-dir','$out','--variants','$v','--seeds','$s','--fold-year','$y','--timesteps','$TIMESTEPS']; runpy.run_module('scripts.run_reward_variant_experiment', run_name='__main__')" \
      > "$RUN_DIR/logs/fold${y}_${v}_seed${s}.log" 2>&1; then
    echo "done $y $v $s"
  else
    echo "FAILED $y $v $s"
  fi
  exit 0
fi

RUN_DIR=${1:?run directory required}
WORKERS=${2:-4}
TIMESTEPS=${3:-100000}
PYTHON=${PYTHON:-python}
CONFIG=${CONFIG:-config/long_term_reward_experiment.yaml}
YEARS=(${=YEARS:-2016 2017 2018 2019 2020 2021 2022})
VARIANTS=(${=VARIANTS:-plan_main_weekly_lstm log_weekly})
SEEDS=(${=SEEDS:-42 43 44 45 46})

mkdir -p "$RUN_DIR/logs"
tasks="$RUN_DIR/tasks.txt"
: > "$tasks"
# LSTM tasks first so the slowest work starts early.
for v in "${VARIANTS[@]}"; do for y in "${YEARS[@]}"; do for s in "${SEEDS[@]}"; do
  echo "$y $v $s" >> "$tasks"
done; done; done
echo "$(wc -l < "$tasks" | tr -d ' ') tasks, $WORKERS workers, $TIMESTEPS timesteps each"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RUN_DIR CONFIG PYTHON TIMESTEPS
xargs -P "$WORKERS" -L 1 zsh "$0" --task < "$tasks"
echo "all tasks finished"
