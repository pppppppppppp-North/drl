#!/bin/bash
# Run a mixed pool of walk-forward tasks from one list, at most WORKERS at once (milestone M6).
#
# Unlike scripts/run_walk_forward_reward.sh, every line names its own run folder, runner, config and
# timesteps, so experiments with different configs share one pool: one SLURM job
# (slurm/walk_forward_pool.sbatch) instead of one array element per task. Each line of the task file:
#
#   <run-dir> <runner module> <config> <timesteps> <fold year> <variant> <seed>
#
# Blank lines and lines starting with # are ignored. Order the lines longest task first. Validation
# years never pass each config's validation_end, so the 2023-2024 test years and the sealed holdouts
# are never read. Finished tasks are skipped and each task takes an atomic lock, so a re-run resumes.
#
# Usage: scripts/run_task_pool.sh <task-file> [workers]      (PYTHON=/path/to/python)
set -euo pipefail

if [[ "${1:-}" == "--task" ]]; then
  run_dir=$2 runner=$3 config=$4 timesteps=$5 y=$6 v=$7 s=$8
  out="$run_dir/fold_${y}/${v}_seed${s}"
  if [[ -f "$out/validation_equity_curves.csv" ]]; then echo "skip $run_dir $y $v $s"; exit 0; fi
  mkdir -p "$run_dir/locks" "$run_dir/logs"
  lock="$run_dir/locks/fold${y}_${v}_seed${s}"
  if ! mkdir "$lock" 2>/dev/null; then echo "busy $run_dir $y $v $s (remove $lock if stale)"; exit 0; fi
  trap 'rmdir "$lock" 2>/dev/null' EXIT
  if "$PYTHON" -c "import torch; torch.set_num_threads(1); import runpy, sys; sys.argv=['x','--config','$config','--output-dir','$out','--variants','$v','--seeds','$s','--fold-year','$y','--timesteps','$timesteps']; runpy.run_module('$runner', run_name='__main__')" \
      > "$run_dir/logs/fold${y}_${v}_seed${s}.log" 2>&1; then
    # A runner given an unknown variant still writes the baselines' curves and exits cleanly, so
    # success needs the trained policy (<variant>_seed<seed>) in the validation curves.
    if grep -q ",${v}_seed${s}," "$out/validation_equity_curves.csv" 2>/dev/null; then
      echo "$(date '+%F %T') done $run_dir $y $v $s"
    else
      echo "$(date '+%F %T') FAILED $run_dir $y $v $s (no ${v}_seed${s} curve written)"
      mv "$out" "$out.failed_$(date +%Y%m%d%H%M%S)" 2>/dev/null || true
    fi
  else
    echo "$(date '+%F %T') FAILED $run_dir $y $v $s"
  fi
  exit 0
fi

TASKS=${1:?task file required}
WORKERS=${2:-4}
PYTHON=${PYTHON:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHON
echo "$(grep -cvE '^[[:space:]]*(#|$)' "$TASKS") tasks, $WORKERS workers"
grep -vE '^[[:space:]]*(#|$)' "$TASKS" | xargs -P "$WORKERS" -L 1 bash "$0" --task
echo "all tasks finished"
