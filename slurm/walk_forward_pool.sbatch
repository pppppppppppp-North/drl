#!/bin/bash
#SBATCH --job-name=drl_pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --partition=compute-long
#SBATCH --account=25sfcs03
#SBATCH --output=logs/pool_%j.out
#SBATCH --error=logs/pool_%j.err
#
# One job for a whole pool of walk-forward tasks (milestone M6): scripts/run_task_pool.sh runs every line
# of POOL with one worker per allocated CPU, instead of one array element per task. Exploratory, not part
# of the gated V2 campaign. Submit only with explicit operator intent and record the job in
# PROGRESS_CHECKLIST.md:
#   cd /lustrefs/project/25sfcs03/drl_thai_stock_20261122 && mkdir -p logs
#   sbatch --export=ALL,PROJECT_DIR=$PWD,POOL=slurm/pools/bistka_20261122.tasks slurm/walk_forward_pool.sbatch
#
# Before the pool, a smoke stage trains one task of every (runner, config, variant) in the pool for
# SMOKE_TIMESTEPS steps (default 3000; 0 skips it) into results/<pool>_smoke_<job id>/ and stops the job
# if any fails. Finished tasks are skipped, so resubmitting the same POOL resumes it.
#
# A helper job can share a running pool: give it the same tasks in reverse order with KEEP_LOCKS=1 (the
# running job's locks are live, not stale) and SMOKE_TIMESTEPS=0. Each task's atomic lock keeps the two
# jobs from training the same task; each job skips the tasks the other has finished or holds. It only
# helps if the project's QOS has CPUs to spare: QOS 25sfcs03 allows 32 CPUs at once across all its jobs,
# so a helper beside a 32-CPU pool waits (QOSGrpCpuLimit) until the pool ends (job 4411, 2026-11-23).
#   sbatch --mem=96G --export=ALL,PROJECT_DIR=$PWD,POOL=slurm/pools/bistka_20261122_reverse.tasks,KEEP_LOCKS=1,SMOKE_TIMESTEPS=0 slurm/walk_forward_pool.sbatch

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lustrefs/project/25sfcs03/drl_thai_stock}"
ENV_DIR="${ENV_DIR:-/lustrefs/project/25sfcs03/envs/drl_env}"
PYTHON="${PYTHON:-$ENV_DIR/bin/python}"
POOL="${POOL:?set POOL to a task file}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
SMOKE_TIMESTEPS="${SMOKE_TIMESTEPS:-3000}"
KEEP_LOCKS="${KEEP_LOCKS:-0}"

cd "$PROJECT_DIR"
test -x "$PYTHON"
test -f "$POOL"
set -u
export PYTHON OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/pycache_${USER:-drl}_${SLURM_JOB_ID:-local}"
name=$(basename "$POOL" .tasks)
lines() { grep -vE '^[[:space:]]*(#|$)' "$1"; }

echo "host $(hostname) start $(date -Is) pool $POOL: $(lines "$POOL" | wc -l) tasks, $WORKERS workers"
"$PYTHON" --version
# With one pool job at a time, locks left by a killed job are stale. A helper job (KEEP_LOCKS=1) keeps them.
if [[ "$KEEP_LOCKS" != "1" ]]; then
  for d in $(lines "$POOL" | awk '{print $1}' | sort -u); do rm -rf "$d/locks"; done
fi

if [[ "$SMOKE_TIMESTEPS" != "0" ]]; then
  smoke="results/${name}_smoke_${SLURM_JOB_ID:-local}"
  mkdir -p "$smoke"
  lines "$POOL" | awk -v d="$smoke" -v t="$SMOKE_TIMESTEPS" '!seen[$2" "$3" "$6]++ {$1=d; $4=t; print}' > "$smoke/tasks.txt"
  echo "smoke stage: $(wc -l < "$smoke/tasks.txt") tasks at $SMOKE_TIMESTEPS timesteps"
  bash scripts/run_task_pool.sh "$smoke/tasks.txt" "$WORKERS" | tee "$smoke/pool.log"
  if grep -q FAILED "$smoke/pool.log"; then
    echo "smoke stage failed; the pool was not started (see $smoke/logs/)"
    exit 1
  fi
fi

bash scripts/run_task_pool.sh "$POOL" "$WORKERS"
echo "end $(date -Is)"
