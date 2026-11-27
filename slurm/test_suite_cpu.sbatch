#!/bin/bash
#SBATCH --job-name=drl_tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --partition=compute-devel
#SBATCH --account=25sfcs03
#SBATCH --output=logs/tests_%j.out
#SBATCH --error=logs/tests_%j.err
#
# The repository's test suite on BistKA (milestone M6): checks that the code runs in the cluster's Python
# environment before any training job. It trains no model and reads neither the 2023-2024 test years nor
# the sealed holdouts. Submit only with explicit operator intent and record the job in PROGRESS_CHECKLIST.md.
# The logs/ folder must exist before submission:
#   cd /lustrefs/project/25sfcs03/drl_thai_stock_20261122 && mkdir -p logs
#   sbatch --export=ALL,PROJECT_DIR=$PWD slurm/test_suite_cpu.sbatch
# TESTS selects test files instead of the whole suite, for example:
#   export TESTS="tests/test_pit_heuristic_baselines.py tests/test_pit_lstm_forecast_baseline.py"
# The pytest report is written to logs/tests_<job id>.txt; the job ends with "pytest exit <code>".

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lustrefs/project/25sfcs03/drl_thai_stock}"
ENV_DIR="${ENV_DIR:-/lustrefs/project/25sfcs03/envs/drl_env}"
PYTHON="${PYTHON:-$ENV_DIR/bin/python}"
TESTS="${TESTS:-tests/}"

cd "$PROJECT_DIR"
test -x "$PYTHON"
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg

REPORT="logs/tests_${SLURM_JOB_ID:-local}.txt"
echo "host $(hostname) start $(date -Is) project $PROJECT_DIR"
"$PYTHON" --version
"$PYTHON" -m pip list 2>/dev/null | grep -i -E "^(torch|stable.baselines3|gymnasium|numpy|pandas|scipy|pytest) " || true

status=0
# A test module that imports a package missing from the cluster environment (scipy, pdfplumber) is
# reported as a collection error instead of stopping the whole run.
"$PYTHON" -m pytest -p no:cacheprovider -q -rf --continue-on-collection-errors $TESTS > "$REPORT" 2>&1 || status=$?
tail -n 3 "$REPORT"
echo "end $(date -Is) pytest exit $status"
