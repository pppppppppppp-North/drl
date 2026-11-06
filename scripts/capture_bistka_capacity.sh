#!/bin/bash

set -uo pipefail

OUTPUT_PATH="${1:-reports/bistka_capacity_$(date +%Y%m%d_%H%M%S).txt}"
mkdir -p "$(dirname "$OUTPUT_PATH")"

run_if_available() {
  local command_name="$1"
  shift
  echo
  echo "## ${command_name} $*"
  if command -v "$command_name" >/dev/null 2>&1; then
    "$command_name" "$@" 2>&1
  else
    echo "unavailable"
  fi
}

{
  echo "# BistKA Capacity Snapshot"
  echo "captured_at=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "hostname=$(hostname)"
  echo "user=${USER:-unknown}"
  run_if_available mycredit
  run_if_available myquota
  run_if_available sinfo
  run_if_available squeue -u "${USER:-}"
  run_if_available sacctmgr -nP show assoc where user="${USER:-}" account="${BISTKA_ACCOUNT:-25sfcs03}" format=User,Account,Partition,QOS
  run_if_available sacctmgr -nP show qos where name="${BISTKA_ACCOUNT:-25sfcs03}" format=Name,GrpJobs,GrpSubmitJobs,GrpTRES,Flags
  run_if_available scontrol show partition -o
  echo
  echo "## live scheduler policy"
  if command -v sacctmgr >/dev/null 2>&1; then
    policy_row="$(sacctmgr -nP show qos where name="${BISTKA_ACCOUNT:-25sfcs03}" format=Name,GrpJobs,GrpSubmitJobs,GrpTRES 2>/dev/null | head -n 1)"
    IFS='|' read -r _policy_name policy_running policy_submitted policy_tres <<<"$policy_row"
    policy_cpus="$(sed -n 's/.*cpu=\([0-9][0-9]*\).*/\1/p' <<<"$policy_tres")"
    policy_gpus="$(sed -n 's/.*gres\/gpu=\([0-9][0-9]*\).*/\1/p' <<<"$policy_tres")"
    echo "Maximum Job Submission: ${policy_submitted:-unavailable} jobs/group"
    echo "Maximum Job Running: ${policy_running:-unavailable} jobs/group"
    echo "Maximum CPU-cores = ${policy_cpus:-unavailable} CPU-cores/group"
    echo "Maximum GPUs = ${policy_gpus:-unavailable} GPUs/group"
  else
    echo "unavailable"
  fi
  if command -v scontrol >/dev/null 2>&1; then
    for partition in compute-devel compute-normal compute-long; do
      partition_row="$(scontrol show partition "$partition" -o 2>/dev/null || true)"
      max_time="$(sed -n 's/.* MaxTime=\([^ ]*\).*/\1/p' <<<"$partition_row")"
      case "$max_time" in
        06:00:00) echo "$partition: 6 hours" ;;
        2-00:00:00) echo "$partition: 2 days" ;;
        4-00:00:00) echo "$partition: 4 days" ;;
        *) echo "$partition: ${max_time:-unavailable}" ;;
      esac
    done
  fi
  echo
  echo "## global scheduler configuration"
  if command -v scontrol >/dev/null 2>&1; then
    scontrol show config 2>&1 | grep -E 'MaxArraySize|MaxJobCount|MaxStepCount|SchedulerType|SelectType|DefMemPerCPU|DefMemPerNode' || true
  else
    echo "unavailable"
  fi
  echo
  echo "## loaded modules"
  module list 2>&1 || true
  echo
  echo "## Python modules"
  ml av Python 2>&1 || true
  echo
  echo "## Miniforge modules"
  ml av Miniforge3 2>&1 || true
  echo
  echo "## CUDA modules"
  ml spider cuda 2>&1 || true
} >"$OUTPUT_PATH"

echo "$OUTPUT_PATH"
