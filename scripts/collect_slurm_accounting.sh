#!/bin/bash

set -euo pipefail

JOB_ID="${1:?Usage: collect_slurm_accounting.sh JOB_ID [OUTPUT_CSV]}"
OUTPUT_PATH="${2:-reports/slurm_accounting_${JOB_ID}.csv}"
mkdir -p "$(dirname "$OUTPUT_PATH")"
command -v sacct >/dev/null 2>&1

sacct \
  --jobs "$JOB_ID" \
  --parsable2 \
  --noheader \
  --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,TotalCPU,NodeList \
  >"$OUTPUT_PATH"

echo "$OUTPUT_PATH"
