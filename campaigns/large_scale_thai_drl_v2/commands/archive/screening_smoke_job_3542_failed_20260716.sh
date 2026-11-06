#!/bin/bash
set -euo pipefail

# Historical command for rejected BistKA job 3542.
# Preserved for incident audit only; do not resubmit.
mkdir -p logs
SCREENING_SMOKE_SUBMISSION=$(sbatch --parsable --export=BISTKA_PREFLIGHT_SHA256=99ff875694ff47b6a02541718b6af93699de1f789ef35effaefe24d2e9655951,CAMPAIGN_ATTEMPT_ID=screening_smoke_20260716T160537Z_132d2047,CAMPAIGN_ATTEMPT_SHA256=70fddb26a0d3d21e9fdade74291352905f093a2721c34f474eed3bca52bda944,CAMPAIGN_ID=large_scale_thai_drl_v2,CAMPAIGN_STAGE_ID=screening_smoke,CONFIG_PATH=campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml,DEPLOYMENT_RELEASE_ID=ef89ff5a738c5b25460ae40ad7789239134d761ce511a36665f857e01e3d74b4,MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,PATH=/usr/local/bin:/usr/bin:/bin slurm/screening_smoke.sbatch)
SCREENING_SMOKE_JOB_ID="${SCREENING_SMOKE_SUBMISSION%%;*}"
echo "run_dir=results/screening_smoke_${SCREENING_SMOKE_JOB_ID}"
echo "stage=screening_smoke job_id=${SCREENING_SMOKE_JOB_ID}"
