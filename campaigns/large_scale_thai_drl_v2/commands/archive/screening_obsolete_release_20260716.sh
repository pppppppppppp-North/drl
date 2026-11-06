#!/bin/bash
set -euo pipefail

mkdir -p logs
SCREENING_SUBMISSION=$(sbatch --parsable --export=BISTKA_PREFLIGHT_SHA256=f41d894604333a753653e776df48119d8d6087e46b42734f8c54b17cefa458de,CAMPAIGN_ATTEMPT_ID=screening_20260716T150537Z_5529f328,CAMPAIGN_ATTEMPT_SHA256=842ed98c000a78b0243edf0f3948c4a7e236248332f5a569ab202af3598074a1,CAMPAIGN_ID=large_scale_thai_drl_v2,CAMPAIGN_STAGE_ID=screening,CONFIG_PATH=campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml,DEPLOYMENT_RELEASE_ID=4d5c4cbc49f18739879a215854ca97d919aa117870be95facf824f82e389f270,MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,PATH=/usr/local/bin:/usr/bin:/bin slurm/large_scale_screening.sbatch)
SCREENING_JOB_ID="${SCREENING_SUBMISSION%%;*}"
echo "run_dir=results/large_scale_screening_${SCREENING_JOB_ID}"
echo "stage=screening job_id=${SCREENING_JOB_ID}"
