#!/bin/bash
set -euo pipefail

mkdir -p logs
VECTOR_BENCHMARK_SUBMISSION=$(sbatch --parsable --export=BISTKA_PREFLIGHT_SHA256=6c3b217f1c55240e85eddeb8d8cb1e659c89d8a7632004fc95a3294c83142aa3,CAMPAIGN_ATTEMPT_ID=vector_benchmark_20260715T161221Z_8ac0554b,CAMPAIGN_ATTEMPT_SHA256=1e3fecdf3e5eeb590f923ce0895b9cffd75195e0298bcd987fcf5025f0b64dd1,CAMPAIGN_ID=large_scale_thai_drl,CAMPAIGN_STAGE_ID=vector_benchmark,CONFIG_PATH=config/large_scale_screening.yaml,DEPLOYMENT_RELEASE_ID=6d66448341c780c5b90a2d2d54cf2b4d4bebd972a18885aad727a05ff8236aaa,MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,PATH=/usr/local/bin:/usr/bin:/bin slurm/benchmark_vectorized_training.sbatch)
VECTOR_BENCHMARK_JOB_ID="${VECTOR_BENCHMARK_SUBMISSION%%;*}"
echo "stage=vector_benchmark job_id=${VECTOR_BENCHMARK_JOB_ID}"
