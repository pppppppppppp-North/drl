#!/bin/bash
set -euo pipefail

python -m scripts.materialize_calibrated_screening --project-root . --campaign-dir campaigns/large_scale_thai_drl --base-config config/large_scale_screening.yaml --output campaigns/large_scale_thai_drl/derived/locked_large_scale_screening.yaml
