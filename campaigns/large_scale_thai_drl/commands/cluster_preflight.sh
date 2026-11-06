#!/bin/bash
set -euo pipefail

MANIFEST=campaigns/large_scale_thai_drl/deployment/bistka_deployment_manifest.json BUNDLE=campaigns/large_scale_thai_drl/deployment/bistka_release.tar.gz PREFLIGHT_OUTPUT=reports/bistka_calibration_preflight.json bash scripts/stage_bistka_deployment.sh --dry-run
