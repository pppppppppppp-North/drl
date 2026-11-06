#!/bin/bash

set -euo pipefail
umask 077

STAMP="${1:?usage: bash capture_screening_execution_window_evidence.sh YYYYMMDDTHHMMSSZ}"
OUTPUT_DIR="${2:-/tmp}"
PROJECT_ROOT="/lustrefs/project/25sfcs03/drl_thai_stock"
PYTHON="/lustrefs/project/25sfcs03/envs/drl_env/bin/python"
ACCOUNT="25sfcs03"

[[ "$STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || {
  echo "timestamp must use YYYYMMDDTHHMMSSZ" >&2
  exit 2
}
[[ "$OUTPUT_DIR" == "/tmp" ]] || {
  echo "operational evidence must be staged under /tmp" >&2
  exit 2
}
[[ "$(hostname)" == "bistka-master" ]] || {
  echo "capture must run on bistka-master" >&2
  exit 2
}
[[ "${USER:-}" == "s00658" ]] || {
  echo "capture must run as s00658" >&2
  exit 2
}
[[ -x "$PYTHON" ]] || {
  echo "locked project Python is unavailable" >&2
  exit 2
}

SCHEDULER="$OUTPUT_DIR/scheduler_snapshot_${STAMP}.txt"
QUOTA="$OUTPUT_DIR/quota_snapshot_${STAMP}.json"
LIVE_PREFLIGHT="$OUTPUT_DIR/live_preflight_${STAMP}.json"

for path in \
  "$SCHEDULER" "$SCHEDULER.sha256" \
  "$QUOTA" "$QUOTA.sha256" \
  "$LIVE_PREFLIGHT" "$LIVE_PREFLIGHT.sha256"; do
  [[ ! -e "$path" ]] || {
    echo "refusing to overwrite existing evidence: $path" >&2
    exit 2
  }
done

cd "$PROJECT_ROOT"

bash scripts/capture_bistka_capacity.sh "$SCHEDULER" >/dev/null
{
  echo
  echo "## squeue -A $ACCOUNT"
  squeue -A "$ACCOUNT"
  echo
  echo "## deployed release identity"
  printf 'deployed_bundle_sha256='
  tr -d '\r\n' < .bistka/deployed_bundle.sha256
  echo
  printf 'deployment_manifest_sha256='
  sha256sum .bistka/deployment_manifest.json | cut -d ' ' -f 1
} >> "$SCHEDULER"
sha256sum "$SCHEDULER" | cut -d ' ' -f 1 > "$SCHEDULER.sha256"

"$PYTHON" -m scripts.capture_screening_quota_snapshot \
  --filesystem /lustrefs/project/25sfcs03 \
  --output-parent "$PROJECT_ROOT/results" \
  --group s00658 \
  --gid 1039 \
  --project-id 25110303 \
  --output "$QUOTA" >/dev/null

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m scripts.check_bistka_calibration_preflight \
  --project-root "$PROJECT_ROOT" \
  --manifest campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_storage_safe_cutover_v2_manifest.json \
  --output "$LIVE_PREFLIGHT" >/dev/null
sha256sum "$LIVE_PREFLIGHT" | cut -d ' ' -f 1 > "$LIVE_PREFLIGHT.sha256"

printf 'scheduler=%s\nquota=%s\nlive_preflight=%s\n' \
  "$SCHEDULER" "$QUOTA" "$LIVE_PREFLIGHT"
