#!/bin/bash

set -euo pipefail

MODE="${1:---dry-run}"
LOCAL_PYTHON="${LOCAL_PYTHON:-.venv/bin/python}"
STAGE_SCRIPT="scripts/stage_bistka_deployment_v2.sh"
CANDIDATE_MANIFEST="campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_storage_safe_cutover_v2_manifest.json"
CANDIDATE_BUNDLE="campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_storage_safe_cutover_v2.tar.gz"
CANDIDATE_PREFLIGHT="reports/bistka_screening_storage_safe_cutover_v2_preflight.json"
ROLLBACK_MANIFEST="campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2_manifest.json"
ROLLBACK_BUNDLE="campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2.tar.gz"
ROLLBACK_PREFLIGHT="reports/bistka_screening_v2_rollback_preflight.json"

fail() {
  echo "$1" >&2
  exit 2
}

if [[ "$MODE" != "--dry-run" && "$MODE" != "--apply" ]]; then
  fail "usage: bash scripts/apply_bistka_storage_safe_cutover.sh [--dry-run|--apply]"
fi
[[ "$(pwd -P)" == "$(cd "$(dirname "$0")/.." && pwd -P)" ]] \
  || fail "run the storage-safe cutover from the project root"
[[ -x "$LOCAL_PYTHON" ]] \
  || fail "LOCAL_PYTHON must be an executable file"
[[ -f "$STAGE_SCRIPT" && ! -L "$STAGE_SCRIPT" ]] \
  || fail "the V2 deployment stage script is missing or unsafe"
for path in \
  "$CANDIDATE_MANIFEST" \
  "$CANDIDATE_BUNDLE" \
  "${CANDIDATE_BUNDLE}.sha256" \
  "$ROLLBACK_MANIFEST" \
  "$ROLLBACK_BUNDLE" \
  "${ROLLBACK_BUNDLE}.sha256"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required cutover artifact is missing: $path"
done
[[ "$CANDIDATE_PREFLIGHT" != "$ROLLBACK_PREFLIGHT" ]] \
  || fail "candidate and rollback preflight paths must be distinct"

run_release() {
  local manifest="$1"
  local bundle="$2"
  local preflight="$3"
  LOCAL_PYTHON="$LOCAL_PYTHON" \
  MANIFEST="$manifest" \
  BUNDLE="$bundle" \
  PREFLIGHT_OUTPUT="$preflight" \
    bash "$STAGE_SCRIPT" "$MODE"
}

if [[ "$MODE" == "--dry-run" ]]; then
  echo "cutover_validation=candidate"
  run_release "$CANDIDATE_MANIFEST" "$CANDIDATE_BUNDLE" "$CANDIDATE_PREFLIGHT"
  echo "cutover_validation=rollback"
  run_release "$ROLLBACK_MANIFEST" "$ROLLBACK_BUNDLE" "$ROLLBACK_PREFLIGHT"
  echo "status=dry_run"
  echo "candidate and rollback releases passed local validation; no remote file was changed"
  exit 0
fi

rollback_after_signal() {
  local signal_name="$1"
  trap - HUP INT TERM
  echo "cutover interrupted by ${signal_name}; attempting exact active-release rollback" >&2
  set +e
  run_release "$ROLLBACK_MANIFEST" "$ROLLBACK_BUNDLE" "$ROLLBACK_PREFLIGHT"
  local rollback_status="$?"
  set -e
  if [[ "$rollback_status" -ne 0 ]]; then
    echo "critical: interrupted cutover rollback failed with status ${rollback_status}" >&2
    exit 3
  fi
  echo "status=rolled_back_after_interrupt" >&2
  exit 130
}
trap 'rollback_after_signal HUP' HUP
trap 'rollback_after_signal INT' INT
trap 'rollback_after_signal TERM' TERM

echo "cutover_apply=candidate"
set +e
run_release "$CANDIDATE_MANIFEST" "$CANDIDATE_BUNDLE" "$CANDIDATE_PREFLIGHT"
CANDIDATE_STATUS="$?"
set -e
if [[ "$CANDIDATE_STATUS" -eq 0 ]]; then
  trap - HUP INT TERM
  echo "status=applied"
  echo "rollback=not_required"
  exit 0
fi

echo "candidate cutover or preflight failed with status ${CANDIDATE_STATUS}" >&2
echo "cutover_apply=rollback" >&2
set +e
run_release "$ROLLBACK_MANIFEST" "$ROLLBACK_BUNDLE" "$ROLLBACK_PREFLIGHT"
ROLLBACK_STATUS="$?"
set -e
trap - HUP INT TERM
if [[ "$ROLLBACK_STATUS" -ne 0 ]]; then
  echo "critical: exact active-release rollback failed with status ${ROLLBACK_STATUS}" >&2
  exit 3
fi
echo "status=rolled_back" >&2
echo "candidate_status=${CANDIDATE_STATUS}" >&2
exit "$CANDIDATE_STATUS"
