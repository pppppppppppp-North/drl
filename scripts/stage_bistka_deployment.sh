#!/bin/bash

set -euo pipefail

MODE="${1:---dry-run}"
MANIFEST="${MANIFEST:-campaigns/large_scale_thai_drl/deployment/bistka_deployment_manifest.json}"
BUNDLE="${BUNDLE:-campaigns/large_scale_thai_drl/deployment/bistka_release.tar.gz}"
REMOTE="${BISTKA_REMOTE:-s00658@10.205.100.101}"
LOCAL_PYTHON="${LOCAL_PYTHON:-python3}"
PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:-reports/bistka_calibration_preflight.json}"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--apply" ]]; then
  echo "usage: bash scripts/stage_bistka_deployment.sh [--dry-run|--apply]" >&2
  exit 2
fi
test -f "$MANIFEST"
test -f "$BUNDLE"
test -f "${BUNDLE}.sha256"

RELEASE_ID="$("$LOCAL_PYTHON" - "$MANIFEST" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["release_id"])
PY
)"
TARGET_ROOT="$("$LOCAL_PYTHON" - "$MANIFEST" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["target_project_root"])
PY
)"
PYTHON_EXECUTABLE="$("$LOCAL_PYTHON" - "$MANIFEST" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["python_executable"])
PY
)"
MANIFEST_RELATIVE="${MANIFEST#./}"
if [[ "$MANIFEST_RELATIVE" == /* || "$MANIFEST_RELATIVE" == *".."* ]]; then
  echo "MANIFEST must be a safe project-relative path" >&2
  exit 2
fi
if [[ "$PREFLIGHT_OUTPUT" == /* || "$PREFLIGHT_OUTPUT" == *".."* ]]; then
  echo "PREFLIGHT_OUTPUT must be a safe project-relative path" >&2
  exit 2
fi
EXPECTED_SHA256="$(tr -d '[:space:]' <"${BUNDLE}.sha256")"
REMOTE_BUNDLE="/tmp/bistka_release_${RELEASE_ID}.tar.gz"
BACKUP_ROOT="${TARGET_ROOT}/.bistka/backups/${RELEASE_ID}"

echo "release_id=${RELEASE_ID}"
echo "target=${REMOTE}:${TARGET_ROOT}"
echo "bundle_sha256=${EXPECTED_SHA256}"
echo "preflight_output=${PREFLIGHT_OUTPUT}"

if [[ "$MODE" == "--dry-run" ]]; then
  echo "status=dry_run"
  echo "No remote file was changed. Re-run with --apply to upload, back up replaced manifest files, and extract the exact bundle."
  echo "The apply path also runs the bundled BistKA preflight and returns its receipt locally."
  exit 0
fi

scp "$BUNDLE" "${REMOTE}:${REMOTE_BUNDLE}"
ssh "$REMOTE" bash -s -- \
  "$TARGET_ROOT" \
  "$REMOTE_BUNDLE" \
  "$EXPECTED_SHA256" \
  "$BACKUP_ROOT" \
  "$PYTHON_EXECUTABLE" \
  "$MANIFEST_RELATIVE" \
  "$PREFLIGHT_OUTPUT" <<'REMOTE_SCRIPT'
set -euo pipefail
TARGET_ROOT="$1"
REMOTE_BUNDLE="$2"
EXPECTED_SHA256="$3"
BACKUP_ROOT="$4"
PYTHON_EXECUTABLE="$5"
MANIFEST_RELATIVE="$6"
PREFLIGHT_OUTPUT="$7"

ACTUAL_SHA256="$(sha256sum "$REMOTE_BUNDLE" | awk '{print $1}')"
test "$ACTUAL_SHA256" = "$EXPECTED_SHA256"
mkdir -p "$TARGET_ROOT" "$BACKUP_ROOT"

while IFS= read -r relative; do
  [[ -z "$relative" || "$relative" == .bistka/* ]] && continue
  if [[ -f "$TARGET_ROOT/$relative" ]]; then
    mkdir -p "$BACKUP_ROOT/$(dirname "$relative")"
    cp -p "$TARGET_ROOT/$relative" "$BACKUP_ROOT/$relative"
  fi
done < <(tar -tzf "$REMOTE_BUNDLE")

tar -xzf "$REMOTE_BUNDLE" -C "$TARGET_ROOT"
printf '%s\n' "$EXPECTED_SHA256" >"$TARGET_ROOT/.bistka/deployed_bundle.sha256"
rm -f "$REMOTE_BUNDLE"
cd "$TARGET_ROOT"
"$PYTHON_EXECUTABLE" -m scripts.check_bistka_calibration_preflight \
  --project-root . \
  --manifest "$MANIFEST_RELATIVE" \
  --output "$PREFLIGHT_OUTPUT"
echo "status=staged"
REMOTE_SCRIPT

mkdir -p "$(dirname "$PREFLIGHT_OUTPUT")"
scp "${REMOTE}:${TARGET_ROOT}/${PREFLIGHT_OUTPUT}" "$PREFLIGHT_OUTPUT"

echo "status=applied"
echo "status=preflight_receipt_fetched"
echo "Verify the matching campaign preflight stage locally before rendering compute."
