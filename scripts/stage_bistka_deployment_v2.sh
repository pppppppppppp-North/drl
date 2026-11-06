#!/bin/bash

set -euo pipefail

MODE="${1:---dry-run}"
MANIFEST="${MANIFEST:?Set MANIFEST to the exact release manifest}"
BUNDLE="${BUNDLE:?Set BUNDLE to the exact release bundle}"
PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:?Set PREFLIGHT_OUTPUT to a project-relative receipt path}"
REMOTE="${BISTKA_REMOTE:-s00658@10.205.100.101}"
EXPECTED_REMOTE="${BISTKA_EXPECTED_REMOTE:-s00658@10.205.100.101}"
EXPECTED_TARGET_ROOT="${BISTKA_EXPECTED_TARGET_ROOT:-/lustrefs/project/25sfcs03/drl_thai_stock}"
LOCAL_PYTHON="${LOCAL_PYTHON:-python3}"
QUOTA_MARGIN_MIB="${QUOTA_MARGIN_MIB:-1}"
APPLIER="src/campaigns/atomic_release_apply.py"
PROJECT_ROOT="$(pwd -P)"

fail() {
  echo "$1" >&2
  exit 2
}

safe_relative() {
  local value="$1"
  [[ "$value" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || return 1
  [[ "$value" != -* && "/$value/" != *"/./"* && "/$value/" != *"/../"* ]]
}

safe_absolute() {
  local value="$1"
  [[ "$value" =~ ^/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || return 1
  [[ "/$value/" != *"/./"* && "/$value/" != *"/../"* ]]
}

safe_python_command() {
  local value="$1"
  [[ "$value" =~ ^[A-Za-z0-9._+@-]+(/[A-Za-z0-9._+@-]+)*$ \
    || "$value" =~ ^/[A-Za-z0-9._+@-]+(/[A-Za-z0-9._+@-]+)*$ ]] || return 1
  [[ "$value" != -* && "/$value/" != *"/./"* && "/$value/" != *"/../"* ]]
}

if [[ "$MODE" != "--dry-run" && "$MODE" != "--apply" ]]; then
  fail "usage: bash scripts/stage_bistka_deployment_v2.sh [--dry-run|--apply]"
fi

MANIFEST_RELATIVE="${MANIFEST#./}"
BUNDLE_RELATIVE="${BUNDLE#./}"
APPLIER_RELATIVE="${APPLIER#./}"
safe_relative "$MANIFEST_RELATIVE" || fail "MANIFEST must be a strict project-relative path"
safe_relative "$BUNDLE_RELATIVE" || fail "BUNDLE must be a strict project-relative path"
safe_relative "$APPLIER_RELATIVE" || fail "APPLIER must be a strict project-relative path"
safe_relative "$PREFLIGHT_OUTPUT" || fail "PREFLIGHT_OUTPUT must be a strict project-relative path"
safe_absolute "$EXPECTED_TARGET_ROOT" || fail "BISTKA_EXPECTED_TARGET_ROOT is not a safe absolute path"
safe_python_command "$LOCAL_PYTHON" || fail "LOCAL_PYTHON is not a safe executable name or path"
command -v "$LOCAL_PYTHON" >/dev/null || fail "LOCAL_PYTHON is not executable"
[[ "$REMOTE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*$ ]] \
  || fail "BISTKA_REMOTE is not a strict user@host target"
[[ "$EXPECTED_REMOTE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*$ ]] \
  || fail "BISTKA_EXPECTED_REMOTE is not a strict user@host target"
[[ "$REMOTE" == "$EXPECTED_REMOTE" ]] || fail "BISTKA_REMOTE does not match the expected BistKA endpoint"
[[ "$QUOTA_MARGIN_MIB" =~ ^[0-9]+$ ]] || fail "QUOTA_MARGIN_MIB must be a nonnegative integer"
(( QUOTA_MARGIN_MIB <= 102400 )) || fail "QUOTA_MARGIN_MIB is unreasonably large"

test -f "$MANIFEST" || fail "release manifest is missing"
test -f "$BUNDLE" || fail "release bundle is missing"
test -f "${BUNDLE}.sha256" || fail "release bundle SHA-256 sidecar is missing"
test -f "$APPLIER" || fail "standalone release applier is missing"

"$LOCAL_PYTHON" - \
  "$PROJECT_ROOT" \
  "$MANIFEST_RELATIVE" \
  "$BUNDLE_RELATIVE" \
  "$APPLIER_RELATIVE" \
  "$EXPECTED_TARGET_ROOT" \
  "$PREFLIGHT_OUTPUT" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from src.campaigns.bistka_deployment import (
    validate_deployment_bundle,
    validate_deployment_manifest,
)

root = Path(sys.argv[1]).resolve()
manifest_relative, bundle_relative, applier_relative = sys.argv[2:5]
expected_target, preflight_relative = sys.argv[5:7]


def project_path(relative: str, label: str) -> Path:
    candidate = root / relative
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or not candidate.is_file():
        raise SystemExit(f"local deployment validation failed: {label} is missing or unsafe")
    return candidate


manifest_path = project_path(manifest_relative, "manifest")
bundle_path = project_path(bundle_relative, "bundle")
applier_path = project_path(applier_relative, "applier")
preflight_path = (root / preflight_relative).resolve()
if not preflight_path.is_relative_to(root):
    raise SystemExit("local deployment validation failed: preflight output escapes the project")

manifest_issues, _ = validate_deployment_manifest(manifest_path, project_root=root)
bundle_issues, _ = validate_deployment_bundle(
    bundle_path,
    manifest_path=manifest_path,
    project_root=root,
)
issues = [*manifest_issues, *bundle_issues]
if issues:
    detail = "\n - ".join(sorted(set(issues)))
    raise SystemExit(f"local deployment validation failed:\n - {detail}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
release_id = str(manifest.get("release_id", ""))
if re.fullmatch(r"[0-9a-f]{64}", release_id) is None:
    raise SystemExit("local deployment validation failed: release_id is not lowercase SHA-256")
if manifest.get("target_project_root") != expected_target:
    raise SystemExit(
        "local deployment validation failed: manifest target_project_root does not "
        "match BISTKA_EXPECTED_TARGET_ROOT"
    )
python_executable = str(manifest.get("python_executable", ""))
if re.fullmatch(r"/[A-Za-z0-9._+-]+(?:/[A-Za-z0-9._+-]+)*", python_executable) is None:
    raise SystemExit("local deployment validation failed: unsafe remote Python executable")

bundle_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
bundle_sidecar = Path(str(bundle_path) + ".sha256")
raw_bundle_sidecar = bundle_sidecar.read_text(encoding="utf-8")
if re.fullmatch(r"[0-9a-f]{64}\n?", raw_bundle_sidecar) is None:
    raise SystemExit("local deployment validation failed: malformed bundle SHA-256 sidecar")
if raw_bundle_sidecar.strip() != bundle_hash:
    raise SystemExit("local deployment validation failed: bundle SHA-256 mismatch")

manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
manifest_sidecar = manifest_path.with_suffix(".sha256")
raw_manifest_sidecar = manifest_sidecar.read_text(encoding="utf-8")
if re.fullmatch(r"[0-9a-f]{64}\n?", raw_manifest_sidecar) is None:
    raise SystemExit("local deployment validation failed: malformed manifest SHA-256 sidecar")
if raw_manifest_sidecar.strip() != manifest_hash:
    raise SystemExit("local deployment validation failed: manifest SHA-256 mismatch")

applier_records = [
    record
    for record in manifest.get("files", [])
    if isinstance(record, dict) and record.get("path") == applier_relative
]
if len(applier_records) != 1:
    raise SystemExit(
        "local deployment validation failed: manifest must bind exactly one standalone applier"
    )
applier_hash = hashlib.sha256(applier_path.read_bytes()).hexdigest()
if applier_records[0].get("sha256") != applier_hash:
    raise SystemExit("local deployment validation failed: standalone applier hash mismatch")

print("local_validation=pass")
PY

RELEASE_ID="$("$LOCAL_PYTHON" - "$MANIFEST" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["release_id"])
PY
)"
TARGET_ROOT="$("$LOCAL_PYTHON" - "$MANIFEST" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["target_project_root"])
PY
)"
PYTHON_EXECUTABLE="$("$LOCAL_PYTHON" - "$MANIFEST" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["python_executable"])
PY
)"
EXPECTED_SHA256="$(tr -d '\n' <"${BUNDLE}.sha256")"
EXPECTED_MANIFEST_SHA256="$("$LOCAL_PYTHON" - "$MANIFEST" <<'PY'
import hashlib
import sys

print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)"
EXPECTED_APPLIER_SHA256="$("$LOCAL_PYTHON" - "$MANIFEST" "$APPLIER_RELATIVE" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
matches = [record for record in manifest["files"] if record.get("path") == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit("release manifest must bind exactly one applier record")
print(matches[0]["sha256"])
PY
)"

[[ "$RELEASE_ID" =~ ^[0-9a-f]{64}$ ]] || fail "release ID is not a strict SHA-256 value"
[[ "$EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail "bundle SHA-256 is malformed"
[[ "$EXPECTED_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail "manifest SHA-256 is malformed"
[[ "$EXPECTED_APPLIER_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail "applier SHA-256 is malformed"
[[ "$TARGET_ROOT" == "$EXPECTED_TARGET_ROOT" ]] || fail "release target changed after validation"
safe_absolute "$TARGET_ROOT" || fail "release target is not a strict absolute path"
safe_absolute "$PYTHON_EXECUTABLE" || fail "remote Python executable is not a strict absolute path"

REMOTE_BUNDLE="/tmp/bistka_release_${RELEASE_ID}.tar.gz"
REMOTE_MANIFEST="/tmp/bistka_manifest_${RELEASE_ID}.json"
REMOTE_APPLIER="/tmp/bistka_apply_${RELEASE_ID}.py"

echo "release_id=${RELEASE_ID}"
echo "target=${REMOTE}:${TARGET_ROOT}"
echo "bundle_sha256=${EXPECTED_SHA256}"
echo "manifest_sha256=${EXPECTED_MANIFEST_SHA256}"
echo "applier_sha256=${EXPECTED_APPLIER_SHA256}"
echo "preflight_output=${PREFLIGHT_OUTPUT}"
echo "quota_margin_mib=${QUOTA_MARGIN_MIB}"

if [[ "$MODE" == "--dry-run" ]]; then
  echo "status=dry_run"
  echo "No remote file was changed. Apply uploads verified artifacts to node-local /tmp, reserves changed-file headroom, stages sibling files, rolls back a failed commit, promotes manifests last, and writes the release marker last."
  exit 0
fi

scp "$BUNDLE" "${REMOTE}:${REMOTE_BUNDLE}"
scp "$MANIFEST" "${REMOTE}:${REMOTE_MANIFEST}"
scp "$APPLIER" "${REMOTE}:${REMOTE_APPLIER}"
ssh "$REMOTE" bash -s -- \
  "$TARGET_ROOT" \
  "$REMOTE_BUNDLE" \
  "$REMOTE_MANIFEST" \
  "$REMOTE_APPLIER" \
  "$EXPECTED_SHA256" \
  "$EXPECTED_MANIFEST_SHA256" \
  "$EXPECTED_APPLIER_SHA256" \
  "$PYTHON_EXECUTABLE" \
  "$MANIFEST_RELATIVE" \
  "$PREFLIGHT_OUTPUT" \
  "$QUOTA_MARGIN_MIB" \
  "$RELEASE_ID" \
  "$EXPECTED_TARGET_ROOT" <<'REMOTE_SCRIPT'
set -euo pipefail

if [[ "$#" -ne 13 ]]; then
  echo "remote staging received an unexpected argument count" >&2
  exit 2
fi

TARGET_ROOT="$1"
REMOTE_BUNDLE="$2"
REMOTE_MANIFEST="$3"
REMOTE_APPLIER="$4"
EXPECTED_SHA256="$5"
EXPECTED_MANIFEST_SHA256="$6"
EXPECTED_APPLIER_SHA256="$7"
PYTHON_EXECUTABLE="$8"
MANIFEST_RELATIVE="$9"
PREFLIGHT_OUTPUT="${10}"
QUOTA_MARGIN_MIB="${11}"
RELEASE_ID="${12}"
EXPECTED_TARGET_ROOT="${13}"

safe_relative() {
  local value="$1"
  [[ "$value" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || return 1
  [[ "$value" != -* && "/$value/" != *"/./"* && "/$value/" != *"/../"* ]]
}

safe_absolute() {
  local value="$1"
  [[ "$value" =~ ^/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || return 1
  [[ "/$value/" != *"/./"* && "/$value/" != *"/../"* ]]
}

safe_absolute "$TARGET_ROOT"
safe_absolute "$EXPECTED_TARGET_ROOT"
safe_absolute "$REMOTE_BUNDLE"
safe_absolute "$REMOTE_MANIFEST"
safe_absolute "$REMOTE_APPLIER"
safe_absolute "$PYTHON_EXECUTABLE"
safe_relative "$MANIFEST_RELATIVE"
safe_relative "$PREFLIGHT_OUTPUT"
[[ "$TARGET_ROOT" == "$EXPECTED_TARGET_ROOT" ]]
[[ "$RELEASE_ID" =~ ^[0-9a-f]{64}$ ]]
[[ "$EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$EXPECTED_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$EXPECTED_APPLIER_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$QUOTA_MARGIN_MIB" =~ ^[0-9]+$ ]]
(( QUOTA_MARGIN_MIB <= 102400 ))
[[ "$REMOTE_BUNDLE" == "/tmp/bistka_release_${RELEASE_ID}.tar.gz" ]]
[[ "$REMOTE_MANIFEST" == "/tmp/bistka_manifest_${RELEASE_ID}.json" ]]
[[ "$REMOTE_APPLIER" == "/tmp/bistka_apply_${RELEASE_ID}.py" ]]

cleanup() {
  rm -f "$REMOTE_BUNDLE" "$REMOTE_MANIFEST" "$REMOTE_APPLIER"
}
trap cleanup EXIT

ACTUAL_SHA256="$(sha256sum "$REMOTE_BUNDLE" | awk '{print $1}')"
test "$ACTUAL_SHA256" = "$EXPECTED_SHA256"
ACTUAL_MANIFEST_SHA256="$(sha256sum "$REMOTE_MANIFEST" | awk '{print $1}')"
test "$ACTUAL_MANIFEST_SHA256" = "$EXPECTED_MANIFEST_SHA256"
ACTUAL_APPLIER_SHA256="$(sha256sum "$REMOTE_APPLIER" | awk '{print $1}')"
test "$ACTUAL_APPLIER_SHA256" = "$EXPECTED_APPLIER_SHA256"

"$PYTHON_EXECUTABLE" - "$REMOTE_MANIFEST" "$RELEASE_ID" "$TARGET_ROOT" "$PYTHON_EXECUTABLE" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("release_id") != sys.argv[2]:
    raise SystemExit("remote manifest release ID mismatch")
if manifest.get("target_project_root") != sys.argv[3]:
    raise SystemExit("remote manifest target mismatch")
if manifest.get("python_executable") != sys.argv[4]:
    raise SystemExit("remote manifest Python executable mismatch")
PY

"$PYTHON_EXECUTABLE" "$REMOTE_APPLIER" \
  --bundle "$REMOTE_BUNDLE" \
  --manifest "$REMOTE_MANIFEST" \
  --manifest-relative "$MANIFEST_RELATIVE" \
  --target-root "$TARGET_ROOT" \
  --expected-bundle-sha256 "$EXPECTED_SHA256" \
  --staging-root /tmp \
  --quota-margin-mib "$QUOTA_MARGIN_MIB"

cd "$TARGET_ROOT"
"$PYTHON_EXECUTABLE" -m scripts.check_bistka_calibration_preflight \
  --project-root . \
  --manifest "$MANIFEST_RELATIVE" \
  --output "$PREFLIGHT_OUTPUT"
echo "status=applied"
REMOTE_SCRIPT

mkdir -p "$(dirname "$PREFLIGHT_OUTPUT")"
scp "${REMOTE}:${TARGET_ROOT}/${PREFLIGHT_OUTPUT}" "$PREFLIGHT_OUTPUT"

echo "status=preflight_receipt_fetched"
echo "Verify the matching campaign preflight stage locally before rendering compute."
