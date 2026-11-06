#!/bin/bash

set -euo pipefail

fail() {
  echo "$1" >&2
  exit 2
}

safe_absolute() {
  local value="$1"
  [[ "$value" =~ ^/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || return 1
  [[ "/$value/" != *"/./"* && "/$value/" != *"/../"* ]]
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    fail "neither sha256sum nor shasum is available"
  fi
}

if [[ "$#" -ne 1 ]]; then
  fail "usage: bash scripts/fetch_bistka_screening_smoke_results.sh JOB_ID"
fi

JOB_ID="$1"
[[ "$JOB_ID" =~ ^[1-9][0-9]*$ ]] || fail "JOB_ID must be one positive numeric SLURM job ID"

REMOTE="${BISTKA_REMOTE:-s00658@10.205.100.101}"
EXPECTED_REMOTE="${BISTKA_EXPECTED_REMOTE:-s00658@10.205.100.101}"
REMOTE_PROJECT_DIR="${BISTKA_PROJECT_DIR:-/lustrefs/project/25sfcs03/drl_thai_stock}"
EXPECTED_PROJECT_DIR="${BISTKA_EXPECTED_PROJECT_DIR:-/lustrefs/project/25sfcs03/drl_thai_stock}"
LOCAL_ROOT_INPUT="${LOCAL_ROOT:-.}"
EXPECTED_LOCAL_ROOT_INPUT="${BISTKA_EXPECTED_LOCAL_ROOT:-$(pwd -P)}"
EXPECTED_ACCOUNT="25sfcs03"
EXPECTED_PARTITION="compute-devel"
EXPECTED_JOB_NAME="drl_screen_smoke"

[[ "$REMOTE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*$ ]] \
  || fail "BISTKA_REMOTE is not a strict user@host target"
[[ "$EXPECTED_REMOTE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*$ ]] \
  || fail "BISTKA_EXPECTED_REMOTE is not a strict user@host target"
[[ "$REMOTE" == "$EXPECTED_REMOTE" ]] \
  || fail "BISTKA_REMOTE does not match the expected BistKA endpoint"
safe_absolute "$REMOTE_PROJECT_DIR" || fail "BISTKA_PROJECT_DIR is not a strict absolute path"
safe_absolute "$EXPECTED_PROJECT_DIR" \
  || fail "BISTKA_EXPECTED_PROJECT_DIR is not a strict absolute path"
[[ "$REMOTE_PROJECT_DIR" == "$EXPECTED_PROJECT_DIR" ]] \
  || fail "BISTKA_PROJECT_DIR does not match the expected BistKA project root"

[[ -d "$LOCAL_ROOT_INPUT" && ! -L "$LOCAL_ROOT_INPUT" ]] \
  || fail "LOCAL_ROOT must be an existing non-symlink directory"
[[ -d "$EXPECTED_LOCAL_ROOT_INPUT" && ! -L "$EXPECTED_LOCAL_ROOT_INPUT" ]] \
  || fail "BISTKA_EXPECTED_LOCAL_ROOT must be an existing non-symlink directory"
LOCAL_ROOT="$(cd "$LOCAL_ROOT_INPUT" && pwd -P)"
EXPECTED_LOCAL_ROOT="$(cd "$EXPECTED_LOCAL_ROOT_INPUT" && pwd -P)"
[[ "$LOCAL_ROOT" == "$EXPECTED_LOCAL_ROOT" ]] \
  || fail "LOCAL_ROOT does not match the expected local project root"

for command_name in ssh rsync scp; do
  command -v "$command_name" >/dev/null 2>&1 || fail "required command is missing: $command_name"
done

REMOTE_RUN_DIR="${REMOTE_PROJECT_DIR}/results/screening_smoke_${JOB_ID}"
REMOTE_STDOUT="${REMOTE_PROJECT_DIR}/logs/screening_smoke_${JOB_ID}.out"
REMOTE_STDERR="${REMOTE_PROJECT_DIR}/logs/screening_smoke_${JOB_ID}.err"
LOCAL_RUN_DIR="${LOCAL_ROOT}/results/screening_smoke_${JOB_ID}"
LOCAL_STDOUT="${LOCAL_ROOT}/logs/screening_smoke_${JOB_ID}.out"
LOCAL_STDERR="${LOCAL_ROOT}/logs/screening_smoke_${JOB_ID}.err"
LOCAL_ACCOUNTING="${LOCAL_ROOT}/reports/slurm_accounting_screening_smoke_${JOB_ID}.txt"

for destination in "$LOCAL_RUN_DIR" "$LOCAL_STDOUT" "$LOCAL_STDERR" "$LOCAL_ACCOUNTING"; do
  [[ ! -e "$destination" && ! -L "$destination" ]] \
    || fail "refusing to overwrite existing local smoke evidence: $destination"
done

for directory in results logs reports; do
  local_directory="${LOCAL_ROOT}/${directory}"
  if [[ -e "$local_directory" || -L "$local_directory" ]]; then
    [[ -d "$local_directory" && ! -L "$local_directory" ]] \
      || fail "local evidence parent is not a non-symlink directory: $local_directory"
  fi
done

STAGING_DIR="$(mktemp -d "${LOCAL_ROOT}/.screening_smoke_fetch_${JOB_ID}.XXXXXX")"
cleanup() {
  if [[ -n "${STAGING_DIR:-}" && -d "$STAGING_DIR" ]]; then
    case "$STAGING_DIR" in
      "${LOCAL_ROOT}/.screening_smoke_fetch_${JOB_ID}."*) rm -rf -- "$STAGING_DIR" ;;
      *) echo "refusing to clean an unexpected staging path: $STAGING_DIR" >&2 ;;
    esac
  fi
}
trap cleanup EXIT

STAGED_RUN="${STAGING_DIR}/results/screening_smoke_${JOB_ID}"
STAGED_STDOUT="${STAGING_DIR}/logs/screening_smoke_${JOB_ID}.out"
STAGED_STDERR="${STAGING_DIR}/logs/screening_smoke_${JOB_ID}.err"
STAGED_ACCOUNTING="${STAGING_DIR}/reports/slurm_accounting_screening_smoke_${JOB_ID}.txt"
mkdir -p "$STAGED_RUN" "$(dirname "$STAGED_STDOUT")" "$(dirname "$STAGED_ACCOUNTING")"

ssh "$REMOTE" bash -s -- \
  "$REMOTE_PROJECT_DIR" \
  "$JOB_ID" \
  "$EXPECTED_ACCOUNT" \
  "$EXPECTED_PARTITION" \
  "$EXPECTED_JOB_NAME" >"$STAGED_ACCOUNTING" <<'REMOTE_SCRIPT'
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "remote smoke fetch validation received an unexpected argument count" >&2
  exit 2
fi

PROJECT_ROOT="$1"
JOB_ID="$2"
EXPECTED_ACCOUNT="$3"
EXPECTED_PARTITION="$4"
EXPECTED_JOB_NAME="$5"

safe_absolute() {
  local value="$1"
  [[ "$value" =~ ^/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || return 1
  [[ "/$value/" != *"/./"* && "/$value/" != *"/../"* ]]
}

safe_absolute "$PROJECT_ROOT"
[[ "$JOB_ID" =~ ^[1-9][0-9]*$ ]]
[[ "$EXPECTED_ACCOUNT" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "$EXPECTED_PARTITION" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "$EXPECTED_JOB_NAME" =~ ^[A-Za-z0-9._-]+$ ]]
[[ -d "$PROJECT_ROOT" && ! -L "$PROJECT_ROOT" ]]
[[ "$(cd "$PROJECT_ROOT" && pwd -P)" == "$PROJECT_ROOT" ]]

RUN_DIR="${PROJECT_ROOT}/results/screening_smoke_${JOB_ID}"
STDOUT_PATH="${PROJECT_ROOT}/logs/screening_smoke_${JOB_ID}.out"
STDERR_PATH="${PROJECT_ROOT}/logs/screening_smoke_${JOB_ID}.err"

[[ -d "${PROJECT_ROOT}/results" && ! -L "${PROJECT_ROOT}/results" ]]
[[ -d "${PROJECT_ROOT}/logs" && ! -L "${PROJECT_ROOT}/logs" ]]
[[ -d "$RUN_DIR" && ! -L "$RUN_DIR" ]]
[[ "$(cd "$RUN_DIR" && pwd -P)" == "$RUN_DIR" ]]
[[ -f "$STDOUT_PATH" && ! -L "$STDOUT_PATH" ]]
[[ -f "$STDERR_PATH" && ! -L "$STDERR_PATH" ]]
if find "$RUN_DIR" -type l -print -quit | grep -q .; then
  echo "remote smoke run contains a symlink" >&2
  exit 2
fi
find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .

command -v sacct >/dev/null 2>&1
FORMAT="JobIDRaw,JobName%64,Account%64,Partition%64,State%32,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,TotalCPU,NodeList%128,WorkDir%512"
ACCOUNTING="$(
  sacct \
    --jobs "$JOB_ID" \
    --noheader \
    --parsable2 \
    --format="$FORMAT"
)"
[[ -n "$ACCOUNTING" ]]

PARENT_COUNT="$(
  printf '%s\n' "$ACCOUNTING" |
    awk -F'|' -v expected="$JOB_ID" '$1 == expected {count += 1} END {print count + 0}'
)"
[[ "$PARENT_COUNT" == "1" ]]
PARENT_ROW="$(
  printf '%s\n' "$ACCOUNTING" |
    awk -F'|' -v expected="$JOB_ID" '$1 == expected {print}'
)"
IFS='|' read -r \
  ACCOUNTING_JOB_ID \
  JOB_NAME \
  ACCOUNT \
  PARTITION \
  STATE \
  EXIT_CODE \
  ELAPSED_RAW \
  ALLOC_CPUS \
  REQ_MEM \
  MAX_RSS \
  TOTAL_CPU \
  NODE_LIST \
  WORK_DIR \
  EXTRA_FIELD <<<"$PARENT_ROW"

[[ "$ACCOUNTING_JOB_ID" == "$JOB_ID" ]]
[[ "$JOB_NAME" == "$EXPECTED_JOB_NAME" ]]
[[ "$ACCOUNT" == "$EXPECTED_ACCOUNT" ]]
[[ "$PARTITION" == "$EXPECTED_PARTITION" ]]
[[ "$ALLOC_CPUS" == "2" ]]
[[ "$WORK_DIR" == "$PROJECT_ROOT" ]]

NORMALIZED_STATE="${STATE%% *}"
NORMALIZED_STATE="${NORMALIZED_STATE%%+}"
case "$NORMALIZED_STATE" in
  COMPLETED|FAILED|TIMEOUT|OUT_OF_MEMORY|CANCELLED|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT) ;;
  *)
    echo "screening smoke job is not in an accepted terminal state: $STATE" >&2
    exit 2
    ;;
esac
if [[ "$NORMALIZED_STATE" == "COMPLETED" ]]; then
  [[ "$EXIT_CODE" == "0:0" ]]
fi

printf '# schema_version=1\n'
printf '# stage=screening_smoke\n'
printf '# job_id=%s\n' "$JOB_ID"
printf '# remote_project_root=%s\n' "$PROJECT_ROOT"
printf 'JobIDRaw|JobName|Account|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|MaxRSS|TotalCPU|NodeList|WorkDir|\n'
printf '%s\n' "$ACCOUNTING"
REMOTE_SCRIPT

[[ -s "$STAGED_ACCOUNTING" ]] || fail "remote accounting capture is empty"
grep -Fqx "# schema_version=1" "$STAGED_ACCOUNTING" \
  || fail "remote accounting capture has no schema marker"
grep -Fqx "# stage=screening_smoke" "$STAGED_ACCOUNTING" \
  || fail "remote accounting capture has the wrong stage marker"
grep -Fqx "# job_id=${JOB_ID}" "$STAGED_ACCOUNTING" \
  || fail "remote accounting capture has the wrong job ID"
grep -Fqx "# remote_project_root=${REMOTE_PROJECT_DIR}" "$STAGED_ACCOUNTING" \
  || fail "remote accounting capture has the wrong project root"

PARENT_ROW="$(
  awk -F'|' -v expected="$JOB_ID" '$1 == expected {print}' "$STAGED_ACCOUNTING"
)"
PARENT_COUNT="$(
  awk -F'|' -v expected="$JOB_ID" '$1 == expected {count += 1} END {print count + 0}' \
    "$STAGED_ACCOUNTING"
)"
[[ "$PARENT_COUNT" == "1" ]] || fail "accounting does not contain exactly one parent job row"
IFS='|' read -r \
  ACCOUNTING_JOB_ID \
  JOB_NAME \
  ACCOUNT \
  PARTITION \
  STATE \
  EXIT_CODE \
  ELAPSED_RAW \
  ALLOC_CPUS \
  REQ_MEM \
  MAX_RSS \
  TOTAL_CPU \
  NODE_LIST \
  WORK_DIR \
  EXTRA_FIELD <<<"$PARENT_ROW"
[[ "$ACCOUNTING_JOB_ID" == "$JOB_ID" ]] || fail "accounting parent job ID mismatch"
[[ "$JOB_NAME" == "$EXPECTED_JOB_NAME" ]] || fail "accounting job name mismatch"
[[ "$ACCOUNT" == "$EXPECTED_ACCOUNT" ]] || fail "accounting account mismatch"
[[ "$PARTITION" == "$EXPECTED_PARTITION" ]] || fail "accounting partition mismatch"
[[ "$ALLOC_CPUS" == "2" ]] || fail "accounting CPU allocation mismatch"
[[ "$WORK_DIR" == "$REMOTE_PROJECT_DIR" ]] || fail "accounting working directory mismatch"

NORMALIZED_STATE="${STATE%% *}"
NORMALIZED_STATE="${NORMALIZED_STATE%%+}"
case "$NORMALIZED_STATE" in
  COMPLETED|FAILED|TIMEOUT|OUT_OF_MEMORY|CANCELLED|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT) ;;
  *) fail "accounting state is not terminal: $STATE" ;;
esac
if [[ "$NORMALIZED_STATE" == "COMPLETED" && "$EXIT_CODE" != "0:0" ]]; then
  fail "completed accounting row has a nonzero exit code: $EXIT_CODE"
fi

rsync -a --checksum --safe-links \
  "${REMOTE}:${REMOTE_RUN_DIR}/" \
  "${STAGED_RUN}/"
scp "${REMOTE}:${REMOTE_STDOUT}" "$STAGED_STDOUT"
scp "${REMOTE}:${REMOTE_STDERR}" "$STAGED_STDERR"

[[ -d "$STAGED_RUN" && ! -L "$STAGED_RUN" ]] || fail "staged smoke run directory is missing"
find "$STAGED_RUN" -mindepth 1 -print -quit | grep -q . \
  || fail "staged smoke run directory is empty"
if find "$STAGED_RUN" -type l -print -quit | grep -q .; then
  fail "staged smoke run contains a symlink"
fi
[[ -f "$STAGED_STDOUT" && ! -L "$STAGED_STDOUT" ]] \
  || fail "staged screening smoke stdout is missing"
[[ -f "$STAGED_STDERR" && ! -L "$STAGED_STDERR" ]] \
  || fail "staged screening smoke stderr is missing"

SUMMARY_PATH="${STAGED_RUN}/screening_smoke_summary.json"
SUMMARY_SIDECAR="${SUMMARY_PATH}.sha256"
if [[ -e "$SUMMARY_PATH" || -L "$SUMMARY_PATH" || -e "$SUMMARY_SIDECAR" || -L "$SUMMARY_SIDECAR" ]]; then
  [[ -f "$SUMMARY_PATH" && ! -L "$SUMMARY_PATH" ]] \
    || fail "screening smoke summary is not a regular file"
  [[ -f "$SUMMARY_SIDECAR" && ! -L "$SUMMARY_SIDECAR" ]] \
    || fail "screening smoke summary sidecar is not a regular file"
  SIDECAR_VALUE="$(tr -d '\r\n' <"$SUMMARY_SIDECAR")"
  [[ "$SIDECAR_VALUE" =~ ^[0-9a-f]{64}$ ]] \
    || fail "screening smoke summary sidecar is malformed"
  [[ "$(sha256_file "$SUMMARY_PATH")" == "$SIDECAR_VALUE" ]] \
    || fail "screening smoke summary SHA-256 mismatch"
fi

if [[ "$NORMALIZED_STATE" == "COMPLETED" ]]; then
  [[ -f "$SUMMARY_PATH" && -f "$SUMMARY_SIDECAR" ]] \
    || fail "completed screening smoke job has no verified summary"
  MANIFEST_COUNT="$(
    find "$STAGED_RUN" -type f -name experiment_manifest.json | wc -l | tr -d '[:space:]'
  )"
  [[ "$MANIFEST_COUNT" == "5" ]] \
    || fail "completed screening smoke job must contain exactly five experiment manifests"
fi

mkdir -p "${LOCAL_ROOT}/results" "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/reports"
mv "$STAGED_RUN" "$LOCAL_RUN_DIR"
mv "$STAGED_STDOUT" "$LOCAL_STDOUT"
mv "$STAGED_STDERR" "$LOCAL_STDERR"
mv "$STAGED_ACCOUNTING" "$LOCAL_ACCOUNTING"
rmdir "${STAGING_DIR}/results" "${STAGING_DIR}/logs" "${STAGING_DIR}/reports" "$STAGING_DIR"
STAGING_DIR=""

echo "status=fetched"
echo "job_id=${JOB_ID}"
echo "job_state=${NORMALIZED_STATE}"
echo "run_dir=${LOCAL_RUN_DIR}"
echo "stdout=${LOCAL_STDOUT}"
echo "stderr=${LOCAL_STDERR}"
echo "accounting=${LOCAL_ACCOUNTING}"
echo "Run the screening_smoke campaign gate locally after inspecting the copied evidence."
