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

if [[ "$#" -ne 2 ]]; then
  fail "usage: bash scripts/fetch_bistka_screening_storage_safe_results.sh JOB_ID ATTEMPT_ID"
fi

JOB_ID="$1"
ATTEMPT_ID="$2"
[[ "$JOB_ID" =~ ^[1-9][0-9]*$ ]] \
  || fail "JOB_ID must be one positive numeric SLURM array job ID"
[[ "$ATTEMPT_ID" =~ ^screening_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$ ]] \
  || fail "ATTEMPT_ID must be one immutable screening attempt ID"

REMOTE="${BISTKA_REMOTE:-s00658@10.205.100.101}"
EXPECTED_REMOTE="${BISTKA_EXPECTED_REMOTE:-s00658@10.205.100.101}"
REMOTE_PROJECT_DIR="${BISTKA_PROJECT_DIR:-/lustrefs/project/25sfcs03/drl_thai_stock}"
EXPECTED_PROJECT_DIR="${BISTKA_EXPECTED_PROJECT_DIR:-/lustrefs/project/25sfcs03/drl_thai_stock}"
LOCAL_ROOT_INPUT="${LOCAL_ROOT:-.}"
EXPECTED_LOCAL_ROOT_INPUT="${BISTKA_EXPECTED_LOCAL_ROOT:-$(pwd -P)}"
LOCAL_PYTHON_INPUT="${LOCAL_PYTHON:-.venv/bin/python}"
EXPECTED_ACCOUNT="25sfcs03"
EXPECTED_PARTITION="compute-long"
EXPECTED_JOB_NAME="drl_screen_safe"
EXPECTED_ARRAY_TASKS="5"
EXPECTED_CPUS="2"
RUN_NAME="large_scale_screening_${JOB_ID}"
LOCKED_CONFIG_RELATIVE="campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml"
ATTEMPT_RELATIVE="campaigns/large_scale_thai_drl_v2/attempts/${ATTEMPT_ID}.json"

[[ "$REMOTE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*$ ]] \
  || fail "BISTKA_REMOTE is not a strict user@host target"
[[ "$EXPECTED_REMOTE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*$ ]] \
  || fail "BISTKA_EXPECTED_REMOTE is not a strict user@host target"
[[ "$REMOTE" == "$EXPECTED_REMOTE" ]] \
  || fail "BISTKA_REMOTE does not match the expected BistKA endpoint"
safe_absolute "$REMOTE_PROJECT_DIR" \
  || fail "BISTKA_PROJECT_DIR is not a strict absolute path"
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

case "$LOCAL_PYTHON_INPUT" in
  /*) LOCAL_PYTHON="$LOCAL_PYTHON_INPUT" ;;
  *) LOCAL_PYTHON="${LOCAL_ROOT}/${LOCAL_PYTHON_INPUT}" ;;
esac
[[ -x "$LOCAL_PYTHON" && ! -L "$LOCAL_PYTHON" ]] \
  || fail "LOCAL_PYTHON must be an executable non-symlink file"
for command_name in ssh rsync scp; do
  command -v "$command_name" >/dev/null 2>&1 \
    || fail "required command is missing: $command_name"
done

LOCKED_CONFIG="${LOCAL_ROOT}/${LOCKED_CONFIG_RELATIVE}"
ATTEMPT_PATH="${LOCAL_ROOT}/${ATTEMPT_RELATIVE}"
[[ -f "$LOCKED_CONFIG" && ! -L "$LOCKED_CONFIG" ]] \
  || fail "locked screening configuration is missing or unsafe"
[[ -f "$ATTEMPT_PATH" && ! -L "$ATTEMPT_PATH" ]] \
  || fail "immutable screening attempt is missing or unsafe"

REMOTE_RUN_DIR="${REMOTE_PROJECT_DIR}/results/${RUN_NAME}"
LOCAL_RUN_DIR="${LOCAL_ROOT}/results/${RUN_NAME}"
LOCAL_ACCOUNTING="${LOCAL_ROOT}/reports/slurm_accounting_${RUN_NAME}.txt"
LOCAL_VALIDATION="${LOCAL_ROOT}/reports/${RUN_NAME}_import_validation.json"

DESTINATIONS=(
  "$LOCAL_RUN_DIR"
  "$LOCAL_ACCOUNTING"
  "$LOCAL_VALIDATION"
  "${LOCAL_VALIDATION}.sha256"
)
for task_id in 0 1 2 3 4; do
  DESTINATIONS+=(
    "${LOCAL_ROOT}/logs/screening_safe_${JOB_ID}_${task_id}.out"
    "${LOCAL_ROOT}/logs/screening_safe_${JOB_ID}_${task_id}.err"
  )
done
for destination in "${DESTINATIONS[@]}"; do
  [[ ! -e "$destination" && ! -L "$destination" ]] \
    || fail "refusing to overwrite existing local full-screening evidence: $destination"
done
for directory in results logs reports; do
  local_directory="${LOCAL_ROOT}/${directory}"
  if [[ -e "$local_directory" || -L "$local_directory" ]]; then
    [[ -d "$local_directory" && ! -L "$local_directory" ]] \
      || fail "local evidence parent is not a non-symlink directory: $local_directory"
  fi
done

STAGING_DIR="$(mktemp -d "${LOCAL_ROOT}/.screening_fetch_${JOB_ID}.XXXXXX")"
cleanup() {
  if [[ -n "${STAGING_DIR:-}" && -d "$STAGING_DIR" ]]; then
    case "$STAGING_DIR" in
      "${LOCAL_ROOT}/.screening_fetch_${JOB_ID}."*) rm -rf -- "$STAGING_DIR" ;;
      *) echo "refusing to clean an unexpected staging path: $STAGING_DIR" >&2 ;;
    esac
  fi
}
trap cleanup EXIT

STAGED_RUN="${STAGING_DIR}/results/${RUN_NAME}"
STAGED_ACCOUNTING="${STAGING_DIR}/reports/slurm_accounting_${RUN_NAME}.txt"
STAGED_VALIDATION="${STAGING_DIR}/reports/${RUN_NAME}_import_validation.json"
STAGED_CONFIG="${STAGING_DIR}/${LOCKED_CONFIG_RELATIVE}"
STAGED_ATTEMPT="${STAGING_DIR}/${ATTEMPT_RELATIVE}"
mkdir -p \
  "$STAGED_RUN" \
  "${STAGING_DIR}/logs" \
  "${STAGING_DIR}/reports" \
  "$(dirname "$STAGED_CONFIG")" \
  "$(dirname "$STAGED_ATTEMPT")"
cp -p -- "$LOCKED_CONFIG" "$STAGED_CONFIG"
cp -p -- "$ATTEMPT_PATH" "$STAGED_ATTEMPT"

ssh "$REMOTE" bash -s -- \
  "$REMOTE_PROJECT_DIR" \
  "$JOB_ID" \
  "$EXPECTED_ACCOUNT" \
  "$EXPECTED_PARTITION" \
  "$EXPECTED_JOB_NAME" \
  "$EXPECTED_ARRAY_TASKS" \
  "$EXPECTED_CPUS" \
  "$RUN_NAME" >"$STAGED_ACCOUNTING" <<'REMOTE_SCRIPT'
set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "remote full-screening validation received an unexpected argument count" >&2
  exit 2
fi

PROJECT_ROOT="$1"
JOB_ID="$2"
EXPECTED_ACCOUNT="$3"
EXPECTED_PARTITION="$4"
EXPECTED_JOB_NAME="$5"
EXPECTED_ARRAY_TASKS="$6"
EXPECTED_CPUS="$7"
RUN_NAME="$8"

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
[[ "$EXPECTED_ARRAY_TASKS" == "5" ]]
[[ "$EXPECTED_CPUS" == "2" ]]
[[ "$RUN_NAME" == "large_scale_screening_${JOB_ID}" ]]
[[ -d "$PROJECT_ROOT" && ! -L "$PROJECT_ROOT" ]]
[[ "$(cd "$PROJECT_ROOT" && pwd -P)" == "$PROJECT_ROOT" ]]

RUN_DIR="${PROJECT_ROOT}/results/${RUN_NAME}"
[[ -d "${PROJECT_ROOT}/results" && ! -L "${PROJECT_ROOT}/results" ]]
[[ -d "${PROJECT_ROOT}/logs" && ! -L "${PROJECT_ROOT}/logs" ]]
[[ -d "$RUN_DIR" && ! -L "$RUN_DIR" ]]
[[ "$(cd "$RUN_DIR" && pwd -P)" == "$RUN_DIR" ]]
if find "$RUN_DIR" -type l -print -quit | grep -q .; then
  echo "remote full-screening run contains a symlink" >&2
  exit 2
fi
find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .
for task_id in 0 1 2 3 4; do
  [[ -s "${PROJECT_ROOT}/logs/screening_safe_${JOB_ID}_${task_id}.out" ]]
  [[ -f "${PROJECT_ROOT}/logs/screening_safe_${JOB_ID}_${task_id}.err" ]]
  [[ ! -L "${PROJECT_ROOT}/logs/screening_safe_${JOB_ID}_${task_id}.out" ]]
  [[ ! -L "${PROJECT_ROOT}/logs/screening_safe_${JOB_ID}_${task_id}.err" ]]
done

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

for task_id in 0 1 2 3 4; do
  expected_id="${JOB_ID}_${task_id}"
  count="$(
    printf '%s\n' "$ACCOUNTING" |
      awk -F'|' -v expected="$expected_id" '$1 == expected {count += 1} END {print count + 0}'
  )"
  [[ "$count" == "1" ]]
  row="$(
    printf '%s\n' "$ACCOUNTING" |
      awk -F'|' -v expected="$expected_id" '$1 == expected {print}'
  )"
  IFS='|' read -r \
    ACCOUNTING_JOB_ID JOB_NAME ACCOUNT PARTITION STATE EXIT_CODE ELAPSED_RAW \
    ALLOC_CPUS REQ_MEM MAX_RSS TOTAL_CPU NODE_LIST WORK_DIR EXTRA_FIELD <<<"$row"
  [[ "$ACCOUNTING_JOB_ID" == "$expected_id" ]]
  [[ "$JOB_NAME" == "$EXPECTED_JOB_NAME" ]]
  [[ "$ACCOUNT" == "$EXPECTED_ACCOUNT" ]]
  [[ "$PARTITION" == "$EXPECTED_PARTITION" ]]
  [[ "$STATE" == "COMPLETED" || "$STATE" == "COMPLETED+" ]]
  [[ "$EXIT_CODE" == "0:0" ]]
  [[ "$ALLOC_CPUS" == "$EXPECTED_CPUS" ]]
  [[ "$WORK_DIR" == "$PROJECT_ROOT" ]]
done

printf '# schema_version=1\n'
printf '# stage=screening\n'
printf '# job_id=%s\n' "$JOB_ID"
printf '# array_tasks=%s\n' "$EXPECTED_ARRAY_TASKS"
printf '# remote_project_root=%s\n' "$PROJECT_ROOT"
printf 'JobIDRaw|JobName|Account|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|MaxRSS|TotalCPU|NodeList|WorkDir|\n'
printf '%s\n' "$ACCOUNTING"
REMOTE_SCRIPT

[[ -s "$STAGED_ACCOUNTING" ]] || fail "remote accounting capture is empty"
grep -Fqx "# schema_version=1" "$STAGED_ACCOUNTING" \
  || fail "remote accounting capture has no schema marker"
grep -Fqx "# stage=screening" "$STAGED_ACCOUNTING" \
  || fail "remote accounting capture has the wrong stage marker"
grep -Fqx "# job_id=${JOB_ID}" "$STAGED_ACCOUNTING" \
  || fail "remote accounting capture has the wrong job ID"
grep -Fqx "# array_tasks=${EXPECTED_ARRAY_TASKS}" "$STAGED_ACCOUNTING" \
  || fail "remote accounting capture has the wrong array-task count"
grep -Fqx "# remote_project_root=${REMOTE_PROJECT_DIR}" "$STAGED_ACCOUNTING" \
  || fail "remote accounting capture has the wrong project root"

for task_id in 0 1 2 3 4; do
  expected_id="${JOB_ID}_${task_id}"
  count="$(
    awk -F'|' -v expected="$expected_id" \
      '$1 == expected {count += 1} END {print count + 0}' "$STAGED_ACCOUNTING"
  )"
  [[ "$count" == "1" ]] \
    || fail "accounting does not contain exactly one row for array task ${task_id}"
  row="$(awk -F'|' -v expected="$expected_id" '$1 == expected {print}' "$STAGED_ACCOUNTING")"
  IFS='|' read -r \
    ACCOUNTING_JOB_ID JOB_NAME ACCOUNT PARTITION STATE EXIT_CODE ELAPSED_RAW \
    ALLOC_CPUS REQ_MEM MAX_RSS TOTAL_CPU NODE_LIST WORK_DIR EXTRA_FIELD <<<"$row"
  [[ "$JOB_NAME" == "$EXPECTED_JOB_NAME" ]] || fail "accounting job name mismatch"
  [[ "$ACCOUNT" == "$EXPECTED_ACCOUNT" ]] || fail "accounting account mismatch"
  [[ "$PARTITION" == "$EXPECTED_PARTITION" ]] || fail "accounting partition mismatch"
  [[ "$STATE" == "COMPLETED" || "$STATE" == "COMPLETED+" ]] \
    || fail "array task ${task_id} did not complete"
  [[ "$EXIT_CODE" == "0:0" ]] || fail "array task ${task_id} has a nonzero exit code"
  [[ "$ALLOC_CPUS" == "$EXPECTED_CPUS" ]] || fail "accounting CPU allocation mismatch"
  [[ "$WORK_DIR" == "$REMOTE_PROJECT_DIR" ]] || fail "accounting working directory mismatch"
done

rsync -a --checksum --safe-links "${REMOTE}:${REMOTE_RUN_DIR}/" "${STAGED_RUN}/"
for task_id in 0 1 2 3 4; do
  scp \
    "${REMOTE}:${REMOTE_PROJECT_DIR}/logs/screening_safe_${JOB_ID}_${task_id}.out" \
    "${STAGING_DIR}/logs/screening_safe_${JOB_ID}_${task_id}.out"
  scp \
    "${REMOTE}:${REMOTE_PROJECT_DIR}/logs/screening_safe_${JOB_ID}_${task_id}.err" \
    "${STAGING_DIR}/logs/screening_safe_${JOB_ID}_${task_id}.err"
done
[[ -d "$STAGED_RUN" && ! -L "$STAGED_RUN" ]] \
  || fail "staged full-screening run directory is missing"
find "$STAGED_RUN" -mindepth 1 -print -quit | grep -q . \
  || fail "staged full-screening run directory is empty"
if find "$STAGED_RUN" -type l -print -quit | grep -q .; then
  fail "staged full-screening run contains a symlink"
fi

"$LOCAL_PYTHON" -m scripts.validate_screening_storage_safe_results \
  --project-root "$STAGING_DIR" \
  --run-dir "results/${RUN_NAME}" \
  --config "$LOCKED_CONFIG_RELATIVE" \
  --attempt "$ATTEMPT_RELATIVE" \
  --job-id "$JOB_ID" >"$STAGED_VALIDATION"
[[ -s "$STAGED_VALIDATION" ]] \
  || fail "full-screening import validation is empty"
printf '%s\n' "$(sha256_file "$STAGED_VALIDATION")" \
  >"${STAGED_VALIDATION}.sha256"

mkdir -p "${LOCAL_ROOT}/results" "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/reports"
mv "$STAGED_RUN" "$LOCAL_RUN_DIR"
for task_id in 0 1 2 3 4; do
  mv \
    "${STAGING_DIR}/logs/screening_safe_${JOB_ID}_${task_id}.out" \
    "${LOCAL_ROOT}/logs/screening_safe_${JOB_ID}_${task_id}.out"
  mv \
    "${STAGING_DIR}/logs/screening_safe_${JOB_ID}_${task_id}.err" \
    "${LOCAL_ROOT}/logs/screening_safe_${JOB_ID}_${task_id}.err"
done
mv "$STAGED_ACCOUNTING" "$LOCAL_ACCOUNTING"
mv "$STAGED_VALIDATION" "$LOCAL_VALIDATION"
mv "${STAGED_VALIDATION}.sha256" "${LOCAL_VALIDATION}.sha256"
rm -f -- "$STAGED_CONFIG" "$STAGED_ATTEMPT"
rmdir \
  "${STAGING_DIR}/campaigns/large_scale_thai_drl_v2/derived" \
  "${STAGING_DIR}/campaigns/large_scale_thai_drl_v2/attempts" \
  "${STAGING_DIR}/campaigns/large_scale_thai_drl_v2" \
  "${STAGING_DIR}/campaigns" \
  "${STAGING_DIR}/results" \
  "${STAGING_DIR}/logs" \
  "${STAGING_DIR}/reports" \
  "$STAGING_DIR"
STAGING_DIR=""

echo "status=fetched"
echo "job_id=${JOB_ID}"
echo "array_tasks=${EXPECTED_ARRAY_TASKS}"
echo "job_state=COMPLETED"
echo "run_dir=${LOCAL_RUN_DIR}"
echo "accounting=${LOCAL_ACCOUNTING}"
echo "validation=${LOCAL_VALIDATION}"
