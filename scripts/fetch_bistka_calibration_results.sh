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

if [[ "$#" -ne 2 ]]; then
  fail "usage: bash scripts/fetch_bistka_calibration_results.sh VECTOR_JOB_ID DETERMINISM_JOB_ID"
fi

VECTOR_JOB_ID="$1"
DETERMINISM_JOB_ID="$2"
[[ "$VECTOR_JOB_ID" =~ ^[1-9][0-9]*$ ]] \
  || fail "VECTOR_JOB_ID must be one positive numeric SLURM job ID"
[[ "$DETERMINISM_JOB_ID" =~ ^[1-9][0-9]*$ ]] \
  || fail "DETERMINISM_JOB_ID must be one positive numeric SLURM job ID"
[[ "$VECTOR_JOB_ID" != "$DETERMINISM_JOB_ID" ]] \
  || fail "VECTOR_JOB_ID and DETERMINISM_JOB_ID must be distinct allocations"

REMOTE="${BISTKA_REMOTE:-s00658@10.205.100.101}"
EXPECTED_REMOTE="${BISTKA_EXPECTED_REMOTE:-s00658@10.205.100.101}"
REMOTE_PROJECT_DIR="${BISTKA_PROJECT_DIR:-${PROJECT_DIR:-/lustrefs/project/25sfcs03/drl_thai_stock}}"
EXPECTED_PROJECT_DIR="${BISTKA_EXPECTED_PROJECT_DIR:-/lustrefs/project/25sfcs03/drl_thai_stock}"
LOCAL_ROOT_INPUT="${LOCAL_ROOT:-.}"
EXPECTED_LOCAL_ROOT_INPUT="${BISTKA_EXPECTED_LOCAL_ROOT:-$(pwd -P)}"
LOCAL_PYTHON="${LOCAL_PYTHON:-python3}"
EXPECTED_ACCOUNT="25sfcs03"
EXPECTED_PARTITION="compute-devel"

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

for command_name in ssh scp "$LOCAL_PYTHON"; do
  command -v "$command_name" >/dev/null 2>&1 \
    || fail "required command is missing: $command_name"
done

VECTOR_CSV_NAME="vectorized_training_benchmark_${VECTOR_JOB_ID}.csv"
VECTOR_METADATA_NAME="vectorized_training_benchmark_${VECTOR_JOB_ID}.json"
VECTOR_RECEIPT_NAME="${VECTOR_CSV_NAME}.receipt.json"
VECTOR_STDOUT_NAME="vector_benchmark_${VECTOR_JOB_ID}.out"
VECTOR_STDERR_NAME="vector_benchmark_${VECTOR_JOB_ID}.err"
VECTOR_ACCOUNTING_NAME="slurm_accounting_vector_benchmark_${VECTOR_JOB_ID}.txt"

DETERMINISM_RESULT_NAME="determinism_${DETERMINISM_JOB_ID}.json"
DETERMINISM_RECEIPT_NAME="${DETERMINISM_RESULT_NAME}.receipt.json"
DETERMINISM_STDOUT_NAME="determinism_${DETERMINISM_JOB_ID}.out"
DETERMINISM_STDERR_NAME="determinism_${DETERMINISM_JOB_ID}.err"
DETERMINISM_ACCOUNTING_NAME="slurm_accounting_determinism_${DETERMINISM_JOB_ID}.txt"

LOCAL_VECTOR_CSV="${LOCAL_ROOT}/reports/${VECTOR_CSV_NAME}"
LOCAL_VECTOR_METADATA="${LOCAL_ROOT}/reports/${VECTOR_METADATA_NAME}"
LOCAL_VECTOR_RECEIPT="${LOCAL_ROOT}/reports/${VECTOR_RECEIPT_NAME}"
LOCAL_VECTOR_STDOUT="${LOCAL_ROOT}/logs/${VECTOR_STDOUT_NAME}"
LOCAL_VECTOR_STDERR="${LOCAL_ROOT}/logs/${VECTOR_STDERR_NAME}"
LOCAL_VECTOR_ACCOUNTING="${LOCAL_ROOT}/reports/${VECTOR_ACCOUNTING_NAME}"
LOCAL_DETERMINISM_RESULT="${LOCAL_ROOT}/reports/${DETERMINISM_RESULT_NAME}"
LOCAL_DETERMINISM_RECEIPT="${LOCAL_ROOT}/reports/${DETERMINISM_RECEIPT_NAME}"
LOCAL_DETERMINISM_STDOUT="${LOCAL_ROOT}/logs/${DETERMINISM_STDOUT_NAME}"
LOCAL_DETERMINISM_STDERR="${LOCAL_ROOT}/logs/${DETERMINISM_STDERR_NAME}"
LOCAL_DETERMINISM_ACCOUNTING="${LOCAL_ROOT}/reports/${DETERMINISM_ACCOUNTING_NAME}"

DESTINATIONS=(
  "$LOCAL_VECTOR_CSV"
  "$LOCAL_VECTOR_METADATA"
  "$LOCAL_VECTOR_RECEIPT"
  "$LOCAL_VECTOR_STDOUT"
  "$LOCAL_VECTOR_STDERR"
  "$LOCAL_VECTOR_ACCOUNTING"
  "$LOCAL_DETERMINISM_RESULT"
  "$LOCAL_DETERMINISM_RECEIPT"
  "$LOCAL_DETERMINISM_STDOUT"
  "$LOCAL_DETERMINISM_STDERR"
  "$LOCAL_DETERMINISM_ACCOUNTING"
)

for destination in "${DESTINATIONS[@]}"; do
  [[ ! -e "$destination" && ! -L "$destination" ]] \
    || fail "refusing to overwrite existing local calibration evidence: $destination"
done

for directory in reports logs; do
  local_directory="${LOCAL_ROOT}/${directory}"
  if [[ -e "$local_directory" || -L "$local_directory" ]]; then
    [[ -d "$local_directory" && ! -L "$local_directory" ]] \
      || fail "local evidence parent is not a non-symlink directory: $local_directory"
  fi
done

STAGING_DIR="$(
  mktemp -d \
    "${LOCAL_ROOT}/.calibration_fetch_${VECTOR_JOB_ID}_${DETERMINISM_JOB_ID}.XXXXXX"
)"
cleanup() {
  if [[ -n "${STAGING_DIR:-}" && -d "$STAGING_DIR" ]]; then
    case "$STAGING_DIR" in
      "${LOCAL_ROOT}/.calibration_fetch_${VECTOR_JOB_ID}_${DETERMINISM_JOB_ID}."*)
        rm -rf -- "$STAGING_DIR"
        ;;
      *)
        echo "refusing to clean an unexpected staging path: $STAGING_DIR" >&2
        ;;
    esac
  fi
}
trap cleanup EXIT

STAGED_REPORTS="${STAGING_DIR}/reports"
STAGED_LOGS="${STAGING_DIR}/logs"
mkdir -p "$STAGED_REPORTS" "$STAGED_LOGS"

STAGED_VECTOR_CSV="${STAGED_REPORTS}/${VECTOR_CSV_NAME}"
STAGED_VECTOR_METADATA="${STAGED_REPORTS}/${VECTOR_METADATA_NAME}"
STAGED_VECTOR_RECEIPT="${STAGED_REPORTS}/${VECTOR_RECEIPT_NAME}"
STAGED_VECTOR_STDOUT="${STAGED_LOGS}/${VECTOR_STDOUT_NAME}"
STAGED_VECTOR_STDERR="${STAGED_LOGS}/${VECTOR_STDERR_NAME}"
STAGED_VECTOR_ACCOUNTING="${STAGED_REPORTS}/${VECTOR_ACCOUNTING_NAME}"
STAGED_DETERMINISM_RESULT="${STAGED_REPORTS}/${DETERMINISM_RESULT_NAME}"
STAGED_DETERMINISM_RECEIPT="${STAGED_REPORTS}/${DETERMINISM_RECEIPT_NAME}"
STAGED_DETERMINISM_STDOUT="${STAGED_LOGS}/${DETERMINISM_STDOUT_NAME}"
STAGED_DETERMINISM_STDERR="${STAGED_LOGS}/${DETERMINISM_STDERR_NAME}"
STAGED_DETERMINISM_ACCOUNTING="${STAGED_REPORTS}/${DETERMINISM_ACCOUNTING_NAME}"

capture_accounting() {
  local stage_id="$1"
  local job_id="$2"
  local expected_job_name="$3"
  local expected_cpus="$4"
  local output_path="$5"

  ssh "$REMOTE" bash -s -- \
    "$REMOTE_PROJECT_DIR" \
    "$stage_id" \
    "$job_id" \
    "$EXPECTED_ACCOUNT" \
    "$EXPECTED_PARTITION" \
    "$expected_job_name" \
    "$expected_cpus" >"$output_path" <<'REMOTE_SCRIPT'
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "remote calibration fetch validation received an unexpected argument count" >&2
  exit 2
fi

PROJECT_ROOT="$1"
STAGE_ID="$2"
JOB_ID="$3"
EXPECTED_ACCOUNT="$4"
EXPECTED_PARTITION="$5"
EXPECTED_JOB_NAME="$6"
EXPECTED_CPUS="$7"

safe_absolute() {
  local value="$1"
  [[ "$value" =~ ^/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || return 1
  [[ "/$value/" != *"/./"* && "/$value/" != *"/../"* ]]
}

safe_absolute "$PROJECT_ROOT"
[[ "$STAGE_ID" == "vector_benchmark" || "$STAGE_ID" == "determinism" ]]
[[ "$JOB_ID" =~ ^[1-9][0-9]*$ ]]
[[ "$EXPECTED_ACCOUNT" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "$EXPECTED_PARTITION" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "$EXPECTED_JOB_NAME" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "$EXPECTED_CPUS" =~ ^[1-9][0-9]*$ ]]
[[ -d "$PROJECT_ROOT" && ! -L "$PROJECT_ROOT" ]]
[[ "$(cd "$PROJECT_ROOT" && pwd -P)" == "$PROJECT_ROOT" ]]
[[ -d "${PROJECT_ROOT}/reports" && ! -L "${PROJECT_ROOT}/reports" ]]
[[ -d "${PROJECT_ROOT}/logs" && ! -L "${PROJECT_ROOT}/logs" ]]

case "$STAGE_ID" in
  vector_benchmark)
    REQUIRED_NONEMPTY=(
      "${PROJECT_ROOT}/reports/vectorized_training_benchmark_${JOB_ID}.csv"
      "${PROJECT_ROOT}/reports/vectorized_training_benchmark_${JOB_ID}.json"
      "${PROJECT_ROOT}/reports/vectorized_training_benchmark_${JOB_ID}.csv.receipt.json"
      "${PROJECT_ROOT}/logs/vector_benchmark_${JOB_ID}.out"
    )
    REQUIRED_FILES=("${PROJECT_ROOT}/logs/vector_benchmark_${JOB_ID}.err")
    ;;
  determinism)
    REQUIRED_NONEMPTY=(
      "${PROJECT_ROOT}/reports/determinism_${JOB_ID}.json"
      "${PROJECT_ROOT}/reports/determinism_${JOB_ID}.json.receipt.json"
      "${PROJECT_ROOT}/logs/determinism_${JOB_ID}.out"
    )
    REQUIRED_FILES=("${PROJECT_ROOT}/logs/determinism_${JOB_ID}.err")
    ;;
esac

for path in "${REQUIRED_NONEMPTY[@]}"; do
  [[ -f "$path" && ! -L "$path" && -s "$path" ]]
done
for path in "${REQUIRED_FILES[@]}"; do
  [[ -f "$path" && ! -L "$path" ]]
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
[[ "$ALLOC_CPUS" == "$EXPECTED_CPUS" ]]
[[ "$WORK_DIR" == "$PROJECT_ROOT" ]]
NORMALIZED_STATE="${STATE%% *}"
NORMALIZED_STATE="${NORMALIZED_STATE%%+}"
[[ "$NORMALIZED_STATE" == "COMPLETED" ]]
[[ "$EXIT_CODE" == "0:0" ]]

printf '# schema_version=1\n'
printf '# stage=%s\n' "$STAGE_ID"
printf '# job_id=%s\n' "$JOB_ID"
printf '# remote_project_root=%s\n' "$PROJECT_ROOT"
printf 'JobIDRaw|JobName|Account|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|MaxRSS|TotalCPU|NodeList|WorkDir|\n'
printf '%s\n' "$ACCOUNTING"
REMOTE_SCRIPT
}

validate_accounting() {
  local stage_id="$1"
  local job_id="$2"
  local expected_job_name="$3"
  local expected_cpus="$4"
  local accounting_path="$5"

  [[ -s "$accounting_path" && ! -L "$accounting_path" ]] \
    || fail "${stage_id} accounting capture is missing or empty"
  grep -Fqx "# schema_version=1" "$accounting_path" \
    || fail "${stage_id} accounting capture has no schema marker"
  grep -Fqx "# stage=${stage_id}" "$accounting_path" \
    || fail "${stage_id} accounting capture has the wrong stage marker"
  grep -Fqx "# job_id=${job_id}" "$accounting_path" \
    || fail "${stage_id} accounting capture has the wrong job ID"
  grep -Fqx "# remote_project_root=${REMOTE_PROJECT_DIR}" "$accounting_path" \
    || fail "${stage_id} accounting capture has the wrong project root"

  local parent_count
  local parent_row
  parent_count="$(
    awk -F'|' -v expected="$job_id" \
      '$1 == expected {count += 1} END {print count + 0}' "$accounting_path"
  )"
  [[ "$parent_count" == "1" ]] \
    || fail "${stage_id} accounting does not contain exactly one parent job row"
  parent_row="$(
    awk -F'|' -v expected="$job_id" '$1 == expected {print}' "$accounting_path"
  )"

  local accounting_job_id
  local job_name
  local account
  local partition
  local state
  local exit_code
  local elapsed_raw
  local alloc_cpus
  local req_mem
  local max_rss
  local total_cpu
  local node_list
  local work_dir
  local extra_field
  IFS='|' read -r \
    accounting_job_id \
    job_name \
    account \
    partition \
    state \
    exit_code \
    elapsed_raw \
    alloc_cpus \
    req_mem \
    max_rss \
    total_cpu \
    node_list \
    work_dir \
    extra_field <<<"$parent_row"

  [[ "$accounting_job_id" == "$job_id" ]] \
    || fail "${stage_id} accounting parent job ID mismatch"
  [[ "$job_name" == "$expected_job_name" ]] \
    || fail "${stage_id} accounting job name mismatch"
  [[ "$account" == "$EXPECTED_ACCOUNT" ]] \
    || fail "${stage_id} accounting account mismatch"
  [[ "$partition" == "$EXPECTED_PARTITION" ]] \
    || fail "${stage_id} accounting partition mismatch"
  [[ "$alloc_cpus" == "$expected_cpus" ]] \
    || fail "${stage_id} accounting CPU allocation mismatch"
  [[ "$work_dir" == "$REMOTE_PROJECT_DIR" ]] \
    || fail "${stage_id} accounting working directory mismatch"
  local normalized_state="${state%% *}"
  normalized_state="${normalized_state%%+}"
  [[ "$normalized_state" == "COMPLETED" ]] \
    || fail "${stage_id} job did not complete successfully: $state"
  [[ "$exit_code" == "0:0" ]] \
    || fail "${stage_id} completed job has a nonzero exit code: $exit_code"
}

capture_accounting \
  "vector_benchmark" \
  "$VECTOR_JOB_ID" \
  "drl_vec_bench" \
  "18" \
  "$STAGED_VECTOR_ACCOUNTING"
capture_accounting \
  "determinism" \
  "$DETERMINISM_JOB_ID" \
  "drl_determinism" \
  "2" \
  "$STAGED_DETERMINISM_ACCOUNTING"
validate_accounting \
  "vector_benchmark" \
  "$VECTOR_JOB_ID" \
  "drl_vec_bench" \
  "18" \
  "$STAGED_VECTOR_ACCOUNTING"
validate_accounting \
  "determinism" \
  "$DETERMINISM_JOB_ID" \
  "drl_determinism" \
  "2" \
  "$STAGED_DETERMINISM_ACCOUNTING"

fetch_exact() {
  local remote_path="$1"
  local staged_path="$2"
  scp "${REMOTE}:${remote_path}" "$staged_path"
}

fetch_exact \
  "${REMOTE_PROJECT_DIR}/reports/${VECTOR_CSV_NAME}" \
  "$STAGED_VECTOR_CSV"
fetch_exact \
  "${REMOTE_PROJECT_DIR}/reports/${VECTOR_METADATA_NAME}" \
  "$STAGED_VECTOR_METADATA"
fetch_exact \
  "${REMOTE_PROJECT_DIR}/reports/${VECTOR_RECEIPT_NAME}" \
  "$STAGED_VECTOR_RECEIPT"
fetch_exact \
  "${REMOTE_PROJECT_DIR}/logs/${VECTOR_STDOUT_NAME}" \
  "$STAGED_VECTOR_STDOUT"
fetch_exact \
  "${REMOTE_PROJECT_DIR}/logs/${VECTOR_STDERR_NAME}" \
  "$STAGED_VECTOR_STDERR"
fetch_exact \
  "${REMOTE_PROJECT_DIR}/reports/${DETERMINISM_RESULT_NAME}" \
  "$STAGED_DETERMINISM_RESULT"
fetch_exact \
  "${REMOTE_PROJECT_DIR}/reports/${DETERMINISM_RECEIPT_NAME}" \
  "$STAGED_DETERMINISM_RECEIPT"
fetch_exact \
  "${REMOTE_PROJECT_DIR}/logs/${DETERMINISM_STDOUT_NAME}" \
  "$STAGED_DETERMINISM_STDOUT"
fetch_exact \
  "${REMOTE_PROJECT_DIR}/logs/${DETERMINISM_STDERR_NAME}" \
  "$STAGED_DETERMINISM_STDERR"

REQUIRED_NONEMPTY=(
  "$STAGED_VECTOR_CSV"
  "$STAGED_VECTOR_METADATA"
  "$STAGED_VECTOR_RECEIPT"
  "$STAGED_VECTOR_STDOUT"
  "$STAGED_VECTOR_ACCOUNTING"
  "$STAGED_DETERMINISM_RESULT"
  "$STAGED_DETERMINISM_RECEIPT"
  "$STAGED_DETERMINISM_STDOUT"
  "$STAGED_DETERMINISM_ACCOUNTING"
)
REQUIRED_FILES=(
  "$STAGED_VECTOR_STDERR"
  "$STAGED_DETERMINISM_STDERR"
)
for path in "${REQUIRED_NONEMPTY[@]}"; do
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] \
    || fail "required fetched calibration artifact is missing, empty, or unsafe: $path"
done
for path in "${REQUIRED_FILES[@]}"; do
  [[ -f "$path" && ! -L "$path" ]] \
    || fail "required fetched calibration log is missing or unsafe: $path"
done

"$LOCAL_PYTHON" - \
  "$REMOTE_PROJECT_DIR" \
  "$VECTOR_JOB_ID" \
  "$STAGED_VECTOR_CSV" \
  "$STAGED_VECTOR_METADATA" \
  "$STAGED_VECTOR_RECEIPT" \
  "$DETERMINISM_JOB_ID" \
  "$STAGED_DETERMINISM_RESULT" \
  "$STAGED_DETERMINISM_RECEIPT" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


(
    remote_root,
    vector_job_id,
    vector_csv_raw,
    vector_metadata_raw,
    vector_receipt_raw,
    determinism_job_id,
    determinism_result_raw,
    determinism_receipt_raw,
) = sys.argv[1:]

vector_csv = Path(vector_csv_raw)
vector_metadata = Path(vector_metadata_raw)
vector_receipt = Path(vector_receipt_raw)
determinism_result = Path(determinism_result_raw)
determinism_receipt = Path(determinism_receipt_raw)
sha_pattern = re.compile(r"^[0-9a-f]{64}$")


def load_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid JSON calibration artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"calibration artifact must contain one JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_receipt(
    *,
    path: Path,
    stage_id: str,
    job_id: str,
    calibration_kind: str,
    expected_outputs: dict[str, tuple[str, Path]],
) -> None:
    receipt = load_object(path)
    expected_values = {
        "schema_version": 1,
        "status": "success",
        "calibration_kind": calibration_kind,
        "campaign_stage_id": stage_id,
        "slurm_job_id": job_id,
    }
    for field, expected in expected_values.items():
        if receipt.get(field) != expected:
            raise SystemExit(
                f"{stage_id} receipt {field} mismatch: "
                f"expected {expected!r}, got {receipt.get(field)!r}"
            )
    attempt_id = receipt.get("campaign_attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.startswith(f"{stage_id}_"):
        raise SystemExit(f"{stage_id} receipt has no stage-bound campaign attempt ID")
    for field in (
        "campaign_attempt_sha256",
        "deployment_release_id",
        "bistka_preflight_sha256",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or sha_pattern.fullmatch(value) is None:
            raise SystemExit(f"{stage_id} receipt has an invalid {field}")
    hostname = receipt.get("hostname")
    if not isinstance(hostname, str) or not hostname.startswith("bistka-"):
        raise SystemExit(f"{stage_id} receipt was not produced on a BistKA host")
    raw_outputs = receipt.get("outputs")
    if not isinstance(raw_outputs, list):
        raise SystemExit(f"{stage_id} receipt outputs are missing")
    outputs = {
        item.get("name"): item
        for item in raw_outputs
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if set(outputs) != set(expected_outputs):
        raise SystemExit(
            f"{stage_id} receipt output names mismatch: "
            f"expected {sorted(expected_outputs)}, got {sorted(outputs)}"
        )
    for name, (remote_path, local_path) in expected_outputs.items():
        record = outputs[name]
        if record.get("path") != remote_path or record.get("exists") is not True:
            raise SystemExit(f"{stage_id} receipt output path mismatch for {name}")
        expected_bytes = local_path.stat().st_size
        if record.get("bytes") != expected_bytes:
            raise SystemExit(f"{stage_id} receipt byte count mismatch for {name}")
        expected_sha256 = sha256_file(local_path)
        if record.get("sha256") != expected_sha256:
            raise SystemExit(f"{stage_id} receipt SHA-256 mismatch for {name}")


vector_metadata_payload = load_object(vector_metadata)
if vector_metadata_payload.get("schema_version") != 2:
    raise SystemExit("vector benchmark metadata does not use schema version 2")
expected_vector_csv = f"{remote_root}/reports/vectorized_training_benchmark_{vector_job_id}.csv"
expected_vector_metadata = (
    f"{remote_root}/reports/vectorized_training_benchmark_{vector_job_id}.json"
)
if vector_metadata_payload.get("results_file") != expected_vector_csv:
    raise SystemExit("vector benchmark metadata results_file is not job-bound")
validate_receipt(
    path=vector_receipt,
    stage_id="vector_benchmark",
    job_id=vector_job_id,
    calibration_kind="vectorized_training_benchmark",
    expected_outputs={
        "benchmark_csv": (expected_vector_csv, vector_csv),
        "benchmark_metadata": (expected_vector_metadata, vector_metadata),
    },
)

determinism_payload = load_object(determinism_result)
if determinism_payload.get("deterministic") is not True:
    raise SystemExit("determinism result does not report deterministic=true")
expected_determinism_result = (
    f"{remote_root}/reports/determinism_{determinism_job_id}.json"
)
validate_receipt(
    path=determinism_receipt,
    stage_id="determinism",
    job_id=determinism_job_id,
    calibration_kind="fixed_seed_determinism",
    expected_outputs={
        "determinism_result": (expected_determinism_result, determinism_result),
    },
)
PY

for destination in "${DESTINATIONS[@]}"; do
  [[ ! -e "$destination" && ! -L "$destination" ]] \
    || fail "refusing to overwrite calibration evidence created during fetch: $destination"
done

mkdir -p "${LOCAL_ROOT}/reports" "${LOCAL_ROOT}/logs"

publish_file() {
  local staged_path="$1"
  local destination="$2"
  mv -n -- "$staged_path" "$destination"
  [[ ! -e "$staged_path" && -f "$destination" && ! -L "$destination" ]] \
    || fail "could not publish calibration evidence without overwriting: $destination"
}

publish_file "$STAGED_VECTOR_CSV" "$LOCAL_VECTOR_CSV"
publish_file "$STAGED_VECTOR_METADATA" "$LOCAL_VECTOR_METADATA"
publish_file "$STAGED_VECTOR_RECEIPT" "$LOCAL_VECTOR_RECEIPT"
publish_file "$STAGED_VECTOR_STDOUT" "$LOCAL_VECTOR_STDOUT"
publish_file "$STAGED_VECTOR_STDERR" "$LOCAL_VECTOR_STDERR"
publish_file "$STAGED_VECTOR_ACCOUNTING" "$LOCAL_VECTOR_ACCOUNTING"
publish_file "$STAGED_DETERMINISM_RESULT" "$LOCAL_DETERMINISM_RESULT"
publish_file "$STAGED_DETERMINISM_RECEIPT" "$LOCAL_DETERMINISM_RECEIPT"
publish_file "$STAGED_DETERMINISM_STDOUT" "$LOCAL_DETERMINISM_STDOUT"
publish_file "$STAGED_DETERMINISM_STDERR" "$LOCAL_DETERMINISM_STDERR"
publish_file "$STAGED_DETERMINISM_ACCOUNTING" "$LOCAL_DETERMINISM_ACCOUNTING"

rmdir "$STAGED_REPORTS" "$STAGED_LOGS" "$STAGING_DIR"
STAGING_DIR=""

echo "status=fetched"
echo "vector_job_id=${VECTOR_JOB_ID}"
echo "vector_job_state=COMPLETED"
echo "determinism_job_id=${DETERMINISM_JOB_ID}"
echo "determinism_job_state=COMPLETED"
echo "vector_csv=${LOCAL_VECTOR_CSV}"
echo "vector_metadata=${LOCAL_VECTOR_METADATA}"
echo "vector_receipt=${LOCAL_VECTOR_RECEIPT}"
echo "vector_accounting=${LOCAL_VECTOR_ACCOUNTING}"
echo "determinism_result=${LOCAL_DETERMINISM_RESULT}"
echo "determinism_receipt=${LOCAL_DETERMINISM_RECEIPT}"
echo "determinism_accounting=${LOCAL_DETERMINISM_ACCOUNTING}"
echo "Run the vector_benchmark and determinism campaign gates locally after inspecting the copied evidence."
