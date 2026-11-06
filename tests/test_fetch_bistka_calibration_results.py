from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "fetch_bistka_calibration_results.sh"
REMOTE_ROOT = "/lustrefs/project/25sfcs03/drl_thai_stock"


def _run(
    local_root: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "LOCAL_ROOT": str(local_root),
            "BISTKA_EXPECTED_LOCAL_ROOT": str(local_root),
        }
    )
    env.update(environment or {})
    return subprocess.run(
        ["bash", str(FETCH_SCRIPT), *arguments],
        cwd=local_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _fake_transfer_commands(directory: Path) -> Path:
    fake_bin = directory / "fake-bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "ssh",
        """#!/bin/bash
set -euo pipefail
cat >/dev/null
root="${5:?}"
stage="${6:?}"
job_id="${7:?}"
job_name="${10:?}"
cpus="${11:?}"
state="COMPLETED"
exit_code="0:0"
if [[ "${TEST_BAD_STAGE:-}" == "$stage" ]]; then
  state="${TEST_BAD_STATE:-FAILED}"
  exit_code="${TEST_BAD_EXIT_CODE:-1:0}"
fi
cat <<EOF
# schema_version=1
# stage=${stage}
# job_id=${job_id}
# remote_project_root=${root}
JobIDRaw|JobName|Account|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|MaxRSS|TotalCPU|NodeList|WorkDir|
${job_id}|${job_name}|25sfcs03|compute-devel|${state}|${exit_code}|120|${cpus}|4Gn|||bistka-compute-01|${root}|
${job_id}.batch|batch|25sfcs03|compute-devel|${state}|${exit_code}|120|${cpus}|||00:01:30|bistka-compute-01|${root}|
EOF
""",
    )
    _write_executable(
        fake_bin / "scp",
        """#!/bin/bash
set -euo pipefail
destination="${!#}"
mkdir -p "$(dirname "$destination")"
base="$(basename "$destination")"
root="${TEST_REMOTE_ROOT:?}"

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

case "$base" in
  vectorized_training_benchmark_*.csv)
    printf 'num_envs,repeat,samples_per_second\\n1,1,10.0\\n' >"$destination"
    ;;
  vectorized_training_benchmark_*.csv.receipt.json)
    job_id="${base#vectorized_training_benchmark_}"
    job_id="${job_id%.csv.receipt.json}"
    csv="$(dirname "$destination")/vectorized_training_benchmark_${job_id}.csv"
    metadata="$(dirname "$destination")/vectorized_training_benchmark_${job_id}.json"
    csv_sha="$(hash_file "$csv")"
    metadata_sha="$(hash_file "$metadata")"
    csv_bytes="$(wc -c <"$csv" | tr -d '[:space:]')"
    metadata_bytes="$(wc -c <"$metadata" | tr -d '[:space:]')"
    cat >"$destination" <<EOF
{"schema_version":1,"status":"success","calibration_kind":"vectorized_training_benchmark","campaign_attempt_id":"vector_benchmark_20260716T000000Z_deadbeef","campaign_attempt_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","campaign_stage_id":"vector_benchmark","deployment_release_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bistka_preflight_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","slurm_job_id":"${job_id}","hostname":"bistka-compute-01","outputs":[{"name":"benchmark_csv","path":"${root}/reports/vectorized_training_benchmark_${job_id}.csv","exists":true,"sha256":"${csv_sha}","bytes":${csv_bytes}},{"name":"benchmark_metadata","path":"${root}/reports/vectorized_training_benchmark_${job_id}.json","exists":true,"sha256":"${metadata_sha}","bytes":${metadata_bytes}}]}
EOF
    ;;
  vectorized_training_benchmark_*.json)
    job_id="${base#vectorized_training_benchmark_}"
    job_id="${job_id%.json}"
    printf '{"schema_version":2,"results_file":"%s/reports/vectorized_training_benchmark_%s.csv"}\\n' \
      "$root" "$job_id" >"$destination"
    ;;
  determinism_*.json.receipt.json)
    job_id="${base#determinism_}"
    job_id="${job_id%.json.receipt.json}"
    result="$(dirname "$destination")/determinism_${job_id}.json"
    result_sha="$(hash_file "$result")"
    result_bytes="$(wc -c <"$result" | tr -d '[:space:]')"
    cat >"$destination" <<EOF
{"schema_version":1,"status":"success","calibration_kind":"fixed_seed_determinism","campaign_attempt_id":"determinism_20260716T000000Z_deadbeef","campaign_attempt_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","campaign_stage_id":"determinism","deployment_release_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bistka_preflight_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","slurm_job_id":"${job_id}","hostname":"bistka-compute-01","outputs":[{"name":"determinism_result","path":"${root}/reports/determinism_${job_id}.json","exists":true,"sha256":"${result_sha}","bytes":${result_bytes}}]}
EOF
    ;;
  determinism_*.json)
    printf '{"deterministic":true}\\n' >"$destination"
    ;;
  vector_benchmark_*.out|determinism_*.out)
    printf 'calibration complete\\n' >"$destination"
    ;;
  vector_benchmark_*.err|determinism_*.err)
    : >"$destination"
    ;;
  *)
    echo "unexpected exact fetch destination: $destination" >&2
    exit 2
    ;;
esac

if [[ "${TEST_EMPTY_BASENAME:-}" == "$base" ]]; then
  : >"$destination"
fi
""",
    )
    return fake_bin


def _fake_environment(
    fake_bin: Path,
    **overrides: str,
) -> dict[str, str]:
    environment = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TEST_REMOTE_ROOT": REMOTE_ROOT,
    }
    environment.update(overrides)
    return environment


def test_fetch_calibration_requires_exactly_two_distinct_positive_ids(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()

    missing = _run(local_root, "123")
    non_numeric = _run(local_root, "123", "45x")
    duplicate = _run(local_root, "123", "123")
    extra = _run(local_root, "123", "456", "789")

    assert missing.returncode == 2
    assert "usage:" in missing.stderr
    assert non_numeric.returncode == 2
    assert "positive numeric" in non_numeric.stderr
    assert duplicate.returncode == 2
    assert "distinct allocations" in duplicate.stderr
    assert extra.returncode == 2
    assert "usage:" in extra.stderr


def test_fetch_calibration_rejects_endpoint_or_project_root_mismatch(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()

    unsafe_remote = _run(
        local_root,
        "123",
        "456",
        environment={"BISTKA_REMOTE": "s00658@10.205.100.101;touch-pwned"},
    )
    wrong_root = _run(
        local_root,
        "123",
        "456",
        environment={"BISTKA_PROJECT_DIR": "/lustrefs/project/wrong"},
    )

    assert unsafe_remote.returncode == 2
    assert "strict user@host" in unsafe_remote.stderr
    assert wrong_root.returncode == 2
    assert "does not match the expected BistKA project root" in wrong_root.stderr
    assert not (local_root / "touch-pwned").exists()


def test_fetch_calibration_publishes_only_exact_job_bound_artifacts(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()
    fake_bin = _fake_transfer_commands(tmp_path)
    vector_job_id = "12345"
    determinism_job_id = "12346"

    result = _run(
        local_root,
        vector_job_id,
        determinism_job_id,
        environment=_fake_environment(fake_bin),
    )

    assert result.returncode == 0, result.stderr
    expected_reports = {
        f"vectorized_training_benchmark_{vector_job_id}.csv",
        f"vectorized_training_benchmark_{vector_job_id}.json",
        f"vectorized_training_benchmark_{vector_job_id}.csv.receipt.json",
        f"slurm_accounting_vector_benchmark_{vector_job_id}.txt",
        f"determinism_{determinism_job_id}.json",
        f"determinism_{determinism_job_id}.json.receipt.json",
        f"slurm_accounting_determinism_{determinism_job_id}.txt",
    }
    expected_logs = {
        f"vector_benchmark_{vector_job_id}.out",
        f"vector_benchmark_{vector_job_id}.err",
        f"determinism_{determinism_job_id}.out",
        f"determinism_{determinism_job_id}.err",
    }
    assert {path.name for path in (local_root / "reports").iterdir()} == expected_reports
    assert {path.name for path in (local_root / "logs").iterdir()} == expected_logs
    assert (
        local_root / "reports" / f"vectorized_training_benchmark_{vector_job_id}.csv"
    ).stat().st_size > 0
    assert (
        local_root / "reports" / f"determinism_{determinism_job_id}.json"
    ).stat().st_size > 0
    assert "status=fetched" in result.stdout
    assert "vector_job_state=COMPLETED" in result.stdout
    assert "determinism_job_state=COMPLETED" in result.stdout


def test_fetch_calibration_rejects_noncompleted_accounting_before_transfer(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()
    fake_bin = _fake_transfer_commands(tmp_path)

    result = _run(
        local_root,
        "12345",
        "12346",
        environment=_fake_environment(
            fake_bin,
            TEST_BAD_STAGE="determinism",
            TEST_BAD_STATE="FAILED",
            TEST_BAD_EXIT_CODE="1:0",
        ),
    )

    assert result.returncode == 2
    assert "determinism job did not complete successfully" in result.stderr
    assert not (local_root / "reports").exists()
    assert not (local_root / "logs").exists()


def test_fetch_calibration_refuses_overwrite_before_remote_access(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "project"
    reports = local_root / "reports"
    reports.mkdir(parents=True)
    existing = reports / "vectorized_training_benchmark_12345.csv"
    existing.write_text("user evidence\n", encoding="utf-8")
    fake_bin = _fake_transfer_commands(tmp_path)

    result = _run(
        local_root,
        "12345",
        "12346",
        environment=_fake_environment(fake_bin),
    )

    assert result.returncode == 2
    assert "refusing to overwrite existing local calibration evidence" in result.stderr
    assert existing.read_text(encoding="utf-8") == "user evidence\n"


def test_fetch_calibration_rejects_empty_required_artifact_without_publishing(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()
    fake_bin = _fake_transfer_commands(tmp_path)

    result = _run(
        local_root,
        "12345",
        "12346",
        environment=_fake_environment(
            fake_bin,
            TEST_EMPTY_BASENAME="determinism_12346.json",
        ),
    )

    assert result.returncode == 2
    assert "missing, empty, or unsafe" in result.stderr
    assert not (local_root / "reports").exists()
    assert not (local_root / "logs").exists()
