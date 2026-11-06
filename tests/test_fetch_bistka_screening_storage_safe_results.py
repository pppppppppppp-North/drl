from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = (
    PROJECT_ROOT / "scripts/fetch_bistka_screening_storage_safe_results.sh"
)
REMOTE_ROOT = "/lustrefs/project/25sfcs03/drl_thai_stock"
ATTEMPT_ID = "screening_20260725T120000Z_deadbeef"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _fake_commands(directory: Path) -> tuple[Path, Path]:
    fake_bin = directory / "fake-bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "ssh",
        """#!/bin/bash
set -euo pipefail
cat >/dev/null
job_id="${TEST_JOB_ID:?}"
root="${TEST_REMOTE_ROOT:?}"
bad_task="${TEST_BAD_TASK:-none}"
printf '# schema_version=1\n'
printf '# stage=screening\n'
printf '# job_id=%s\n' "$job_id"
printf '# array_tasks=5\n'
printf '# remote_project_root=%s\n' "$root"
printf 'JobIDRaw|JobName|Account|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|MaxRSS|TotalCPU|NodeList|WorkDir|\n'
for task_id in 0 1 2 3 4; do
  state="COMPLETED"
  exit_code="0:0"
  if [[ "$task_id" == "$bad_task" ]]; then
    state="FAILED"
    exit_code="1:0"
  fi
  printf '%s_%s|drl_screen_safe|25sfcs03|compute-long|%s|%s|120|2|12Gn|||bistka-compute-01|%s|\n' \
    "$job_id" "$task_id" "$state" "$exit_code" "$root"
done
""",
    )
    _write_executable(
        fake_bin / "rsync",
        """#!/bin/bash
set -euo pipefail
destination="${!#}"
mkdir -p "$destination"
printf '{"staged":true}\n' >"${destination}/placeholder.json"
""",
    )
    _write_executable(
        fake_bin / "scp",
        """#!/bin/bash
set -euo pipefail
destination="${!#}"
mkdir -p "$(dirname "$destination")"
case "$destination" in
  *.out) printf 'screening complete\n' >"$destination" ;;
  *.err) : >"$destination" ;;
  *) exit 2 ;;
esac
""",
    )
    fake_python = directory / "fake-python"
    _write_executable(
        fake_python,
        """#!/bin/bash
set -euo pipefail
printf '{"schema_version":1,"status":"pass","complete_trials":150}\n'
""",
    )
    return fake_bin, fake_python


def _materialize_local_contract(local_root: Path) -> None:
    config = (
        local_root
        / "campaigns/large_scale_thai_drl_v2/derived/"
        "locked_large_scale_screening.yaml"
    )
    attempt = (
        local_root
        / "campaigns/large_scale_thai_drl_v2/attempts"
        / f"{ATTEMPT_ID}.json"
    )
    config.parent.mkdir(parents=True)
    attempt.parent.mkdir(parents=True)
    config.write_text("screening: {}\n", encoding="utf-8")
    attempt.write_text("{}\n", encoding="utf-8")


def _run(
    local_root: Path,
    fake_bin: Path,
    fake_python: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "LOCAL_ROOT": str(local_root),
            "BISTKA_EXPECTED_LOCAL_ROOT": str(local_root),
            "LOCAL_PYTHON": str(fake_python),
            "TEST_JOB_ID": "12345",
            "TEST_REMOTE_ROOT": REMOTE_ROOT,
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


def test_full_screening_fetch_requires_job_and_immutable_attempt(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()
    fake_bin, fake_python = _fake_commands(tmp_path)

    missing = _run(local_root, fake_bin, fake_python, "12345")
    bad_job = _run(local_root, fake_bin, fake_python, "12x", ATTEMPT_ID)
    bad_attempt = _run(
        local_root,
        fake_bin,
        fake_python,
        "12345",
        "screening_latest",
    )

    assert missing.returncode == 2
    assert "usage:" in missing.stderr
    assert bad_job.returncode == 2
    assert "positive numeric" in bad_job.stderr
    assert bad_attempt.returncode == 2
    assert "immutable screening attempt" in bad_attempt.stderr


def test_full_screening_fetch_publishes_only_after_all_tasks_validate(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()
    _materialize_local_contract(local_root)
    fake_bin, fake_python = _fake_commands(tmp_path)

    result = _run(
        local_root,
        fake_bin,
        fake_python,
        "12345",
        ATTEMPT_ID,
    )

    assert result.returncode == 0, result.stderr
    assert (
        local_root / "results/large_scale_screening_12345/placeholder.json"
    ).is_file()
    assert (
        local_root / "reports/slurm_accounting_large_scale_screening_12345.txt"
    ).is_file()
    validation = (
        local_root
        / "reports/large_scale_screening_12345_import_validation.json"
    )
    assert validation.is_file()
    assert Path(str(validation) + ".sha256").is_file()
    for task_id in range(5):
        assert (
            local_root / f"logs/screening_safe_12345_{task_id}.out"
        ).is_file()
        assert (
            local_root / f"logs/screening_safe_12345_{task_id}.err"
        ).is_file()
    assert "array_tasks=5" in result.stdout
    assert "job_state=COMPLETED" in result.stdout
    assert not list(local_root.glob(".screening_fetch_12345.*"))


def test_full_screening_fetch_rejects_one_failed_array_task_atomically(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()
    _materialize_local_contract(local_root)
    fake_bin, fake_python = _fake_commands(tmp_path)

    result = _run(
        local_root,
        fake_bin,
        fake_python,
        "12345",
        ATTEMPT_ID,
        environment={"TEST_BAD_TASK": "3"},
    )

    assert result.returncode != 0
    assert not (local_root / "results/large_scale_screening_12345").exists()
    assert not (
        local_root
        / "reports/large_scale_screening_12345_import_validation.json"
    ).exists()
    assert not list(local_root.glob(".screening_fetch_12345.*"))


def test_full_screening_fetch_refuses_overwrite(tmp_path: Path) -> None:
    local_root = tmp_path / "project"
    existing = local_root / "results/large_scale_screening_12345"
    existing.mkdir(parents=True)
    _materialize_local_contract(local_root)
    fake_bin, fake_python = _fake_commands(tmp_path)

    result = _run(
        local_root,
        fake_bin,
        fake_python,
        "12345",
        ATTEMPT_ID,
    )

    assert result.returncode == 2
    assert "refusing to overwrite" in result.stderr
