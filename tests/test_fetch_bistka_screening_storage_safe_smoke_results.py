from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = (
    PROJECT_ROOT / "scripts/fetch_bistka_screening_storage_safe_smoke_results.sh"
)
REMOTE_ROOT = "/lustrefs/project/25sfcs03/drl_thai_stock"


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
job_name="${TEST_JOB_NAME:-drl_safe_smoke}"
root="${TEST_REMOTE_ROOT:?}"
cat <<EOF
# schema_version=1
# stage=screening_storage_safe_smoke
# job_id=${job_id}
# remote_project_root=${root}
JobIDRaw|JobName|Account|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|MaxRSS|TotalCPU|NodeList|WorkDir|
${job_id}|${job_name}|25sfcs03|compute-devel|COMPLETED|0:0|120|2|8Gn|||bistka-master|${root}|
${job_id}.batch|batch|25sfcs03|compute-devel|COMPLETED|0:0|120|2|||00:01:30|bistka-master|${root}|
EOF
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
  *.out) printf 'storage-safe smoke complete\n' >"$destination" ;;
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
printf '{"schema_version":1,"status":"pass","checkpoint_paths":0}\n'
""",
    )
    return fake_bin, fake_python


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


def test_storage_safe_fetch_rejects_non_numeric_job_id(tmp_path: Path) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()
    fake_bin, fake_python = _fake_commands(tmp_path)

    result = _run(local_root, fake_bin, fake_python, "12x")

    assert result.returncode == 2
    assert "positive numeric" in result.stderr


def test_storage_safe_fetch_imports_exact_paths_atomically(tmp_path: Path) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()
    fake_bin, fake_python = _fake_commands(tmp_path)

    result = _run(local_root, fake_bin, fake_python, "12345")

    assert result.returncode == 0, result.stderr
    run = local_root / "results/screening_storage_safe_smoke_12345"
    assert (run / "placeholder.json").is_file()
    assert (local_root / "logs/screening_safe_smoke_12345.out").is_file()
    assert (local_root / "logs/screening_safe_smoke_12345.err").is_file()
    accounting = (
        local_root
        / "reports/slurm_accounting_screening_storage_safe_smoke_12345.txt"
    )
    assert accounting.is_file()
    validation = (
        local_root
        / "reports/screening_storage_safe_smoke_12345_import_validation.json"
    )
    assert validation.is_file()
    assert Path(str(validation) + ".sha256").is_file()
    assert "job_state=COMPLETED" in result.stdout
    assert not list(local_root.glob(".storage_safe_smoke_fetch_12345.*"))


def test_storage_safe_fetch_rejects_another_job_name_without_import(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()
    fake_bin, fake_python = _fake_commands(tmp_path)

    result = _run(
        local_root,
        fake_bin,
        fake_python,
        "12345",
        environment={"TEST_JOB_NAME": "drl_screen_smoke"},
    )

    assert result.returncode == 2
    assert "accounting job name mismatch" in result.stderr
    assert not (local_root / "results/screening_storage_safe_smoke_12345").exists()


def test_storage_safe_fetch_refuses_overwrite(tmp_path: Path) -> None:
    local_root = tmp_path / "project"
    existing = local_root / "results/screening_storage_safe_smoke_12345"
    existing.mkdir(parents=True)
    fake_bin, fake_python = _fake_commands(tmp_path)

    result = _run(local_root, fake_bin, fake_python, "12345")

    assert result.returncode == 2
    assert "refusing to overwrite" in result.stderr
