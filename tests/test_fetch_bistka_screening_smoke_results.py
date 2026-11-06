from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "fetch_bistka_screening_smoke_results.sh"
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
job_id="${TEST_JOB_ID:?}"
job_name="${TEST_JOB_NAME:-drl_screen_smoke}"
root="${TEST_REMOTE_ROOT:?}"
cat <<EOF
# schema_version=1
# stage=screening_smoke
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
printf '{"schema_version":1,"status":"complete"}\n' >"${destination}/screening_smoke_summary.json"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${destination}/screening_smoke_summary.json" | awk '{print $1}' \
    >"${destination}/screening_smoke_summary.json.sha256"
else
  shasum -a 256 "${destination}/screening_smoke_summary.json" | awk '{print $1}' \
    >"${destination}/screening_smoke_summary.json.sha256"
fi
for algorithm in ppo a2c ddpg td3 sac; do
  mkdir -p "${destination}/algorithm_${algorithm}"
  printf '{"status":"complete"}\n' \
    >"${destination}/algorithm_${algorithm}/experiment_manifest.json"
done
""",
    )
    _write_executable(
        fake_bin / "scp",
        """#!/bin/bash
set -euo pipefail
destination="${!#}"
mkdir -p "$(dirname "$destination")"
case "$destination" in
  *.out) printf 'screening smoke complete\n' >"$destination" ;;
  *.err) : >"$destination" ;;
  *) exit 2 ;;
esac
""",
    )
    return fake_bin


def test_fetch_smoke_rejects_non_numeric_or_extra_job_ids(tmp_path: Path) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()

    non_numeric = _run(local_root, "12x")
    extra = _run(local_root, "123", "456")

    assert non_numeric.returncode == 2
    assert "positive numeric" in non_numeric.stderr
    assert extra.returncode == 2
    assert "usage:" in extra.stderr


def test_fetch_smoke_rejects_remote_or_project_root_mismatch_before_transfer(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()

    unsafe_remote = _run(
        local_root,
        "123",
        environment={"BISTKA_REMOTE": "s00658@10.205.100.101;touch-pwned"},
    )
    wrong_root = _run(
        local_root,
        "123",
        environment={"BISTKA_PROJECT_DIR": "/lustrefs/project/wrong"},
    )

    assert unsafe_remote.returncode == 2
    assert "strict user@host" in unsafe_remote.stderr
    assert wrong_root.returncode == 2
    assert "does not match the expected BistKA project root" in wrong_root.stderr
    assert not (local_root / "touch-pwned").exists()


def test_fetch_smoke_stages_exact_run_logs_and_accounting_without_network(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()
    fake_bin = _fake_transfer_commands(tmp_path)
    job_id = "12345"
    environment = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TEST_JOB_ID": job_id,
        "TEST_REMOTE_ROOT": REMOTE_ROOT,
    }

    result = _run(local_root, job_id, environment=environment)

    assert result.returncode == 0, result.stderr
    run_dir = local_root / "results" / f"screening_smoke_{job_id}"
    assert (run_dir / "screening_smoke_summary.json").is_file()
    assert len(list(run_dir.rglob("experiment_manifest.json"))) == 5
    assert (local_root / "logs" / f"screening_smoke_{job_id}.out").is_file()
    assert (local_root / "logs" / f"screening_smoke_{job_id}.err").is_file()
    accounting = (
        local_root
        / "reports"
        / f"slurm_accounting_screening_smoke_{job_id}.txt"
    )
    assert accounting.is_file()
    assert f"# job_id={job_id}" in accounting.read_text(encoding="utf-8")
    assert "status=fetched" in result.stdout
    assert "job_state=COMPLETED" in result.stdout


def test_fetch_smoke_rejects_accounting_for_another_stage(tmp_path: Path) -> None:
    local_root = tmp_path / "project"
    local_root.mkdir()
    fake_bin = _fake_transfer_commands(tmp_path)
    environment = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TEST_JOB_ID": "12345",
        "TEST_JOB_NAME": "unrelated_job",
        "TEST_REMOTE_ROOT": REMOTE_ROOT,
    }

    result = _run(local_root, "12345", environment=environment)

    assert result.returncode == 2
    assert "accounting job name mismatch" in result.stderr
    assert not (local_root / "results" / "screening_smoke_12345").exists()
