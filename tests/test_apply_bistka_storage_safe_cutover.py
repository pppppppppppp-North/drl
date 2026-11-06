from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = PROJECT_ROOT / "scripts/apply_bistka_storage_safe_cutover.sh"


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    script = root / "scripts/apply_bistka_storage_safe_cutover.sh"
    _write_executable(script, WRAPPER.read_text(encoding="utf-8"))
    _write_executable(
        root / "scripts/stage_bistka_deployment_v2.sh",
        """#!/bin/bash
set -euo pipefail
case "${MANIFEST:?}" in
  *storage_safe_cutover*)
    printf 'fake_stage=candidate\n'
    exit "${FAKE_CANDIDATE_STATUS:-0}"
    ;;
  *bistka_screening_v2_manifest.json)
    printf 'fake_stage=rollback\n'
    exit "${FAKE_ROLLBACK_STATUS:-0}"
    ;;
  *) exit 99 ;;
esac
""",
    )
    _write_executable(root / ".venv/bin/python", "#!/bin/bash\nexit 0\n")
    artifacts = (
        "campaigns/large_scale_thai_drl_v2/deployment/"
        "bistka_screening_storage_safe_cutover_v2_manifest.json",
        "campaigns/large_scale_thai_drl_v2/deployment/"
        "bistka_screening_storage_safe_cutover_v2.tar.gz",
        "campaigns/large_scale_thai_drl_v2/deployment/"
        "bistka_screening_storage_safe_cutover_v2.tar.gz.sha256",
        "campaigns/large_scale_thai_drl_v2/deployment/"
        "bistka_screening_v2_manifest.json",
        "campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2.tar.gz",
        "campaigns/large_scale_thai_drl_v2/deployment/"
        "bistka_screening_v2.tar.gz.sha256",
    )
    for relative in artifacts:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    return root


def _run(
    root: Path,
    mode: str,
    *,
    candidate_status: int = 0,
    rollback_status: int = 0,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "FAKE_CANDIDATE_STATUS": str(candidate_status),
            "FAKE_ROLLBACK_STATUS": str(rollback_status),
        }
    )
    return subprocess.run(
        ["bash", "scripts/apply_bistka_storage_safe_cutover.sh", mode],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cutover_dry_run_validates_candidate_and_rollback(tmp_path: Path) -> None:
    result = _run(_project(tmp_path), "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "fake_stage=candidate" in result.stdout
    assert "fake_stage=rollback" in result.stdout
    assert "status=dry_run" in result.stdout


def test_cutover_apply_does_not_rollback_success(tmp_path: Path) -> None:
    result = _run(_project(tmp_path), "--apply")

    assert result.returncode == 0, result.stderr
    assert "fake_stage=candidate" in result.stdout
    assert "fake_stage=rollback" not in result.stdout
    assert "rollback=not_required" in result.stdout


def test_cutover_apply_rolls_back_candidate_or_preflight_failure(
    tmp_path: Path,
) -> None:
    result = _run(_project(tmp_path), "--apply", candidate_status=7)

    assert result.returncode == 7
    assert "fake_stage=candidate" in result.stdout
    assert "fake_stage=rollback" in result.stdout
    assert "status=rolled_back" in result.stderr


def test_cutover_reports_critical_rollback_failure(tmp_path: Path) -> None:
    result = _run(
        _project(tmp_path),
        "--apply",
        candidate_status=7,
        rollback_status=8,
    )

    assert result.returncode == 3
    assert "critical: exact active-release rollback failed" in result.stderr
