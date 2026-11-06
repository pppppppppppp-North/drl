from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.campaigns.bistka_deployment import build_deployment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_SCRIPT = PROJECT_ROOT / "scripts" / "stage_bistka_deployment_v2.sh"


def test_stage_v2_dry_run_validates_exact_release_without_remote_action(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    applier = project / "src" / "campaigns" / "atomic_release_apply.py"
    applier.parent.mkdir(parents=True)
    shutil.copyfile(
        PROJECT_ROOT / "src" / "campaigns" / "atomic_release_apply.py",
        applier,
    )
    payload = project / "payload.txt"
    payload.write_text("verified payload\n", encoding="utf-8")
    target = tmp_path / "remote-target"
    config = project / "deployment.yaml"
    config.write_text(
        "deployment:\n"
        "  campaign_id: dry_run_test\n"
        f"  target_project_root: {target}\n"
        "  python_executable: /safe/python3\n"
        "  runtime_configs: []\n"
        "  inputs:\n"
        "    - payload.txt\n"
        "    - src/campaigns/atomic_release_apply.py\n",
        encoding="utf-8",
    )
    manifest = project / "campaigns" / "test" / "deployment" / "manifest.json"
    bundle = project / "campaigns" / "test" / "deployment" / "release.tar.gz"
    build_deployment(
        project_root=project,
        config_path=config,
        output_manifest=manifest,
        output_bundle=bundle,
    )

    environment = os.environ.copy()
    environment.update(
        {
            "MANIFEST": str(manifest.relative_to(project)),
            "BUNDLE": str(bundle.relative_to(project)),
            "PREFLIGHT_OUTPUT": "reports/preflight.json",
            "BISTKA_EXPECTED_TARGET_ROOT": str(target),
            "LOCAL_PYTHON": sys.executable,
            "PYTHONPATH": str(PROJECT_ROOT),
        }
    )
    result = subprocess.run(
        ["bash", str(STAGE_SCRIPT), "--dry-run"],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "local_validation=pass" in result.stdout
    assert "status=dry_run" in result.stdout
    assert not target.exists()


def test_stage_v2_rejects_remote_shell_metacharacters_before_dry_run(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "MANIFEST": "missing/manifest.json",
            "BUNDLE": "missing/release.tar.gz",
            "PREFLIGHT_OUTPUT": "reports/preflight.json",
            "BISTKA_REMOTE": "s00658@10.205.100.101;touch-pwned",
        }
    )
    result = subprocess.run(
        ["bash", str(STAGE_SCRIPT), "--dry-run"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "not a strict user@host target" in result.stderr
    assert not (tmp_path / "touch-pwned").exists()
