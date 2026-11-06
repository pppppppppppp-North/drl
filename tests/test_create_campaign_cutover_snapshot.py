from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.create_campaign_cutover_snapshot import create_snapshot


def _project(tmp_path: Path) -> tuple[Path, list[str]]:
    root = tmp_path / "project"
    manifest = (
        root
        / "campaigns/large_scale_thai_drl_v2/deployment/"
        "bistka_screening_v2_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"release_id": "a" * 64}) + "\n",
        encoding="utf-8",
    )
    spec = root / "config/research_campaign.yaml"
    spec.parent.mkdir(parents=True)
    spec.write_text("campaign:\n  id: test\n", encoding="utf-8")
    state = root / "campaigns/large_scale_thai_drl_v2/state.json"
    state.write_text("{}\n", encoding="utf-8")
    return root, [
        str(spec.relative_to(root)),
        str(state.relative_to(root)),
        str(manifest.relative_to(root)),
    ]


def test_snapshot_is_deterministic_and_checksum_bound(tmp_path: Path) -> None:
    root, inputs = _project(tmp_path)

    first = create_snapshot(
        project_root=root,
        output_dir="snapshots/first",
        inputs=inputs,
    )
    second = create_snapshot(
        project_root=root,
        output_dir="snapshots/second",
        inputs=inputs,
    )

    assert first["archive"]["sha256"] == second["archive"]["sha256"]
    assert first["archive"]["file_count"] == 3
    assert first["active_release"]["release_id"] == "a" * 64
    archive = root / "snapshots/first/pre_cutover_control_and_release.tar.gz"
    assert Path(str(archive) + ".sha256").is_file()
    receipt = root / "snapshots/first/snapshot_receipt.json"
    assert Path(str(receipt) + ".sha256").is_file()
    with tarfile.open(archive, "r:gz") as handle:
        names = sorted(handle.getnames())
        spec_member = handle.extractfile("config/research_campaign.yaml")
        assert spec_member is not None
        assert io.BytesIO(spec_member.read()).getvalue() == b"campaign:\n  id: test\n"
    assert names == sorted(inputs)


def test_snapshot_refuses_existing_output(tmp_path: Path) -> None:
    root, inputs = _project(tmp_path)
    output = root / "snapshot"
    output.mkdir()

    with pytest.raises(ValueError, match="must be a new directory"):
        create_snapshot(project_root=root, output_dir=output, inputs=inputs)


def test_snapshot_rejects_symlinked_input(tmp_path: Path) -> None:
    root, inputs = _project(tmp_path)
    linked = root / "linked-state.json"
    try:
        linked.symlink_to(root / inputs[1])
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(ValueError, match="missing or unsafe"):
        create_snapshot(
            project_root=root,
            output_dir="snapshot",
            inputs=[*inputs, "linked-state.json"],
        )
