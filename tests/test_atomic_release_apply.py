from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

import src.campaigns.atomic_release_apply as release_apply
from src.campaigns.bistka_deployment import build_deployment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release(tmp_path: Path, target_root: Path, content: bytes = b"certified-data\n") -> tuple[Path, Path]:
    source = tmp_path / "source"
    artifact = source / "data" / "exact.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(content)
    manifest_path = source / "deployment" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "campaign_id": "test",
        "target_project_root": str(target_root.resolve()),
        "runtime_feature_bindings": [],
        "files": [
            {
                "path": "data/exact.csv",
                "bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
                "executable": False,
            }
        ],
    }
    manifest["release_id"] = release_apply._expected_release_id(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    bundle = tmp_path / "release.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(artifact, arcname="data/exact.csv")
        archive.add(manifest_path, arcname=".bistka/deployment_manifest.json")
        archive.add(manifest_path, arcname="deployment/manifest.json")
    return bundle, manifest_path


def test_apply_release_skips_byte_identical_large_input(tmp_path: Path) -> None:
    target = tmp_path / "target"
    existing = target / "data" / "exact.csv"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"certified-data\n")
    original_inode = existing.stat().st_ino
    original_mtime = existing.stat().st_mtime_ns
    bundle, manifest = _release(tmp_path, target)

    result = release_apply.apply_release(
        bundle_path=bundle,
        manifest_path=manifest,
        target_root=target,
        expected_bundle_sha256=_sha256(bundle),
        manifest_relative_path="deployment/manifest.json",
        staging_root=tmp_path / "staging",
        quota_margin_bytes=0,
    )

    assert result["status"] == "applied"
    assert "data/exact.csv" in result["unchanged_files"]
    assert existing.stat().st_ino == original_inode
    assert existing.stat().st_mtime_ns == original_mtime
    assert (target / "deployment" / "manifest.json").is_file()
    assert (target / ".bistka" / "deployed_bundle.sha256").read_text().strip() == _sha256(bundle)


def test_apply_release_promotes_changed_file_after_verification(tmp_path: Path) -> None:
    target = tmp_path / "target"
    existing = target / "data" / "exact.csv"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old\n")
    bundle, manifest = _release(tmp_path, target, b"new-certified-data\n")

    result = release_apply.apply_release(
        bundle_path=bundle,
        manifest_path=manifest,
        target_root=target,
        manifest_relative_path="deployment/manifest.json",
        staging_root=tmp_path / "staging",
        quota_margin_bytes=0,
    )

    assert "data/exact.csv" in result["changed_files"]
    assert existing.read_bytes() == b"new-certified-data\n"
    assert not list(existing.parent.glob("*.tmp"))


def test_apply_release_quota_failure_leaves_existing_target_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    existing = target / "data" / "exact.csv"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old\n")
    bundle, manifest = _release(tmp_path, target, b"new-certified-data\n")

    def fail_reservation(_target: Path, _required: int) -> None:
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(release_apply, "_reserve_space", fail_reservation)
    with pytest.raises(OSError, match="Disk quota exceeded"):
        release_apply.apply_release(
            bundle_path=bundle,
            manifest_path=manifest,
            target_root=target,
            manifest_relative_path="deployment/manifest.json",
            staging_root=tmp_path / "staging",
            quota_margin_bytes=0,
        )

    assert existing.read_bytes() == b"old\n"
    assert not (target / "deployment" / "manifest.json").exists()


def test_apply_release_rejects_unsafe_bundle_member(tmp_path: Path) -> None:
    target = tmp_path / "target"
    bundle, manifest = _release(tmp_path, target)
    unsafe_bundle = tmp_path / "unsafe.tar.gz"
    with tarfile.open(bundle, "r:gz") as source, tarfile.open(unsafe_bundle, "w:gz") as output:
        for member in source.getmembers():
            extracted = source.extractfile(member)
            assert extracted is not None
            payload = extracted.read()
            copy = tarfile.TarInfo(member.name)
            copy.size = len(payload)
            output.addfile(copy, io.BytesIO(payload))
        attack = b"escape"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(attack)
        output.addfile(member, io.BytesIO(attack))

    with pytest.raises(ValueError, match="unsafe bundle member"):
        release_apply.apply_release(
            bundle_path=unsafe_bundle,
            manifest_path=manifest,
            target_root=target,
            manifest_relative_path="deployment/manifest.json",
            staging_root=tmp_path / "staging",
            quota_margin_bytes=0,
        )
    assert not (tmp_path / "escape.txt").exists()


def test_apply_release_rejects_bundle_hash_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "target"
    bundle, manifest = _release(tmp_path, target)

    with pytest.raises(ValueError, match="bundle SHA-256"):
        release_apply.apply_release(
            bundle_path=bundle,
            manifest_path=manifest,
            target_root=target,
            expected_bundle_sha256="b" * 64,
            manifest_relative_path="deployment/manifest.json",
            staging_root=tmp_path / "staging",
            quota_margin_bytes=0,
        )


@pytest.mark.parametrize("linked_parent", ["data", ".bistka"])
def test_apply_release_rejects_symlinked_destination_parent(
    tmp_path: Path, linked_parent: str
) -> None:
    target = tmp_path / "target"
    outside = tmp_path / "outside"
    target.mkdir()
    outside.mkdir()
    (target / linked_parent).symlink_to(outside, target_is_directory=True)
    bundle, manifest = _release(tmp_path, target, b"must-not-escape\n")

    with pytest.raises(ValueError, match="symlinked destination parent"):
        release_apply.apply_release(
            bundle_path=bundle,
            manifest_path=manifest,
            target_root=target,
            manifest_relative_path="deployment/manifest.json",
            staging_root=tmp_path / "staging",
            quota_margin_bytes=0,
        )

    assert not (outside / "exact.csv").exists()
    assert not (outside / "deployment_manifest.json").exists()
    assert not (outside / "deployed_bundle.sha256").exists()


def test_apply_release_rejects_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target"
    destination = target / "data" / "exact.csv"
    outside = tmp_path / "outside.csv"
    destination.parent.mkdir(parents=True)
    outside.write_bytes(b"outside-old\n")
    destination.symlink_to(outside)
    bundle, manifest = _release(tmp_path, target, b"must-not-overwrite\n")

    with pytest.raises(ValueError, match="symlink destination"):
        release_apply.apply_release(
            bundle_path=bundle,
            manifest_path=manifest,
            target_root=target,
            manifest_relative_path="deployment/manifest.json",
            staging_root=tmp_path / "staging",
            quota_margin_bytes=0,
        )

    assert outside.read_bytes() == b"outside-old\n"


def test_apply_release_mid_commit_failure_restores_prior_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    old_files = {
        "data/exact.csv": b"old-data\n",
        ".bistka/deployment_manifest.json": b"old-internal-manifest\n",
        "deployment/manifest.json": b"old-external-manifest\n",
        ".bistka/deployed_bundle.sha256": ("a" * 64 + "\n").encode(),
    }
    for relative, payload in old_files.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    bundle, manifest = _release(tmp_path, target, b"new-data\n")

    actual_promote = release_apply._promote_staged
    calls = 0

    def fail_second_promotion(staged: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated mid-commit failure")
        actual_promote(staged, destination)

    monkeypatch.setattr(release_apply, "_promote_staged", fail_second_promotion)
    with pytest.raises(OSError, match="simulated mid-commit failure"):
        release_apply.apply_release(
            bundle_path=bundle,
            manifest_path=manifest,
            target_root=target,
            manifest_relative_path="deployment/manifest.json",
            staging_root=tmp_path / "staging",
            quota_margin_bytes=0,
        )

    assert calls == 2
    for relative, payload in old_files.items():
        assert (target / relative).read_bytes() == payload
    leftovers = [
        path
        for path in target.rglob("*")
        if path.is_file() and path.name.endswith((".tmp", ".bak"))
    ]
    assert leftovers == []


def test_apply_release_rejects_embedded_manifest_that_differs_from_supplied(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    bundle, manifest = _release(tmp_path, target)
    altered_bundle = tmp_path / "altered.tar.gz"
    with tarfile.open(bundle, "r:gz") as source, tarfile.open(altered_bundle, "w:gz") as output:
        for member in source.getmembers():
            extracted = source.extractfile(member)
            assert extracted is not None
            payload = extracted.read()
            if member.name == ".bistka/deployment_manifest.json":
                embedded = json.loads(payload)
                embedded["required_packages"] = ["unexpected"]
                payload = (json.dumps(embedded, indent=2) + "\n").encode()
            copy = tarfile.TarInfo(member.name)
            copy.size = len(payload)
            output.addfile(copy, io.BytesIO(payload))

    with pytest.raises(ValueError, match="bundled manifest differs"):
        release_apply.apply_release(
            bundle_path=altered_bundle,
            manifest_path=manifest,
            target_root=target,
            manifest_relative_path="deployment/manifest.json",
            staging_root=tmp_path / "staging",
            quota_margin_bytes=0,
        )


def test_apply_release_requires_explicit_manifest_destination(tmp_path: Path) -> None:
    target = tmp_path / "target"
    bundle, manifest = _release(tmp_path, target)

    with pytest.raises(ValueError, match="manifest_relative_path is required"):
        release_apply.apply_release(
            bundle_path=bundle,
            manifest_path=manifest,
            target_root=target,
            staging_root=tmp_path / "staging",
            quota_margin_bytes=0,
        )


def test_apply_release_accepts_manifest_emitted_by_real_builder(tmp_path: Path) -> None:
    project = tmp_path / "project"
    target = tmp_path / "target"
    source = project / "payload.txt"
    source.parent.mkdir(parents=True)
    source.write_text("real-builder-payload\n", encoding="utf-8")
    config = project / "deployment.yaml"
    config.write_text(
        "deployment:\n"
        "  campaign_id: integration\n"
        f"  target_project_root: {target}\n"
        "  data_readiness_path: payload.txt\n"
        "  runtime_configs: []\n"
        "  inputs:\n"
        "    - payload.txt\n",
        encoding="utf-8",
    )
    manifest = project / "deployment" / "manifest.json"
    bundle = project / "deployment" / "release.tar.gz"
    build_deployment(
        project_root=project,
        config_path=config,
        output_manifest=manifest,
        output_bundle=bundle,
    )

    result = release_apply.apply_release(
        bundle_path=bundle,
        manifest_path=manifest,
        target_root=target,
        expected_bundle_sha256=_sha256(bundle),
        manifest_relative_path="deployment/manifest.json",
        staging_root=tmp_path / "staging",
        quota_margin_bytes=0,
    )

    assert result["release_id"] == json.loads(manifest.read_text())["release_id"]
    assert (target / "payload.txt").read_text(encoding="utf-8") == "real-builder-payload\n"
    assert (target / "deployment" / "manifest.json").read_bytes() == manifest.read_bytes()
