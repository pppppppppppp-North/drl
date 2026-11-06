from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.capture_screening_quota_snapshot import (
    QuotaCaptureError,
    QuotaParseError,
    capture_quota_snapshot,
    parse_lfs_quota_table,
    perform_write_probe,
    query_lfs_quota,
    write_snapshot,
)


FILESYSTEM = "/lustrefs/project/25sfcs03"
GROUP = "s00658"
GID = 1039
PROJECT_ID = 25110303


def _quota_output(
    *,
    kind: str = "grp",
    subject: str = GROUP,
    id_label: str = "gid",
    numeric_id: int = GID,
    filesystem: str = FILESYSTEM,
    blocks: str = "24693389",
    block_soft: str = "31457280",
    block_hard: str = "52428800",
    block_grace: str = "-",
    files: str = "170185",
    file_soft: str = "1000000",
    file_hard: str = "1100000",
    file_grace: str = "-",
) -> str:
    return (
        f"Disk quotas for {kind} {subject} ({id_label} {numeric_id}):\n"
        "      Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace\n"
        f"{filesystem} {blocks} {block_soft} {block_hard} {block_grace} "
        f"{files} {file_soft} {file_hard} {file_grace}\n"
    )


def _project_output(**overrides: object) -> str:
    values: dict[str, object] = {
        "kind": "prj",
        "subject": str(PROJECT_ID),
        "id_label": "pid",
        "numeric_id": PROJECT_ID,
        "blocks": "38299560",
        "block_soft": "262144000",
        "block_hard": "314572800",
        "files": "278628",
        "file_soft": "10000000",
        "file_hard": "11000000",
    }
    values.update(overrides)
    return _quota_output(**values)  # type: ignore[arg-type]


def test_parse_real_observed_group_and_project_tables() -> None:
    group = parse_lfs_quota_table(
        _quota_output(),
        filesystem=FILESYSTEM,
        subject_kind="group",
        subject=GROUP,
        numeric_id=GID,
    )
    project = parse_lfs_quota_table(
        _project_output(),
        filesystem=FILESYSTEM,
        subject_kind="project",
        subject=PROJECT_ID,
        numeric_id=PROJECT_ID,
    )

    assert group == {
        "bytes": {
            "used": 24_693_389 * 1024,
            "soft_limit": 31_457_280 * 1024,
            "hard_limit": 52_428_800 * 1024,
            "grace_expired": False,
        },
        "inodes": {
            "used": 170_185,
            "soft_limit": 1_000_000,
            "hard_limit": 1_100_000,
            "grace_expired": False,
        },
    }
    assert project["bytes"]["used"] == 38_299_560 * 1024
    assert project["inodes"]["hard_limit"] == 11_000_000


@pytest.mark.parametrize(
    ("used", "grace", "expected_expired"),
    [
        ("100", "-", False),
        ("101*", "6d23h58m", False),
        ("101*", "-", True),
        ("101*", "*", True),
        ("101*", "none", True),
        ("101*", "expired", True),
    ],
)
def test_grace_marker_countdown_and_dash_semantics(
    used: str,
    grace: str,
    expected_expired: bool,
) -> None:
    parsed = parse_lfs_quota_table(
        _quota_output(
            blocks=used,
            block_soft="100",
            block_hard="200",
            block_grace=grace,
        ),
        filesystem=FILESYSTEM,
        subject_kind="group",
        subject=GROUP,
        numeric_id=GID,
    )
    assert parsed["bytes"]["grace_expired"] is expected_expired


@pytest.mark.parametrize(
    "output",
    [
        "",
        _quota_output().replace(f"(gid {GID})", "(gid 9999)"),
        _quota_output(filesystem="/lustrefs/project/wrong"),
        _quota_output().replace("Filesystem  kbytes", "Filesystem blocks"),
        _quota_output(blocks="not-a-number"),
        _quota_output(block_soft="600", block_hard="500"),
        _quota_output(block_grace="later"),
        _quota_output() + "unexpected extra row\n",
    ],
)
def test_malformed_or_mismatched_quota_output_is_rejected(output: str) -> None:
    with pytest.raises(QuotaParseError):
        parse_lfs_quota_table(
            output,
            filesystem=FILESYSTEM,
            subject_kind="group",
            subject=GROUP,
            numeric_id=GID,
        )


def test_write_probe_fsyncs_checks_gid_and_removes_file(tmp_path: Path) -> None:
    fsync_calls: list[int] = []
    removed_paths: list[Path] = []

    def recording_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        os.fsync(descriptor)

    def recording_unlink(path: str | os.PathLike[str]) -> None:
        removed_paths.append(Path(path))
        os.unlink(path)

    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    receipt = perform_write_probe(
        output_parent=tmp_path,
        filesystem=str(tmp_path),
        expected_gid=os.getgid(),
        probe_bytes=12_345,
        clock=lambda: now,
        fsync=recording_fsync,
        unlink=recording_unlink,
    )

    assert receipt == {
        "completed_at": "2026-07-17T12:00:00Z",
        "filesystem": str(tmp_path),
        "path": str(removed_paths[0]),
        "gid": os.getgid(),
        "bytes_written": 12_345,
        "fsync_succeeded": True,
        "removed": True,
    }
    assert len(fsync_calls) == 1
    assert len(removed_paths) == 1
    assert removed_paths[0].parent == tmp_path
    assert not removed_paths[0].exists()


def test_write_probe_gid_mismatch_fails_and_still_removes(tmp_path: Path) -> None:
    removed_paths: list[Path] = []

    def recording_unlink(path: str | os.PathLike[str]) -> None:
        removed_paths.append(Path(path))
        os.unlink(path)

    with pytest.raises(QuotaCaptureError, match="unexpected effective GID"):
        perform_write_probe(
            output_parent=tmp_path,
            filesystem=str(tmp_path),
            expected_gid=os.getgid(),
            probe_bytes=64,
            fstat=lambda _fd: SimpleNamespace(st_gid=os.getgid() + 1),
            unlink=recording_unlink,
        )

    assert len(removed_paths) == 1
    assert not removed_paths[0].exists()


def test_write_probe_fsync_failure_still_removes_and_fails(tmp_path: Path) -> None:
    removed_paths: list[Path] = []

    def failing_fsync(_descriptor: int) -> None:
        raise OSError("simulated fsync failure")

    def recording_unlink(path: str | os.PathLike[str]) -> None:
        removed_paths.append(Path(path))
        os.unlink(path)

    with pytest.raises(QuotaCaptureError, match="failed before durable verification"):
        perform_write_probe(
            output_parent=tmp_path,
            filesystem=str(tmp_path),
            expected_gid=os.getgid(),
            probe_bytes=64,
            fsync=failing_fsync,
            unlink=recording_unlink,
        )

    assert len(removed_paths) == 1
    assert not removed_paths[0].exists()


def test_write_probe_cleanup_failure_fails_closed(tmp_path: Path) -> None:
    created_path: Path | None = None

    def failing_unlink(path: str | os.PathLike[str]) -> None:
        nonlocal created_path
        created_path = Path(path)
        raise OSError("simulated unlink failure")

    with pytest.raises(QuotaCaptureError, match="cleanup failed"):
        perform_write_probe(
            output_parent=tmp_path,
            filesystem=str(tmp_path),
            expected_gid=os.getgid(),
            probe_bytes=64,
            unlink=failing_unlink,
        )

    assert created_path is not None
    assert created_path.exists()
    created_path.unlink()


def test_quota_command_failure_does_not_leak_stderr() -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def failing_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="private diagnostics")

    with pytest.raises(QuotaCaptureError) as caught:
        query_lfs_quota(
            filesystem=FILESYSTEM,
            subject_kind="group",
            subject=GROUP,
            numeric_id=GID,
            runner=failing_runner,
        )

    assert "private diagnostics" not in str(caught.value)
    assert calls[0][0] == ["lfs", "quota", "-g", GROUP, FILESYSTEM]
    assert "shell" not in calls[0][1]
    assert calls[0][1]["check"] is False


def test_capture_builds_complete_schema_without_raw_command_output(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    timestamps = iter(
        [
            datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 17, 12, 0, 1, tzinfo=timezone.utc),
        ]
    )

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        output = (
            _quota_output(filesystem=str(tmp_path))
            if "-g" in command
            else _project_output(filesystem=str(tmp_path))
        )
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    def probe(**kwargs: object) -> dict[str, object]:
        assert kwargs["output_parent"] == tmp_path
        assert kwargs["expected_gid"] == GID
        return {
            "completed_at": "2026-07-17T12:00:00Z",
            "filesystem": str(tmp_path),
            "path": str(tmp_path / ".screening-quota-probe-fixed"),
            "gid": GID,
            "bytes_written": 4096,
            "fsync_succeeded": True,
            "removed": True,
        }

    snapshot = capture_quota_snapshot(
        filesystem=str(tmp_path),
        output_parent=tmp_path,
        group=GROUP,
        gid=GID,
        project_id=PROJECT_ID,
        probe_bytes=4096,
        runner=runner,
        clock=lambda: next(timestamps),
        probe=probe,
    )

    assert snapshot["schema_version"] == 1
    assert snapshot["captured_at"] == "2026-07-17T12:00:00Z"
    assert snapshot["group"]["name"] == GROUP
    assert snapshot["group"]["gid"] == GID
    assert snapshot["project"]["id"] == PROJECT_ID
    assert snapshot["write_probe"]["removed"] is True
    assert "raw" not in snapshot
    assert commands == [
        ["lfs", "quota", "-g", GROUP, str(tmp_path)],
        ["lfs", "quota", "-p", str(PROJECT_ID), str(tmp_path)],
    ]


def test_write_snapshot_sidecar_is_exact_digest_of_exact_json(tmp_path: Path) -> None:
    destination = tmp_path / "quota.json"
    snapshot = {
        "schema_version": 1,
        "filesystem": FILESYSTEM,
        "nested": {"z": 2, "a": 1},
    }

    receipt = write_snapshot(destination, snapshot)
    payload = destination.read_bytes()
    expected_payload = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    expected_digest = hashlib.sha256(expected_payload).hexdigest()

    assert payload == expected_payload
    assert Path(str(destination) + ".sha256").read_bytes() == (
        expected_digest + "\n"
    ).encode("ascii")
    assert receipt == {
        "path": str(destination),
        "sha256": expected_digest,
        "sidecar_path": str(destination) + ".sha256",
    }
