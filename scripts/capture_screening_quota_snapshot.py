from __future__ import annotations

"""Capture fail-closed Lustre quota evidence for the screening storage gate.

This is intentionally a small login-node diagnostic.  It runs two ``lfs quota``
commands and performs a bounded, fsynced write probe; it never submits work to
SLURM.  Command output is parsed in memory and is not copied into the receipt.
"""

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_PROBE_BYTES = 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 30
MAX_COMMAND_OUTPUT_BYTES = 64 * 1024

_TABLE_HEADER = (
    "Filesystem",
    "kbytes",
    "quota",
    "limit",
    "grace",
    "files",
    "quota",
    "limit",
    "grace",
)
_UNSIGNED_INTEGER = re.compile(r"[0-9]+")
_USED_VALUE = re.compile(r"(?P<value>[0-9]+)(?P<over>\*)?")
_GRACE_COUNTDOWN = re.compile(r"(?:[0-9]+[ywdhms])+")
_GROUP_NAME = re.compile(r"[A-Za-z0-9_.-]+")


class QuotaCaptureError(RuntimeError):
    """Raised when evidence cannot be captured without ambiguity."""


class QuotaParseError(QuotaCaptureError):
    """Raised when ``lfs quota`` output violates the expected table contract."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise QuotaCaptureError("capture clock must return a timezone-aware timestamp")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalise_absolute_path(value: str | os.PathLike[str], *, name: str) -> str:
    raw = os.fspath(value)
    if not raw or "\x00" in raw:
        raise QuotaCaptureError(f"{name} must be a nonempty absolute path")
    normalised = os.path.normpath(raw)
    if not os.path.isabs(normalised):
        raise QuotaCaptureError(f"{name} must be an absolute path")
    return normalised


def _parse_used(token: str, *, dimension: str) -> tuple[int, bool]:
    match = _USED_VALUE.fullmatch(token)
    if match is None:
        raise QuotaParseError(f"{dimension} usage is not an unsigned Lustre value")
    return int(match.group("value")), match.group("over") is not None


def _parse_limit(token: str, *, dimension: str, label: str) -> int:
    if _UNSIGNED_INTEGER.fullmatch(token) is None:
        raise QuotaParseError(f"{dimension} {label} limit is not an unsigned integer")
    value = int(token)
    if value <= 0:
        raise QuotaParseError(f"{dimension} {label} limit must be positive")
    return value


def _grace_is_expired(token: str, *, over_limit: bool, dimension: str) -> bool:
    """Convert Lustre's marker/countdown representation into a fail-closed bool.

    A dash is the normal representation below quota.  If usage has Lustre's
    trailing ``*`` marker and there is no countdown, the grace period is no
    longer usable and is therefore reported as expired.  A finite countdown is
    explicitly unexpired.  ``*``, ``none``, and ``expired`` are accepted as
    explicit expired spellings used by Lustre/site wrappers.
    """

    if token == "-":
        return over_limit
    if token in {"*", "none", "expired"}:
        return True
    if _GRACE_COUNTDOWN.fullmatch(token) is not None:
        return False
    raise QuotaParseError(f"{dimension} grace value is not a recognised Lustre value")


def _quota_dimension(
    *,
    used_token: str,
    soft_token: str,
    hard_token: str,
    grace_token: str,
    dimension: str,
    scale: int,
) -> dict[str, int | bool]:
    used, over_limit = _parse_used(used_token, dimension=dimension)
    soft = _parse_limit(soft_token, dimension=dimension, label="soft")
    hard = _parse_limit(hard_token, dimension=dimension, label="hard")
    if hard < soft:
        raise QuotaParseError(f"{dimension} hard limit is below the soft limit")
    return {
        "used": used * scale,
        "soft_limit": soft * scale,
        "hard_limit": hard * scale,
        "grace_expired": _grace_is_expired(
            grace_token,
            over_limit=over_limit,
            dimension=dimension,
        ),
    }


def parse_lfs_quota_table(
    output: str,
    *,
    filesystem: str,
    subject_kind: str,
    subject: str | int,
    numeric_id: int,
) -> dict[str, dict[str, int | bool]]:
    """Strictly parse one group or project quota table.

    ``subject_kind`` is ``"group"`` or ``"project"``.  The identity in the
    heading, the numeric GID/PID, and the one data-row filesystem must all match
    the requested values.
    """

    expected_filesystem = _normalise_absolute_path(filesystem, name="filesystem")
    if not isinstance(output, str) or not output.strip() or "\x00" in output:
        raise QuotaParseError("quota command produced no valid text table")
    if len(output.encode("utf-8")) > MAX_COMMAND_OUTPUT_BYTES:
        raise QuotaParseError("quota command output exceeds the bounded table size")

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 3:
        raise QuotaParseError("quota output must contain one heading, header, and data row")

    if subject_kind == "group":
        subject_text = str(subject)
        if _GROUP_NAME.fullmatch(subject_text) is None:
            raise QuotaParseError("requested group name is invalid")
        expected_heading = f"Disk quotas for grp {subject_text} (gid {numeric_id}):"
    elif subject_kind == "project":
        if isinstance(subject, bool) or not isinstance(subject, (str, int)):
            raise QuotaParseError("requested project ID is invalid")
        subject_text = str(subject)
        if _UNSIGNED_INTEGER.fullmatch(subject_text) is None:
            raise QuotaParseError("requested project ID is invalid")
        expected_heading = f"Disk quotas for prj {subject_text} (pid {numeric_id}):"
    else:
        raise QuotaParseError("subject_kind must be 'group' or 'project'")

    if lines[0] != expected_heading:
        raise QuotaParseError("quota heading does not match the requested identity")
    if tuple(lines[1].split()) != _TABLE_HEADER:
        raise QuotaParseError("quota table header does not match the Lustre contract")

    fields = lines[2].split()
    if len(fields) != len(_TABLE_HEADER):
        raise QuotaParseError("quota table must contain exactly one nine-field data row")
    observed_filesystem = _normalise_absolute_path(fields[0], name="quota filesystem")
    if observed_filesystem != expected_filesystem:
        raise QuotaParseError("quota data row filesystem does not match the request")

    return {
        "bytes": _quota_dimension(
            used_token=fields[1],
            soft_token=fields[2],
            hard_token=fields[3],
            grace_token=fields[4],
            dimension="bytes",
            scale=1024,
        ),
        "inodes": _quota_dimension(
            used_token=fields[5],
            soft_token=fields[6],
            hard_token=fields[7],
            grace_token=fields[8],
            dimension="inodes",
            scale=1,
        ),
    }


def query_lfs_quota(
    *,
    filesystem: str,
    subject_kind: str,
    subject: str | int,
    numeric_id: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_seconds: int = COMMAND_TIMEOUT_SECONDS,
) -> dict[str, dict[str, int | bool]]:
    """Run one bounded ``lfs quota`` query without a command shell."""

    if subject_kind == "group":
        flag = "-g"
    elif subject_kind == "project":
        flag = "-p"
    else:
        raise QuotaCaptureError("subject_kind must be 'group' or 'project'")
    if timeout_seconds <= 0:
        raise QuotaCaptureError("quota command timeout must be positive")

    filesystem_text = _normalise_absolute_path(filesystem, name="filesystem")
    command = ["lfs", "quota", flag, str(subject), filesystem_text]
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QuotaCaptureError(f"lfs {subject_kind} quota command could not complete") from exc

    if completed.returncode != 0:
        raise QuotaCaptureError(
            f"lfs {subject_kind} quota command failed with exit code {completed.returncode}"
        )
    if not isinstance(completed.stdout, str) or not isinstance(completed.stderr, str):
        raise QuotaCaptureError(f"lfs {subject_kind} quota command returned invalid streams")
    if completed.stderr.strip():
        raise QuotaCaptureError(f"lfs {subject_kind} quota command emitted diagnostics")
    return parse_lfs_quota_table(
        completed.stdout,
        filesystem=filesystem_text,
        subject_kind=subject_kind,
        subject=subject,
        numeric_id=numeric_id,
    )


def _validated_probe_parent(
    output_parent: str | os.PathLike[str],
    filesystem: str,
) -> Path:
    raw_parent = Path(output_parent)
    if raw_parent.is_symlink():
        raise QuotaCaptureError("write probe output parent must not be a symlink")
    try:
        parent = raw_parent.resolve(strict=True)
    except OSError as exc:
        raise QuotaCaptureError("write probe output parent is unavailable") from exc
    try:
        parent_mode = parent.stat().st_mode
    except OSError as exc:
        raise QuotaCaptureError("write probe output parent cannot be inspected") from exc
    if not stat.S_ISDIR(parent_mode):
        raise QuotaCaptureError("write probe output parent must be a directory")

    filesystem_text = _normalise_absolute_path(filesystem, name="filesystem")
    try:
        filesystem_root = Path(filesystem_text).resolve(strict=True)
    except OSError as exc:
        raise QuotaCaptureError("quota filesystem is unavailable") from exc
    if parent != filesystem_root and filesystem_root not in parent.parents:
        raise QuotaCaptureError("write probe output parent is outside the quota filesystem")
    return parent


def perform_write_probe(
    *,
    output_parent: str | os.PathLike[str],
    filesystem: str,
    expected_gid: int,
    probe_bytes: int = DEFAULT_PROBE_BYTES,
    clock: Callable[[], datetime] = _utc_now,
    mkstemp: Callable[..., tuple[int, str]] = tempfile.mkstemp,
    fdopen: Callable[..., Any] = os.fdopen,
    fstat: Callable[[int], os.stat_result] = os.fstat,
    fsync: Callable[[int], None] = os.fsync,
    close_fd: Callable[[int], None] = os.close,
    unlink: Callable[[str | os.PathLike[str]], None] = os.unlink,
) -> dict[str, Any]:
    """Write, fsync, verify ownership of, and remove one exclusive probe file."""

    if isinstance(expected_gid, bool) or not isinstance(expected_gid, int) or expected_gid < 0:
        raise QuotaCaptureError("expected GID must be a nonnegative integer")
    if isinstance(probe_bytes, bool) or not isinstance(probe_bytes, int) or probe_bytes <= 0:
        raise QuotaCaptureError("probe size must be a positive integer")

    filesystem_text = _normalise_absolute_path(filesystem, name="filesystem")
    parent = _validated_probe_parent(output_parent, filesystem_text)
    descriptor: int | None = None
    handle: Any | None = None
    probe_path: Path | None = None
    created = False
    primary_error: BaseException | None = None

    try:
        descriptor, temporary_name = mkstemp(
            prefix=".screening-quota-probe-",
            dir=str(parent),
        )
        created = True
        probe_path = Path(temporary_name)
        if probe_path.parent != parent:
            raise QuotaCaptureError("exclusive probe was not created directly under output parent")

        metadata = fstat(descriptor)
        if metadata.st_gid != expected_gid:
            raise QuotaCaptureError("exclusive probe inherited an unexpected effective GID")

        handle = fdopen(descriptor, "wb")
        descriptor = None
        remaining = probe_bytes
        chunk = bytes(min(DEFAULT_PROBE_BYTES, probe_bytes))
        while remaining:
            requested = min(len(chunk), remaining)
            written = handle.write(chunk[:requested])
            if written != requested:
                raise QuotaCaptureError("write probe did not persist the requested byte count")
            remaining -= written
        handle.flush()
        fsync(handle.fileno())
        handle.close()
        handle = None
    except BaseException as exc:
        primary_error = exc
    finally:
        if handle is not None:
            try:
                handle.close()
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
        elif descriptor is not None:
            try:
                close_fd(descriptor)
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc

        cleanup_error: BaseException | None = None
        if created and probe_path is not None:
            try:
                unlink(probe_path)
            except BaseException as exc:
                cleanup_error = exc
        if cleanup_error is not None:
            raise QuotaCaptureError("write probe cleanup failed; quota evidence is unusable") from cleanup_error

    if primary_error is not None:
        if isinstance(primary_error, QuotaCaptureError):
            raise primary_error
        raise QuotaCaptureError("write probe failed before durable verification") from primary_error
    if probe_path is None:
        raise QuotaCaptureError("write probe did not create an evidence path")

    return {
        "completed_at": _utc_text(clock()),
        "filesystem": filesystem_text,
        "path": str(probe_path),
        "gid": expected_gid,
        "bytes_written": probe_bytes,
        "fsync_succeeded": True,
        "removed": True,
    }


def capture_quota_snapshot(
    *,
    filesystem: str,
    output_parent: str | os.PathLike[str],
    group: str,
    gid: int,
    project_id: int,
    probe_bytes: int = DEFAULT_PROBE_BYTES,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    clock: Callable[[], datetime] = _utc_now,
    probe: Callable[..., dict[str, Any]] = perform_write_probe,
) -> dict[str, Any]:
    """Capture one schema-v1 quota snapshot without retaining raw command data."""

    filesystem_text = _normalise_absolute_path(filesystem, name="filesystem")
    if _GROUP_NAME.fullmatch(group) is None:
        raise QuotaCaptureError("group must be a nonempty POSIX-style group name")
    if isinstance(gid, bool) or not isinstance(gid, int) or gid < 0:
        raise QuotaCaptureError("GID must be a nonnegative integer")
    if isinstance(project_id, bool) or not isinstance(project_id, int) or project_id <= 0:
        raise QuotaCaptureError("project ID must be a positive integer")

    write_probe = probe(
        output_parent=output_parent,
        filesystem=filesystem_text,
        expected_gid=gid,
        probe_bytes=probe_bytes,
        clock=clock,
    )
    group_quota = query_lfs_quota(
        filesystem=filesystem_text,
        subject_kind="group",
        subject=group,
        numeric_id=gid,
        runner=runner,
    )
    project_quota = query_lfs_quota(
        filesystem=filesystem_text,
        subject_kind="project",
        subject=project_id,
        numeric_id=project_id,
        runner=runner,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": _utc_text(clock()),
        "filesystem": filesystem_text,
        "group": {
            "name": group,
            "gid": gid,
            **group_quota,
        },
        "project": {
            "id": project_id,
            **project_quota,
        },
        "write_probe": write_probe,
    }


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise QuotaCaptureError("snapshot destination must not be a symlink")
    if path.exists() and not path.is_file():
        raise QuotaCaptureError("snapshot destination must be a regular file")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_snapshot(
    path: str | os.PathLike[str],
    snapshot: Mapping[str, Any],
) -> dict[str, str]:
    """Atomically write canonical JSON and its exact SHA-256 sidecar."""

    destination = Path(path)
    payload = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(str(destination) + ".sha256")
    _atomic_bytes(destination, payload)
    _atomic_bytes(sidecar, (digest + "\n").encode("ascii"))
    return {
        "path": str(destination),
        "sha256": digest,
        "sidecar_path": str(sidecar),
    }


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture signed group/project Lustre quota evidence for screening.",
    )
    parser.add_argument("--filesystem", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--gid", required=True, type=_nonnegative_integer)
    parser.add_argument("--project-id", required=True, type=_positive_integer)
    parser.add_argument("--probe-bytes", type=_positive_integer, default=DEFAULT_PROBE_BYTES)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        snapshot = capture_quota_snapshot(
            filesystem=args.filesystem,
            output_parent=args.output_parent,
            group=args.group,
            gid=args.gid,
            project_id=args.project_id,
            probe_bytes=args.probe_bytes,
        )
        receipt = write_snapshot(args.output, snapshot)
    except QuotaCaptureError as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
