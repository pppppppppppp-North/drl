from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _safe_relative(raw: str) -> Path:
    pure = PurePosixPath(raw)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe bundle member path: {raw!r}")
    return Path(*pure.parts)


def _absolute_without_symlink_resolution(path: Path) -> Path:
    """Return a normalized absolute path without following filesystem links."""

    return Path(os.path.abspath(os.fspath(path)))


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _validate_directory_chain(path: Path, *, allow_missing: bool) -> None:
    """Reject links and non-directories in an absolute directory chain.

    Deployment destinations are security boundaries.  ``Path.resolve`` and
    ``mkdir(parents=True)`` both follow links, so neither is appropriate for
    validating the target tree.
    """

    if not path.is_absolute():
        raise ValueError(f"destination directory must be absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        metadata = _lstat(current)
        if metadata is None:
            if allow_missing:
                continue
            raise ValueError(f"destination directory is missing: {current}")
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"symlinked destination parent is forbidden: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"destination parent is not a directory: {current}")


def _validate_regular_or_absent(path: Path) -> None:
    metadata = _lstat(path)
    if metadata is None:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"symlink destination is forbidden: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"destination is not a regular file: {path}")


def _validate_destination(target_root: Path, relative: Path) -> Path:
    destination = target_root / relative
    _validate_directory_chain(destination.parent, allow_missing=True)
    _validate_regular_or_absent(destination)
    return destination


def _ensure_directory_chain(path: Path) -> list[Path]:
    """Create missing directories one component at a time without following links."""

    if not path.is_absolute():
        raise ValueError(f"destination directory must be absolute: {path}")
    created: list[Path] = []
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        metadata = _lstat(current)
        if metadata is None:
            try:
                current.mkdir()
                created.append(current)
            except FileExistsError:
                # A concurrent creator is safe only when it created a real
                # directory.  Re-check below rather than following it.
                pass
            metadata = _lstat(current)
        if metadata is None:
            raise ValueError(f"destination directory could not be created: {current}")
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"symlinked destination parent is forbidden: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"destination parent is not a directory: {current}")
    return created


def _promote_staged(staged: Path, destination: Path) -> None:
    """Atomic same-filesystem promotion, split out for fault-injection tests."""

    os.replace(staged, destination)


def _unused_sibling_backup(destination: Path, release_id: str) -> Path:
    descriptor, backup_name = tempfile.mkstemp(
        prefix=f".{destination.name}.{release_id[:12]}.",
        suffix=".bak",
        dir=destination.parent,
    )
    os.close(descriptor)
    backup = Path(backup_name)
    backup.unlink()
    return backup


def _atomic_marker(path: Path, value: str) -> None:
    _validate_directory_chain(path.parent, allow_missing=False)
    _validate_regular_or_absent(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _reserve_space(target_root: Path, required_bytes: int) -> None:
    if required_bytes <= 0:
        return
    # Probe in the target directory when it exists, otherwise in its nearest
    # existing ancestor.  This tests the destination filesystem without
    # persistently creating any target path before the quota check succeeds.
    probe_dir = target_root
    while _lstat(probe_dir) is None:
        parent = probe_dir.parent
        if parent == probe_dir:
            raise ValueError(f"no existing parent for quota probe: {target_root}")
        probe_dir = parent
    _validate_directory_chain(probe_dir, allow_missing=False)
    descriptor, probe_name = tempfile.mkstemp(prefix=".quota_probe.", dir=probe_dir)
    probe = Path(probe_name)
    try:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(descriptor, 0, required_bytes)
        else:
            block = b"\0" * min(required_bytes, 1024 * 1024)
            remaining = required_bytes
            while remaining:
                written = os.write(descriptor, block[: min(len(block), remaining)])
                if written <= 0:
                    raise OSError("quota probe made no forward progress")
                remaining -= written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
        probe.unlink(missing_ok=True)


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("deployment manifest must be a schema-version-1 JSON object")
    release_id = str(value.get("release_id", ""))
    if not _is_sha256(release_id):
        raise ValueError("deployment manifest has an invalid release_id")
    records = value.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("deployment manifest has no file inventory")
    return value


def _expected_release_id(manifest: dict[str, Any]) -> str:
    payload = {
        "schema_version": 1,
        "campaign_id": str(manifest.get("campaign_id", "")),
        "files": manifest["files"],
        "runtime_feature_bindings": manifest.get("runtime_feature_bindings", []),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def apply_release(
    *,
    bundle_path: Path,
    manifest_path: Path,
    target_root: Path,
    expected_bundle_sha256: str | None = None,
    manifest_relative_path: str | Path | None = None,
    staging_root: Path | None = None,
    quota_margin_bytes: int = 1024 * 1024,
) -> dict[str, Any]:
    """Apply a verified release without rewriting byte-identical targets.

    Every changed file is copied to a sibling temporary path and verified before
    any destination is promoted. Deployment manifests are promoted after normal
    files, and the deployed-bundle marker is written last.
    """

    bundle_path = bundle_path.resolve()
    manifest_path = manifest_path.resolve()
    raw_target_root = Path(target_root)
    if not raw_target_root.is_absolute() or ".." in raw_target_root.parts:
        raise ValueError("target root must be an absolute path without parent traversal")
    target_root = _absolute_without_symlink_resolution(raw_target_root)
    if not bundle_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("bundle and manifest must exist")
    if quota_margin_bytes < 0:
        raise ValueError("quota_margin_bytes must be nonnegative")

    manifest = _load_manifest(manifest_path)
    if manifest.get("release_id") != _expected_release_id(manifest):
        raise ValueError("deployment manifest release_id is inconsistent with its inventory")
    raw_expected_target = Path(str(manifest.get("target_project_root", "")))
    if not raw_expected_target.is_absolute() or ".." in raw_expected_target.parts:
        raise ValueError("manifest target root must be an absolute path without parent traversal")
    expected_target = _absolute_without_symlink_resolution(raw_expected_target)
    if expected_target != target_root:
        raise ValueError(
            f"target root does not match manifest: expected={expected_target} actual={target_root}"
        )
    _validate_directory_chain(target_root, allow_missing=True)
    bundle_sha256 = sha256_file(bundle_path)
    if expected_bundle_sha256 is not None:
        expected_bundle_sha256 = expected_bundle_sha256.strip().lower()
        if not _is_sha256(expected_bundle_sha256) or bundle_sha256 != expected_bundle_sha256:
            raise ValueError("bundle SHA-256 does not match the expected value")

    record_map: dict[str, dict[str, Any]] = {}
    for record in manifest["files"]:
        if not isinstance(record, dict):
            raise ValueError("deployment file records must be objects")
        relative = _safe_relative(str(record.get("path", ""))).as_posix()
        expected_hash = str(record.get("sha256", "")).lower()
        expected_bytes = int(record.get("bytes", -1))
        if relative in record_map or not _is_sha256(expected_hash) or expected_bytes < 0:
            raise ValueError(f"invalid or duplicate deployment record: {relative}")
        record_map[relative] = record

    raw_manifest_relative = (
        manifest_relative_path
        if manifest_relative_path is not None
        else manifest.get("manifest_path")
    )
    if raw_manifest_relative is None:
        raise ValueError("manifest_relative_path is required for manifests without manifest_path")
    manifest_relative = _safe_relative(str(raw_manifest_relative)).as_posix()
    internal_manifest = ".bistka/deployment_manifest.json"
    expected_members = set(record_map) | {manifest_relative, internal_manifest}
    staging_parent = staging_root.resolve() if staging_root else None
    if staging_parent is not None:
        if staging_parent == target_root or staging_parent.is_relative_to(target_root):
            raise ValueError("staging root must stay outside the deployment target")
        staging_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="bistka-release-", dir=staging_parent) as directory:
        extracted_root = Path(directory)
        with tarfile.open(bundle_path, "r:gz") as archive:
            actual_members: set[str] = set()
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"bundle contains a non-regular member: {member.name}")
                relative = _safe_relative(member.name)
                normalized = relative.as_posix()
                if normalized in actual_members:
                    raise ValueError(f"bundle contains a duplicate member: {normalized}")
                actual_members.add(normalized)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"bundle member cannot be read: {normalized}")
                destination = extracted_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
            if actual_members != expected_members:
                missing = sorted(expected_members - actual_members)
                extra = sorted(actual_members - expected_members)
                raise ValueError(f"bundle inventory mismatch: missing={missing} extra={extra}")

        for relative, record in record_map.items():
            staged = extracted_root / relative
            if staged.stat().st_size != int(record["bytes"]) or sha256_file(staged) != str(
                record["sha256"]
            ):
                raise ValueError(f"extracted deployment file failed verification: {relative}")
        external_manifest_bytes = manifest_path.read_bytes()
        for relative in (manifest_relative, internal_manifest):
            staged_path = extracted_root / relative
            _load_manifest(staged_path)
            if staged_path.read_bytes() != external_manifest_bytes:
                raise ValueError(f"bundled manifest differs from supplied manifest: {relative}")

        changed: list[str] = []
        skipped: list[str] = []
        for relative in sorted(expected_members):
            source = extracted_root / relative
            destination = _validate_destination(target_root, Path(relative))
            if (
                _lstat(destination) is not None
                and destination.stat().st_size == source.stat().st_size
                and sha256_file(destination) == sha256_file(source)
            ):
                skipped.append(relative)
            else:
                changed.append(relative)

        marker_path = _validate_destination(
            target_root, Path(".bistka/deployed_bundle.sha256")
        )
        expected_marker = (bundle_sha256 + "\n").encode("utf-8")
        marker_changed = not (
            _lstat(marker_path) is not None
            and marker_path.stat().st_size == len(expected_marker)
            and marker_path.read_bytes() == expected_marker
        )
        changed_bytes = sum((extracted_root / relative).stat().st_size for relative in changed)
        reserved_bytes = changed_bytes + (len(expected_marker) if marker_changed else 0)
        if changed or marker_changed:
            reserved_bytes += quota_margin_bytes
        _reserve_space(target_root, reserved_bytes)

        staged_destinations: dict[str, Path] = {}
        created_directories: set[Path] = set()
        transactions: list[tuple[Path, Path | None]] = []
        try:
            created_directories.update(_ensure_directory_chain(target_root))
            for relative in changed:
                source = extracted_root / relative
                destination = target_root / relative
                created_directories.update(_ensure_directory_chain(destination.parent))
                _validate_regular_or_absent(destination)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.{manifest['release_id'][:12]}.",
                    suffix=".tmp",
                    dir=destination.parent,
                )
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_handle:
                        shutil.copyfileobj(input_handle, output)
                        output.flush()
                        os.fsync(output.fileno())
                    os.chmod(temporary, source.stat().st_mode & 0o777)
                    if (
                        temporary.stat().st_size != source.stat().st_size
                        or sha256_file(temporary) != sha256_file(source)
                    ):
                        raise ValueError(f"staged destination failed verification: {relative}")
                    staged_destinations[relative] = temporary
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise

            manifest_members = {manifest_relative, internal_manifest}
            commit_order = sorted(
                changed,
                key=lambda relative: (relative in manifest_members, relative),
            )
            for relative in commit_order:
                destination = target_root / relative
                _validate_directory_chain(destination.parent, allow_missing=False)
                _validate_regular_or_absent(destination)
                backup: Path | None = None
                if _lstat(destination) is not None:
                    backup = _unused_sibling_backup(
                        destination, str(manifest["release_id"])
                    )
                if backup is not None:
                    os.replace(destination, backup)
                transactions.append((destination, backup))
                staged = staged_destinations[relative]
                _promote_staged(staged, destination)
                staged_destinations.pop(relative)

            if marker_changed:
                created_directories.update(_ensure_directory_chain(marker_path.parent))
                _atomic_marker(marker_path, bundle_sha256)
        except BaseException as exc:
            rollback_issues: list[str] = []
            for destination, backup in reversed(transactions):
                try:
                    destination_metadata = _lstat(destination)
                    if destination_metadata is not None:
                        if stat.S_ISLNK(destination_metadata.st_mode) or not stat.S_ISREG(
                            destination_metadata.st_mode
                        ):
                            raise ValueError(
                                f"rollback refuses unexpected destination type: {destination}"
                            )
                        destination.unlink()
                    if backup is not None and _lstat(backup) is not None:
                        backup_metadata = backup.lstat()
                        if not stat.S_ISREG(backup_metadata.st_mode):
                            raise ValueError(
                                f"rollback backup is not a regular file: {backup}"
                            )
                        os.replace(backup, destination)
                except BaseException as rollback_exc:
                    rollback_issues.append(f"{destination}: {rollback_exc}")
            for temporary in staged_destinations.values():
                temporary.unlink(missing_ok=True)
            for created in sorted(
                created_directories, key=lambda value: len(value.parts), reverse=True
            ):
                try:
                    created.rmdir()
                except FileNotFoundError:
                    pass
                except OSError:
                    # Non-empty directories either pre-dated this attempt or
                    # still contain evidence needed to diagnose rollback.
                    pass
            if rollback_issues:
                raise RuntimeError(
                    "release commit failed and rollback was incomplete: "
                    + "; ".join(rollback_issues)
                ) from exc
            raise
        else:
            for _, backup in transactions:
                if backup is not None:
                    try:
                        backup.unlink(missing_ok=True)
                    except OSError:
                        # The release and marker are already committed.  A
                        # hidden same-filesystem backup is safer than reporting
                        # a false rollback-capable failure at this point.
                        pass
        finally:
            for temporary in staged_destinations.values():
                temporary.unlink(missing_ok=True)
    return {
        "status": "applied",
        "release_id": manifest["release_id"],
        "bundle_sha256": bundle_sha256,
        "changed_file_count": len(changed),
        "unchanged_file_count": len(skipped),
        "changed_bytes": changed_bytes,
        "changed_files": changed,
        "unchanged_files": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a release without rewriting exact files")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", default=None)
    parser.add_argument(
        "--manifest-relative",
        required=True,
        help="Project-relative destination of the deployment manifest",
    )
    parser.add_argument("--staging-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--quota-margin-mib", type=int, default=1)
    args = parser.parse_args()
    result = apply_release(
        bundle_path=args.bundle,
        manifest_path=args.manifest,
        target_root=args.target_root,
        expected_bundle_sha256=args.expected_bundle_sha256,
        manifest_relative_path=args.manifest_relative,
        staging_root=args.staging_root,
        quota_margin_bytes=args.quota_margin_mib * 1024 * 1024,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
