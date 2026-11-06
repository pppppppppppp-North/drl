from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUTS = (
    "config/research_campaign.yaml",
    "campaigns/large_scale_thai_drl_v2/state.json",
    "campaigns/large_scale_thai_drl_v2/plan.json",
    "campaigns/large_scale_thai_drl_v2/resolved_campaign.json",
    "campaigns/large_scale_thai_drl_v2/events.jsonl",
    "campaigns/large_scale_thai_drl_v2/stages",
    "campaigns/large_scale_thai_drl_v2/attempts",
    "campaigns/large_scale_thai_drl_v2/commands",
    "campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2_deployment.yaml",
    "campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2_manifest.json",
    "campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2_manifest.sha256",
    "campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2.tar.gz",
    "campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2.tar.gz.sha256",
    "reports/bistka_screening_v2_preflight.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _safe_input(root: Path, raw: str | Path) -> Path:
    candidate = Path(raw)
    unresolved = candidate if candidate.is_absolute() else root / candidate
    path = unresolved.resolve()
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or unresolved.is_symlink()
        or not path.is_relative_to(root)
        or not path.exists()
        or path.is_symlink()
    ):
        raise ValueError(f"snapshot input is missing or unsafe: {raw}")
    return path


def _collect_files(root: Path, inputs: Iterable[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for raw in inputs:
        path = _safe_input(root, raw)
        if path.is_file():
            files.add(path)
            continue
        if not path.is_dir():
            raise ValueError(f"snapshot input is not a file or directory: {raw}")
        for child in path.rglob("*"):
            if child.is_symlink():
                raise ValueError(f"snapshot tree contains a symlink: {child}")
            if child.is_file():
                files.add(child.resolve())
    if not files:
        raise ValueError("snapshot input inventory is empty")
    return sorted(files, key=lambda path: str(path.relative_to(root)))


def _release_identity(manifest_path: Path) -> tuple[str, str]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"active deployment manifest is unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("active deployment manifest must be a JSON object")
    release_id = str(manifest.get("release_id", ""))
    if len(release_id) != 64 or any(character not in "0123456789abcdef" for character in release_id):
        raise ValueError("active deployment manifest has no valid release id")
    return release_id, _sha256(manifest_path)


def create_snapshot(
    *,
    project_root: str | Path,
    output_dir: str | Path,
    inputs: Iterable[str | Path] = DEFAULT_INPUTS,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_dir)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    if (
        not root.is_dir()
        or root.is_symlink()
        or not output.is_relative_to(root)
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("snapshot output must be a new directory inside the project")
    files = _collect_files(root, inputs)
    output.mkdir(parents=True)
    archive_path = output / "pre_cutover_control_and_release.tar.gz"
    receipt_path = output / "snapshot_receipt.json"
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{archive_path.name}.",
            suffix=".tmp",
            dir=output,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as raw_handle:
                with gzip.GzipFile(
                    filename="",
                    fileobj=raw_handle,
                    mode="wb",
                    mtime=0,
                ) as gzip_handle:
                    with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                        for path in files:
                            relative = path.relative_to(root)
                            info = archive.gettarinfo(str(path), arcname=str(relative))
                            info.uid = 0
                            info.gid = 0
                            info.uname = ""
                            info.gname = ""
                            info.mtime = 0
                            info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                            with path.open("rb") as handle:
                                archive.addfile(info, handle)
                raw_handle.flush()
                os.fsync(raw_handle.fileno())
            os.replace(temporary, archive_path)
        finally:
            temporary.unlink(missing_ok=True)

        active_manifest = root / (
            "campaigns/large_scale_thai_drl_v2/deployment/"
            "bistka_screening_v2_manifest.json"
        )
        release_id, manifest_sha256 = _release_identity(active_manifest)
        receipt = {
            "schema_version": 1,
            "snapshot_kind": "pre_storage_safe_campaign_cutover",
            "created_at_utc": _utc_now(),
            "project_root": ".",
            "active_release": {
                "release_id": release_id,
                "manifest_path": str(active_manifest.relative_to(root)),
                "manifest_sha256": manifest_sha256,
            },
            "archive": {
                "path": archive_path.name,
                "bytes": archive_path.stat().st_size,
                "sha256": _sha256(archive_path),
                "file_count": len(files),
            },
            "files": [
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in files
            ],
            "restore_scope": (
                "Restore this archive only for an intentional rollback to the original "
                "research_campaign.yaml control lineage. Remote release rollback remains "
                "an explicit deployment operation."
            ),
        }
        _atomic_json(receipt_path, receipt)
        _atomic_text(Path(str(receipt_path) + ".sha256"), _sha256(receipt_path) + "\n")
        _atomic_text(Path(str(archive_path) + ".sha256"), _sha256(archive_path) + "\n")
        return receipt
    except BaseException:
        for path in sorted(output.glob("*")):
            if path.is_file() and not path.is_symlink():
                path.unlink()
        output.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic pre-cutover campaign rollback snapshot"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            create_snapshot(
                project_root=args.project_root,
                output_dir=args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
