from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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


def verify_snapshot(snapshot_path: Path, project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    snapshot_path = snapshot_path.resolve()
    if not snapshot_path.is_relative_to(project_root) or not snapshot_path.is_file():
        raise FileNotFoundError("snapshot must be an existing file inside the project")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        raise ValueError("snapshot must be a schema-version-1 JSON object")
    if snapshot.get("status") != "accepted_historical_snapshot":
        raise ValueError("snapshot status is not accepted_historical_snapshot")
    raw_artifacts = snapshot.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("snapshot must contain at least one artifact")

    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            raise ValueError("snapshot artifact records must be objects")
        relative = str(raw.get("path", ""))
        expected = str(raw.get("sha256", "")).lower()
        if not relative or relative in seen or not _is_sha256(expected):
            raise ValueError(f"invalid or duplicate snapshot artifact record: {relative!r}")
        path = (project_root / relative).resolve()
        if not path.is_relative_to(project_root):
            raise ValueError(f"snapshot artifact escapes the project: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"snapshot artifact is missing: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"snapshot artifact changed: {relative} expected={expected} observed={observed}"
            )
        seen.add(relative)
        verified.append({"path": relative, "bytes": path.stat().st_size, "sha256": observed})

    sidecar = snapshot_path.with_suffix(snapshot_path.suffix + ".sha256")
    if sidecar.is_file():
        expected_snapshot_hash = sidecar.read_text(encoding="utf-8").strip().lower()
        if not _is_sha256(expected_snapshot_hash):
            raise ValueError("snapshot sidecar does not contain one SHA-256 value")
        observed_snapshot_hash = sha256_file(snapshot_path)
        if observed_snapshot_hash != expected_snapshot_hash:
            raise ValueError("snapshot JSON does not match its SHA-256 sidecar")
    else:
        observed_snapshot_hash = sha256_file(snapshot_path)

    return {
        "status": "verified",
        "snapshot_id": snapshot.get("snapshot_id"),
        "snapshot_sha256": observed_snapshot_hash,
        "artifact_count": len(verified),
        "total_bytes": sum(record["bytes"] for record in verified),
        "artifacts": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a frozen accepted campaign snapshot")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(verify_snapshot(args.snapshot, args.project_root), indent=2))


if __name__ == "__main__":
    main()
