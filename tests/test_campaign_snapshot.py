from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_campaign_snapshot import verify_snapshot


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path, artifact: Path, *, recorded_path: str | None = None) -> Path:
    path = root / "snapshot.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_id": "test_snapshot",
                "status": "accepted_historical_snapshot",
                "artifacts": [
                    {
                        "path": recorded_path or str(artifact.relative_to(root)),
                        "sha256": _sha256(artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    path.with_suffix(".json.sha256").write_text(_sha256(path) + "\n", encoding="utf-8")
    return path


def test_verify_snapshot_accepts_unchanged_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.json"
    artifact.write_text('{"accepted": true}\n', encoding="utf-8")
    snapshot = _snapshot(tmp_path, artifact)

    result = verify_snapshot(snapshot, tmp_path)

    assert result["status"] == "verified"
    assert result["artifact_count"] == 1
    assert result["artifacts"][0]["path"] == "evidence.json"


def test_verify_snapshot_rejects_changed_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.json"
    artifact.write_text('{"accepted": true}\n', encoding="utf-8")
    snapshot = _snapshot(tmp_path, artifact)
    artifact.write_text('{"accepted": false}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="snapshot artifact changed"):
        verify_snapshot(snapshot, tmp_path)


def test_verify_snapshot_rejects_project_escape(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.json"
    artifact.write_text('{"accepted": true}\n', encoding="utf-8")
    snapshot = _snapshot(tmp_path, artifact, recorded_path="../evidence.json")

    with pytest.raises(ValueError, match="escapes the project"):
        verify_snapshot(snapshot, tmp_path)
