from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from src.campaigns.calibration import sha256_file
from src.campaigns.evidence_migration import (
    _diff_paths,
    _nested_without,
    create_calibration_import,
    validate_calibration_import,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(sha256_file(path) + "\n", encoding="utf-8")
    return sidecar


def _fixture(tmp_path: Path) -> tuple[SimpleNamespace, SimpleNamespace, Path, dict]:
    project = tmp_path / "project"
    campaign = project / "campaigns" / "target_campaign"
    snapshot = project / "campaigns" / "source_campaign" / "snapshot.json"
    _write_json(snapshot, {"schema_version": 1})
    receipt_path = campaign / "imports" / "v1_calibration_evidence.json"
    receipt = {
        "schema_version": 1,
        "evidence_kind": "v1_calibration_evidence_import",
        "status": "accepted",
        "submitted_hpc_jobs": 0,
        "created_at_utc": "2026-07-16T00:00:00+00:00",
        "source_snapshot": {
            "path": str(snapshot.relative_to(project)),
            "bytes": snapshot.stat().st_size,
            "sha256": sha256_file(snapshot),
        },
    }
    _write_json(receipt_path, receipt)
    _sidecar(receipt_path)
    definition = SimpleNamespace(project_root=project, campaign_dir=campaign)
    stage = SimpleNamespace(id="vector_benchmark")
    return definition, stage, receipt_path, receipt


def test_config_compatibility_removes_only_the_declared_metadata_path() -> None:
    source = {
        "project": {"name": "v1", "seed": 42},
        "training": {"num_envs": 8},
    }
    target = {
        "project": {"name": "v2", "seed": 42},
        "training": {"num_envs": 8},
    }

    assert _diff_paths(source, target) == ["project.name"]
    assert _nested_without(source, "project.name") == _nested_without(
        target,
        "project.name",
    )
    changed_protocol = {
        **target,
        "training": {"num_envs": 16},
    }
    assert _diff_paths(source, changed_protocol) == [
        "project.name",
        "training.num_envs",
    ]


def test_import_validator_accepts_only_a_reproducible_sidecar_bound_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    definition, stage, receipt_path, receipt = _fixture(tmp_path)
    snapshot_path = (
        definition.project_root / receipt["source_snapshot"]["path"]
    ).resolve()

    monkeypatch.setattr(
        "src.campaigns.evidence_migration._build_calibration_import",
        lambda definition, source_snapshot_path, created_at_utc: (
            receipt,
            [snapshot_path],
        ),
    )

    issues, references = validate_calibration_import(
        definition,
        stage,
        receipt_path,
        {},
    )

    assert issues == []
    assert receipt_path in references
    assert receipt_path.with_suffix(".json.sha256") in references
    assert snapshot_path in references


def test_import_validator_rejects_sidecar_or_rebuilt_content_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    definition, stage, receipt_path, receipt = _fixture(tmp_path)
    sidecar = receipt_path.with_suffix(".json.sha256")
    sidecar.write_text("0" * 64 + "\n", encoding="utf-8")

    issues, _ = validate_calibration_import(definition, stage, receipt_path, {})

    assert any("sidecar does not match" in issue for issue in issues)
    _sidecar(receipt_path)
    rebuilt = {**receipt, "status": "rejected"}
    monkeypatch.setattr(
        "src.campaigns.evidence_migration._build_calibration_import",
        lambda definition, source_snapshot_path, created_at_utc: (rebuilt, []),
    )

    issues, _ = validate_calibration_import(definition, stage, receipt_path, {})

    assert any("does not reproduce" in issue for issue in issues)


def test_create_import_writes_an_atomic_checksum_and_never_submits_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    campaign = project / "campaigns" / "target_campaign"
    snapshot = project / "campaigns" / "source_campaign" / "snapshot.json"
    _write_json(snapshot, {"schema_version": 1})
    definition = SimpleNamespace(project_root=project, campaign_dir=campaign)
    receipt = {
        "schema_version": 1,
        "evidence_kind": "v1_calibration_evidence_import",
        "status": "accepted",
        "submitted_hpc_jobs": 0,
    }
    monkeypatch.setattr(
        "src.campaigns.evidence_migration._build_calibration_import",
        lambda definition, source_snapshot_path, created_at_utc: (receipt, []),
    )

    created = create_calibration_import(definition, snapshot)
    output = campaign / "imports" / "v1_calibration_evidence.json"

    assert created["submitted_hpc_jobs"] == 0
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    assert output.with_suffix(".json.sha256").read_text(
        encoding="utf-8"
    ).strip() == sha256_file(output)
    assert "sbatch" not in inspect.getsource(create_calibration_import)
