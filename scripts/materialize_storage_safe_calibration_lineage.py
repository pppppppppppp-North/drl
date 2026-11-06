from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from src.campaigns.control import load_definition
from src.campaigns.gates import evaluate_stage_gate, sha256_file


STAGE_IDS = ("vector_benchmark", "determinism")
RESTORED_PATHS = (
    "campaigns/large_scale_thai_drl_v2/stages/cluster_preflight/gate_report.json",
    "campaigns/large_scale_thai_drl_v2/stages/cluster_preflight/_SUCCESS.json",
    "campaigns/large_scale_thai_drl_v2/stages/vector_benchmark/gate_report.json",
    "campaigns/large_scale_thai_drl_v2/stages/vector_benchmark/_SUCCESS.json",
    "campaigns/large_scale_thai_drl_v2/stages/determinism/gate_report.json",
    "campaigns/large_scale_thai_drl_v2/stages/determinism/_SUCCESS.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _safe_path(
    root: Path,
    raw: str | Path,
    *,
    label: str,
    must_exist: bool = True,
) -> Path:
    candidate = Path(raw)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if (
        ".." in candidate.parts
        or not path.is_relative_to(root)
        or (must_exist and (not path.is_file() or path.is_symlink()))
    ):
        raise ValueError(f"{label} is missing or unsafe: {raw}")
    return path


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_sidecar(path: Path, *, label: str) -> Path:
    sidecar = Path(str(path) + ".sha256")
    if path.is_symlink() or sidecar.is_symlink() or not path.is_file() or not sidecar.is_file():
        raise ValueError(f"{label} and sidecar must be regular files")
    if sidecar.read_text(encoding="utf-8") != f"{sha256_file(path)}\n":
        raise ValueError(f"{label} SHA-256 sidecar does not match")
    return sidecar


def _snapshot_members(
    *,
    root: Path,
    snapshot_dir: Path,
) -> tuple[dict[str, Any], Path, dict[str, bytes]]:
    receipt_path = snapshot_dir / "snapshot_receipt.json"
    archive_path = snapshot_dir / "pre_cutover_control_and_release.tar.gz"
    _verify_sidecar(receipt_path, label="cutover snapshot receipt")
    _verify_sidecar(archive_path, label="cutover snapshot archive")
    receipt = _json_object(receipt_path, label="cutover snapshot receipt")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("snapshot_kind") != "pre_storage_safe_campaign_cutover"
    ):
        raise ValueError("cutover snapshot receipt contract does not match")
    archive_record = receipt.get("archive")
    if not isinstance(archive_record, dict) or archive_record != {
        "path": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "file_count": archive_record.get("file_count"),
    }:
        raise ValueError("cutover snapshot archive binding does not match")
    records_value = receipt.get("files")
    if not isinstance(records_value, list) or not records_value:
        raise ValueError("cutover snapshot file inventory is empty")
    records: dict[str, dict[str, Any]] = {}
    for record in records_value:
        if not isinstance(record, dict):
            raise ValueError("cutover snapshot contains a malformed file record")
        relative = str(record.get("path", ""))
        safe = PurePosixPath(relative)
        if (
            not relative
            or safe.is_absolute()
            or ".." in safe.parts
            or relative in records
            or not isinstance(record.get("bytes"), int)
            or len(str(record.get("sha256", ""))) != 64
        ):
            raise ValueError(f"cutover snapshot contains an unsafe file record: {relative}")
        records[relative] = record
    members: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            safe = PurePosixPath(member.name)
            if (
                not member.isfile()
                or safe.is_absolute()
                or ".." in safe.parts
                or member.name in members
            ):
                raise ValueError(f"cutover snapshot archive member is unsafe: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cutover snapshot member is unreadable: {member.name}")
            members[member.name] = handle.read()
    if set(members) != set(records) or len(members) != archive_record.get("file_count"):
        raise ValueError("cutover snapshot archive inventory does not match its receipt")
    for relative, value in members.items():
        record = records[relative]
        if len(value) != record["bytes"] or _sha256_bytes(value) != record["sha256"]:
            raise ValueError(f"cutover snapshot member hash or size differs: {relative}")
    return receipt, archive_path, members


def _artifact(root: Path, path: Path, **extra: Any) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"lineage artifact is missing or unsafe: {path}")
    return {
        "path": str(path.resolve().relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **extra,
    }


def _lineage_stage_records(definition: Any) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for stage_id in STAGE_IDS:
        gate = evaluate_stage_gate(definition, definition.stage_map[stage_id])
        if not gate.passed:
            raise ValueError(
                f"accepted {stage_id} evidence does not survive the cutover: {gate.issues}"
            )
        attempt_path = definition.campaign_dir / "attempts" / f"{stage_id}_latest.json"
        attempt = _json_object(attempt_path, label=f"{stage_id} latest attempt")
        attempt_id = str(attempt.get("attempt_id", ""))
        attempt_sha256 = str(attempt.get("attempt_sha256", ""))
        if not attempt_id or len(attempt_sha256) != 64:
            raise ValueError(f"accepted {stage_id} attempt identity is invalid")
        receipts: list[str] = []
        for path in gate.evidence_files:
            candidate = definition.project_root / path
            if not path.endswith(".receipt.json") or not candidate.is_file():
                continue
            try:
                candidate_value = _json_object(
                    candidate,
                    label=f"{stage_id} receipt candidate",
                )
            except ValueError:
                continue
            if candidate_value.get("campaign_attempt_id") == attempt_id:
                receipts.append(path)
        if len(receipts) != 1:
            raise ValueError(
                f"accepted {stage_id} must bind exactly one native receipt, "
                f"found {len(receipts)}"
            )
        records[stage_id] = {
            "stage_id": stage_id,
            "attempt_id": attempt_id,
            "attempt_sha256": attempt_sha256,
            "gate_evidence_fingerprint": gate.evidence_fingerprint,
            "evidence_file_count": len(gate.evidence_files),
            "receipts": [
                _artifact(definition.project_root, definition.project_root / path)
                for path in sorted(receipts)
            ],
        }
    return records


def validate_lineage_receipt(
    receipt_path: str | Path,
    *,
    project_root: str | Path = ".",
    spec_path: str | Path = "config/research_campaign_storage_safe.yaml",
    campaign_dir: str | Path = "campaigns/large_scale_thai_drl_v2",
) -> list[str]:
    root = Path(project_root).resolve()
    try:
        receipt_file = _safe_path(root, receipt_path, label="calibration lineage receipt")
        _verify_sidecar(receipt_file, label="calibration lineage receipt")
        receipt = _json_object(receipt_file, label="calibration lineage receipt")
        if (
            receipt.get("schema_version") != 1
            or receipt.get("status") != "accepted"
            or receipt.get("decision") != "reuse_without_compute"
            or receipt.get("submitted_hpc_jobs") != 0
        ):
            return ["calibration lineage receipt contract does not match"]
        references = receipt.get("references")
        if not isinstance(references, list) or not references:
            return ["calibration lineage receipt has no reference inventory"]
        for record in references:
            if not isinstance(record, dict):
                return ["calibration lineage receipt contains a malformed reference"]
            path = _safe_path(root, str(record.get("path", "")), label="lineage reference")
            if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get(
                "sha256"
            ):
                return [f"calibration lineage reference changed: {record.get('path')}"]
        definition, _ = load_definition(
            _safe_path(root, spec_path, label="storage-safe campaign spec"),
            project_root=root,
            campaign_dir=(root / campaign_dir).resolve(),
        )
        superseded = {"cluster_preflight", *STAGE_IDS} & set(definition.stage_map)
        if superseded:
            return [f"storage-safe campaign still declares superseded stages: {sorted(superseded)}"]
        if "calibration_lineage" not in definition.stage_map:
            return ["storage-safe campaign lacks its calibration-lineage stage"]
        original_definition, _ = load_definition(
            _safe_path(
                root,
                "config/research_campaign.yaml",
                label="original campaign spec",
            ),
            project_root=root,
            campaign_dir=(root / campaign_dir).resolve(),
        )
        _lineage_stage_records(original_definition)
    except (OSError, TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def materialize_lineage(
    *,
    project_root: str | Path,
    spec_path: str | Path,
    campaign_dir: str | Path,
    snapshot_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    spec = _safe_path(root, spec_path, label="storage-safe campaign spec")
    campaign = (root / campaign_dir).resolve()
    snapshot = (root / snapshot_dir).resolve()
    output = (root / output_path).resolve()
    if (
        not campaign.is_relative_to(root)
        or not campaign.is_dir()
        or campaign.is_symlink()
        or not snapshot.is_relative_to(root)
        or not snapshot.is_dir()
        or snapshot.is_symlink()
        or not output.is_relative_to(campaign)
        or output.exists()
        or Path(str(output) + ".sha256").exists()
    ):
        raise ValueError("calibration lineage paths are missing, unsafe, or already exist")
    snapshot_receipt, archive_path, members = _snapshot_members(
        root=root,
        snapshot_dir=snapshot,
    )
    original_values: dict[Path, bytes] = {}
    restored: list[Path] = []
    backups: list[Path] = []
    try:
        for index, relative in enumerate(RESTORED_PATHS):
            destination = _safe_path(root, relative, label="live predecessor evidence")
            original_values[destination] = destination.read_bytes()
            expected = members.get(relative)
            if expected is None:
                raise ValueError(f"cutover snapshot lacks predecessor evidence: {relative}")
            backup = snapshot / f"superseded_transition_{index}_{destination.name}"
            if backup.exists():
                _verify_sidecar(backup, label="superseded transition backup")
            else:
                _atomic_bytes(backup, original_values[destination])
                _atomic_bytes(Path(str(backup) + ".sha256"), f"{sha256_file(backup)}\n".encode())
            backups.append(backup)
            if original_values[destination] != expected:
                _atomic_bytes(destination, expected)
            restored.append(destination)

        restored_stage_ids = ("cluster_preflight", *STAGE_IDS)
        for index, stage_id in enumerate(restored_stage_ids):
            gate = _json_object(
                restored[index * 2],
                label=f"restored {stage_id} gate",
            )
            marker = _json_object(
                restored[index * 2 + 1],
                label=f"restored {stage_id} marker",
            )
            if (
                gate.get("stage_id") != stage_id
                or gate.get("passed") is not True
                or marker.get("stage_id") != stage_id
                or marker.get("evidence_fingerprint") != gate.get("evidence_fingerprint")
            ):
                raise ValueError(f"restored {stage_id} gate and marker are inconsistent")

        definition, _ = load_definition(
            spec,
            project_root=root,
            campaign_dir=campaign,
        )
        superseded = {"cluster_preflight", *STAGE_IDS} & set(definition.stage_map)
        if superseded:
            raise ValueError(
                f"storage-safe campaign did not supersede stages: {sorted(superseded)}"
            )
        original_spec = _safe_path(
            root,
            "config/research_campaign.yaml",
            label="original campaign spec",
        )
        original_definition, _ = load_definition(
            original_spec,
            project_root=root,
            campaign_dir=campaign,
        )
        stage_records = _lineage_stage_records(original_definition)
        snapshot_receipt_path = snapshot / "snapshot_receipt.json"
        references = [
            _artifact(root, snapshot_receipt_path),
            _artifact(root, Path(str(snapshot_receipt_path) + ".sha256")),
            _artifact(root, archive_path),
            _artifact(root, Path(str(archive_path) + ".sha256")),
            _artifact(root, original_spec),
            _artifact(root, spec),
            *(_artifact(root, path) for path in restored),
            *(_artifact(root, path) for path in backups),
        ]
        receipt = {
            "schema_version": 1,
            "status": "accepted",
            "decision": "reuse_without_compute",
            "created_at_utc": _utc_now(),
            "submitted_hpc_jobs": 0,
            "source": {
                "snapshot_kind": snapshot_receipt.get("snapshot_kind"),
                "active_release": snapshot_receipt.get("active_release"),
                "original_campaign_spec": next(
                    record
                    for record in snapshot_receipt["files"]
                    if record.get("path") == "config/research_campaign.yaml"
                ),
            },
            "target": {
                "campaign_id": definition.campaign_id,
                "campaign_spec": _artifact(root, spec),
                "replacement_stage": "calibration_lineage",
            },
            "restored_predecessor_records": [
                _artifact(root, path) for path in restored
            ],
            "superseded_transition_records": [
                _artifact(root, path) for path in backups
            ],
            "accepted_stages": stage_records,
            "references": sorted(references, key=lambda record: record["path"]),
            "scientific_claim": (
                "Only control-lineage predecessor records were restored. The accepted "
                "observed-data calibration outputs, attempts, deployment, preflight, "
                "configuration, executable code, and receipt hashes remain unchanged."
            ),
        }
        _atomic_json(output, receipt)
        _atomic_bytes(Path(str(output) + ".sha256"), f"{sha256_file(output)}\n".encode())
        issues = validate_lineage_receipt(
            output,
            project_root=root,
            spec_path=spec,
            campaign_dir=campaign,
        )
        if issues:
            raise ValueError(f"materialized calibration lineage is invalid: {issues}")
        return receipt
    except BaseException:
        for destination, value in original_values.items():
            _atomic_bytes(destination, value)
        output.unlink(missing_ok=True)
        Path(str(output) + ".sha256").unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accept pre-cutover calibration under the storage-safe control lineage"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize_lineage(
                project_root=args.project_root,
                spec_path=args.spec,
                campaign_dir=args.campaign_dir,
                snapshot_dir=args.snapshot_dir,
                output_path=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
