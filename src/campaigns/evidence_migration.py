from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.verify_campaign_snapshot import verify_snapshot
from src.campaigns.bistka_deployment import (
    validate_deployment_manifest,
    validate_preflight_receipt,
)
from src.campaigns.calibration import (
    atomic_json,
    atomic_text,
    canonical_sha256,
    code_provenance,
    load_observed_inputs,
    sha256_file,
    training_feature_subset,
)
from src.campaigns.model import CampaignDefinition, CampaignStage
from src.utils.config import load_config


IMPORT_RELATIVE_PATH = "imports/v1_calibration_evidence.json"
ALLOWED_CONFIG_DIFFERENCES = ("project.name",)
STAGE_IDS = ("vector_benchmark", "determinism")
RUNTIME_FIELDS = (
    "python",
    "implementation",
    "platform",
    "numpy",
    "pandas",
    "gymnasium",
    "stable_baselines3",
    "torch",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _relative_path(root: Path, raw: str | Path, label: str, *, must_exist: bool = True) -> Path:
    candidate = Path(raw)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if ".." in candidate.parts or not path.is_relative_to(root):
        raise ValueError(f"{label} must stay inside the project")
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {candidate}")
    return path


def _artifact(root: Path, path: Path, **extra: Any) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise FileNotFoundError(f"artifact is missing or outside the project: {path}")
    return {
        "path": str(resolved.relative_to(root)),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        **extra,
    }


def _attempt_sha256(attempt: dict[str, Any]) -> str:
    unhashed = dict(attempt)
    unhashed.pop("attempt_sha256", None)
    return canonical_sha256(unhashed)


def _remote_output_path(
    *,
    root: Path,
    remote_root: Path,
    raw_path: Any,
    label: str,
) -> Path:
    remote = Path(str(raw_path))
    if not remote.is_absolute() or not remote_root.is_absolute():
        raise ValueError(f"{label} must use an absolute remote path")
    try:
        relative = remote.relative_to(remote_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the source deployment root") from exc
    if ".." in relative.parts:
        raise ValueError(f"{label} contains an unsafe path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"{label} is missing locally: {relative}")
    return path


def _nested_without(value: dict[str, Any], dotted: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    parts = dotted.split(".")
    current: Any = result
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return result
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)
    return result


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        paths: list[str] = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            child = f"{prefix}[{index}]"
            paths.extend(_diff_paths(left_item, right_item, child))
        return paths
    return [] if left == right else [prefix]


def _require_sidecar(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"{label} SHA-256 sidecar is missing: {sidecar}")
    recorded = sidecar.read_text(encoding="utf-8").strip()
    if not _is_sha256(recorded) or recorded != sha256_file(path):
        raise ValueError(f"{label} SHA-256 sidecar does not match")
    return sidecar


def _snapshot_artifacts(snapshot: dict[str, Any]) -> dict[str, str]:
    records = snapshot.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("accepted snapshot artifacts must be a list")
    result: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("accepted snapshot contains a malformed artifact record")
        path = str(record.get("path", ""))
        digest = str(record.get("sha256", ""))
        if not path or path in result or not _is_sha256(digest):
            raise ValueError(f"accepted snapshot contains an invalid artifact record: {path}")
        result[path] = digest
    return result


def _source_campaign_from_snapshot(root: Path, snapshot: dict[str, Any]) -> tuple[str, Path]:
    candidates = {
        Path(str(record.get("path", ""))).parent
        for record in snapshot.get("artifacts", [])
        if isinstance(record, dict)
        and Path(str(record.get("path", ""))).name == "state.json"
        and len(Path(str(record.get("path", ""))).parts) == 3
        and Path(str(record.get("path", ""))).parts[0] == "campaigns"
    }
    if len(candidates) != 1:
        raise ValueError("accepted snapshot must identify exactly one source campaign state")
    relative = candidates.pop()
    campaign_dir = (root / relative).resolve()
    if not campaign_dir.is_relative_to(root) or not campaign_dir.is_dir():
        raise FileNotFoundError("accepted snapshot source campaign directory is missing")
    return campaign_dir.name, campaign_dir


def _gate_binding(root: Path, campaign_dir: Path, stage_id: str) -> dict[str, Any]:
    gate_path = campaign_dir / "stages" / stage_id / "gate_report.json"
    marker_path = campaign_dir / "stages" / stage_id / "_SUCCESS.json"
    gate = _json_object(gate_path, f"source {stage_id} gate report")
    marker = _json_object(marker_path, f"source {stage_id} success marker")
    fingerprint = str(gate.get("evidence_fingerprint", ""))
    if (
        gate.get("stage_id") != stage_id
        or gate.get("passed") is not True
        or marker.get("stage_id") != stage_id
        or not _is_sha256(fingerprint)
        or marker.get("evidence_fingerprint") != fingerprint
    ):
        raise ValueError(f"source {stage_id} gate report and success marker are inconsistent")
    evidence_files = gate.get("evidence_files")
    if not isinstance(evidence_files, list) or not evidence_files:
        raise ValueError(f"source {stage_id} gate report contains no evidence inventory")
    return {
        "stage_id": stage_id,
        "evidence_fingerprint": fingerprint,
        "gate_report": _artifact(root, gate_path),
        "success_marker": _artifact(root, marker_path),
        "evidence_files": sorted(str(value) for value in evidence_files),
    }


def _stage_validator_options(stage: CampaignStage, validator: str) -> dict[str, Any]:
    matches = [
        requirement.options
        for requirement in stage.requirements
        if requirement.validator == validator
    ]
    if len(matches) != 1:
        raise ValueError(f"target stage {stage.id} must define exactly one {validator} validator")
    return dict(matches[0])


def _migration_options(stage: CampaignStage) -> dict[str, Any]:
    matches = [
        requirement.options
        for requirement in stage.requirements
        if requirement.validator == "calibration_evidence"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"target stage {stage.id} must define exactly one calibration_evidence validator"
        )
    return dict(matches[0])


def _source_attempt(
    root: Path,
    source_campaign_id: str,
    source_campaign_dir: Path,
    stage_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    latest_path = source_campaign_dir / "attempts" / f"{stage_id}_latest.json"
    attempt = _json_object(latest_path, f"source {stage_id} latest attempt")
    attempt_id = str(attempt.get("attempt_id", ""))
    if (
        attempt.get("schema_version") not in {1, 2}
        or attempt.get("status") != "rendered"
        or attempt.get("campaign_id") != source_campaign_id
        or attempt.get("stage_id") != stage_id
        or not attempt_id
    ):
        raise ValueError(f"source {stage_id} latest attempt has an invalid identity")
    observed_hash = _attempt_sha256(attempt)
    if not _is_sha256(attempt.get("attempt_sha256")) or attempt["attempt_sha256"] != observed_hash:
        raise ValueError(f"source {stage_id} attempt hash is invalid")
    immutable_path = source_campaign_dir / "attempts" / f"{attempt_id}.json"
    immutable = _json_object(immutable_path, f"source {stage_id} immutable attempt")
    if immutable != attempt:
        raise ValueError(f"source {stage_id} latest and immutable attempts differ")
    return attempt, {
        "latest_path": str(latest_path.relative_to(root)),
        "latest_sha256": sha256_file(latest_path),
        "immutable_path": str(immutable_path.relative_to(root)),
        "immutable_sha256": sha256_file(immutable_path),
        "attempt_id": attempt_id,
        "attempt_sha256": observed_hash,
        "schema_version": attempt["schema_version"],
        "rendered_at_utc": attempt.get("rendered_at_utc"),
    }


def _source_manifest_and_preflight(
    root: Path,
    source_campaign_dir: Path,
    attempt: dict[str, Any],
    snapshot_hashes: dict[str, str],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    manifest_path = source_campaign_dir / "deployment" / "bistka_deployment_manifest.json"
    preflight_path = root / "reports" / "bistka_calibration_preflight.json"
    for path, label in (
        (manifest_path, "source deployment manifest"),
        (preflight_path, "source calibration preflight"),
    ):
        relative = str(path.relative_to(root))
        if snapshot_hashes.get(relative) != sha256_file(path):
            raise ValueError(f"{label} is not frozen by the accepted snapshot")
    manifest = _json_object(manifest_path, "source deployment manifest")
    preflight = _json_object(preflight_path, "source calibration preflight")
    if (
        manifest.get("release_id") != attempt.get("deployment_release_id")
        or preflight.get("release_id") != manifest.get("release_id")
        or preflight.get("manifest_sha256") != sha256_file(manifest_path)
        or preflight.get("status") != "pass"
        or sha256_file(preflight_path) != attempt.get("preflight_sha256")
    ):
        raise ValueError("source attempt, deployment manifest, and preflight do not agree")
    remote_root = Path(str(manifest.get("target_project_root", "")))
    if not remote_root.is_absolute() or preflight.get("project_root") != str(remote_root):
        raise ValueError("source deployment root and preflight project root do not agree")
    return manifest_path, manifest, preflight_path, preflight


def _receipt_for_attempt(
    root: Path,
    *,
    stage_id: str,
    source_campaign_id: str,
    attempt: dict[str, Any],
    manifest: dict[str, Any],
    preflight: dict[str, Any],
    snapshot_hashes: dict[str, str],
    gate: dict[str, Any],
    options: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    candidates = []
    for path in (root / "reports").glob("**/*.receipt.json"):
        try:
            value = _json_object(path, "calibration receipt candidate")
        except ValueError:
            continue
        if value.get("campaign_attempt_id") == attempt.get("attempt_id"):
            candidates.append((path, value))
    if len(candidates) != 1:
        raise ValueError(
            f"source {stage_id} must have exactly one receipt for its accepted attempt"
        )
    receipt_path, receipt = candidates[0]
    expected_kind = str(options.get("kind", ""))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "success"
        or receipt.get("error")
        or receipt.get("campaign_id") != source_campaign_id
        or receipt.get("campaign_stage_id") != stage_id
        or receipt.get("campaign_attempt_sha256") != attempt.get("attempt_sha256")
        or receipt.get("deployment_release_id") != manifest.get("release_id")
        or receipt.get("bistka_preflight_sha256") != attempt.get("preflight_sha256")
        or (expected_kind and receipt.get("calibration_kind") != expected_kind)
    ):
        raise ValueError(f"source {stage_id} receipt is not bound to its accepted attempt")
    if str(receipt.get("slurm_job_id", "")).isdigit() is False:
        raise ValueError(f"source {stage_id} receipt has no numeric SLURM job id")
    if receipt.get("hostname") != preflight.get("hostname"):
        raise ValueError(f"source {stage_id} receipt hostname differs from its preflight")
    parameters = receipt.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"source {stage_id} receipt parameters must be an object")
    for key, expected in dict(options.get("parameters", {})).items():
        if parameters.get(key) != expected:
            raise ValueError(f"source {stage_id} parameter {key} differs from V2")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"source {stage_id} receipt runtime must be an object")
    for field in options.get("runtime_packages", RUNTIME_FIELDS):
        if runtime.get(str(field)) in {None, "", "missing"}:
            raise ValueError(f"source {stage_id} receipt runtime is missing {field}")

    remote_root = Path(str(manifest["target_project_root"]))
    attempt_inputs = {
        str(record.get("path", "")): record
        for record in attempt.get("inputs", [])
        if isinstance(record, dict)
    }
    input_names: set[str] = set()
    for record in receipt.get("inputs", []):
        if not isinstance(record, dict):
            raise ValueError(f"source {stage_id} receipt contains a malformed input")
        name = str(record.get("name", ""))
        if not name or name in input_names:
            raise ValueError(f"source {stage_id} receipt contains duplicate input names")
        input_names.add(name)
        if record.get("exists") is not True or not _is_sha256(record.get("sha256")):
            raise ValueError(f"source {stage_id} receipt input is missing or unhashed: {name}")
        if record.get("bytes") is None:
            continue
        local = _remote_output_path(
            root=root,
            remote_root=remote_root,
            raw_path=record.get("path"),
            label=f"source {stage_id} receipt input {name}",
        )
        relative = str(local.relative_to(root))
        frozen = attempt_inputs.get(relative)
        if (
            not isinstance(frozen, dict)
            or frozen.get("sha256") != record.get("sha256")
            or frozen.get("bytes") != record.get("bytes")
        ):
            raise ValueError(f"source {stage_id} receipt input was not frozen: {name}")
        if sha256_file(local) != record.get("sha256") or local.stat().st_size != record.get(
            "bytes"
        ):
            raise ValueError(f"source {stage_id} scientific input changed: {relative}")
    required_names = {str(value) for value in options.get("required_input_names", [])}
    if not required_names.issubset(input_names):
        raise ValueError(
            f"source {stage_id} receipt is missing required inputs: "
            f"{sorted(required_names - input_names)}"
        )

    outputs: dict[str, Path] = {}
    output_records = receipt.get("outputs")
    if not isinstance(output_records, list):
        raise ValueError(f"source {stage_id} receipt outputs must be a list")
    for record in output_records:
        if not isinstance(record, dict):
            raise ValueError(f"source {stage_id} receipt contains a malformed output")
        name = str(record.get("name", ""))
        if not name or name in outputs:
            raise ValueError(f"source {stage_id} receipt contains duplicate output names")
        path = _remote_output_path(
            root=root,
            remote_root=remote_root,
            raw_path=record.get("path"),
            label=f"source {stage_id} output {name}",
        )
        if (
            record.get("exists") is not True
            or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != record.get("bytes")
        ):
            raise ValueError(f"source {stage_id} output hash or size differs: {name}")
        relative = str(path.relative_to(root))
        if snapshot_hashes.get(relative) != sha256_file(path):
            raise ValueError(f"source {stage_id} output is not frozen by the snapshot: {relative}")
        outputs[name] = path
    expected_output_names = {str(value) for value in options.get("output_names", [])}
    if set(outputs) != expected_output_names:
        raise ValueError(
            f"source {stage_id} output names differ from V2: "
            f"{sorted(outputs)} != {sorted(expected_output_names)}"
        )
    receipt_relative = str(receipt_path.relative_to(root))
    if snapshot_hashes.get(receipt_relative) != sha256_file(receipt_path):
        raise ValueError(f"source {stage_id} receipt is not frozen by the snapshot")
    gate_evidence = set(gate["evidence_files"])
    required_gate_evidence = {
        receipt_relative,
        *(str(path.relative_to(root)) for path in outputs.values()),
    }
    if not required_gate_evidence.issubset(gate_evidence):
        raise ValueError(f"source {stage_id} accepted gate does not bind its receipt and outputs")
    return receipt_path, receipt, outputs


def _validate_result_contract(
    *,
    stage: CampaignStage,
    receipt: dict[str, Any],
    outputs: dict[str, Path],
) -> dict[str, Any]:
    from src.campaigns.gates import (
        _benchmark_validator,
        _determinism_validator,
        _vector_metadata_validator,
    )

    if stage.id == "vector_benchmark":
        csv_path = outputs.get("benchmark_csv")
        metadata_path = outputs.get("benchmark_metadata")
        if csv_path is None or metadata_path is None:
            raise ValueError("source vector benchmark outputs are incomplete")
        csv_issues = _benchmark_validator(
            csv_path,
            _stage_validator_options(stage, "vector_benchmark"),
        )
        metadata_issues = _vector_metadata_validator(
            metadata_path,
            _stage_validator_options(stage, "vector_benchmark_metadata"),
        )
        if csv_issues or metadata_issues:
            raise ValueError(
                "source vector benchmark evidence is invalid: "
                + "; ".join(sorted(set([*csv_issues, *metadata_issues])))
            )
        metadata = _json_object(metadata_path, "source vector benchmark metadata")
        selected = int(metadata.get("selected_num_envs", 0))
        if receipt.get("result", {}).get("selected_num_envs") != selected:
            raise ValueError("source vector receipt selection differs from its metadata")
        return {
            "selected_num_envs": selected,
            "trial_count": int(metadata.get("trial_count", 0)),
            "selection_rule": metadata.get("selection_rule"),
            "selection_fraction_of_max": metadata.get("selection_fraction_of_max"),
            "throughput_statistic": metadata.get("throughput_statistic"),
            "summary_by_env_count": metadata.get("summary_by_env_count"),
        }
    result_path = outputs.get("determinism_result")
    if result_path is None:
        raise ValueError("source determinism output is incomplete")
    issues = _determinism_validator(
        result_path,
        dict(_migration_options(stage).get("determinism_options", {})),
    )
    if issues:
        raise ValueError(
            "source determinism evidence is invalid: " + "; ".join(sorted(set(issues)))
        )
    result = _json_object(result_path, "source determinism result")
    if (
        result.get("deterministic") is not True
        or receipt.get("result", {}).get("deterministic") is not True
        or receipt.get("result", {}).get("trajectory_sha256") != result.get("first_sha256")
    ):
        raise ValueError("source determinism receipt does not reproduce its result")
    return {
        "deterministic": True,
        "steps": int(result.get("steps", 0)),
        "trajectory_sha256": result.get("first_sha256"),
        "independent_subprocess_runs": result.get("independent_subprocess_runs"),
    }


def _receipt_record(root: Path, path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    return _artifact(
        root,
        path,
        slurm_job_id=str(receipt.get("slurm_job_id", "")),
        hostname=receipt.get("hostname"),
        runtime=receipt.get("runtime"),
        parameters=receipt.get("parameters"),
        result=receipt.get("result"),
        calibration_kind=receipt.get("calibration_kind"),
    )


def _target_binding(
    definition: CampaignDefinition,
    stages: dict[str, CampaignStage],
) -> tuple[dict[str, Any], list[Path]]:
    pairs = {definition.resolve_deployment_pair(stage) for stage in stages.values()}
    if len(pairs) != 1:
        raise ValueError("V2 calibration stages do not share one deployment/preflight pair")
    raw_manifest, raw_preflight = pairs.pop()
    manifest_path = _relative_path(
        definition.project_root,
        definition.expand_project_relative(raw_manifest),
        "target deployment manifest",
    )
    preflight_path = _relative_path(
        definition.project_root,
        definition.expand_project_relative(raw_preflight),
        "target calibration preflight",
    )
    manifest_issues, manifest_files = validate_deployment_manifest(
        manifest_path,
        project_root=definition.project_root,
    )
    preflight_issues, preflight_files = validate_preflight_receipt(
        preflight_path,
        expected_manifest_path=manifest_path,
    )
    if manifest_issues or preflight_issues:
        raise ValueError(
            "target V2 deployment/preflight is stale or invalid: "
            + "; ".join(sorted(set([*manifest_issues, *preflight_issues])))
        )
    manifest = _json_object(manifest_path, "target deployment manifest")
    preflight = _json_object(preflight_path, "target calibration preflight")
    if (
        preflight.get("release_id") != manifest.get("release_id")
        or preflight.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("target V2 manifest and preflight identities differ")
    return {
        "manifest": _artifact(
            definition.project_root,
            manifest_path,
            release_id=manifest.get("release_id"),
            target_project_root=manifest.get("target_project_root"),
        ),
        "preflight": _artifact(
            definition.project_root,
            preflight_path,
            release_id=preflight.get("release_id"),
            hostname=preflight.get("hostname"),
            project_root=preflight.get("project_root"),
            python_executable=preflight.get("python_executable"),
            python_version=preflight.get("python_version"),
            platform=preflight.get("platform"),
            packages=preflight.get("packages"),
        ),
    }, [*manifest_files, *preflight_files]


def _config_binding(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = load_config(path)
    if not isinstance(resolved, dict):
        raise ValueError(f"resolved configuration must be a mapping: {path}")
    return resolved, _artifact(
        root,
        path,
        resolved_sha256=canonical_sha256(resolved),
        project_name=resolved.get("project", {}).get("name"),
    )


def _runtime_comparison(
    *,
    source_preflight: dict[str, Any],
    target_preflight_record: dict[str, Any],
    source_receipts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_packages = dict(target_preflight_record.get("packages") or {})
    normalized_target = {
        "python": target_preflight_record.get("python_version"),
        "platform": target_preflight_record.get("platform"),
        "numpy": target_packages.get("numpy"),
        "pandas": target_packages.get("pandas"),
        "gymnasium": target_packages.get("gymnasium"),
        "stable_baselines3": target_packages.get("stable-baselines3"),
        "torch": target_packages.get("torch"),
    }
    source_preflight_runtime = {
        "python": source_preflight.get("python_version"),
        "platform": source_preflight.get("platform"),
        "numpy": source_preflight.get("packages", {}).get("numpy"),
        "pandas": source_preflight.get("packages", {}).get("pandas"),
        "gymnasium": source_preflight.get("packages", {}).get("gymnasium"),
        "stable_baselines3": source_preflight.get("packages", {}).get(
            "stable-baselines3"
        ),
        "torch": source_preflight.get("packages", {}).get("torch"),
    }
    if normalized_target != source_preflight_runtime:
        raise ValueError("source and target calibration runtime versions differ")
    source_runtimes = {
        stage_id: {
            field: receipt.get("runtime", {}).get(field)
            for field in (
                "python",
                "platform",
                "numpy",
                "pandas",
                "gymnasium",
                "stable_baselines3",
                "torch",
            )
        }
        for stage_id, receipt in source_receipts.items()
    }
    for stage_id, runtime in source_runtimes.items():
        if runtime != normalized_target:
            raise ValueError(f"source {stage_id} runtime differs from the V2 preflight")
    if (
        source_preflight.get("hostname") != target_preflight_record.get("hostname")
        or source_preflight.get("project_root") != target_preflight_record.get(
            "project_root"
        )
        or source_preflight.get("python_executable")
        != target_preflight_record.get("python_executable")
    ):
        raise ValueError("source and target host, project root, or Python executable differ")
    return {
        "hostname": source_preflight.get("hostname"),
        "project_root": source_preflight.get("project_root"),
        "python_executable": source_preflight.get("python_executable"),
        "versions": normalized_target,
    }


def _scientific_inventory(
    definition: CampaignDefinition,
    stages: dict[str, CampaignStage],
    source_attempts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for stage_id, stage in stages.items():
        source_inputs = {
            str(record.get("path", "")): record
            for record in source_attempts[stage_id].get("inputs", [])
            if isinstance(record, dict)
        }
        raw_paths = [stage.job_script, *stage.fingerprint_paths]
        for raw_path in raw_paths:
            if not raw_path:
                continue
            relative = definition.expand_project_relative(str(raw_path))
            path = _relative_path(
                definition.project_root,
                relative,
                f"target {stage_id} scientific input",
            )
            source_record = source_inputs.get(str(path.relative_to(definition.project_root)))
            if not isinstance(source_record, dict):
                raise ValueError(
                    f"source {stage_id} attempt did not freeze scientific input: {relative}"
                )
            observed_hash = sha256_file(path)
            if (
                source_record.get("sha256") != observed_hash
                or source_record.get("bytes") != path.stat().st_size
            ):
                raise ValueError(
                    f"V1 and V2 scientific input differs for {stage_id}: {relative}"
                )
            existing = records.get(relative)
            record = {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": observed_hash,
                "stages": sorted(
                    {
                        stage_id,
                        *(existing.get("stages", []) if existing else []),
                    }
                ),
            }
            records[relative] = record
    return [records[path] for path in sorted(records)]


def _referenced_paths(root: Path, value: Any) -> list[Path]:
    paths: set[Path] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            raw_path = item.get("path")
            digest = item.get("sha256")
            if isinstance(raw_path, str) and _is_sha256(digest):
                try:
                    path = _relative_path(root, raw_path, "migration evidence artifact")
                except (FileNotFoundError, ValueError):
                    pass
                else:
                    paths.add(path)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return sorted(paths)


def _build_calibration_import(
    definition: CampaignDefinition,
    source_snapshot_path: Path,
    *,
    created_at_utc: str,
) -> tuple[dict[str, Any], list[Path]]:
    root = definition.project_root
    snapshot_path = _relative_path(root, source_snapshot_path, "source snapshot")
    snapshot_sidecar = _require_sidecar(snapshot_path, "source snapshot")
    snapshot_verification = verify_snapshot(snapshot_path, root)
    snapshot = _json_object(snapshot_path, "source snapshot")
    snapshot_hashes = _snapshot_artifacts(snapshot)
    source_campaign_id, source_campaign_dir = _source_campaign_from_snapshot(root, snapshot)
    if source_campaign_id == definition.campaign_id:
        raise ValueError("source and target campaign ids must differ for evidence migration")

    stages = {stage_id: definition.stage_map[stage_id] for stage_id in STAGE_IDS}
    target_pair, target_references = _target_binding(definition, stages)
    source_attempts: dict[str, dict[str, Any]] = {}
    stage_records: dict[str, Any] = {}
    source_receipts: dict[str, dict[str, Any]] = {}
    source_manifest_path: Path | None = None
    source_manifest: dict[str, Any] | None = None
    source_preflight_path: Path | None = None
    source_preflight: dict[str, Any] | None = None
    source_config_path: Path | None = None

    for stage_id, target_stage in stages.items():
        gate = _gate_binding(root, source_campaign_dir, stage_id)
        attempt, attempt_record = _source_attempt(
            root,
            source_campaign_id,
            source_campaign_dir,
            stage_id,
        )
        source_attempts[stage_id] = attempt
        (
            current_manifest_path,
            current_manifest,
            current_preflight_path,
            current_preflight,
        ) = _source_manifest_and_preflight(
            root,
            source_campaign_dir,
            attempt,
            snapshot_hashes,
        )
        if source_manifest_path is None:
            source_manifest_path = current_manifest_path
            source_manifest = current_manifest
            source_preflight_path = current_preflight_path
            source_preflight = current_preflight
        elif (
            current_manifest_path != source_manifest_path
            or current_preflight_path != source_preflight_path
        ):
            raise ValueError("source calibration stages used different releases or preflights")
        options = _migration_options(target_stage)
        receipt_path, receipt, outputs = _receipt_for_attempt(
            root,
            stage_id=stage_id,
            source_campaign_id=source_campaign_id,
            attempt=attempt,
            manifest=current_manifest,
            preflight=current_preflight,
            snapshot_hashes=snapshot_hashes,
            gate=gate,
            options=options,
        )
        source_receipts[stage_id] = receipt
        result = _validate_result_contract(
            stage=target_stage,
            receipt=receipt,
            outputs=outputs,
        )
        config_input = next(
            (
                record
                for record in receipt.get("inputs", [])
                if isinstance(record, dict) and record.get("name") == "config_file"
            ),
            None,
        )
        if not isinstance(config_input, dict):
            raise ValueError(f"source {stage_id} receipt has no config_file input")
        candidate_config = _remote_output_path(
            root=root,
            remote_root=Path(str(current_manifest["target_project_root"])),
            raw_path=config_input.get("path"),
            label=f"source {stage_id} config file",
        )
        if source_config_path is None:
            source_config_path = candidate_config
        elif source_config_path != candidate_config:
            raise ValueError("source calibration stages used different configurations")
        stage_records[stage_id] = {
            "stage_id": stage_id,
            "gate": gate,
            "attempt": attempt_record,
            "receipt": _receipt_record(root, receipt_path, receipt),
            "outputs": {
                name: _artifact(root, path) for name, path in sorted(outputs.items())
            },
            "result": result,
        }

    assert source_manifest_path is not None
    assert source_manifest is not None
    assert source_preflight_path is not None
    assert source_preflight is not None
    assert source_config_path is not None

    target_config_paths = {
        _relative_path(
            root,
            definition.expand_project_relative(str(stage.config_path)),
            f"target {stage.id} calibration config",
        )
        for stage in stages.values()
    }
    if len(target_config_paths) != 1:
        raise ValueError("V2 calibration stages do not share one configuration")
    target_config_path = target_config_paths.pop()
    source_config, source_config_record = _config_binding(root, source_config_path)
    target_config, target_config_record = _config_binding(root, target_config_path)
    config_differences = sorted(set(_diff_paths(source_config, target_config)))
    if config_differences != list(ALLOWED_CONFIG_DIFFERENCES):
        raise ValueError(
            "V1 and V2 resolved calibration configurations differ outside the "
            f"allowed metadata path: {config_differences}"
        )
    normalized_source = source_config
    normalized_target = target_config
    for allowed_path in ALLOWED_CONFIG_DIFFERENCES:
        normalized_source = _nested_without(normalized_source, allowed_path)
        normalized_target = _nested_without(normalized_target, allowed_path)
    source_compatibility_hash = canonical_sha256(normalized_source)
    target_compatibility_hash = canonical_sha256(normalized_target)
    if source_compatibility_hash != target_compatibility_hash:
        raise ValueError("normalized V1 and V2 calibration configurations differ")

    source_inputs = load_observed_inputs(source_config_path)
    target_inputs = load_observed_inputs(target_config_path)
    source_features, source_split = training_feature_subset(source_inputs)
    target_features, target_split = training_feature_subset(target_inputs)
    data_contract = {
        "source": source_inputs.data_source,
        "path": str(source_inputs.feature_path.relative_to(root)),
        "bytes": source_inputs.feature_path.stat().st_size,
        "sha256": source_inputs.data_sha256,
        "feature_rows": int(len(source_features)),
        "feature_dates": int(source_features["date"].nunique()),
        "feature_tickers": int(source_features["ticker"].nunique()),
        "training_split": source_split,
        "synthetic": False,
    }
    target_data_contract = {
        "source": target_inputs.data_source,
        "path": str(target_inputs.feature_path.relative_to(root)),
        "bytes": target_inputs.feature_path.stat().st_size,
        "sha256": target_inputs.data_sha256,
        "feature_rows": int(len(target_features)),
        "feature_dates": int(target_features["date"].nunique()),
        "feature_tickers": int(target_features["ticker"].nunique()),
        "training_split": target_split,
        "synthetic": False,
    }
    if target_data_contract != data_contract:
        raise ValueError("V1 and V2 observed calibration data contracts differ")

    scientific_inventory = _scientific_inventory(definition, stages, source_attempts)
    for stage_id, stage in stages.items():
        code_paths = [path for path in stage.fingerprint_paths if Path(path).suffix == ".py"]
        current_code = code_provenance(root, code_paths)
        if source_receipts[stage_id].get("result", {}).get("code_sha256") != current_code.get(
            "sha256"
        ):
            raise ValueError(f"V1 and V2 executable calibration code differs for {stage_id}")

    target_preflight_record = target_pair["preflight"]
    runtime = _runtime_comparison(
        source_preflight=source_preflight,
        target_preflight_record=target_preflight_record,
        source_receipts=source_receipts,
    )
    accepted_result = snapshot.get("accepted_result")
    scheduler = snapshot.get("scheduler")
    if not isinstance(accepted_result, dict) or not isinstance(scheduler, dict):
        raise ValueError("accepted snapshot is missing result or scheduler metadata")
    accepted_job = str(scheduler.get("accepted_job_id", ""))
    if (
        accepted_result.get("selected_num_envs")
        != stage_records["vector_benchmark"]["result"]["selected_num_envs"]
        or accepted_result.get("deterministic")
        != stage_records["determinism"]["result"]["deterministic"]
        or accepted_result.get("feature_sha256") != data_contract["sha256"]
        or {
            stage_records[stage_id]["receipt"]["slurm_job_id"] for stage_id in STAGE_IDS
        }
        != {accepted_job}
    ):
        raise ValueError("accepted snapshot summary differs from its calibration evidence")

    receipt = {
        "schema_version": 1,
        "evidence_kind": "v1_calibration_evidence_import",
        "status": "accepted",
        "decision": "migrated_without_rerun",
        "created_at_utc": created_at_utc,
        "submitted_hpc_jobs": 0,
        "source_snapshot": {
            **_artifact(
                root,
                snapshot_path,
                snapshot_id=snapshot.get("snapshot_id"),
                verification=snapshot_verification,
            ),
            "sidecar": _artifact(root, snapshot_sidecar),
        },
        "source": {
            "campaign_id": source_campaign_id,
            "campaign_dir": str(source_campaign_dir.relative_to(root)),
            "deployment": {
                "manifest": _artifact(
                    root,
                    source_manifest_path,
                    release_id=source_manifest.get("release_id"),
                    target_project_root=source_manifest.get("target_project_root"),
                ),
                "preflight": _artifact(
                    root,
                    source_preflight_path,
                    release_id=source_preflight.get("release_id"),
                    hostname=source_preflight.get("hostname"),
                    project_root=source_preflight.get("project_root"),
                    python_executable=source_preflight.get("python_executable"),
                    python_version=source_preflight.get("python_version"),
                    platform=source_preflight.get("platform"),
                    packages=source_preflight.get("packages"),
                ),
            },
            "config": source_config_record,
            "stages": stage_records,
        },
        "target": {
            "campaign_id": definition.campaign_id,
            "campaign_dir": str(definition.campaign_dir.relative_to(root)),
            "deployment": target_pair,
            "config": target_config_record,
        },
        "compatibility": {
            "allowed_config_differences": list(ALLOWED_CONFIG_DIFFERENCES),
            "observed_config_differences": config_differences,
            "normalized_config_sha256": source_compatibility_hash,
            "observed_data": data_contract,
            "scientific_inputs": scientific_inventory,
            "runtime": runtime,
            "combined_job_caveat": (
                f"Accepted job {accepted_job} executed both calibration protocols in one "
                "allocation; "
                "the vector benchmark used its declared 18-CPU/64-GiB envelope and the "
                "determinism protocol used no resource-sensitive selection."
            ),
        },
        "accepted_result": {
            "selected_num_envs": accepted_result.get("selected_num_envs"),
            "deterministic": accepted_result.get("deterministic"),
            "determinism_steps": stage_records["determinism"]["result"]["steps"],
            "vector_trials": stage_records["vector_benchmark"]["result"]["trial_count"],
            "feature_sha256": data_contract["sha256"],
            "source_slurm_job_id": accepted_job,
        },
    }
    references = {
        snapshot_path,
        snapshot_sidecar,
        source_manifest_path,
        source_preflight_path,
        source_config_path,
        target_config_path,
        *_referenced_paths(root, receipt),
        *target_references,
    }
    return receipt, sorted(references)


def create_calibration_import(
    definition: CampaignDefinition,
    source_snapshot_path: str | Path,
) -> dict[str, Any]:
    receipt, _ = _build_calibration_import(
        definition,
        Path(source_snapshot_path),
        created_at_utc=_utc_now(),
    )
    output = definition.campaign_dir / IMPORT_RELATIVE_PATH
    atomic_json(output, receipt)
    atomic_text(
        output.with_suffix(output.suffix + ".sha256"),
        sha256_file(output) + "\n",
    )
    return receipt


def validate_calibration_import(
    definition: CampaignDefinition,
    stage: CampaignStage,
    receipt_path: Path,
    options: dict[str, Any],
) -> tuple[list[str], list[Path]]:
    del options
    try:
        expected_path = (definition.campaign_dir / IMPORT_RELATIVE_PATH).resolve()
        path = receipt_path.resolve()
        if path != expected_path or not path.is_file():
            raise ValueError("calibration import must use the exact campaign import path")
        sidecar = _require_sidecar(path, "calibration import")
        receipt = _json_object(path, "calibration import")
        if stage.id not in STAGE_IDS:
            raise ValueError(f"calibration import cannot satisfy stage {stage.id}")
        if receipt.get("schema_version") != 1:
            raise ValueError("calibration import schema_version must equal 1")
        if receipt.get("evidence_kind") != "v1_calibration_evidence_import":
            raise ValueError("calibration import has an unsupported evidence kind")
        if receipt.get("status") != "accepted" or receipt.get("submitted_hpc_jobs") != 0:
            raise ValueError("calibration import must be accepted and record zero submitted jobs")
        source_snapshot = receipt.get("source_snapshot")
        if not isinstance(source_snapshot, dict):
            raise ValueError("calibration import source_snapshot must be an object")
        snapshot_path = _relative_path(
            definition.project_root,
            str(source_snapshot.get("path", "")),
            "calibration import source snapshot",
        )
        rebuilt, references = _build_calibration_import(
            definition,
            snapshot_path,
            created_at_utc=str(receipt.get("created_at_utc", "")),
        )
        if rebuilt != receipt:
            raise ValueError("calibration import content does not reproduce from current evidence")
        return [], sorted({path, sidecar, *references})
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        return [str(exc)], []
