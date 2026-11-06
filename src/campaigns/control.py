from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.campaigns.decisions import create_protocol_lock, materialize_decisions
from src.campaigns.gates import (
    atomic_json,
    evaluate_campaign,
    gate_report_path,
    latest_run_dir,
    protocol_lock_status,
    sha256_file,
    success_marker_path,
    verify_stage,
)
from src.campaigns.model import CampaignDefinition, CampaignStage, StageState
from src.campaigns.models import build_locked_stress_config
from src.campaigns.report import write_campaign_outputs
from src.campaigns.slurm import render_stage_command, validate_all_slurm_stages
from src.utils.config import load_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_definition(
    spec_path: Path,
    *,
    project_root: Path | None = None,
    campaign_dir: Path | None = None,
) -> tuple[CampaignDefinition, dict[str, Any]]:
    spec_path = spec_path.resolve()
    project_root = (project_root or spec_path.parent.parent).resolve()
    config = load_config(spec_path)
    raw_id = str(config.get("campaign", {}).get("id", "large_scale_thai_drl"))
    campaign_dir = (campaign_dir or project_root / "campaigns" / raw_id).resolve()
    definition = CampaignDefinition.from_mapping(
        config,
        project_root=project_root,
        campaign_dir=campaign_dir,
        spec_path=spec_path,
    )
    return definition, config


def _append_event(definition: CampaignDefinition, event: dict[str, Any]) -> None:
    path = definition.campaign_dir / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _utc_now(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def build_plan(definition: CampaignDefinition, config: dict[str, Any]) -> dict[str, Any]:
    resources = validate_all_slurm_stages(definition)
    statuses = evaluate_campaign(definition)
    definition.campaign_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(definition.campaign_dir / "resolved_campaign.json", config)
    plan = {
        "campaign_id": definition.campaign_id,
        "title": definition.title,
        "spec": str(definition.spec_path.relative_to(definition.project_root)),
        "workload": definition.workload_summary(),
        "stage_order": [stage.id for stage in definition.topological_stages()],
        "resources": resources,
        "safety": {
            "render_only": True,
            "max_rendered_submissions": definition.limits.max_rendered_submissions,
            "max_array_tasks": definition.limits.max_array_tasks,
            "max_concurrent_tasks": definition.limits.max_concurrent_tasks,
        },
    }
    atomic_json(definition.campaign_dir / "plan.json", plan)
    write_campaign_outputs(definition, statuses)
    _append_event(definition, {"event": "plan_built", "workload": plan["workload"]})
    return plan


def _write_status(definition: CampaignDefinition) -> list:
    statuses = evaluate_campaign(definition)
    write_campaign_outputs(definition, statuses)
    return statuses


def _required_project_file(
    definition: CampaignDefinition,
    raw_path: str,
    *,
    label: str,
) -> Path:
    relative = Path(definition.expand_project_relative(raw_path))
    path = (definition.project_root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.is_relative_to(definition.project_root)
        or not path.is_file()
    ):
        raise FileNotFoundError(f"{label} is missing or unsafe: {raw_path}")
    return path


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


_ACTIVE_ATTEMPT_STATUSES = frozenset({"submitted", "running"})
_INACTIVE_ATTEMPT_STATUSES = frozenset(
    {"rendered", "complete", "failed", "cancelled", "superseded"}
)


def _stale_slurm_rerender_context(
    definition: CampaignDefinition,
    stage: CampaignStage,
    status: Any,
    status_map: dict[str, Any],
) -> dict[str, Any]:
    """Authorize an explicit strict-SLURM rerender of stale verified evidence."""

    if stage.kind != "slurm":
        raise ValueError(
            f"stage {stage.id} is STALE and is not a SLURM stage eligible for rerender"
        )
    if not definition.strict_deployment_resolution:
        raise ValueError(
            f"stale SLURM stage {stage.id} requires strict deployment resolution "
            "before rerender"
        )

    predecessor_ids = list(stage.depends_on)
    if stage.run_source and stage.run_source not in predecessor_ids:
        predecessor_ids.append(stage.run_source)
    unverified = [
        predecessor_id
        for predecessor_id in predecessor_ids
        if (
            predecessor_id not in status_map
            or status_map[predecessor_id].state != StageState.VERIFIED
        )
    ]
    if unverified:
        raise ValueError(
            f"stale SLURM stage {stage.id} cannot rerender before all direct and "
            f"run-source predecessors are VERIFIED: {unverified}"
        )

    if status.state in {StageState.SUBMITTED, StageState.RUNNING}:
        raise ValueError(
            f"stale SLURM stage {stage.id} cannot rerender while it is "
            f"{status.state.value}"
        )
    if status.gate.manifest_running:
        raise ValueError(
            f"stale SLURM stage {stage.id} cannot rerender while "
            f"{status.gate.manifest_running} experiment manifest(s) are running"
        )

    latest_attempt_path = (
        definition.campaign_dir / "attempts" / f"{stage.id}_latest.json"
    )
    if latest_attempt_path.is_file():
        latest_attempt = _load_json_object(
            latest_attempt_path,
            label=f"latest attempt for stale stage {stage.id}",
        )
        if (
            latest_attempt.get("campaign_id") != definition.campaign_id
            or latest_attempt.get("stage_id") != stage.id
        ):
            raise ValueError(
                f"latest attempt for stale stage {stage.id} has inconsistent identity"
            )
        attempt_status = str(latest_attempt.get("status", "")).strip().lower()
        if attempt_status in _ACTIVE_ATTEMPT_STATUSES:
            raise ValueError(
                f"stale SLURM stage {stage.id} cannot rerender while its latest "
                f"attempt is {attempt_status}"
            )
        if attempt_status not in _INACTIVE_ATTEMPT_STATUSES:
            raise ValueError(
                f"stale SLURM stage {stage.id} latest attempt has unknown status "
                f"{attempt_status!r}; inactive execution cannot be established"
            )

    marker_path = success_marker_path(definition, stage.id).resolve()
    if (
        not marker_path.is_relative_to(definition.project_root)
        or not marker_path.is_file()
    ):
        raise FileNotFoundError(
            f"stale SLURM stage {stage.id} has no safe _SUCCESS marker to supersede"
        )
    marker = _load_json_object(
        marker_path,
        label=f"stale stage {stage.id} _SUCCESS marker",
    )
    fingerprint = str(marker.get("evidence_fingerprint", ""))
    if (
        marker.get("stage_id") != stage.id
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError(
            f"stale SLURM stage {stage.id} has an invalid _SUCCESS marker"
        )
    return {
        "stage_id": stage.id,
        "success_marker_path": str(marker_path.relative_to(definition.project_root)),
        "success_marker_sha256": sha256_file(marker_path),
        "verified_evidence_fingerprint": fingerprint,
        "current_evidence_fingerprint": status.gate.evidence_fingerprint,
    }


def _strict_predecessor_records(
    definition: CampaignDefinition,
    stage: CampaignStage,
    status_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Bind a rendered attempt to every direct or run-source success gate."""

    predecessor_ids = list(stage.depends_on)
    if stage.run_source and stage.run_source not in predecessor_ids:
        predecessor_ids.append(stage.run_source)
    records: list[dict[str, Any]] = []
    paths: list[Path] = []
    for predecessor_id in predecessor_ids:
        status = status_map.get(predecessor_id)
        if status is None or status.state != StageState.VERIFIED:
            observed = status.state.value if status is not None else "missing"
            raise ValueError(
                f"strict stage {stage.id} predecessor {predecessor_id} is not VERIFIED: "
                f"{observed}"
            )
        gate_path = gate_report_path(definition, predecessor_id).resolve()
        marker_path = success_marker_path(definition, predecessor_id).resolve()
        for path, label in (
            (gate_path, "predecessor gate report"),
            (marker_path, "predecessor _SUCCESS marker"),
        ):
            if not path.is_relative_to(definition.project_root) or not path.is_file():
                raise FileNotFoundError(
                    f"strict stage {stage.id} {label} is missing or unsafe: {path}"
                )
        gate = _load_json_object(gate_path, label="predecessor gate report")
        marker = _load_json_object(marker_path, label="predecessor _SUCCESS marker")
        fingerprint = str(gate.get("evidence_fingerprint", ""))
        if (
            gate.get("stage_id") != predecessor_id
            or gate.get("passed") is not True
            or marker.get("stage_id") != predecessor_id
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
            or marker.get("evidence_fingerprint") != fingerprint
        ):
            raise ValueError(
                f"strict stage {stage.id} predecessor {predecessor_id} has an "
                "inconsistent gate/_SUCCESS fingerprint"
            )
        records.append(
            {
                "stage_id": predecessor_id,
                "evidence_fingerprint": fingerprint,
                "gate_report_path": str(gate_path.relative_to(definition.project_root)),
                "gate_report_sha256": sha256_file(gate_path),
                "success_marker_path": str(marker_path.relative_to(definition.project_root)),
                "success_marker_sha256": sha256_file(marker_path),
            }
        )
        paths.extend((gate_path, marker_path))
    return records, paths


def _strict_deployment_context(
    definition: CampaignDefinition,
    stage: CampaignStage,
    *,
    config_path: Path,
) -> tuple[dict[str, Any], list[Path]]:
    """Resolve and validate a strict stage's exact release and preflight."""

    from src.campaigns.bistka_deployment import (
        validate_deployment_manifest,
        validate_preflight_receipt,
    )

    raw_manifest, raw_preflight = definition.resolve_deployment_pair(stage)
    manifest_path = _required_project_file(
        definition,
        raw_manifest,
        label=f"strict stage {stage.id} deployment manifest",
    )
    preflight_path = _required_project_file(
        definition,
        raw_preflight,
        label=f"strict stage {stage.id} preflight receipt",
    )
    manifest_issues, _ = validate_deployment_manifest(
        manifest_path,
        project_root=definition.project_root,
    )
    if manifest_issues:
        raise ValueError(
            f"strict stage {stage.id} deployment manifest is invalid: "
            + "; ".join(manifest_issues)
        )
    preflight_issues, _ = validate_preflight_receipt(
        preflight_path,
        expected_manifest_path=manifest_path,
    )
    if preflight_issues:
        raise ValueError(
            f"strict stage {stage.id} preflight does not validate against its manifest: "
            + "; ".join(preflight_issues)
        )
    manifest = _load_json_object(manifest_path, label="deployment manifest")
    release_id = str(manifest.get("release_id", ""))
    if len(release_id) != 64 or any(
        character not in "0123456789abcdef" for character in release_id
    ):
        raise ValueError(f"strict stage {stage.id} deployment release id is invalid")

    file_records = {
        str(record.get("path", "")): record
        for record in manifest.get("files", [])
        if isinstance(record, dict)
    }

    def require_manifest_record(path: Path, *, label: str) -> str:
        relative = str(path.relative_to(definition.project_root))
        record = file_records.get(relative)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != sha256_file(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise ValueError(
                f"strict stage {stage.id} {label} is not bound to the exact "
                f"deployment: {relative}"
            )
        return relative

    config_relative = str(config_path.relative_to(definition.project_root))
    require_manifest_record(config_path, label="config_path")
    primary_config = load_config(config_path)
    primary_data = primary_config.get("data", {})
    primary_source = (
        str(primary_data.get("source", "")).strip().lower()
        if isinstance(primary_data, dict)
        else ""
    )
    if primary_source in {"synthetic", "generated"}:
        raise ValueError(
            f"strict stage {stage.id} config_path uses prohibited {primary_source} data"
        )

    bindings = manifest.get("runtime_feature_bindings", [])
    if not isinstance(bindings, list):
        raise ValueError(
            f"strict stage {stage.id} deployment runtime_feature_bindings is not a list"
        )

    runtime_records: list[dict[str, Any]] = []
    runtime_paths: list[Path] = []
    observed_binding_count = 0
    for raw_runtime_config in definition.resolve_runtime_config_paths(stage):
        runtime_config_path = _required_project_file(
            definition,
            raw_runtime_config,
            label=f"strict stage {stage.id} runtime config",
        )
        runtime_relative = require_manifest_record(
            runtime_config_path,
            label="runtime config",
        )
        runtime_config = load_config(runtime_config_path)
        data = runtime_config.get("data", {})
        paths = runtime_config.get("paths", {})
        source = (
            str(data.get("source", "")).strip().lower()
            if isinstance(data, dict)
            else ""
        )
        declares_features = isinstance(paths, dict) and bool(paths.get("features"))
        if source in {"synthetic", "generated"}:
            raise ValueError(
                f"strict stage {stage.id} runtime config {runtime_relative} uses "
                f"prohibited {source} data"
            )
        if source and not declares_features:
            raise ValueError(
                f"strict stage {stage.id} observed runtime config {runtime_relative} "
                "does not declare paths.features"
            )
        if declares_features and not source:
            raise ValueError(
                f"strict stage {stage.id} runtime config {runtime_relative} declares "
                "paths.features without an observed data.source"
            )

        runtime_config_sha256 = sha256_file(runtime_config_path)
        feature_relative = None
        feature_sha256 = None
        if source:
            from src.campaigns.calibration import load_observed_inputs

            runtime_feature_path = load_observed_inputs(runtime_config_path).feature_path
            if (
                runtime_feature_path is None
                or not runtime_feature_path.is_relative_to(definition.project_root)
            ):
                raise FileNotFoundError(
                    f"strict stage {stage.id} runtime feature table is outside the project: "
                    f"{runtime_feature_path}"
                )
            feature_relative = require_manifest_record(
                runtime_feature_path,
                label="runtime feature table",
            )
            feature_sha256 = sha256_file(runtime_feature_path)
            config_binding = next(
                (
                    binding
                    for binding in bindings
                    if isinstance(binding, dict)
                    and binding.get("config_path") == runtime_relative
                ),
                None,
            )
            if (
                not isinstance(config_binding, dict)
                or config_binding.get("config_sha256") != runtime_config_sha256
                or config_binding.get("feature_path") != feature_relative
                or config_binding.get("feature_sha256") != feature_sha256
                or config_binding.get("feature_bytes")
                != runtime_feature_path.stat().st_size
            ):
                raise ValueError(
                    f"strict stage {stage.id} runtime feature binding does not match "
                    f"the actual observed config/table: {runtime_relative}"
                )
            runtime_paths.append(runtime_feature_path)
            observed_binding_count += 1
        runtime_records.append(
            {
                "path": runtime_relative,
                "sha256": runtime_config_sha256,
                "feature_path": feature_relative,
                "feature_sha256": feature_sha256,
            }
        )
        runtime_paths.append(runtime_config_path)

    if not stage.run_source and observed_binding_count == 0:
        raise ValueError(
            f"strict compute stage {stage.id} has zero observed runtime feature bindings"
        )

    if len(runtime_records) == 1:
        scalar_config_path = runtime_records[0]["path"]
        scalar_config_sha256 = runtime_records[0]["sha256"]
        scalar_feature_path = runtime_records[0]["feature_path"]
        scalar_feature_sha256 = runtime_records[0]["feature_sha256"]
    else:
        scalar_config_path = config_relative
        scalar_config_sha256 = sha256_file(config_path)
        scalar_feature_path = None
        scalar_feature_sha256 = None

    context = {
        "deployment_manifest_path": str(manifest_path.relative_to(definition.project_root)),
        "deployment_manifest_sha256": sha256_file(manifest_path),
        "deployment_release_id": release_id,
        "preflight_receipt_path": str(preflight_path.relative_to(definition.project_root)),
        "preflight_sha256": sha256_file(preflight_path),
        "runtime_configs": runtime_records,
        "runtime_config_path": scalar_config_path,
        "runtime_config_sha256": scalar_config_sha256,
        "runtime_feature_path": scalar_feature_path,
        "runtime_feature_sha256": scalar_feature_sha256,
    }
    return context, [manifest_path, preflight_path, *runtime_paths]


def _render_command(definition: CampaignDefinition, requested_stage: str | None) -> tuple[str, str, Path]:
    statuses = _write_status(definition)
    status_map = {status.stage_id: status for status in statuses}
    if requested_stage:
        if requested_stage not in status_map:
            raise KeyError(f"unknown campaign stage: {requested_stage}")
        selected = status_map[requested_stage]
    else:
        selected = next((status for status in statuses if status.state == StageState.READY), None)
        if selected is None:
            raise ValueError("no campaign stage is ready for command rendering")
    stage = definition.stage_map[selected.stage_id]
    superseded_stale_evidence: dict[str, Any] | None = None
    if selected.state == StageState.STALE and requested_stage:
        superseded_stale_evidence = _stale_slurm_rerender_context(
            definition,
            stage,
            selected,
            status_map,
        )
    elif selected.state != StageState.READY:
        raise ValueError(
            f"stage {selected.stage_id} is {selected.state.value}, not READY: "
            f"{selected.reason}"
        )
    config_path: Path | None = None
    runtime_feature_path: Path | None = None
    if stage.config_path:
        config_path = _required_project_file(
            definition,
            stage.config_path,
            label=f"stage {stage.id} configuration",
        )
        resolved_stage_config = load_config(config_path)
        source = str(resolved_stage_config.get("data", {}).get("source", "")).strip().lower()
        paths = resolved_stage_config.get("paths", {})
        if (
            source not in {"", "synthetic", "generated"}
            and isinstance(paths, dict)
            and paths.get("features")
        ):
            from src.campaigns.calibration import load_observed_inputs

            runtime_feature_path = load_observed_inputs(config_path).feature_path
    run_dir = latest_run_dir(definition, stage) if stage.run_source else None
    lock_valid, lock_hash, lock_issues = protocol_lock_status(definition)
    if stage.require_protocol_hash and not lock_valid:
        raise ValueError(f"protocol lock is not valid: {lock_issues}")
    extra_exports: dict[str, str] = {}
    if stage.kind == "slurm":
        attempt_id = (
            f"{stage.id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
            f"{secrets.token_hex(4)}"
        )
        input_paths = [definition.spec_path]
        for raw_path in (stage.job_script, stage.config_path, *stage.fingerprint_paths):
            if raw_path:
                path = (
                    definition.project_root / definition.expand_project_relative(str(raw_path))
                ).resolve()
                if not path.is_relative_to(definition.project_root) or not path.is_file():
                    raise FileNotFoundError(f"attempt input is missing or unsafe: {raw_path}")
                input_paths.append(path)
        if runtime_feature_path is not None:
            if not runtime_feature_path.is_relative_to(definition.project_root):
                raise FileNotFoundError(
                    f"runtime feature table is outside the project: {runtime_feature_path}"
                )
            input_paths.append(runtime_feature_path)
        deployment_context: dict[str, Any]
        predecessor_gates: list[dict[str, Any]] = []
        if definition.strict_deployment_resolution:
            if config_path is None:
                raise ValueError(f"strict SLURM stage {stage.id} has no config_path")
            deployment_context, deployment_paths = _strict_deployment_context(
                definition,
                stage,
                config_path=config_path,
            )
            predecessor_gates, predecessor_paths = _strict_predecessor_records(
                definition,
                stage,
                status_map,
            )
            input_paths.extend(deployment_paths)
            input_paths.extend(predecessor_paths)
        else:
            deployment_release_id = None
            deployment_manifest = (
                definition.project_root
                / definition.expand_project_relative(stage.deployment_manifest)
            ).resolve() if stage.deployment_manifest else (
                definition.campaign_dir / "deployment" / "bistka_deployment_manifest.json"
            )
            deployment_manifest_sha256 = None
            if deployment_manifest.is_file():
                input_paths.append(deployment_manifest)
                deployment_manifest_sha256 = sha256_file(deployment_manifest)
                deployment_value = json.loads(deployment_manifest.read_text(encoding="utf-8"))
                if isinstance(deployment_value, dict):
                    deployment_release_id = str(deployment_value.get("release_id", "")) or None
            preflight_path = (
                definition.project_root
                / definition.expand_project_relative(stage.preflight_receipt)
            ).resolve() if stage.preflight_receipt else (
                definition.project_root / "reports" / "bistka_calibration_preflight.json"
            )
            preflight_sha256 = None
            if preflight_path.is_file():
                input_paths.append(preflight_path)
                preflight_sha256 = sha256_file(preflight_path)
            deployment_context = {
                "deployment_manifest_path": (
                    str(deployment_manifest.relative_to(definition.project_root))
                    if deployment_manifest.is_file()
                    else None
                ),
                "deployment_manifest_sha256": deployment_manifest_sha256,
                "deployment_release_id": deployment_release_id,
                "preflight_receipt_path": (
                    str(preflight_path.relative_to(definition.project_root))
                    if preflight_path.is_file()
                    and preflight_path.is_relative_to(definition.project_root)
                    else None
                ),
                "preflight_sha256": preflight_sha256,
            }
        input_records = [
            {
                "path": str(path.relative_to(definition.project_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(set(input_paths))
        ]
        attempt = {
            "schema_version": 2 if definition.strict_deployment_resolution else 1,
            "status": "rendered",
            "campaign_id": definition.campaign_id,
            "stage_id": stage.id,
            "attempt_id": attempt_id,
            "rendered_at_utc": _utc_now(),
            "inputs": input_records,
            **deployment_context,
            "predecessor_gates": predecessor_gates,
        }
        if superseded_stale_evidence is not None:
            attempt["supersedes_stale_evidence"] = superseded_stale_evidence
        attempt["attempt_sha256"] = hashlib.sha256(
            json.dumps(attempt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        attempt_dir = definition.campaign_dir / "attempts"
        atomic_json(attempt_dir / f"{attempt_id}.json", attempt)
        atomic_json(attempt_dir / f"{stage.id}_latest.json", attempt)
        extra_exports["CAMPAIGN_ATTEMPT_ID"] = attempt_id
        extra_exports["CAMPAIGN_ATTEMPT_SHA256"] = attempt["attempt_sha256"]
        extra_exports["CAMPAIGN_STAGE_ID"] = stage.id
        deployment_release_id = deployment_context.get("deployment_release_id")
        preflight_sha256 = deployment_context.get("preflight_sha256")
        if deployment_release_id:
            extra_exports["DEPLOYMENT_RELEASE_ID"] = deployment_release_id
        if preflight_sha256:
            extra_exports["BISTKA_PREFLIGHT_SHA256"] = preflight_sha256
    command = render_stage_command(
        definition,
        stage,
        run_dir=run_dir,
        protocol_hash=lock_hash if stage.require_protocol_hash else None,
        extra_exports=extra_exports,
    )
    if command.count("sbatch ") > definition.limits.max_rendered_submissions:
        raise ValueError("rendered command exceeds the per-step submission safety limit")
    output = definition.campaign_dir / "commands" / f"{stage.id}.sh"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("#!/bin/bash\nset -euo pipefail\n\n" + command + "\n", encoding="utf-8")
    _append_event(definition, {"event": "command_rendered", "stage": stage.id, "path": str(output)})
    return stage.id, command, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render-only, dependency-gated research campaign control plane")
    parser.add_argument(
        "command",
        choices=[
            "plan",
            "status",
            "verify",
            "commands",
            "materialize",
            "migrate-calibration",
            "lock",
            "report",
        ],
    )
    parser.add_argument("--spec", default="config/research_campaign.yaml")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--campaign-dir", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument(
        "--target", choices=["budget", "nested", "robustness", "ablation", "stress"], default=None
    )
    parser.add_argument("--source-snapshot", default=None)
    args = parser.parse_args(argv)
    try:
        definition, config = load_definition(
            Path(args.spec),
            project_root=Path(args.project_root) if args.project_root else None,
            campaign_dir=Path(args.campaign_dir) if args.campaign_dir else None,
        )
        if args.command == "plan":
            result = build_plan(definition, config)
        elif args.command in {"status", "report"}:
            statuses = _write_status(definition)
            result = {
                "campaign_id": definition.campaign_id,
                "report": str(definition.campaign_dir / "reports" / "campaign_status.md"),
                "states": {status.stage_id: status.state.value for status in statuses},
            }
        elif args.command == "verify":
            if not args.stage:
                raise ValueError("verify requires --stage")
            gate = verify_stage(definition, args.stage)
            _append_event(definition, {"event": "stage_verified", "stage": args.stage})
            _write_status(definition)
            result = gate.to_dict()
        elif args.command == "commands":
            stage_id, command, output = _render_command(definition, args.stage)
            result = {"stage": stage_id, "render_only": True, "command": command, "script": str(output)}
        elif args.command == "materialize":
            if not args.target:
                raise ValueError("materialize requires --target")
            result = (
                build_locked_stress_config(definition)
                if args.target == "stress"
                else materialize_decisions(definition, args.target)
            )
            _append_event(definition, {"event": "decisions_materialized", "target": args.target})
            _write_status(definition)
        elif args.command == "migrate-calibration":
            if not args.source_snapshot:
                raise ValueError("migrate-calibration requires --source-snapshot")
            from src.campaigns.evidence_migration import create_calibration_import

            result = create_calibration_import(
                definition,
                Path(args.source_snapshot),
            )
            _append_event(
                definition,
                {
                    "event": "calibration_evidence_migrated",
                    "source_snapshot": args.source_snapshot,
                    "migration_receipt": str(
                        definition.campaign_dir
                        / "imports"
                        / "v1_calibration_evidence.json"
                    ),
                },
            )
            _write_status(definition)
        else:
            result = create_protocol_lock(definition)
            _append_event(definition, {"event": "protocol_locked", "protocol_sha256": result["protocol_sha256"]})
            _write_status(definition)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
