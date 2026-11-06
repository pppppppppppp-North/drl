from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from src.campaigns.bistka_deployment import (
    _runtime_feature_bindings,
    _safe_files,
    _safe_project_file,
    sha256_file,
    validate_deployment_manifest,
)
from src.utils.config import load_config


_PROFILE_KEYS = {
    "schema_version",
    "name",
    "campaign_dir",
    "base",
    "deployment_overrides",
    "include",
    "runtime_configs",
    "resolve",
    "feature_expectations",
    "require_feature_expectations",
}
_INCLUDE_KEYS = {
    "derived_configs",
    "decisions",
    "protocols",
    "registries",
    "files",
}
_RESOLVE_KEYS = {"feature_ablation_configs", "stress_registries"}
_DEPLOYMENT_FIELDS = {
    "campaign_id",
    "target_project_root",
    "python_executable",
    "required_packages",
    "required_hostname_prefix",
    "data_readiness_path",
}
_GLOB_MAGIC = re.compile(r"[*?[]")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported keys: {unknown}")


def _project_relative_directory(project_root: Path, raw_path: str | Path) -> str:
    root = project_root.resolve()
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"campaign directory is unsafe: {raw_path}")
    return str(resolved.relative_to(root))


def _expand_path(raw_path: Any, *, campaign_dir: str | None, label: str) -> str:
    if not isinstance(raw_path, (str, Path)) or not str(raw_path):
        raise ValueError(f"{label} must be a non-empty path")
    value = str(raw_path)
    if "{campaign_dir}" in value:
        if not campaign_dir:
            raise ValueError(
                f"{label} uses {{campaign_dir}} but no campaign directory was provided"
            )
        value = value.replace("{campaign_dir}", campaign_dir)
    if "{" in value or "}" in value:
        raise ValueError(f"{label} contains an unsupported placeholder: {value}")
    return value


def _exact_path(
    project_root: Path,
    raw_path: Any,
    *,
    campaign_dir: str | None,
    label: str,
) -> tuple[str, Path]:
    expanded = _expand_path(raw_path, campaign_dir=campaign_dir, label=label)
    if _GLOB_MAGIC.search(expanded):
        raise ValueError(f"{label} must be exact; broad glob is not allowed: {expanded}")
    candidate = Path(expanded)
    if candidate.is_absolute():
        root = project_root.resolve()
        resolved_candidate = candidate.resolve()
        if not resolved_candidate.is_relative_to(root):
            raise FileNotFoundError(f"{label} is missing or unsafe: {expanded}")
        expanded = str(resolved_candidate.relative_to(root))
    relative, resolved = _safe_project_file(project_root, expanded, label=label)
    return str(relative), resolved


def _exact_paths(
    project_root: Path,
    values: Any,
    *,
    campaign_dir: str | None,
    label: str,
) -> list[tuple[str, Path]]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    return [
        _exact_path(
            project_root,
            value,
            campaign_dir=campaign_dir,
            label=f"{label}[{index}]",
        )
        for index, value in enumerate(values)
    ]


def _safe_output(project_root: Path, output_path: Path) -> tuple[str, Path]:
    root = project_root.resolve()
    resolved = output_path.resolve() if output_path.is_absolute() else (root / output_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"generated deployment config must stay inside the project: {output_path}")
    return str(resolved.relative_to(root)), resolved


def _config_closure(project_root: Path, config_path: Path) -> set[Path]:
    """Return exact YAML inheritance inputs required to load ``config_path`` remotely."""

    root = project_root.resolve()
    closure: set[Path] = set()
    stack: list[Path] = []

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise FileNotFoundError(f"referenced configuration is missing or unsafe: {path}")
        if resolved in stack:
            cycle = " -> ".join(str(item.relative_to(root)) for item in (*stack, resolved))
            raise ValueError(f"configuration inheritance cycle: {cycle}")
        if resolved in closure:
            return
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"configuration root must be a mapping: {resolved.relative_to(root)}")
        parents = loaded.get("extends", [])
        if isinstance(parents, (str, Path)):
            parents = [parents]
        if not isinstance(parents, list):
            raise ValueError(f"configuration extends must be a path or list: {resolved.relative_to(root)}")
        stack.append(resolved)
        for index, parent in enumerate(parents):
            expanded = str(parent)
            if _GLOB_MAGIC.search(expanded) or "{" in expanded or "}" in expanded:
                raise ValueError(
                    "configuration inheritance must use an exact path: "
                    f"{resolved.relative_to(root)} extends[{index}]={expanded!r}"
                )
            candidate = Path(expanded)
            if not candidate.is_absolute():
                candidate = resolved.parent / candidate
            visit(candidate)
        stack.pop()
        closure.add(resolved)

    visit(config_path)
    return closure


def _deployment_metadata(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    missing = sorted({"campaign_id", "target_project_root", "data_readiness_path"} - set(value))
    if missing:
        raise ValueError(f"{label} lacks required deployment metadata: {missing}")
    packages = value.get("required_packages", [])
    if not isinstance(packages, list):
        raise ValueError(f"{label} required_packages must be a list")
    return {
        "campaign_id": str(value["campaign_id"]),
        "target_project_root": str(value["target_project_root"]),
        "python_executable": str(
            value.get(
                "python_executable",
                "/lustrefs/project/25sfcs03/envs/drl_env/bin/python",
            )
        ),
        "required_hostname_prefix": str(value.get("required_hostname_prefix", "bistka-")),
        "data_readiness_path": str(value["data_readiness_path"]),
        "required_packages": [str(item) for item in packages],
    }


def _base_from_manifest(
    project_root: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], set[Path], set[str]]:
    issues, _ = validate_deployment_manifest(manifest_path, project_root=project_root)
    if issues:
        raise ValueError(f"base deployment manifest is invalid: {issues}")
    manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        label="base deployment manifest",
    )
    files: set[Path] = {manifest_path.resolve(), manifest_path.with_suffix(".sha256").resolve()}
    for record in manifest.get("files", []):
        if not isinstance(record, dict) or not record.get("path"):
            raise ValueError("base deployment manifest contains a malformed file record")
        _, path = _safe_project_file(
            project_root,
            str(record["path"]),
            label="base deployment file",
        )
        if sha256_file(path) != record.get("sha256") or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"base deployment file changed: {record['path']}")
        files.add(path)
    runtime_configs = {
        str(binding["config_path"])
        for binding in manifest.get("runtime_feature_bindings", [])
        if isinstance(binding, dict) and binding.get("config_path")
    }
    return _deployment_metadata(manifest, label="base manifest"), files, runtime_configs


def _base_from_config(
    project_root: Path,
    config_path: Path,
    *,
    campaign_dir: str | None,
) -> tuple[dict[str, Any], set[Path], set[str]]:
    loaded = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")) or {},
        label="base deployment config",
    )
    deployment = _mapping(loaded.get("deployment"), label="base deployment config.deployment")
    raw_inputs = deployment.get("inputs", [])
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError("base deployment config inputs must be a non-empty list")
    expanded_patterns = [
        _expand_path(
            item,
            campaign_dir=campaign_dir,
            label=f"base deployment input[{index}]",
        )
        for index, item in enumerate(raw_inputs)
    ]
    # A base deployment config is the one trusted place where curated patterns
    # may appear. They are expanded immediately; the generated cumulative config
    # contains only the resulting exact paths.
    files = set(_safe_files(project_root, expanded_patterns))
    files.add(config_path.resolve())
    raw_runtime = deployment.get("runtime_configs", [])
    if not isinstance(raw_runtime, list):
        raise ValueError("base deployment runtime_configs must be a list")
    runtime_configs = {
        _expand_path(
            value,
            campaign_dir=campaign_dir,
            label=f"base runtime_configs[{index}]",
        )
        for index, value in enumerate(raw_runtime)
    }
    return _deployment_metadata(deployment, label="base deployment config"), files, runtime_configs


def _readiness_artifacts(project_root: Path, raw_path: str) -> dict[str, dict[str, Any]]:
    _, readiness_path = _safe_project_file(
        project_root,
        raw_path,
        label="deployment data-readiness receipt",
    )
    try:
        value = json.loads(readiness_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"data-readiness receipt is not valid JSON: {readiness_path}") from exc
    if (
        not isinstance(value, dict)
        or value.get("status") != "pass"
        or value.get("data_origin") != "observed_market"
        or value.get("execution_eligible") is not True
    ):
        raise ValueError(
            "data-readiness receipt is not a passing execution-eligible observed-market receipt: "
            f"{readiness_path}"
        )
    artifacts: dict[str, dict[str, Any]] = {}
    for record in value.get("artifacts", {}).values():
        if isinstance(record, dict) and record.get("path") and record.get("sha256"):
            artifacts[str(record["path"])] = record
    return artifacts


def _validate_runtime_feature_policy(
    project_root: Path,
    config_relative: str,
    resolved_config: Mapping[str, Any],
    readiness_artifacts: Mapping[str, Mapping[str, Any]],
) -> str:
    paths = resolved_config.get("paths", {})
    if not isinstance(paths, dict) or not paths.get("features"):
        raise ValueError(f"runtime configuration lacks paths.features: {config_relative}")
    primary = str(paths["features"])
    fallback_raw = paths.get("fallback_features_csv")
    if fallback_raw:
        fallback = str(fallback_raw)
        if fallback != primary:
            fallback_path = (project_root / fallback).resolve()
            if fallback in readiness_artifacts or fallback_path.is_file():
                raise ValueError(
                    "runtime configuration has a shadowable or stale preferred feature path; "
                    f"normalize paths.features to the certified exact file: {config_relative} "
                    f"({primary} != {fallback})"
                )
    return primary


def _verify_sidecar_if_present(path: Path) -> Path | None:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        return None
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise ValueError(f"SHA-256 sidecar does not match: {sidecar}")
    return sidecar.resolve()


def _registry_path_records(value: Any) -> Iterable[tuple[str, str, int | None]]:
    if isinstance(value, list):
        for item in value:
            yield from _registry_path_records(item)
        return
    if not isinstance(value, dict):
        return
    for key, raw_path in value.items():
        if key == "path":
            hash_key = "sha256"
            bytes_key = "bytes"
        elif key.endswith("_path"):
            prefix = key.removesuffix("_path")
            hash_key = f"{prefix}_sha256"
            bytes_key = f"{prefix}_bytes"
        else:
            continue
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"stress registry contains an invalid {key}")
        expected_hash = value.get(hash_key)
        if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
            raise ValueError(f"stress registry reference is unbound: {key} lacks {hash_key}")
        expected_bytes = value.get(bytes_key)
        if expected_bytes is not None and (
            isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0
        ):
            raise ValueError(f"stress registry contains invalid byte count: {bytes_key}")
        yield raw_path, expected_hash.lower(), expected_bytes
    for item in value.values():
        yield from _registry_path_records(item)


def _resolve_stress_registry(
    project_root: Path,
    registry_relative: str,
    registry_path: Path,
) -> tuple[set[Path], set[tuple[str, str, str, str]]]:
    try:
        registry = _mapping(
            json.loads(registry_path.read_text(encoding="utf-8")),
            label=f"stress registry {registry_relative}",
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"stress registry is not valid JSON: {registry_relative}") from exc
    models = registry.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"stress registry contains no models: {registry_relative}")
    files: set[Path] = {registry_path.resolve()}
    sidecar = _verify_sidecar_if_present(registry_path)
    if sidecar:
        files.add(sidecar)
    for raw_path, expected_hash, expected_bytes in _registry_path_records(registry):
        relative, path = _safe_project_file(
            project_root,
            raw_path,
            label=f"stress registry referenced file in {registry_relative}",
        )
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"stress registry SHA-256 mismatch: {relative}; "
                f"expected {expected_hash}, got {actual_hash}"
            )
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise ValueError(f"stress registry byte count mismatch: {relative}")
        files.add(path)
    model_records: set[tuple[str, str, str, str]] = set()
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"stress registry model[{index}] is malformed")
        required = ("path", "sha256", "source_test_curve_path", "source_test_curve_sha256")
        missing = [key for key in required if not model.get(key)]
        if missing:
            raise ValueError(f"stress registry model[{index}] has unbound references: {missing}")
        record = (
            str(model["path"]),
            str(model["sha256"]).lower(),
            str(model["source_test_curve_path"]),
            str(model["source_test_curve_sha256"]).lower(),
        )
        if record in model_records:
            raise ValueError(f"stress registry contains a duplicate model reference: {record[0]}")
        model_records.add(record)
    return files, model_records


def _stress_config_registry(
    project_root: Path,
    config_relative: str,
    resolved_config: Mapping[str, Any],
) -> tuple[str, set[tuple[str, str, str, str]]] | None:
    stress = resolved_config.get("execution_stress")
    if not isinstance(stress, dict) or not stress.get("model_registry"):
        return None
    if stress.get("require_models") is not True:
        raise ValueError(
            f"stress configuration names a registry but does not require models: {config_relative}"
        )
    registry_relative, _ = _safe_project_file(
        project_root,
        str(stress["model_registry"]),
        label=f"stress registry referenced by {config_relative}",
    )
    raw_models = stress.get("model_paths")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError(f"stress configuration contains no model paths: {config_relative}")
    records: set[tuple[str, str, str, str]] = set()
    for index, model in enumerate(raw_models):
        if not isinstance(model, dict):
            raise ValueError(f"stress configuration model_paths[{index}] is malformed")
        required = ("path", "sha256", "source_test_curve_path", "source_test_curve_sha256")
        missing = [key for key in required if not model.get(key)]
        if missing:
            raise ValueError(
                f"stress configuration model_paths[{index}] has unbound references: {missing}"
            )
        records.add(
            (
                str(model["path"]),
                str(model["sha256"]).lower(),
                str(model["source_test_curve_path"]),
                str(model["source_test_curve_sha256"]).lower(),
            )
        )
    return str(registry_relative), records


def _normalize_feature_expectations(
    project_root: Path,
    raw_expectations: Any,
    *,
    campaign_dir: str | None,
    readiness_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if raw_expectations is None:
        return {}
    expectations = _mapping(raw_expectations, label="feature_expectations")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_config, raw_record in expectations.items():
        config_relative, _ = _exact_path(
            project_root,
            raw_config,
            campaign_dir=campaign_dir,
            label="feature expectation runtime config",
        )
        record = _mapping(raw_record, label=f"feature expectation for {config_relative}")
        _reject_unknown(
            record,
            {"path", "sha256", "bytes", "require_readiness"},
            label="feature expectation",
        )
        feature_relative, feature_path = _exact_path(
            project_root,
            record.get("path"),
            campaign_dir=campaign_dir,
            label=f"expected feature for {config_relative}",
        )
        require_readiness = record.get("require_readiness", False)
        if not isinstance(require_readiness, bool):
            raise ValueError(
                f"feature expectation require_readiness must be boolean: {config_relative}"
            )
        certified = readiness_artifacts.get(feature_relative) if require_readiness else None
        if require_readiness and not isinstance(certified, Mapping):
            raise ValueError(
                "runtime feature is not certified by the data-readiness receipt: "
                f"{config_relative} -> {feature_relative}"
            )
        certified_hash = str(certified.get("sha256", "")).lower() if certified else ""
        certified_bytes = certified.get("bytes") if certified else None
        if certified and (
            not _SHA256.fullmatch(certified_hash)
            or isinstance(certified_bytes, bool)
            or not isinstance(certified_bytes, int)
            or certified_bytes < 0
        ):
            raise ValueError(
                f"data-readiness feature record is malformed: {feature_relative}"
            )

        declared_hash = str(record.get("sha256", "")).lower()
        expected_hash = certified_hash if require_readiness else declared_hash
        if not _SHA256.fullmatch(expected_hash):
            raise ValueError(f"feature expectation needs a valid SHA-256: {config_relative}")
        if declared_hash and declared_hash != expected_hash:
            raise ValueError(
                f"feature expectation differs from data-readiness SHA-256: {config_relative}"
            )
        declared_bytes = record.get("bytes")
        expected_bytes = certified_bytes if require_readiness else declared_bytes
        if declared_bytes is not None and declared_bytes != expected_bytes:
            raise ValueError(
                f"feature expectation differs from data-readiness byte count: {config_relative}"
            )
        if expected_bytes is not None and expected_bytes != feature_path.stat().st_size:
            raise ValueError(f"feature expectation byte count mismatch: {feature_relative}")
        if sha256_file(feature_path) != expected_hash:
            raise ValueError(f"feature expectation SHA-256 mismatch: {feature_relative}")
        normalized[config_relative] = {
            "path": feature_relative,
            "sha256": expected_hash,
            **({"bytes": int(expected_bytes)} if expected_bytes is not None else {}),
            **({"readiness_certified": True} if require_readiness else {}),
        }
    return normalized


def build_cumulative_deployment_config(
    *,
    project_root: Path,
    profile_path: Path,
    output_path: Path,
    campaign_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve a fail-closed cumulative profile into an exact deployment config.

    The returned YAML is directly consumable by
    :func:`src.campaigns.bistka_deployment.build_deployment`. No timestamps or
    filesystem-order-dependent values are written, so identical inputs produce
    byte-identical output.
    """

    root = project_root.resolve()
    profile_relative, profile_file = _exact_path(
        root,
        profile_path,
        campaign_dir=None,
        label="campaign release profile",
    )
    loaded = _mapping(
        yaml.safe_load(profile_file.read_text(encoding="utf-8")) or {},
        label="profile document",
    )
    profile = _mapping(
        loaded.get("campaign_release_profile"),
        label="campaign_release_profile",
    )
    _reject_unknown(profile, _PROFILE_KEYS, label="campaign_release_profile")
    if profile.get("schema_version") != 1:
        raise ValueError("campaign release profile schema_version must be 1")
    name = str(profile.get("name", "")).strip()
    if not name:
        raise ValueError("campaign release profile needs a name")

    selected_campaign_dir = campaign_dir if campaign_dir is not None else profile.get("campaign_dir")
    campaign_relative = (
        _project_relative_directory(root, selected_campaign_dir)
        if selected_campaign_dir is not None
        else None
    )
    output_relative, output_file = _safe_output(root, output_path)
    if output_file == profile_file:
        raise ValueError("generated deployment config cannot overwrite its source profile")

    base = _mapping(profile.get("base"), label="campaign_release_profile.base")
    _reject_unknown(base, {"manifest", "deployment_config"}, label="profile base")
    if ("manifest" in base) == ("deployment_config" in base):
        raise ValueError("profile base must name exactly one manifest or deployment_config")
    if "manifest" in base:
        base_relative, base_path = _exact_path(
            root,
            base["manifest"],
            campaign_dir=campaign_relative,
            label="base deployment manifest",
        )
        metadata, files, runtime_configs = _base_from_manifest(root, base_path)
        base_record = {"kind": "manifest", "path": base_relative, "sha256": sha256_file(base_path)}
    else:
        base_relative, base_path = _exact_path(
            root,
            base["deployment_config"],
            campaign_dir=campaign_relative,
            label="base deployment config",
        )
        metadata, files, runtime_configs = _base_from_config(
            root,
            base_path,
            campaign_dir=campaign_relative,
        )
        base_record = {
            "kind": "deployment_config",
            "path": base_relative,
            "sha256": sha256_file(base_path),
        }
    base_runtime_configs = set(runtime_configs)

    overrides = _mapping(
        profile.get("deployment_overrides", {}),
        label="deployment_overrides",
    )
    _reject_unknown(overrides, _DEPLOYMENT_FIELDS, label="deployment_overrides")
    metadata.update(overrides)
    metadata = _deployment_metadata(metadata, label="resolved deployment metadata")

    include = _mapping(profile.get("include", {}), label="profile include")
    _reject_unknown(include, _INCLUDE_KEYS, label="profile include")
    files.update({profile_file, output_file})
    for category in sorted(_INCLUDE_KEYS):
        files.update(
            path
            for _, path in _exact_paths(
                root,
                include.get(category, []),
                campaign_dir=campaign_relative,
                label=f"include.{category}",
            )
        )

    runtime_configs.update(
        relative
        for relative, _ in _exact_paths(
            root,
            profile.get("runtime_configs", []),
            campaign_dir=campaign_relative,
            label="runtime_configs",
        )
    )

    resolve = _mapping(profile.get("resolve", {}), label="profile resolve")
    _reject_unknown(resolve, _RESOLVE_KEYS, label="profile resolve")
    for orchestrator_relative, orchestrator_path in _exact_paths(
        root,
        resolve.get("feature_ablation_configs", []),
        campaign_dir=campaign_relative,
        label="resolve.feature_ablation_configs",
    ):
        files.update(_config_closure(root, orchestrator_path))
        orchestrator = load_config(orchestrator_path)
        groups = orchestrator.get("feature_ablation", {}).get("groups", [])
        if not isinstance(groups, list) or not groups:
            raise ValueError(
                f"feature-ablation configuration contains no groups: {orchestrator_relative}"
            )
        names: set[str] = set()
        for index, group in enumerate(groups):
            if not isinstance(group, dict) or not group.get("name") or not group.get("config"):
                raise ValueError(
                    f"feature-ablation group[{index}] is malformed: {orchestrator_relative}"
                )
            group_name = str(group["name"])
            if group_name in names:
                raise ValueError(f"feature-ablation group name is duplicated: {group_name}")
            names.add(group_name)
            group_relative, group_path = _exact_path(
                root,
                group["config"],
                campaign_dir=campaign_relative,
                label=f"feature-ablation group {group_name} config",
            )
            runtime_configs.add(group_relative)
            files.update(_config_closure(root, group_path))

    stress_registries: dict[str, set[tuple[str, str, str, str]]] = {}
    for registry_relative, registry_path in _exact_paths(
        root,
        resolve.get("stress_registries", []),
        campaign_dir=campaign_relative,
        label="resolve.stress_registries",
    ):
        if registry_relative in stress_registries:
            raise ValueError(f"duplicate stress registry: {registry_relative}")
        registry_files, model_records = _resolve_stress_registry(
            root,
            registry_relative,
            registry_path,
        )
        stress_registries[registry_relative] = model_records
        files.update(registry_files)

    readiness_relative, readiness_path = _exact_path(
        root,
        metadata["data_readiness_path"],
        campaign_dir=campaign_relative,
        label="deployment data-readiness receipt",
    )
    metadata["data_readiness_path"] = readiness_relative
    files.add(readiness_path)
    readiness_sidecar = _verify_sidecar_if_present(readiness_path)
    if readiness_sidecar:
        files.add(readiness_sidecar)
    certified_features = _readiness_artifacts(root, readiness_relative)

    resolved_runtime_configs: dict[str, dict[str, Any]] = {}
    for raw_config in sorted(runtime_configs):
        config_relative, config_path = _exact_path(
            root,
            raw_config,
            campaign_dir=campaign_relative,
            label="runtime configuration",
        )
        files.update(_config_closure(root, config_path))
        resolved = load_config(config_path)
        feature_relative = _validate_runtime_feature_policy(
            root,
            config_relative,
            resolved,
            certified_features,
        )
        _, feature_path = _exact_path(
            root,
            feature_relative,
            campaign_dir=campaign_relative,
            label=f"preferred feature table for {config_relative}",
        )
        files.add(feature_path)
        resolved_runtime_configs[config_relative] = resolved
    runtime_configs = set(resolved_runtime_configs)

    deployed_existing = {path for path in files if path.is_file()}
    bindings = _runtime_feature_bindings(root, sorted(runtime_configs), deployed_existing)
    expectation_map = _normalize_feature_expectations(
        root,
        profile.get("feature_expectations"),
        campaign_dir=campaign_relative,
        readiness_artifacts=certified_features,
    )
    binding_map = {str(binding["config_path"]): binding for binding in bindings}
    unknown_expectations = sorted(set(expectation_map) - set(binding_map))
    if unknown_expectations:
        raise ValueError(
            f"feature expectations name configurations outside runtime_configs: {unknown_expectations}"
        )
    require_expectations = profile.get("require_feature_expectations", False)
    if not isinstance(require_expectations, bool):
        raise ValueError("require_feature_expectations must be boolean")
    if require_expectations:
        uncovered = sorted(set(binding_map) - base_runtime_configs - set(expectation_map))
        if uncovered:
            raise ValueError(
                "new runtime configurations lack feature expectations: "
                f"{uncovered}"
            )
    for config_relative, expected in expectation_map.items():
        binding = binding_map[config_relative]
        if (
            binding["feature_path"] != expected["path"]
            or binding["feature_sha256"] != expected["sha256"]
            or (
                "bytes" in expected
                and binding["feature_bytes"] != expected["bytes"]
            )
        ):
            raise ValueError(f"runtime feature binding is stale: {config_relative}")

    bound_stress_registries: set[str] = set()
    feature_hash_by_config = {
        str(binding["config_path"]): str(binding["feature_sha256"])
        for binding in bindings
    }
    for config_relative, resolved in resolved_runtime_configs.items():
        stress_binding = _stress_config_registry(root, config_relative, resolved)
        if stress_binding is None:
            continue
        registry_relative, config_models = stress_binding
        if registry_relative not in stress_registries:
            raise ValueError(
                f"stress configuration references an unbound registry: {config_relative} -> "
                f"{registry_relative}"
            )
        if config_models != stress_registries[registry_relative]:
            raise ValueError(
                f"stress configuration model references differ from registry: {config_relative}"
            )
        feature_hash = feature_hash_by_config[config_relative]
        stress = resolved["execution_stress"]
        declared_data_hashes = {
            str(model.get("feature_data_sha256", ""))
            for model in stress.get("model_paths", [])
            if isinstance(model, dict)
        }
        if declared_data_hashes != {feature_hash}:
            raise ValueError(
                f"stress model feature-data binding is stale: {config_relative}"
            )
        bound_stress_registries.add(registry_relative)
    unbound_registries = sorted(set(stress_registries) - bound_stress_registries)
    if unbound_registries:
        raise ValueError(f"stress registries are not bound by a runtime config: {unbound_registries}")

    exact_inputs = sorted(str(path.resolve().relative_to(root)) for path in files)
    if any(_GLOB_MAGIC.search(value) for value in exact_inputs):
        raise AssertionError("internal error: generated deployment inputs are not exact")
    deployment = {
        **metadata,
        "runtime_configs": sorted(runtime_configs),
        "inputs": exact_inputs,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump({"deployment": deployment}, sort_keys=False).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_file.name}.", suffix=".tmp", dir=output_file.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output_file)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "schema_version": 1,
        "profile_name": name,
        "profile_path": profile_relative,
        "profile_sha256": sha256_file(profile_file),
        "campaign_dir": campaign_relative,
        "base": base_record,
        "deployment_config_path": output_relative,
        "deployment_config_sha256": sha256_file(output_file),
        "input_count": len(exact_inputs),
        "runtime_configs": sorted(runtime_configs),
        "runtime_feature_bindings": bindings,
    }
