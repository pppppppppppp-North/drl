from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import socket
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.utils.config import load_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _safe_files(project_root: Path, patterns: Iterable[str]) -> list[Path]:
    root = project_root.resolve()
    matched: set[Path] = set()
    for raw_pattern in patterns:
        pattern = str(raw_pattern)
        candidate = Path(pattern)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe deployment input pattern: {pattern!r}")
        current = []
        for path in root.glob(pattern):
            resolved = path.resolve()
            if resolved.is_file() and resolved.is_relative_to(root):
                current.append(resolved)
        if not current:
            raise FileNotFoundError(f"deployment input pattern matched no files: {pattern}")
        matched.update(current)
    return sorted(matched, key=lambda value: str(value.relative_to(root)))


def _safe_project_file(root: Path, raw_path: str, *, label: str) -> tuple[Path, Path]:
    relative = Path(str(raw_path))
    resolved = (root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not resolved.is_relative_to(root)
        or not resolved.is_file()
    ):
        raise FileNotFoundError(f"{label} is missing or unsafe: {raw_path}")
    return relative, resolved


def _runtime_feature_bindings(
    project_root: Path,
    runtime_configs: Iterable[str],
    deployed_files: Iterable[Path],
) -> list[dict[str, Any]]:
    """Bind each runtime config to a deployed, preferred feature table.

    Exact releases do not permit a missing preferred path to fall back at runtime:
    an undeclared stale file at that preferred path could otherwise shadow the
    bundled fallback after extraction into an existing project directory.
    """

    root = project_root.resolve()
    deployed = {path.resolve() for path in deployed_files}
    bindings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_config in runtime_configs:
        config_relative, config_path = _safe_project_file(
            root,
            str(raw_config),
            label="runtime configuration",
        )
        config_key = str(config_relative)
        if config_key in seen:
            raise ValueError(f"duplicate runtime configuration: {config_key}")
        seen.add(config_key)
        if config_path not in deployed:
            raise ValueError(
                f"runtime configuration is not included in the deployment: {config_key}"
            )
        resolved_config = load_config(config_path)
        paths = resolved_config.get("paths", {})
        if not isinstance(paths, dict) or not paths.get("features"):
            raise ValueError(
                f"runtime configuration does not declare paths.features: {config_key}"
            )
        feature_relative, feature_path = _safe_project_file(
            root,
            str(paths["features"]),
            label=f"preferred runtime feature table for {config_key}",
        )
        if feature_path not in deployed:
            raise ValueError(
                "preferred runtime feature table is not included in the deployment: "
                f"{feature_relative} (from {config_key})"
            )
        resolved_hash = hashlib.sha256(
            json.dumps(resolved_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        bindings.append(
            {
                "config_path": config_key,
                "config_sha256": sha256_file(config_path),
                "resolved_config_sha256": resolved_hash,
                "feature_path": str(feature_relative),
                "feature_sha256": sha256_file(feature_path),
                "feature_bytes": feature_path.stat().st_size,
            }
        )
    return sorted(bindings, key=lambda value: value["config_path"])


def _canonical_release_payload(
    campaign_id: str,
    records: list[dict[str, Any]],
    runtime_feature_bindings: list[dict[str, Any]] | None = None,
) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "files": records,
            "runtime_feature_bindings": runtime_feature_bindings or [],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _tar_bytes(
    project_root: Path,
    files: list[Path],
    manifest_bytes: bytes,
    manifest_archive_path: Path,
) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in files:
                relative = path.resolve().relative_to(project_root.resolve())
                info = tarfile.TarInfo(name=str(relative))
                payload = path.read_bytes()
                info.size = len(payload)
                info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(payload))
            manifest_names = {
                ".bistka/deployment_manifest.json",
                str(manifest_archive_path),
            }
            for manifest_name in sorted(manifest_names):
                info = tarfile.TarInfo(name=manifest_name)
                info.size = len(manifest_bytes)
                info.mode = 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(manifest_bytes))
    return buffer.getvalue()


def build_deployment(
    *,
    project_root: Path,
    config_path: Path,
    output_manifest: Path,
    output_bundle: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    deployment = config.get("deployment", {})
    campaign_id = str(deployment.get("campaign_id", "large_scale_thai_drl"))
    patterns = deployment.get("inputs", [])
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("deployment.inputs must be a non-empty list")
    files = _safe_files(root, (str(value) for value in patterns))
    excluded = {output_manifest.resolve(), output_bundle.resolve(), output_bundle.with_suffix(output_bundle.suffix + ".sha256").resolve()}
    files = [path for path in files if path.resolve() not in excluded]
    runtime_configs = deployment.get("runtime_configs", [])
    if not isinstance(runtime_configs, list):
        raise ValueError("deployment.runtime_configs must be a list")
    runtime_feature_bindings = _runtime_feature_bindings(
        root,
        (str(value) for value in runtime_configs),
        files,
    )
    records = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "executable": bool(path.stat().st_mode & 0o111),
        }
        for path in files
    ]
    release_id = hashlib.sha256(
        _canonical_release_payload(campaign_id, records, runtime_feature_bindings)
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "release_id": release_id,
        "content_addressed": True,
        "target_project_root": str(deployment.get("target_project_root", "/lustrefs/project/25sfcs03/drl_thai_stock")),
        "python_executable": str(deployment.get("python_executable", "/lustrefs/project/25sfcs03/envs/drl_env/bin/python")),
        "required_packages": [str(value) for value in deployment.get("required_packages", [])],
        "required_hostname_prefix": str(deployment.get("required_hostname_prefix", "bistka-")),
        "data_readiness_path": str(deployment.get("data_readiness_path", "reports/data_quality_market/market_data_readiness.json")),
        "runtime_feature_bindings": runtime_feature_bindings,
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "files": records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(output_manifest, manifest_bytes)
    _atomic_bytes(output_manifest.with_suffix(".sha256"), (hashlib.sha256(manifest_bytes).hexdigest() + "\n").encode("utf-8"))
    try:
        manifest_archive_path = output_manifest.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("deployment manifest output must stay inside the project root") from exc
    bundle_bytes = _tar_bytes(root, files, manifest_bytes, manifest_archive_path)
    _atomic_bytes(output_bundle, bundle_bytes)
    _atomic_bytes(output_bundle.with_suffix(output_bundle.suffix + ".sha256"), (hashlib.sha256(bundle_bytes).hexdigest() + "\n").encode("utf-8"))
    return {
        **manifest,
        "manifest_path": str(output_manifest),
        "manifest_sha256": sha256_file(output_manifest),
        "bundle_path": str(output_bundle),
        "bundle_sha256": sha256_file(output_bundle),
    }


def validate_deployment_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
) -> tuple[list[str], list[Path]]:
    issues: list[str] = []
    referenced: list[Path] = []
    root = project_root.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read BistKA deployment manifest: {exc}"], []
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        return ["BistKA deployment manifest schema is unsupported"], []
    records = manifest.get("files", [])
    if not isinstance(records, list) or not records:
        return ["BistKA deployment manifest contains no files"], []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            issues.append("BistKA deployment manifest contains a malformed file record")
            continue
        relative = Path(str(record.get("path", "")))
        relative_text = str(relative)
        if relative.is_absolute() or ".." in relative.parts or relative_text in seen:
            issues.append(f"unsafe or duplicate deployment file: {relative_text}")
            continue
        seen.add(relative_text)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            issues.append(f"deployment source file is missing: {relative_text}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != record.get("sha256") or path.stat().st_size != record.get("bytes"):
            issues.append(f"deployment source file changed: {relative_text}")
            continue
        referenced.append(path)
        normalized.append(
            {
                "path": relative_text,
                "bytes": path.stat().st_size,
                "sha256": actual_hash,
                "executable": bool(record.get("executable", False)),
            }
        )
    declared_bindings = manifest.get("runtime_feature_bindings", [])
    actual_bindings: list[dict[str, Any]] = []
    if not isinstance(declared_bindings, list):
        issues.append("BistKA deployment runtime feature bindings must be a list")
        declared_bindings = []
    else:
        try:
            actual_bindings = _runtime_feature_bindings(
                root,
                (
                    str(binding.get("config_path", ""))
                    for binding in declared_bindings
                    if isinstance(binding, dict)
                ),
                (
                    (root / str(record["path"])).resolve()
                    for record in normalized
                ),
            )
            if actual_bindings != declared_bindings:
                issues.append(
                    "BistKA deployment runtime feature bindings are inconsistent"
                )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            issues.append(f"BistKA deployment runtime feature binding is invalid: {exc}")
    expected_id = hashlib.sha256(
        _canonical_release_payload(
            str(manifest.get("campaign_id", "")),
            normalized,
            actual_bindings,
        )
    ).hexdigest()
    if manifest.get("release_id") != expected_id or len(normalized) != len(records):
        issues.append("BistKA deployment release id or file inventory is inconsistent")
    if manifest.get("file_count") != len(records):
        issues.append("BistKA deployment file_count is inconsistent")
    if manifest.get("total_bytes") != sum(int(record.get("bytes", 0)) for record in records if isinstance(record, dict)):
        issues.append("BistKA deployment total_bytes is inconsistent")
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(manifest_path):
        issues.append("BistKA deployment manifest SHA-256 sidecar does not match")
    else:
        referenced.append(sidecar)
    return sorted(set(issues)), referenced


def validate_deployment_bundle(
    bundle_path: Path,
    *,
    manifest_path: Path,
    project_root: Path,
) -> tuple[list[str], list[Path]]:
    """Cross-check every tar member against the content-addressed manifest."""

    issues: list[str] = []
    referenced: list[Path] = [manifest_path]
    root = project_root.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_bytes = manifest_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read BistKA deployment manifest for bundle validation: {exc}"], []
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        return ["BistKA deployment manifest is malformed for bundle validation"], referenced
    try:
        manifest_relative = manifest_path.resolve().relative_to(root)
    except ValueError:
        return ["BistKA deployment manifest must stay inside the project root"], referenced
    records = {
        str(record.get("path", "")): record
        for record in manifest["files"]
        if isinstance(record, dict)
    }
    expected_names = {
        *records,
        ".bistka/deployment_manifest.json",
        str(manifest_relative),
    }
    try:
        with tarfile.open(bundle_path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                issues.append("BistKA deployment bundle contains duplicate members")
            if set(names) != expected_names:
                missing = sorted(expected_names - set(names))
                extra = sorted(set(names) - expected_names)
                issues.append(
                    f"BistKA deployment bundle inventory mismatch; missing={missing}, extra={extra}"
                )
            for member in members:
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts or not member.isfile():
                    issues.append(f"unsafe or non-file deployment bundle member: {member.name}")
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    issues.append(f"cannot read deployment bundle member: {member.name}")
                    continue
                payload = extracted.read()
                if member.name in {
                    ".bistka/deployment_manifest.json",
                    str(manifest_relative),
                }:
                    if payload != manifest_bytes:
                        issues.append(f"embedded deployment manifest differs: {member.name}")
                    continue
                record = records.get(member.name)
                if record is None:
                    continue
                source = (root / member_path).resolve()
                if source.is_file() and source.is_relative_to(root):
                    referenced.append(source)
                digest = hashlib.sha256(payload).hexdigest()
                if digest != record.get("sha256") or len(payload) != record.get("bytes"):
                    issues.append(f"deployment bundle payload differs from manifest: {member.name}")
                expected_mode = 0o755 if record.get("executable") else 0o644
                if member.mode != expected_mode:
                    issues.append(f"deployment bundle mode differs from manifest: {member.name}")
    except (OSError, tarfile.TarError) as exc:
        issues.append(f"cannot inspect BistKA deployment bundle: {exc}")
    sidecar = bundle_path.with_suffix(bundle_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(bundle_path):
        issues.append("BistKA deployment bundle SHA-256 sidecar does not match")
    else:
        referenced.append(sidecar)
    return sorted(set(issues)), referenced


def _runtime_packages(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def run_preflight(
    *,
    project_root: Path,
    manifest_path: Path,
    output_path: Path,
    allow_non_bistka: bool = False,
) -> dict[str, Any]:
    started = _utc_now()
    root = project_root.resolve()
    issues: list[str] = []
    checks: list[dict[str, Any]] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        manifest = {}
        issues.append(f"cannot read deployment manifest: {exc}")
    hostname = socket.gethostname()
    required_prefix = str(manifest.get("required_hostname_prefix", "bistka-"))
    host_ok = allow_non_bistka or hostname.startswith(required_prefix)
    checks.append({"check": "bistka_hostname", "passed": host_ok, "observed": hostname, "required": required_prefix})
    if not host_ok:
        issues.append(f"preflight must run on a host beginning with {required_prefix!r}")
    expected_root = str(manifest.get("target_project_root", ""))
    root_ok = allow_non_bistka or str(root) == expected_root
    checks.append({"check": "project_root", "passed": root_ok, "observed": str(root), "required": expected_root})
    if not root_ok:
        issues.append(f"project root mismatch: {root} != {expected_root}")

    records = manifest.get("files", [])
    verified = 0
    referenced: list[dict[str, Any]] = []
    if not isinstance(records, list) or not records:
        issues.append("deployment manifest contains no file records")
        records = []
    for record in records:
        if not isinstance(record, dict):
            issues.append("deployment manifest contains a malformed file record")
            continue
        relative = Path(str(record.get("path", "")))
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root) or not path.is_file():
            issues.append(f"deployment file is missing or unsafe: {relative}")
            continue
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            issues.append(f"deployment file hash mismatch: {relative}")
            continue
        verified += 1
        referenced.append({"path": str(relative), "sha256": actual, "bytes": path.stat().st_size})
    checks.append({"check": "deployment_files", "passed": verified == len(records) and bool(records), "observed": verified, "required": len(records)})

    declared_bindings = manifest.get("runtime_feature_bindings", [])
    actual_bindings: list[dict[str, Any]] = []
    runtime_bindings_ok = isinstance(declared_bindings, list)
    if runtime_bindings_ok:
        try:
            actual_bindings = _runtime_feature_bindings(
                root,
                (
                    str(binding.get("config_path", ""))
                    for binding in declared_bindings
                    if isinstance(binding, dict)
                ),
                (
                    (root / str(record.get("path", ""))).resolve()
                    for record in records
                    if isinstance(record, dict)
                ),
            )
            runtime_bindings_ok = actual_bindings == declared_bindings
        except (FileNotFoundError, TypeError, ValueError) as exc:
            runtime_bindings_ok = False
            issues.append(f"runtime feature binding is invalid: {exc}")
    if not runtime_bindings_ok and not any(
        issue.startswith("runtime feature binding is invalid:") for issue in issues
    ):
        issues.append("runtime feature bindings do not match the exact deployment")
    checks.append(
        {
            "check": "runtime_feature_bindings",
            "passed": runtime_bindings_ok,
            "observed": actual_bindings,
            "required": declared_bindings if isinstance(declared_bindings, list) else [],
        }
    )

    readiness_relative = Path(str(manifest.get("data_readiness_path", "")))
    readiness_path = (root / readiness_relative).resolve()
    data_origin = execution_eligible = evidence_class = None
    if readiness_path.is_relative_to(root) and readiness_path.is_file():
        try:
            readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
            data_origin = readiness.get("data_origin")
            execution_eligible = readiness.get("execution_eligible")
            evidence_class = readiness.get("evidence_class")
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"cannot read market-data readiness receipt: {exc}")
    else:
        issues.append(f"market-data readiness receipt is missing: {readiness_relative}")
    data_ok = data_origin == "observed_market" and execution_eligible is True
    checks.append({"check": "observed_market_data", "passed": data_ok, "observed": {"data_origin": data_origin, "execution_eligible": execution_eligible}, "required": {"data_origin": "observed_market", "execution_eligible": True}})
    if not data_ok and readiness_path.is_file():
        issues.append("market-data readiness receipt is not execution-eligible observed data")

    required_packages = [str(value) for value in manifest.get("required_packages", [])]
    packages = _runtime_packages(required_packages)
    missing_packages = sorted(name for name, version in packages.items() if version == "missing")
    checks.append({"check": "runtime_packages", "passed": not missing_packages, "observed": packages, "required": required_packages})
    if missing_packages:
        issues.append(f"required runtime packages are missing: {missing_packages}")

    receipt = {
        "schema_version": 1,
        "status": "pass" if not issues else "fail",
        "stage": "cluster_preflight",
        "started_at_utc": started,
        "completed_at_utc": _utc_now(),
        "hostname": hostname,
        "platform": platform.platform(),
        "project_root": str(root),
        "python_executable": os.path.realpath(os.sys.executable),
        "python_version": platform.python_version(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "release_id": manifest.get("release_id"),
        "declared_file_count": len(records),
        "verified_file_count": verified,
        "data_origin": data_origin,
        "data_evidence_class": evidence_class,
        "execution_eligible": execution_eligible,
        "packages": packages,
        "runtime_feature_bindings": actual_bindings,
        "checks": checks,
        "issues": issues,
        "verified_files": referenced,
    }
    _atomic_json(output_path, receipt)
    return receipt


def validate_preflight_receipt(
    receipt_path: Path,
    *,
    expected_manifest_path: Path,
) -> tuple[list[str], list[Path]]:
    issues: list[str] = []
    referenced: list[Path] = []
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        manifest = json.loads(expected_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read BistKA preflight evidence: {exc}"], []
    if receipt.get("status") != "pass" or receipt.get("stage") != "cluster_preflight":
        issues.append("BistKA cluster preflight did not pass")
    required_prefix = str(manifest.get("required_hostname_prefix", "bistka-"))
    if not str(receipt.get("hostname", "")).startswith(required_prefix):
        issues.append("BistKA preflight hostname is missing or unexpected")
    if receipt.get("project_root") != manifest.get("target_project_root"):
        issues.append("BistKA preflight project root does not match the deployment target")
    if receipt.get("manifest_sha256") != sha256_file(expected_manifest_path):
        issues.append("BistKA preflight manifest hash does not match the local deployment manifest")
    if receipt.get("release_id") != manifest.get("release_id"):
        issues.append("BistKA preflight release id does not match the local deployment")
    expected_count = int(manifest.get("file_count", 0))
    if expected_count <= 0 or receipt.get("declared_file_count") != expected_count or receipt.get("verified_file_count") != expected_count:
        issues.append("BistKA preflight did not verify the exact deployment file count")
    if receipt.get("data_origin") != "observed_market" or receipt.get("execution_eligible") is not True:
        issues.append("BistKA preflight did not verify execution-eligible observed data")
    packages = receipt.get("packages", {})
    if not isinstance(packages, dict) or any(packages.get(name) in {None, "missing"} for name in manifest.get("required_packages", [])):
        issues.append("BistKA preflight runtime package evidence is incomplete")
    declared_bindings = manifest.get("runtime_feature_bindings", [])
    if declared_bindings:
        checks = receipt.get("checks", [])
        runtime_check = next(
            (
                check
                for check in checks
                if isinstance(check, dict)
                and check.get("check") == "runtime_feature_bindings"
            ),
            None,
        ) if isinstance(checks, list) else None
        if (
            not isinstance(runtime_check, dict)
            or runtime_check.get("passed") is not True
            or receipt.get("runtime_feature_bindings") != declared_bindings
        ):
            issues.append("BistKA preflight did not verify exact runtime feature bindings")
    if receipt.get("issues") not in ([], None):
        issues.append("BistKA preflight receipt contains unresolved issues")
    referenced.append(expected_manifest_path)
    for record in manifest.get("files", []):
        if not isinstance(record, dict):
            continue
        candidate = (expected_manifest_path.parents[3] / str(record.get("path", ""))).resolve()
        if candidate.is_file():
            referenced.append(candidate)
    return sorted(set(issues)), referenced
