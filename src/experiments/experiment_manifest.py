from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "torch",
    "stable-baselines3",
    "gymnasium",
    "optuna",
    "tensorboard",
    "scipy",
    "pyyaml",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(project_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_hashes(config: dict[str, Any], project_dir: Path) -> dict[str, dict[str, Any]]:
    feature_path = config.get("paths", {}).get("features")
    fallback_feature_path = config.get("paths", {}).get("fallback_features_csv")
    if feature_path:
        primary = Path(str(feature_path))
        primary = primary if primary.is_absolute() else project_dir / primary
        if not primary.is_file() and fallback_feature_path:
            feature_path = fallback_feature_path
    candidate_paths = {
        "features": feature_path,
        "raw_prices": config.get("paths", {}).get("raw_prices"),
        "market_indices": config.get("market_indices", {}).get("raw_prices"),
        "market_data_readiness": config.get("data", {}).get(
            "readiness_receipt",
            "reports/data_quality_market/market_data_readiness.json"
            if str(config.get("data", {}).get("source", "")).lower() != "synthetic"
            else None,
        ),
    }
    hashes: dict[str, dict[str, Any]] = {}
    for name, raw_path in candidate_paths.items():
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = project_dir / path
        if path.is_file():
            hashes[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return hashes


def _data_provenance(config: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    source = str(config.get("data", {}).get("source", "")).lower()
    if source == "synthetic":
        return {
            "data_origin": "generated",
            "evidence_class": "engineering_only",
            "execution_eligible": True,
            "publication_claim_eligible": False,
        }
    raw_receipt = config.get("data", {}).get(
        "readiness_receipt", "reports/data_quality_market/market_data_readiness.json"
    )
    receipt_path = Path(str(raw_receipt))
    if not receipt_path.is_absolute():
        receipt_path = project_dir / receipt_path
    if not receipt_path.is_file():
        return {
            "data_origin": "unknown_observed_source",
            "evidence_class": "unverified",
            "execution_eligible": False,
            "publication_claim_eligible": False,
            "readiness_receipt": str(receipt_path),
        }
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        receipt = {}
    return {
        "data_origin": receipt.get("data_origin", "unknown_observed_source"),
        "evidence_class": receipt.get("evidence_class", "unverified"),
        "execution_eligible": receipt.get("execution_eligible") is True,
        "publication_claim_eligible": receipt.get("publication_claim_eligible") is True,
        "usage_boundary": receipt.get("usage_boundary"),
        "readiness_receipt": str(receipt_path),
        "readiness_receipt_sha256": _sha256(receipt_path),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def write_experiment_manifest(
    output_dir: Path,
    *,
    config: dict[str, Any],
    identity: dict[str, Any],
    status: str,
    started_at: str,
    completed_at: str | None = None,
    error: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = Path.cwd().resolve()
    manifest = {
        "experiment_id": identity.get("experiment_id", output_dir.name),
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "identity": identity,
        "git_commit": _git_commit(project_dir),
        "data_files": _data_hashes(config, project_dir),
        "data_provenance": _data_provenance(config, project_dir),
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": _package_versions(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "campaign_id": os.environ.get("CAMPAIGN_ID"),
        "campaign_attempt_id": os.environ.get("CAMPAIGN_ATTEMPT_ID"),
        "campaign_attempt_sha256": os.environ.get("CAMPAIGN_ATTEMPT_SHA256"),
        "campaign_stage_id": os.environ.get("CAMPAIGN_STAGE_ID"),
        "deployment_release_id": os.environ.get("DEPLOYMENT_RELEASE_ID"),
        "bistka_preflight_sha256": os.environ.get("BISTKA_PREFLIGHT_SHA256"),
        "protocol_lock_hash": os.environ.get("PROTOCOL_LOCK_HASH"),
        "error": error,
    }
    manifest_path = output_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    return manifest_path


@contextmanager
def tracked_experiment(
    output_dir: Path,
    *,
    config: dict[str, Any],
    identity: dict[str, Any],
) -> Iterator[None]:
    started_at = _utc_now()
    write_experiment_manifest(
        output_dir,
        config=config,
        identity=identity,
        status="running",
        started_at=started_at,
    )
    try:
        yield
    except BaseException as exc:
        write_experiment_manifest(
            output_dir,
            config=config,
            identity=identity,
            status="failed",
            started_at=started_at,
            completed_at=_utc_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    write_experiment_manifest(
        output_dir,
        config=config,
        identity=identity,
        status="complete",
        started_at=started_at,
        completed_at=_utc_now(),
    )


def collect_experiment_manifests(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_dir.rglob("experiment_manifest.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        identity = data.get("identity", {})
        rows.append(
            {
                "source_path": str(path.relative_to(run_dir)),
                "experiment_id": data.get("experiment_id"),
                "status": data.get("status"),
                "algorithm": identity.get("algorithm"),
                "window": identity.get("window"),
                "seed": identity.get("seed"),
                "feature_group": identity.get("feature_group"),
                "cost_scenario": identity.get("cost_scenario"),
                "git_commit": data.get("git_commit"),
                "hostname": data.get("runtime", {}).get("hostname"),
                "slurm_job_id": data.get("runtime", {}).get("slurm_job_id"),
                "campaign_id": data.get("campaign_id"),
                "campaign_attempt_id": data.get("campaign_attempt_id"),
                "campaign_attempt_sha256": data.get("campaign_attempt_sha256"),
                "campaign_stage_id": data.get("campaign_stage_id"),
                "deployment_release_id": data.get("deployment_release_id"),
                "bistka_preflight_sha256": data.get("bistka_preflight_sha256"),
                "protocol_lock_hash": data.get("protocol_lock_hash"),
                "started_at": data.get("started_at"),
                "completed_at": data.get("completed_at"),
                "error": data.get("error"),
            }
        )
    return rows
