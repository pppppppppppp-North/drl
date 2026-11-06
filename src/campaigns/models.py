from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.campaigns.gates import latest_run_dir, protocol_lock_status, sha256_file
from src.campaigns.model import CampaignDefinition
from src.utils.config import load_config


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _campaign_output(definition: CampaignDefinition, raw_path: str) -> Path:
    path = (definition.project_root / definition.expand_project_relative(raw_path)).resolve()
    if not path.is_relative_to(definition.campaign_dir):
        raise ValueError(f"model-registry output must stay inside the campaign directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _locked_screening_selection(definition: CampaignDefinition) -> tuple[list[str], Path, str]:
    """Read the exact screening decision frozen in the active protocol lock."""
    lock = json.loads(definition.protocol_lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise ValueError("protocol lock must be a JSON object")
    matches = [entry for entry in lock.get("inputs", []) if entry.get("name") == "screening_selection"]
    if len(matches) != 1:
        raise ValueError("protocol lock must contain exactly one screening_selection input")
    record = matches[0]
    source = (definition.project_root / str(record.get("path", ""))).resolve()
    expected_hash = str(record.get("sha256", ""))
    if (
        not source.is_relative_to(definition.project_root)
        or not source.is_file()
        or len(expected_hash) != 64
        or sha256_file(source) != expected_hash
    ):
        raise ValueError("the locked screening selection is missing or has changed")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("locked screening decision must be a JSON object")
    algorithms = [str(item).lower() for item in value.get("selected_algorithms", [])]
    if value.get("selection_split") != "validation" or not 1 <= len(algorithms) <= 2:
        raise ValueError("locked screening decision must be validation-only and select one or two algorithms")
    if len(algorithms) != len(set(algorithms)):
        raise ValueError("locked screening decision contains duplicate algorithms")
    return algorithms, source, expected_hash


def build_locked_stress_config(
    definition: CampaignDefinition,
    *,
    robustness_run_dir: Path | None = None,
) -> dict[str, Any]:
    """Register a predeclared seed for each selected algorithm/window.

    Model choice never reads performance metrics. It uses the fixed seed in the
    campaign specification and binds every model to the active protocol hash.
    """
    lock_valid, protocol_hash, lock_issues = protocol_lock_status(definition)
    if not lock_valid or not protocol_hash:
        raise ValueError(f"a valid protocol lock is required before model registration: {lock_issues}")
    algorithms, screening_source, screening_hash = _locked_screening_selection(definition)
    windows = [int(value) for value in definition.decisions.get("window_ids", [0, 1])]
    fixed_seed = int(definition.decisions.get("stress_model_seed", 42))
    if fixed_seed < 0:
        raise ValueError("decisions.stress_model_seed must be nonnegative")

    robustness_stage = definition.stage_map.get("robustness")
    if robustness_run_dir is None:
        if robustness_stage is None:
            raise ValueError("campaign does not define a robustness stage")
        robustness_run_dir = latest_run_dir(definition, robustness_stage)
    if robustness_run_dir is None or not robustness_run_dir.is_dir():
        raise FileNotFoundError("no robustness result directory is available for model registration")
    robustness_run_dir = robustness_run_dir.resolve()
    if not robustness_run_dir.is_relative_to(definition.project_root):
        raise ValueError("robustness result directory must stay inside the project")

    outputs = dict(definition.decisions.get("outputs", {}))
    base_config = load_config(
        definition.project_root / str(definition.decisions.get("base_configs", {})["stress"])
    )
    locked_robustness = load_config(
        definition.project_root
        / definition.expand_project_relative(str(outputs["robustness"]))
    )
    expected_schema_hash = _canonical_hash(
        {
            "tickers": base_config.get("data", {}).get("tickers", []),
            "lookback": base_config.get("features", {}).get("lookback"),
            "columns": base_config.get("features", {}).get("columns", []),
        }
    )

    entries = []
    schema_hashes: set[str] = set()
    data_hashes: set[str] = set()
    for algorithm in algorithms:
        for window in windows:
            model_dir = (
                robustness_run_dir
                / f"algorithm_{algorithm}"
                / f"window_{window}"
                / f"seed_{fixed_seed}"
            )
            manifest_path = model_dir / "experiment_manifest.json"
            config_path = model_dir / "resolved_config.json"
            model_path = model_dir / f"{algorithm}_window_{window}_model.zip"
            source_curve_path = model_dir / f"{algorithm}_test_equity_curve.csv"
            source_metrics_path = model_dir / "walk_forward_metrics.csv"
            for required in (manifest_path, config_path, model_path, source_curve_path, source_metrics_path):
                if not required.is_file():
                    raise FileNotFoundError(f"fixed-seed model artifact is missing: {required}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity = manifest.get("identity", {})
            if manifest.get("status") != "complete":
                raise ValueError(f"fixed-seed model manifest is not complete: {manifest_path}")
            if manifest.get("protocol_lock_hash") != protocol_hash:
                raise ValueError(f"fixed-seed model belongs to another protocol: {manifest_path}")
            if manifest.get("campaign_id") != definition.campaign_id:
                raise ValueError(f"fixed-seed model belongs to another campaign: {manifest_path}")
            expected_identity = (algorithm, window, fixed_seed)
            actual_identity = (
                str(identity.get("algorithm", "")).lower(),
                int(identity.get("window", -1)),
                int(identity.get("seed", -1)),
            )
            if actual_identity != expected_identity:
                raise ValueError(
                    f"fixed-seed model identity mismatch: expected={expected_identity} actual={actual_identity}"
                )
            resolved = json.loads(config_path.read_text(encoding="utf-8"))
            schema = {
                "tickers": resolved.get("data", {}).get("tickers", []),
                "lookback": resolved.get("features", {}).get("lookback"),
                "columns": resolved.get("features", {}).get("columns", []),
            }
            schema_hash = _canonical_hash(schema)
            if schema_hash != expected_schema_hash:
                raise ValueError(f"fixed-seed model feature schema does not match stress config: {config_path}")
            feature_data_hash = str(manifest.get("data_files", {}).get("features", {}).get("sha256", ""))
            if len(feature_data_hash) != 64:
                raise ValueError(f"fixed-seed manifest lacks a feature-data hash: {manifest_path}")
            schema_hashes.add(schema_hash)
            data_hashes.add(feature_data_hash)
            entries.append(
                {
                    "policy": f"{algorithm}_window_{window}_seed_{fixed_seed}",
                    "algorithm": algorithm,
                    "window": window,
                    "seed": fixed_seed,
                    "path": str(model_path.relative_to(definition.project_root)),
                    "bytes": model_path.stat().st_size,
                    "sha256": sha256_file(model_path),
                    "manifest_path": str(manifest_path.relative_to(definition.project_root)),
                    "manifest_sha256": sha256_file(manifest_path),
                    "resolved_config_path": str(config_path.relative_to(definition.project_root)),
                    "resolved_config_sha256": sha256_file(config_path),
                    "source_test_curve_path": str(source_curve_path.relative_to(definition.project_root)),
                    "source_test_curve_sha256": sha256_file(source_curve_path),
                    "source_metrics_path": str(source_metrics_path.relative_to(definition.project_root)),
                    "source_metrics_sha256": sha256_file(source_metrics_path),
                    "feature_schema_sha256": schema_hash,
                    "feature_data_sha256": feature_data_hash,
                    "protocol_lock_hash": protocol_hash,
                }
            )
    if len(schema_hashes) != 1 or len(data_hashes) != 1:
        raise ValueError("registered models do not share one feature schema and feature-data fingerprint")

    registry_path = _campaign_output(definition, str(outputs["model_registry"]))
    stress_config_path = _campaign_output(definition, str(outputs["stress"]))
    registry = {
        "schema_version": 1,
        "campaign_id": definition.campaign_id,
        "protocol_lock_hash": protocol_hash,
        "selection_rule": "fixed predeclared seed; no test-performance ranking",
        "selected_seed": fixed_seed,
        "screening_selection_path": str(screening_source.relative_to(definition.project_root)),
        "screening_selection_sha256": screening_hash,
        "robustness_run_dir": str(robustness_run_dir.relative_to(definition.project_root)),
        "feature_schema_sha256": next(iter(schema_hashes)),
        "feature_data_sha256": next(iter(data_hashes)),
        "feature_contract": {
            "feature_schema_sha256": next(iter(schema_hashes)),
            "feature_data_sha256": next(iter(data_hashes)),
            "walk_forward": locked_robustness["walk_forward"],
        },
        "models": entries,
    }

    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    registry_sidecar = registry_path.with_suffix(".sha256")
    registry_sidecar.write_text(sha256_file(registry_path) + "\n", encoding="utf-8")
    base_config["walk_forward"] = dict(locked_robustness["walk_forward"])
    stress = base_config.setdefault("execution_stress", {})
    stress["require_models"] = True
    stress["protocol_lock_hash"] = protocol_hash
    stress["model_registry"] = str(registry_path.relative_to(definition.project_root))
    stress["model_paths"] = [
        {
            "policy": entry["policy"],
            "model_id": entry["policy"],
            "algo": entry["algorithm"],
            "window": entry["window"],
            "seed": entry["seed"],
            "path": entry["path"],
            "sha256": entry["sha256"],
            "source_test_curve_path": entry["source_test_curve_path"],
            "source_test_curve_sha256": entry["source_test_curve_sha256"],
            "feature_schema_sha256": entry["feature_schema_sha256"],
            "feature_data_sha256": entry["feature_data_sha256"],
            "protocol_lock_hash": protocol_hash,
        }
        for entry in entries
    ]
    base_config["campaign_model_provenance"] = {
        "campaign_id": definition.campaign_id,
        "protocol_lock_hash": protocol_hash,
        "model_registry": str(registry_path.relative_to(definition.project_root)),
        "model_registry_sha256": sha256_file(registry_path),
        "selection_rule": registry["selection_rule"],
    }
    _write_yaml(stress_config_path, base_config)
    return {
        "model_registry": str(registry_path.relative_to(definition.project_root)),
        "model_registry_sidecar": str(registry_sidecar.relative_to(definition.project_root)),
        "locked_stress_config": str(stress_config_path.relative_to(definition.project_root)),
        "registered_models": len(entries),
        "selected_seed": fixed_seed,
        "protocol_lock_hash": protocol_hash,
    }
