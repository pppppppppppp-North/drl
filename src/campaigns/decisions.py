from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.campaigns.gates import evaluate_campaign, sha256_file
from src.campaigns.model import CampaignDefinition, StageState
from src.utils.config import load_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_file(definition: CampaignDefinition, raw_pattern: str) -> Path:
    pattern = definition.expand_project_relative(raw_pattern)
    matches = []
    for path in definition.project_root.glob(pattern):
        resolved = path.resolve()
        if resolved.is_file() and resolved.is_relative_to(definition.project_root):
            matches.append(resolved)
    if not matches:
        raise FileNotFoundError(f"decision source is missing: {pattern}")
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def _matching_files(definition: CampaignDefinition, raw_pattern: str) -> list[Path]:
    pattern = definition.expand_project_relative(raw_pattern)
    matches = []
    for path in definition.project_root.glob(pattern):
        resolved = path.resolve()
        if resolved.is_file() and resolved.is_relative_to(definition.project_root):
            matches.append(resolved)
    if not matches:
        raise FileNotFoundError(f"protocol-lock input is missing: {pattern}")
    return sorted(matches)


def _safe_output(definition: CampaignDefinition, raw_path: str) -> Path:
    path = (definition.project_root / definition.expand_project_relative(raw_path)).resolve()
    if not path.is_relative_to(definition.campaign_dir):
        raise ValueError(f"derived campaign output must stay in the campaign directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class _DecisionSource:
    path: Path
    source_stage_id: str | None = None
    evidence_fingerprint: str | None = None


def _strict_source_context(definition: CampaignDefinition) -> tuple[dict[str, Any], dict[str, str]]:
    sources = definition.decisions.get("sources", {})
    source_stages = definition.decisions.get("source_stages", {})
    if not isinstance(sources, dict) or not sources:
        raise ValueError("strict decision materialization requires decisions.sources")
    if not isinstance(source_stages, dict):
        raise ValueError("strict decision materialization requires decisions.source_stages")
    missing = sorted(set(sources) - set(source_stages))
    extra = sorted(set(source_stages) - set(sources))
    if missing:
        raise ValueError(
            "strict decision source_stages is missing source key(s): " + ", ".join(missing)
        )
    if extra:
        raise ValueError(
            "strict decision source_stages contains unknown source key(s): " + ", ".join(extra)
        )
    normalized: dict[str, str] = {}
    for key, raw_stage_id in source_stages.items():
        stage_id = str(raw_stage_id).strip()
        stage = definition.stage_map.get(stage_id)
        if not stage_id or stage is None:
            raise ValueError(
                f"strict decision source {key!r} maps to unknown stage {raw_stage_id!r}"
            )
        if not stage.run_source:
            raise ValueError(
                f"strict decision source {key!r} must map to an aggregate stage with run_source; "
                f"got {stage_id!r}"
            )
        normalized[str(key)] = stage_id
    statuses = {status.stage_id: status for status in evaluate_campaign(definition)}
    return statuses, normalized


def _strict_decision_source(
    definition: CampaignDefinition,
    key: str,
    *,
    statuses: dict[str, Any],
    source_stages: dict[str, str],
) -> _DecisionSource:
    raw_pattern = str(definition.decisions["sources"][key])
    pattern = definition.expand_project_relative(raw_pattern)
    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ValueError(f"unsafe strict decision source pattern for {key!r}: {raw_pattern!r}")
    stage_id = source_stages[key]
    status = statuses[stage_id]
    if status.state != StageState.VERIFIED:
        raise ValueError(
            f"strict decision source stage {stage_id!r} for {key!r} must be VERIFIED; "
            f"current state is {status.state.value}"
        )
    candidates: list[Path] = []
    for evidence_file in status.gate.evidence_files:
        relative = Path(evidence_file)
        if not relative.match(pattern):
            continue
        candidate = (definition.project_root / relative).resolve()
        if not candidate.is_relative_to(definition.project_root) or not candidate.is_file():
            raise ValueError(
                f"strict decision source evidence is missing or unsafe: {evidence_file!r}"
            )
        candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError(
            f"strict decision source {key!r} from verified stage {stage_id!r} must match "
            f"exactly one gate evidence file for {pattern!r}; found {len(candidates)}"
        )
    return _DecisionSource(
        path=candidates[0],
        source_stage_id=stage_id,
        evidence_fingerprint=status.gate.evidence_fingerprint,
    )


def _decision_source(
    definition: CampaignDefinition,
    key: str,
    *,
    strict_context: tuple[dict[str, Any], dict[str, str]] | None,
) -> _DecisionSource:
    if strict_context is None:
        return _DecisionSource(
            path=_latest_file(definition, str(definition.decisions["sources"][key]))
        )
    statuses, source_stages = strict_context
    return _strict_decision_source(
        definition,
        key,
        statuses=statuses,
        source_stages=source_stages,
    )


def _source_record(definition: CampaignDefinition, source: _DecisionSource) -> dict[str, Any]:
    record = {
        "path": str(source.path.relative_to(definition.project_root)),
        "bytes": source.path.stat().st_size,
        "sha256": sha256_file(source.path),
    }
    if source.source_stage_id is not None:
        record["source_stage_id"] = source.source_stage_id
        record["evidence_fingerprint"] = source.evidence_fingerprint
    return record


def _selection_inputs(
    definition: CampaignDefinition,
    *,
    strict_context: tuple[dict[str, Any], dict[str, str]] | None,
) -> tuple[list[str], _DecisionSource]:
    source = _decision_source(
        definition,
        "screening_selection",
        strict_context=strict_context,
    )
    value = json.loads(source.path.read_text(encoding="utf-8"))
    algorithms = [str(item).lower() for item in value.get("selected_algorithms", [])]
    if value.get("selection_split") != "validation" or not 1 <= len(algorithms) <= 2:
        raise ValueError("screening decision must be validation-only and select one or two algorithms")
    if len(algorithms) != len(set(algorithms)):
        raise ValueError("screening decision contains duplicate algorithms")
    return algorithms, source


def _budget_input(
    definition: CampaignDefinition,
    *,
    strict_context: tuple[dict[str, Any], dict[str, str]] | None,
) -> tuple[int, _DecisionSource]:
    source = _decision_source(
        definition,
        "budget_selection",
        strict_context=strict_context,
    )
    value = json.loads(source.path.read_text(encoding="utf-8"))
    budget = int(value.get("selected_training_budget", 0))
    if value.get("selection_split") != "validation" or budget <= 0:
        raise ValueError("budget decision must be validation-only and positive")
    return budget, source


def _nested_input(
    definition: CampaignDefinition,
    algorithms: list[str],
    *,
    strict_context: tuple[dict[str, Any], dict[str, str]] | None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], _DecisionSource]:
    source = _decision_source(
        definition,
        "nested_selection",
        strict_context=strict_context,
    )
    all_rows = list(csv.DictReader(source.path.open("r", encoding="utf-8")))
    rows = [row for row in all_rows if str(row.get("algorithm", "")).lower() in algorithms]
    windows = [int(value) for value in definition.decisions.get("window_ids", [0, 1])]
    expected = {(algorithm, window) for algorithm in algorithms for window in windows}
    actual = {(str(row.get("algorithm", "")).lower(), int(row.get("window", -1))) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(f"nested decisions must contain exactly algorithm/window pairs {sorted(expected)}")
    parameters: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        algorithm = str(row["algorithm"]).lower()
        window = str(int(row["window"]))
        params = {}
        for key, raw_value in row.items():
            if not key.startswith("param_") or raw_value in (None, ""):
                continue
            value = yaml.safe_load(raw_value)
            if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
                raise ValueError(f"nested parameter is non-finite: {key}={raw_value}")
            params[key.removeprefix("param_")] = value
        if not params:
            raise ValueError(f"nested decision has no hyperparameters for {algorithm}/window {window}")
        parameters.setdefault(algorithm, {})[window] = params
    return parameters, source


def materialize_decisions(definition: CampaignDefinition, target: str) -> dict[str, Any]:
    decisions = definition.decisions
    base_configs = dict(decisions.get("base_configs", {}))
    outputs = dict(decisions.get("outputs", {}))
    if target not in {"budget", "nested", "robustness", "ablation"}:
        raise ValueError("materialization target must be budget, nested, robustness, or ablation")
    strict_context = (
        _strict_source_context(definition)
        if definition.strict_deployment_resolution
        else None
    )
    algorithms, screening_source = _selection_inputs(
        definition,
        strict_context=strict_context,
    )
    source_records = {"screening_selection": _source_record(definition, screening_source)}

    if target == "budget":
        config = load_config(definition.project_root / str(base_configs["budget"]))
        config.setdefault("budget_sweep", {})["algorithms"] = algorithms
    elif target == "nested":
        budget, budget_source = _budget_input(definition, strict_context=strict_context)
        source_records["budget_selection"] = _source_record(definition, budget_source)
        config = load_config(definition.project_root / str(base_configs["nested"]))
        configured_algorithms = [
            str(item).lower() for item in config.setdefault("nested_screening", {}).get("algorithms", [])
        ]
        if not set(algorithms).issubset(configured_algorithms):
            raise ValueError("nested screening must include every selected screening algorithm")
        config["nested_screening"]["total_timesteps"] = budget
        config.setdefault("training", {})["total_timesteps"] = budget
    elif target == "robustness":
        budget, budget_source = _budget_input(definition, strict_context=strict_context)
        parameters, nested_source = _nested_input(
            definition,
            algorithms,
            strict_context=strict_context,
        )
        source_records["budget_selection"] = _source_record(definition, budget_source)
        source_records["nested_selection"] = _source_record(definition, nested_source)
        config = load_config(definition.project_root / str(base_configs["robustness"]))
        robustness = config.setdefault("robustness", {})
        robustness["algorithms"] = algorithms
        robustness["total_timesteps"] = budget
        robustness["window_algorithm_parameters"] = parameters
        config.setdefault("training", {})["total_timesteps"] = budget
    else:
        budget, budget_source = _budget_input(definition, strict_context=strict_context)
        parameters, nested_source = _nested_input(
            definition,
            algorithms,
            strict_context=strict_context,
        )
        source_records["budget_selection"] = _source_record(definition, budget_source)
        source_records["nested_selection"] = _source_record(definition, nested_source)
        config = load_config(definition.project_root / str(base_configs["ablation"]))
        primary = algorithms[0]
        ablation = config.setdefault("feature_ablation", {})
        ablation["algorithm"] = primary
        ablation["total_timesteps"] = budget
        ablation["window_algorithm_parameters"] = parameters[primary]

    config["campaign_decision_provenance"] = {
        "campaign_id": definition.campaign_id,
        "selection_split": "validation",
        "sources": source_records,
    }
    output = _safe_output(definition, str(outputs[target]))
    _write_yaml(output, config)
    record = {
        "target": target,
        "output": str(output.relative_to(definition.project_root)),
        "selected_algorithms": algorithms,
        "sources": source_records,
        "output_sha256": sha256_file(output),
    }
    if target in {"nested", "robustness"}:
        record["selected_training_budget"] = int(config["training"]["total_timesteps"])
    elif target == "ablation":
        record["selected_training_budget"] = int(config["feature_ablation"]["total_timesteps"])
    record_path = output.with_suffix(".decision.json")
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _semantic_hash(path: Path) -> str:
    if path.suffix.lower() in {".yaml", ".yml"}:
        value: Any = load_config(path)
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        return sha256_file(path)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_protocol_lock(definition: CampaignDefinition) -> dict[str, Any]:
    input_rows = []
    strict_context = None
    decision_source_names = set(definition.decisions.get("sources", {}))
    if (
        definition.strict_deployment_resolution
        and any(lock_input.name in decision_source_names for lock_input in definition.lock_inputs)
    ):
        strict_context = _strict_source_context(definition)
    for lock_input in definition.lock_inputs:
        if strict_context is not None and lock_input.name in decision_source_names:
            selected = [
                _decision_source(
                    definition,
                    lock_input.name,
                    strict_context=strict_context,
                ).path
            ]
        else:
            candidates = _matching_files(definition, lock_input.pattern)
            selected = (
                candidates
                if lock_input.select == "all"
                else [
                    max(
                        candidates,
                        key=lambda path: (path.stat().st_mtime_ns, str(path)),
                    )
                ]
            )
        for index, path in enumerate(selected):
            input_rows.append(
                {
                    "name": lock_input.name if len(selected) == 1 else f"{lock_input.name}_{index}",
                    "path": str(path.relative_to(definition.project_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "semantic_sha256": _semantic_hash(path),
                }
            )
    scientific_payload = {
        "campaign_id": definition.campaign_id,
        "inputs": [
            {"name": row["name"], "path": row["path"], "semantic_sha256": row["semantic_sha256"]}
            for row in input_rows
        ],
    }
    protocol_sha = hashlib.sha256(
        json.dumps(scientific_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = definition.protocol_lock_path
    sidecar = path.with_suffix(".sha256")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != protocol_sha:
            raise FileExistsError("protocol lock is immutable and current inputs differ from the existing lock")
        if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
            raise ValueError("existing protocol-lock sidecar is missing or invalid")
        return existing
    lock = {
        "format": 1,
        "campaign_id": definition.campaign_id,
        "created_at": _utc_now(),
        "selection_firewall": "all locked selection evidence is validation-only; test stages require this lock hash",
        "protocol_sha256": protocol_sha,
        "inputs": input_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    sidecar.write_text(sha256_file(path) + "\n", encoding="utf-8")
    return lock
