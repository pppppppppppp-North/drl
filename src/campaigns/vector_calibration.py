from __future__ import annotations

import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.campaigns.calibration import (
    atomic_json,
    atomic_text,
    canonical_sha256,
    receipt_path_for,
    sha256_file,
)
from src.campaigns.model import CampaignDefinition
from src.utils.config import load_config


SELECTION_FRACTION_OF_MAX = 0.95
SELECTION_RULE = (
    "smallest num_envs with median throughput at least 95% of maximum median throughput"
)
BENCHMARK_GLOB = "reports/vectorized_training_benchmark_*[0-9].csv"
CALIBRATION_IMPORT = "imports/v1_calibration_evidence.json"
JOB_OUTPUT_PATTERN = re.compile(r"^vectorized_training_benchmark_[0-9]+[.]csv$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidatedBenchmark:
    csv_path: Path
    metadata_path: Path
    receipt_path: Path
    metadata: dict[str, Any]
    receipt: dict[str, Any]
    selected_num_envs: int
    median_throughput: dict[int, float]


@dataclass(frozen=True)
class MaterializationEvidence:
    mode: str
    target_campaign_id: str
    source_campaign_id: str
    attempt_path: Path
    attempt: dict[str, Any]
    benchmark: ValidatedBenchmark
    additional_sources: dict[str, Path]
    target_attempt_id: str | None
    target_slurm_job_id: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _sha256(value: Any, label: str) -> str:
    text = str(value)
    if not SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _safe_project_path(
    project_root: Path,
    path: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"{label} must stay inside the project root: {resolved}")
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _source_record(project_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(project_root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _config_closure(project_root: Path, config_path: Path) -> list[Path]:
    closure: list[Path] = []
    seen: set[Path] = set()
    current = config_path.resolve()
    while True:
        if current in seen:
            raise ValueError(f"configuration inheritance cycle detected: {current}")
        if not current.is_relative_to(project_root) or not current.is_file():
            raise ValueError(f"configuration inheritance escapes the project: {current}")
        seen.add(current)
        closure.append(current)
        try:
            value = yaml.safe_load(current.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"configuration is not valid YAML: {current}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"configuration root must be a mapping: {current}")
        parent = value.get("extends")
        if not parent:
            return closure
        parent_path = Path(str(parent))
        if parent_path.is_absolute() or ".." in parent_path.parts:
            raise ValueError(f"configuration extends an unsafe parent: {parent}")
        current = (current.parent / parent_path).resolve()


def _validate_source_record(
    project_root: Path,
    record: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an artifact record")
    path = _safe_project_path(
        project_root,
        str(record.get("path", "")),
        label=label,
        must_exist=True,
    )
    expected_hash = _sha256(record.get("sha256"), f"{label} hash")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{label} hash does not match {path.name}")
    if record.get("bytes") is not None:
        try:
            expected_bytes = int(record["bytes"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} has an invalid byte count") from exc
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"{label} byte count does not match {path.name}")
    return path


def _validate_import_sidecar(import_path: Path) -> Path:
    sidecar = import_path.with_suffix(import_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"calibration import checksum sidecar is missing: {sidecar}")
    try:
        recorded = sidecar.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"calibration import checksum sidecar is unreadable: {sidecar}") from exc
    if _sha256(recorded, "calibration import sidecar") != sha256_file(import_path):
        raise ValueError("calibration import checksum sidecar does not match the import JSON")
    return sidecar


def _receipt_output(receipt: dict[str, Any], name: str) -> dict[str, Any]:
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("benchmark receipt outputs must be a list")
    matches = [item for item in outputs if isinstance(item, dict) and item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"benchmark receipt must contain exactly one {name!r} output")
    return matches[0]


def _validate_receipt_output(receipt: dict[str, Any], name: str, path: Path) -> None:
    record = _receipt_output(receipt, name)
    if record.get("exists") is not True:
        raise ValueError(f"benchmark receipt marks output {name!r} as missing")
    if _sha256(record.get("sha256"), f"receipt output {name} hash") != sha256_file(path):
        raise ValueError(f"benchmark receipt hash does not match {path.name}")
    try:
        expected_bytes = int(record.get("bytes"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"benchmark receipt output {name!r} has an invalid byte count") from exc
    if expected_bytes != path.stat().st_size:
        raise ValueError(f"benchmark receipt byte count does not match {path.name}")


def _read_throughput_rows(path: Path) -> tuple[list[dict[str, str]], dict[int, list[float]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"benchmark CSV is not readable: {path}") from exc
    if not rows:
        raise ValueError("benchmark CSV contains no trials")
    required = {
        "num_envs",
        "repeat",
        "samples_per_second",
        "data_source",
        "data_sha256",
        "config_sha256",
        "code_sha256",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"benchmark CSV is missing columns: {missing}")
    groups: dict[int, list[float]] = {}
    repeats: dict[int, set[int]] = {}
    for row in rows:
        try:
            num_envs = int(row["num_envs"])
            repeat = int(row["repeat"])
            throughput = float(row["samples_per_second"])
        except (TypeError, ValueError) as exc:
            raise ValueError("benchmark CSV has a non-numeric environment, repeat, or throughput") from exc
        if num_envs <= 0 or repeat <= 0 or not math.isfinite(throughput) or throughput <= 0:
            raise ValueError("benchmark environment counts, repeats, and throughputs must be positive")
        if repeat in repeats.setdefault(num_envs, set()):
            raise ValueError(f"benchmark CSV repeats trial ({num_envs}, {repeat})")
        repeats[num_envs].add(repeat)
        groups.setdefault(num_envs, []).append(throughput)
    expected_repeats = None
    for num_envs, identifiers in repeats.items():
        contiguous = set(range(1, len(identifiers) + 1))
        if identifiers != contiguous:
            raise ValueError(f"benchmark repeats for num_envs={num_envs} are not contiguous from one")
        if expected_repeats is None:
            expected_repeats = len(identifiers)
        elif len(identifiers) != expected_repeats:
            raise ValueError("benchmark environment counts do not have equal repeat counts")
    return rows, groups


def _selected_num_envs(groups: dict[int, list[float]]) -> tuple[int, dict[int, float]]:
    medians = {num_envs: float(statistics.median(values)) for num_envs, values in groups.items()}
    peak = max(medians.values())
    threshold = peak * SELECTION_FRACTION_OF_MAX
    selected = min(num_envs for num_envs, throughput in medians.items() if throughput >= threshold)
    return selected, medians


def _validate_summary(metadata: dict[str, Any], groups: dict[int, list[float]]) -> None:
    raw_summary = metadata.get("summary_by_env_count")
    if not isinstance(raw_summary, list) or not raw_summary:
        raise ValueError("benchmark metadata summary_by_env_count must be a non-empty list")
    by_env: dict[int, dict[str, Any]] = {}
    for item in raw_summary:
        if not isinstance(item, dict):
            raise ValueError("benchmark metadata summary rows must be objects")
        try:
            num_envs = int(item["num_envs"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("benchmark metadata summary has an invalid num_envs") from exc
        if num_envs in by_env:
            raise ValueError("benchmark metadata summary repeats an environment count")
        by_env[num_envs] = item
    if set(by_env) != set(groups):
        raise ValueError("benchmark metadata summary environment counts do not match the CSV")
    for num_envs, values in groups.items():
        row = by_env[num_envs]
        try:
            repeats = int(row["repeats"])
            recorded_median = float(row["median_samples_per_second"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("benchmark metadata summary has invalid repeat or median values") from exc
        if repeats != len(values) or not math.isclose(
            recorded_median,
            statistics.median(values),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("benchmark metadata summary does not reproduce the CSV medians")


def validate_benchmark_candidate(
    csv_path: Path,
    *,
    latest_attempt: dict[str, Any],
) -> ValidatedBenchmark:
    if not JOB_OUTPUT_PATTERN.fullmatch(csv_path.name):
        raise ValueError(f"benchmark CSV does not end in a numeric SLURM job id: {csv_path.name}")
    metadata_path = csv_path.with_suffix(".json")
    receipt_path = receipt_path_for(csv_path)
    if not metadata_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(
            f"benchmark candidate is incomplete: {csv_path.name} requires {metadata_path.name} "
            f"and {receipt_path.name}"
        )
    metadata = _json_mapping(metadata_path, "benchmark metadata")
    receipt = _json_mapping(receipt_path, "benchmark receipt")

    attempt_id = str(latest_attempt.get("attempt_id", ""))
    if not attempt_id:
        raise ValueError("latest vector_benchmark attempt has no attempt_id")
    if receipt.get("campaign_attempt_id") != attempt_id:
        raise ValueError(
            f"benchmark receipt attempt {receipt.get('campaign_attempt_id')!r} does not match "
            f"latest vector_benchmark attempt {attempt_id!r}"
        )
    if receipt.get("status") != "success" or receipt.get("calibration_kind") != "vectorized_training_benchmark":
        raise ValueError("benchmark receipt is not a successful vectorized-training calibration")
    if receipt.get("campaign_id") != latest_attempt.get("campaign_id"):
        raise ValueError("benchmark receipt campaign_id does not match the latest attempt")
    slurm_job_id = str(receipt.get("slurm_job_id", ""))
    if not slurm_job_id.isdigit() or not csv_path.name.endswith(f"_{slurm_job_id}.csv"):
        raise ValueError("benchmark receipt SLURM job id does not match the output filename")
    parameters = receipt.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("synthetic") is not False:
        raise ValueError("benchmark receipt must explicitly declare synthetic=false")

    provenance = metadata.get("data_provenance")
    if not isinstance(provenance, dict) or provenance.get("synthetic") is not False:
        raise ValueError("benchmark metadata must explicitly declare observed data with synthetic=false")
    source = str(provenance.get("source", "")).strip().lower()
    if source in {"", "synthetic", "generated"}:
        raise ValueError("benchmark metadata has a generated or unknown data source")
    data_hash = _sha256(provenance.get("data_sha256"), "metadata data hash")
    config_hash = _sha256(provenance.get("config_sha256"), "metadata config hash")
    code = metadata.get("code_provenance")
    if not isinstance(code, dict):
        raise ValueError("benchmark metadata code_provenance must be an object")
    code_hash = _sha256(code.get("sha256"), "metadata code hash")

    if metadata.get("selection_rule") != SELECTION_RULE:
        raise ValueError("benchmark metadata uses an unsupported environment-selection rule")
    try:
        selection_fraction = float(metadata.get("selection_fraction_of_max"))
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark metadata selection fraction is invalid") from exc
    if selection_fraction != SELECTION_FRACTION_OF_MAX:
        raise ValueError("benchmark metadata must use the 95% peak-median threshold")
    if metadata.get("throughput_statistic") != "median_samples_per_second":
        raise ValueError("benchmark metadata must select on median_samples_per_second")

    rows, groups = _read_throughput_rows(csv_path)
    _validate_summary(metadata, groups)
    selected, medians = _selected_num_envs(groups)
    if int(metadata.get("selected_num_envs", 0)) != selected:
        raise ValueError("benchmark metadata selected_num_envs does not reproduce from the CSV")
    if int(metadata.get("trial_count", 0)) != len(rows):
        raise ValueError("benchmark metadata trial_count does not match the CSV")
    expected_repeats = len(next(iter(groups.values())))
    if int(metadata.get("repeats_per_env_count", 0)) != expected_repeats:
        raise ValueError("benchmark metadata repeat count does not match the CSV")

    row_sources = {str(row["data_source"]).strip().lower() for row in rows}
    if row_sources != {source}:
        raise ValueError("benchmark CSV data source does not match the metadata")
    for column, expected in (
        ("data_sha256", data_hash),
        ("config_sha256", config_hash),
        ("code_sha256", code_hash),
    ):
        values = {str(row[column]) for row in rows}
        if values != {expected}:
            raise ValueError(f"benchmark CSV {column} does not match the metadata")

    result = receipt.get("result")
    if not isinstance(result, dict):
        raise ValueError("benchmark receipt result must be an object")
    expected_result = {
        "selected_num_envs": selected,
        "trial_count": len(rows),
        "data_sha256": data_hash,
        "config_sha256": config_hash,
        "code_sha256": code_hash,
    }
    for key, expected in expected_result.items():
        if result.get(key) != expected:
            raise ValueError(f"benchmark receipt result {key} does not match the evidence")
    env_counts = parameters.get("env_counts")
    if not isinstance(env_counts, list) or [int(value) for value in env_counts] != sorted(groups):
        raise ValueError("benchmark receipt environment counts do not match the CSV")
    if int(parameters.get("repeats", 0)) != expected_repeats:
        raise ValueError("benchmark receipt repeat count does not match the CSV")
    if Path(str(metadata.get("results_file", ""))).name != csv_path.name:
        raise ValueError("benchmark metadata results_file does not identify the CSV")
    _validate_receipt_output(receipt, "benchmark_csv", csv_path)
    _validate_receipt_output(receipt, "benchmark_metadata", metadata_path)

    return ValidatedBenchmark(
        csv_path=csv_path,
        metadata_path=metadata_path,
        receipt_path=receipt_path,
        metadata=metadata,
        receipt=receipt,
        selected_num_envs=selected,
        median_throughput=medians,
    )


def _latest_attempt(project_root: Path, campaign_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = campaign_dir / "attempts" / "vector_benchmark_latest.json"
    if not path.is_file():
        raise FileNotFoundError(f"latest vector_benchmark attempt is missing: {path}")
    attempt = _json_mapping(path, "latest vector_benchmark attempt")
    if (
        attempt.get("schema_version") not in {1, 2}
        or attempt.get("status") != "rendered"
        or attempt.get("stage_id") != "vector_benchmark"
        or not str(attempt.get("campaign_id", ""))
        or not str(attempt.get("attempt_id", ""))
    ):
        raise ValueError("latest vector_benchmark attempt has an invalid identity or status")
    if attempt.get("schema_version") == 2:
        attempt_hash = str(attempt.get("attempt_sha256", ""))
        unhashed = dict(attempt)
        unhashed.pop("attempt_sha256", None)
        if _sha256(attempt_hash, "schema-v2 attempt hash") != canonical_sha256(unhashed):
            raise ValueError("schema-v2 vector attempt hash is invalid")
        immutable = campaign_dir / "attempts" / f"{attempt['attempt_id']}.json"
        if not immutable.is_file() or _json_mapping(
            immutable,
            "immutable schema-v2 vector attempt",
        ) != attempt:
            raise ValueError("schema-v2 latest and immutable vector attempts differ")
        for path_field, hash_field, label in (
            (
                "deployment_manifest_path",
                "deployment_manifest_sha256",
                "schema-v2 deployment manifest",
            ),
            (
                "preflight_receipt_path",
                "preflight_sha256",
                "schema-v2 preflight receipt",
            ),
        ):
            bound = _safe_project_path(
                project_root,
                str(attempt.get(path_field, "")),
                label=label,
                must_exist=True,
            )
            if sha256_file(bound) != _sha256(
                attempt.get(hash_field),
                f"{label} hash",
            ):
                raise ValueError(f"{label} no longer matches the rendered attempt")
        manifest = _json_mapping(
            _safe_project_path(
                project_root,
                str(attempt["deployment_manifest_path"]),
                label="schema-v2 deployment manifest",
                must_exist=True,
            ),
            "schema-v2 deployment manifest",
        )
        if manifest.get("release_id") != attempt.get("deployment_release_id"):
            raise ValueError("schema-v2 deployment release differs from the rendered attempt")
    return path, attempt


def _newest_valid_benchmark(
    project_root: Path,
    latest_attempt: dict[str, Any],
    benchmark_glob: str,
) -> ValidatedBenchmark:
    pattern = Path(benchmark_glob)
    if pattern.is_absolute() or ".." in pattern.parts:
        raise ValueError("benchmark glob must be a safe project-relative pattern")
    candidates = sorted(
        path.resolve()
        for path in project_root.glob(benchmark_glob)
        if path.is_file() and path.resolve().is_relative_to(project_root)
    )
    if not candidates:
        raise FileNotFoundError(f"no numeric-job benchmark CSV matches {benchmark_glob}")
    valid: list[ValidatedBenchmark] = []
    errors: list[str] = []
    for candidate in candidates:
        try:
            valid.append(validate_benchmark_candidate(candidate, latest_attempt=latest_attempt))
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{candidate.name}: {exc}")
    if not valid:
        detail = "; ".join(errors[-5:])
        raise ValueError(f"no valid observed benchmark is bound to the latest attempt: {detail}")
    if len(valid) != 1:
        raise ValueError(
            "expected exactly one valid observed benchmark for the latest attempt, "
            f"found {len(valid)}"
        )
    return valid[0]


def _imported_benchmark(
    project_root: Path,
    campaign_dir: Path,
    import_path: Path,
    *,
    definition: CampaignDefinition,
    base_config_path: Path,
) -> MaterializationEvidence:
    from src.campaigns.evidence_migration import validate_calibration_import

    stage = definition.stage_map["vector_benchmark"]
    migration_requirements = [
        requirement
        for requirement in stage.requirements
        if requirement.validator == "calibration_evidence"
    ]
    if len(migration_requirements) != 1:
        raise ValueError(
            "vector_benchmark must define exactly one calibration_evidence requirement"
        )
    import_issues, _ = validate_calibration_import(
        definition,
        stage,
        import_path,
        migration_requirements[0].options,
    )
    if import_issues:
        raise ValueError("calibration evidence import is stale: " + "; ".join(import_issues))
    import_value = _json_mapping(import_path, "calibration evidence import")
    sidecar_path = _validate_import_sidecar(import_path)
    if (
        import_value.get("schema_version") != 1
        or import_value.get("evidence_kind") != "v1_calibration_evidence_import"
        or import_value.get("status") != "accepted"
    ):
        raise ValueError("calibration evidence import has an invalid schema, kind, or status")

    target = import_value.get("target")
    source = import_value.get("source")
    if not isinstance(target, dict) or not isinstance(source, dict):
        raise ValueError("calibration evidence import must contain source and target objects")
    target_campaign_id = str(target.get("campaign_id", "")).strip()
    source_campaign_id = str(source.get("campaign_id", "")).strip()
    if not target_campaign_id or not source_campaign_id:
        raise ValueError("calibration evidence import has an empty source or target campaign id")
    if target_campaign_id != campaign_dir.name:
        raise ValueError(
            f"calibration evidence import targets campaign {target_campaign_id!r}, "
            f"not {campaign_dir.name!r}"
        )
    target_dir = _safe_project_path(
        project_root,
        str(target.get("campaign_dir", "")),
        label="calibration import target campaign directory",
        must_exist=False,
    )
    if target_dir != campaign_dir:
        raise ValueError("calibration evidence import target directory does not match campaign_dir")
    target_config = target.get("config")
    if not isinstance(target_config, dict):
        raise ValueError("calibration evidence import has no target config binding")
    certified_base = _validate_source_record(
        project_root,
        target_config,
        label="calibration import target config",
    )
    if certified_base != base_config_path:
        raise ValueError("base screening config differs from the calibration import target config")
    resolved_base = load_config(base_config_path)
    if canonical_sha256(resolved_base) != _sha256(
        target_config.get("resolved_sha256"),
        "calibration import target resolved-config hash",
    ):
        raise ValueError("base screening resolved config differs from the calibration import")
    source_dir = _safe_project_path(
        project_root,
        str(source.get("campaign_dir", "")),
        label="calibration import source campaign directory",
        must_exist=False,
    )
    if source_dir.name != source_campaign_id:
        raise ValueError("calibration evidence import source directory does not match source campaign id")

    stages = source.get("stages")
    vector = stages.get("vector_benchmark") if isinstance(stages, dict) else None
    if not isinstance(vector, dict) or vector.get("stage_id") != "vector_benchmark":
        raise ValueError("calibration evidence import has no vector_benchmark source stage")
    attempt_record = vector.get("attempt")
    receipt_record = vector.get("receipt")
    outputs = vector.get("outputs")
    if (
        not isinstance(attempt_record, dict)
        or not isinstance(receipt_record, dict)
        or not isinstance(outputs, dict)
    ):
        raise ValueError("calibration evidence import vector stage is incomplete")

    attempt_path = _safe_project_path(
        project_root,
        str(attempt_record.get("latest_path", "")),
        label="imported latest vector attempt",
        must_exist=True,
    )
    if not attempt_path.is_relative_to(source_dir):
        raise ValueError("imported latest vector attempt is outside the source campaign")
    if sha256_file(attempt_path) != _sha256(
        attempt_record.get("latest_sha256"),
        "imported latest vector attempt hash",
    ):
        raise ValueError("imported latest vector attempt hash does not match")
    attempt = _json_mapping(attempt_path, "imported latest vector attempt")
    if (
        attempt.get("schema_version") not in {1, 2}
        or attempt.get("status") != "rendered"
        or attempt.get("campaign_id") != source_campaign_id
        or attempt.get("stage_id") != "vector_benchmark"
        or attempt.get("attempt_id") != attempt_record.get("attempt_id")
        or attempt.get("attempt_sha256") != attempt_record.get("attempt_sha256")
    ):
        raise ValueError("imported latest vector attempt has an invalid or inconsistent identity")

    immutable_path = _safe_project_path(
        project_root,
        str(attempt_record.get("immutable_path", "")),
        label="imported immutable vector attempt",
        must_exist=True,
    )
    if not immutable_path.is_relative_to(source_dir):
        raise ValueError("imported immutable vector attempt is outside the source campaign")
    if sha256_file(immutable_path) != _sha256(
        attempt_record.get("immutable_sha256"),
        "imported immutable vector attempt hash",
    ):
        raise ValueError("imported immutable vector attempt hash does not match")
    if _json_mapping(immutable_path, "imported immutable vector attempt") != attempt:
        raise ValueError("imported latest and immutable vector attempts do not match")

    receipt_path = _validate_source_record(
        project_root,
        receipt_record,
        label="imported vector receipt",
    )
    csv_path = _validate_source_record(
        project_root,
        outputs.get("benchmark_csv"),
        label="imported vector benchmark CSV",
    )
    metadata_path = _validate_source_record(
        project_root,
        outputs.get("benchmark_metadata"),
        label="imported vector benchmark metadata",
    )
    if receipt_path != receipt_path_for(csv_path):
        raise ValueError("imported vector receipt path does not correspond to the benchmark CSV")
    if metadata_path != csv_path.with_suffix(".json"):
        raise ValueError("imported vector metadata path does not correspond to the benchmark CSV")

    benchmark = validate_benchmark_candidate(csv_path, latest_attempt=attempt)
    imported_receipt = benchmark.receipt
    for key in ("slurm_job_id", "hostname", "runtime", "parameters", "result"):
        if receipt_record.get(key) != imported_receipt.get(key):
            raise ValueError(f"imported vector receipt record {key} does not match its source receipt")
    if receipt_path != benchmark.receipt_path or metadata_path != benchmark.metadata_path:
        raise ValueError("imported vector source paths do not match the validated benchmark")

    accepted_result = import_value.get("accepted_result")
    if not isinstance(accepted_result, dict):
        raise ValueError("calibration evidence import has no accepted_result object")
    try:
        accepted_num_envs = int(accepted_result.get("selected_num_envs"))
    except (TypeError, ValueError) as exc:
        raise ValueError("calibration evidence import selected_num_envs is invalid") from exc
    if accepted_num_envs != benchmark.selected_num_envs:
        raise ValueError("calibration evidence import selected_num_envs does not match the benchmark")
    stage_result = receipt_record.get("result")
    if (
        not isinstance(stage_result, dict)
        or stage_result.get("selected_num_envs") != benchmark.selected_num_envs
    ):
        raise ValueError("calibration evidence import vector result does not match the benchmark")

    return MaterializationEvidence(
        mode="imported_v1_calibration",
        target_campaign_id=target_campaign_id,
        source_campaign_id=source_campaign_id,
        attempt_path=attempt_path,
        attempt=attempt,
        benchmark=benchmark,
        additional_sources={
            "immutable_attempt": immutable_path,
            "calibration_import": import_path,
            "calibration_import_sidecar": sidecar_path,
        },
        target_attempt_id=None,
        target_slurm_job_id=None,
    )


def _materialization_evidence(
    project_root: Path,
    campaign_dir: Path,
    benchmark_glob: str,
    *,
    definition: CampaignDefinition | None,
    base_config_path: Path,
) -> MaterializationEvidence:
    native_attempt_path = campaign_dir / "attempts" / "vector_benchmark_latest.json"
    import_path = campaign_dir / CALIBRATION_IMPORT
    current_native_receipts: list[Path] = []
    if native_attempt_path.is_file():
        try:
            attempt_value = _json_mapping(
                native_attempt_path,
                "latest vector_benchmark attempt",
            )
        except ValueError:
            attempt_value = {}
        attempt_id = str(attempt_value.get("attempt_id", ""))
        if attempt_id:
            for candidate in project_root.glob(
                "reports/vectorized_training_benchmark_*[0-9].csv.receipt.json"
            ):
                try:
                    value = _json_mapping(candidate, "vector benchmark receipt")
                except ValueError:
                    continue
                if value.get("campaign_attempt_id") == attempt_id:
                    current_native_receipts.append(candidate.resolve())
    if current_native_receipts and import_path.is_file():
        raise ValueError(
            "ambiguous vector calibration evidence: both a native target receipt and "
            "a V1 calibration import exist"
        )
    if current_native_receipts:
        attempt_path, attempt = _latest_attempt(project_root, campaign_dir)
        benchmark = _newest_valid_benchmark(project_root, attempt, benchmark_glob)
        campaign_id = str(attempt["campaign_id"])
        if campaign_id != campaign_dir.name:
            raise ValueError("native vector attempt campaign_id does not match campaign_dir")
        return MaterializationEvidence(
            mode="native",
            target_campaign_id=campaign_id,
            source_campaign_id=campaign_id,
            attempt_path=attempt_path,
            attempt=attempt,
            benchmark=benchmark,
            additional_sources={},
            target_attempt_id=str(attempt["attempt_id"]),
            target_slurm_job_id=str(benchmark.receipt["slurm_job_id"]),
        )
    if import_path.is_file():
        if definition is None:
            raise ValueError("imported calibration materialization requires a campaign definition")
        return _imported_benchmark(
            project_root,
            campaign_dir,
            import_path,
            definition=definition,
            base_config_path=base_config_path,
        )
    if native_attempt_path.is_file():
        attempt_path, attempt = _latest_attempt(project_root, campaign_dir)
        benchmark = _newest_valid_benchmark(project_root, attempt, benchmark_glob)
        campaign_id = str(attempt["campaign_id"])
        return MaterializationEvidence(
            mode="native",
            target_campaign_id=campaign_id,
            source_campaign_id=campaign_id,
            attempt_path=attempt_path,
            attempt=attempt,
            benchmark=benchmark,
            additional_sources={},
            target_attempt_id=str(attempt["attempt_id"]),
            target_slurm_job_id=str(benchmark.receipt["slurm_job_id"]),
        )
    raise FileNotFoundError(
        "vector calibration evidence is missing: expected either "
        f"{native_attempt_path} or {import_path}"
    )


def materialize_calibrated_screening(
    *,
    project_root: str | Path,
    campaign_dir: str | Path,
    base_config: str | Path,
    output: str | Path,
    benchmark_glob: str = BENCHMARK_GLOB,
    spec_path: str | Path = "config/research_campaign.yaml",
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {root}")
    campaign = _safe_project_path(
        root,
        campaign_dir,
        label="campaign directory",
        must_exist=False,
    )
    if not campaign.is_dir():
        raise FileNotFoundError(f"campaign directory does not exist: {campaign}")
    base_path = _safe_project_path(root, base_config, label="base screening config", must_exist=True)
    output_path = _safe_project_path(root, output, label="locked screening config", must_exist=False)
    if not output_path.is_relative_to(campaign):
        raise ValueError("locked screening config must stay inside the campaign directory")
    if output_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("locked screening config must be a YAML file")

    definition: CampaignDefinition | None = None
    if (campaign / CALIBRATION_IMPORT).is_file():
        resolved_spec = _safe_project_path(
            root,
            spec_path,
            label="research campaign specification",
            must_exist=True,
        )
        spec = load_config(resolved_spec)
        definition = CampaignDefinition.from_mapping(
            spec,
            project_root=root,
            campaign_dir=campaign,
            spec_path=resolved_spec,
        )
        if definition.campaign_id != campaign.name:
            raise ValueError("campaign specification id does not match campaign_dir")
    evidence = _materialization_evidence(
        root,
        campaign,
        benchmark_glob,
        definition=definition,
        base_config_path=base_path,
    )
    attempt_path = evidence.attempt_path
    attempt = evidence.attempt
    benchmark = evidence.benchmark
    try:
        config = load_config(base_path)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"base screening config is not valid YAML: {base_path}") from exc
    training = config.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("base screening config training section must be a mapping")
    base_resolved_sha256 = canonical_sha256(config)
    previous_num_envs = training.get("num_envs")
    training["num_envs"] = benchmark.selected_num_envs

    yaml_text = yaml.safe_dump(config, sort_keys=False)
    atomic_text(output_path, yaml_text)
    atomic_text(
        output_path.with_suffix(output_path.suffix + ".sha256"),
        sha256_file(output_path) + "\n",
    )
    decision_path = output_path.with_suffix(".decision.json")
    config_sources = _config_closure(root, base_path)
    source_paths = {
        "base_config": base_path,
        "benchmark_csv": benchmark.csv_path,
        "benchmark_metadata": benchmark.metadata_path,
        "benchmark_receipt": benchmark.receipt_path,
        "latest_attempt": attempt_path,
    }
    for index, config_source in enumerate(config_sources[1:], start=1):
        source_paths[f"base_config_parent_{index}"] = config_source
    source_paths.update(evidence.additional_sources)
    decision = {
        "schema_version": 2,
        "decision_kind": "vectorized_training_num_envs",
        "status": "success",
        "created_at_utc": _utc_now(),
        "campaign_id": evidence.target_campaign_id,
        "campaign_attempt_id": evidence.target_attempt_id,
        "slurm_job_id": evidence.target_slurm_job_id,
        "evidence_mode": evidence.mode,
        "target_campaign_id": evidence.target_campaign_id,
        "source_campaign_id": evidence.source_campaign_id,
        "source_campaign_attempt_id": attempt["attempt_id"],
        "source_slurm_job_id": benchmark.receipt["slurm_job_id"],
        "calibration_source": {
            "campaign_id": evidence.source_campaign_id,
            "campaign_attempt_id": attempt["attempt_id"],
            "slurm_job_id": benchmark.receipt["slurm_job_id"],
        },
        "resolved_base_config_sha256": base_resolved_sha256,
        "resolved_locked_config_sha256": canonical_sha256(config),
        "calibration_source": {
            "campaign_id": evidence.source_campaign_id,
            "campaign_attempt_id": attempt["attempt_id"],
            "slurm_job_id": benchmark.receipt["slurm_job_id"],
        },
        "selection": {
            "selected_num_envs": benchmark.selected_num_envs,
            "previous_num_envs": previous_num_envs,
            "selection_rule": SELECTION_RULE,
            "selection_fraction_of_max": SELECTION_FRACTION_OF_MAX,
            "throughput_statistic": "median_samples_per_second",
            "median_samples_per_second": {
                str(num_envs): throughput
                for num_envs, throughput in sorted(benchmark.median_throughput.items())
            },
        },
        "sources": {
            name: _source_record(root, path) for name, path in source_paths.items()
        },
        "output": _source_record(root, output_path),
    }
    atomic_json(decision_path, decision)
    atomic_text(
        decision_path.with_suffix(decision_path.suffix + ".sha256"),
        sha256_file(decision_path) + "\n",
    )
    return decision
