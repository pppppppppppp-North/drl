from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tarfile
import tempfile
from collections import Counter
from functools import lru_cache
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from src.campaigns.model import CampaignDefinition, CampaignStage, StageState


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=4096)
def _sha256_cached(path_string: str, modified_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    path = Path(path_string)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class RequirementResult:
    name: str
    passed: bool
    matches: list[str]
    issues: list[str]


@dataclass
class GateResult:
    stage_id: str
    passed: bool
    base_dir: str
    run_dir: str | None
    requirements: list[RequirementResult]
    manifest_expected: int
    manifest_found: int
    manifest_complete: int
    manifest_running: int
    manifest_failed: int
    issues: list[str]
    evidence_files: list[str]
    evidence_fingerprint: str
    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["requirements"] = [asdict(item) for item in self.requirements]
        return value


@dataclass
class StageStatus:
    stage_id: str
    title: str
    state: StageState
    reason: str
    dependencies: list[str]
    run_dir: str | None
    progress: str
    gate: GateResult

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        value["gate"] = self.gate.to_dict()
        return value


def _safe_matches(root: Path, base: Path, patterns: tuple[str, ...]) -> list[Path]:
    matches: set[Path] = set()
    for pattern in patterns:
        for path in base.glob(pattern):
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"artifact path escapes project root: {path}")
            if resolved.is_file():
                matches.add(resolved)
    return sorted(matches)


def latest_run_dir(definition: CampaignDefinition, stage: CampaignStage) -> Path | None:
    source = stage
    if stage.run_source:
        source = definition.stage_map[stage.run_source]
    if not source.run_glob:
        return None
    candidates = []
    for path in definition.project_root.glob(source.run_glob):
        resolved = path.resolve()
        if resolved.is_dir() and resolved.is_relative_to(definition.project_root):
            candidates.append(resolved)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def _capacity_validator(path: Path, options: dict[str, Any]) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    issues = []
    for command in ("mycredit", "myquota", "sinfo", "squeue"):
        marker = f"## {command}"
        start = text.find(marker)
        if start < 0:
            issues.append(f"capacity snapshot is missing {command}")
            continue
        end = text.find("\n## ", start + len(marker))
        section = text[start : end if end >= 0 else len(text)]
        if "unavailable" in section.lower() or len(section.splitlines()) < 2:
            issues.append(f"capacity snapshot has no usable {command} result")
    required_fragments = options.get("required_fragments", [])
    if not isinstance(required_fragments, list):
        issues.append("capacity required_fragments option must be a list")
    else:
        for fragment in required_fragments:
            value = str(fragment)
            if value not in text:
                issues.append(f"capacity snapshot is missing live policy evidence: {value}")
    return issues


def _benchmark_validator(path: Path, options: dict[str, Any]) -> list[str]:
    expected = {int(value) for value in options.get("env_counts", [1, 2, 4, 8, 16])}
    try:
        rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    except (OSError, csv.Error) as exc:
        return [f"cannot read vector benchmark: {exc}"]
    issues = []
    repeats = int(options.get("repeats", 1))
    expected_timesteps = options.get("timesteps")
    expected_backend = options.get("backend")
    expected_seed = options.get("seed")
    require_observed = bool(options.get("observed", False))
    counts = [int(row.get("num_envs", 0)) for row in rows]
    count_frequency = Counter(counts)
    if set(counts) != expected or any(count_frequency[value] != repeats for value in expected):
        issues.append(
            f"vector benchmark must contain exactly {repeats} repeat(s) for env counts {sorted(expected)}"
        )
    identities: list[tuple[int, int]] = []
    data_hashes: set[str] = set()
    config_hashes: set[str] = set()
    config_file_hashes: set[str] = set()
    code_hashes: set[str] = set()
    observed_dimensions: set[tuple[int, int, int]] = set()
    observed_hosts: set[str] = set()
    observed_processes: set[int] = set()
    for row in rows:
        try:
            num_envs = int(row["num_envs"])
            repeat = int(row.get("repeat", 1))
            requested = float(row["requested_timesteps"])
            completed = float(row["completed_timesteps"])
            throughput = float(row["samples_per_second"])
            memory = float(row["peak_rss_mib"])
            elapsed = float(row.get("elapsed_seconds", 0)) if require_observed else 1.0
        except (KeyError, TypeError, ValueError):
            issues.append("vector benchmark contains malformed numeric fields")
            continue
        identities.append((num_envs, repeat))
        if not all(math.isfinite(value) for value in (requested, completed, throughput, memory, elapsed)):
            issues.append("vector benchmark contains non-finite values")
        elif requested <= 0 or completed < requested or throughput <= 0 or memory <= 0 or elapsed <= 0:
            issues.append("vector benchmark contains incomplete or nonpositive measurements")
        elif require_observed and not math.isclose(
            throughput, completed / elapsed, rel_tol=1e-6, abs_tol=1e-6
        ):
            issues.append("vector benchmark throughput is inconsistent with timesteps and elapsed time")
        if repeat < 1 or repeat > repeats:
            issues.append("vector benchmark repeat indices are outside the declared range")
        if expected_timesteps is not None and requested != float(expected_timesteps):
            issues.append(f"vector benchmark requested_timesteps must equal {expected_timesteps}")
        if options.get("exact_completed") and completed != requested:
            issues.append("vector benchmark completed_timesteps must exactly equal requested_timesteps")
        if expected_backend is not None:
            expected_row_backend = "monitor" if num_envs == 1 else str(expected_backend)
            if str(row.get("backend", "")) != expected_row_backend:
                issues.append(f"vector benchmark backend must be {expected_row_backend} for {num_envs} env(s)")
        if expected_seed is not None and int(row.get("seed", -1)) != int(expected_seed):
            issues.append(f"vector benchmark seed must equal {expected_seed}")
        if require_observed:
            source = str(row.get("data_source", "")).strip().lower()
            expected_source = options.get("data_source")
            if not source or source in {"synthetic", "generated", "synthetic_unit_fixture"}:
                issues.append("vector benchmark must use observed, non-generated data")
            if expected_source is not None and source != str(expected_source).strip().lower():
                issues.append(f"vector benchmark data_source must equal {expected_source}")
            for field, collection in (
                ("data_sha256", data_hashes),
                ("config_sha256", config_hashes),
                ("config_file_sha256", config_file_hashes),
                ("code_sha256", code_hashes),
            ):
                value = str(row.get(field, ""))
                if not _is_sha256(value):
                    issues.append(f"vector benchmark contains an invalid {field}")
                else:
                    collection.add(value)
            for field in ("feature_rows", "feature_dates", "feature_tickers"):
                try:
                    if int(row.get(field, 0)) <= 0:
                        issues.append(f"vector benchmark contains a nonpositive {field}")
                except (TypeError, ValueError):
                    issues.append(f"vector benchmark contains a malformed {field}")
            try:
                observed_dimensions.add(
                    (
                        int(row.get("feature_rows", 0)),
                        int(row.get("feature_dates", 0)),
                        int(row.get("feature_tickers", 0)),
                    )
                )
                process_id = int(row.get("process_id", 0))
                if process_id <= 0:
                    issues.append("vector benchmark contains a nonpositive process id")
                observed_processes.add(process_id)
            except (TypeError, ValueError):
                issues.append("vector benchmark contains malformed dimensions or process id")
            hostname = str(row.get("hostname", ""))
            observed_hosts.add(hostname)
            hostname_prefix = str(options.get("hostname_prefix", ""))
            if hostname_prefix and not hostname.startswith(hostname_prefix):
                issues.append(f"vector benchmark hostname must begin with {hostname_prefix!r}")
            if str(row.get("training_split", "")) != str(options.get("training_split", "train")):
                issues.append("vector benchmark did not use the declared training split")
            expected_rss = options.get("rss_measurement")
            if expected_rss is not None and row.get("rss_measurement") != expected_rss:
                issues.append(f"vector benchmark RSS measurement must equal {expected_rss}")
            try:
                self_rss = float(row.get("peak_self_rss_mib", 0))
                child_rss = float(row.get("peak_child_rss_mib", -1))
                if not math.isfinite(self_rss) or self_rss <= 0:
                    issues.append("vector benchmark self RSS must be finite and positive")
                if not math.isfinite(child_rss) or child_rss < 0:
                    issues.append("vector benchmark child RSS must be finite and nonnegative")
            except (TypeError, ValueError):
                issues.append("vector benchmark contains malformed component RSS values")
            for field in (
                "python_version",
                "numpy_version",
                "pandas_version",
                "gymnasium_version",
                "stable_baselines3_version",
                "torch_version",
            ):
                if not str(row.get(field, "")).strip() or str(row.get(field)).lower() == "none":
                    issues.append(f"vector benchmark is missing runtime version {field}")
    if len(identities) != len(set(identities)):
        issues.append("vector benchmark contains duplicate environment-count/repeat trials")
    if require_observed and any(
        len(values) != 1
        for values in (data_hashes, config_hashes, config_file_hashes, code_hashes)
    ):
        issues.append("vector benchmark trials disagree on data, configuration, or code provenance")
    if require_observed and len(observed_dimensions) != 1:
        issues.append("vector benchmark trials disagree on observed feature dimensions")
    if require_observed and len(observed_hosts) != 1:
        issues.append("vector benchmark trials ran on multiple hosts")
    if require_observed and options.get("distinct_processes", True) and len(observed_processes) != len(rows):
        issues.append("vector benchmark trials were not executed in distinct processes")
    return sorted(set(issues))


def _vector_metadata_validator(path: Path, options: dict[str, Any]) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read vector benchmark metadata: {exc}"]
    if not isinstance(value, dict):
        return ["vector benchmark metadata must be a JSON object"]
    issues: list[str] = []
    expected_counts = sorted(int(item) for item in options.get("env_counts", [1, 2, 4, 8, 16]))
    repeats = int(options.get("repeats", 3))
    fraction = float(options.get("selection_fraction_of_max", 0.95))
    if value.get("schema_version") != 2:
        issues.append("vector benchmark metadata schema must equal 2")
    if int(value.get("repeats_per_env_count", -1)) != repeats:
        issues.append(f"vector benchmark metadata must record {repeats} repeats per environment count")
    if int(value.get("trial_count", -1)) != len(expected_counts) * repeats:
        issues.append("vector benchmark metadata trial_count is inconsistent")
    if value.get("trial_isolation") != "one fresh Python subprocess per environment count and repeat":
        issues.append("vector benchmark trials were not isolated in fresh processes")
    if value.get("throughput_statistic") != "median_samples_per_second":
        issues.append("vector benchmark selection must use median throughput")
    try:
        recorded_fraction = float(value.get("selection_fraction_of_max", -1))
    except (TypeError, ValueError):
        recorded_fraction = -1
    if not math.isclose(recorded_fraction, fraction, rel_tol=0.0, abs_tol=1e-12):
        issues.append(f"vector benchmark selection fraction must equal {fraction}")

    csv_path = path.with_suffix(".csv")
    try:
        rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    except (OSError, csv.Error) as exc:
        return [*issues, f"cannot read companion vector benchmark CSV: {exc}"]
    grouped: dict[int, list[tuple[float, float]]] = {count: [] for count in expected_counts}
    for row in rows:
        try:
            grouped[int(row["num_envs"])].append(
                (float(row["samples_per_second"]), float(row["peak_rss_mib"]))
            )
        except (KeyError, TypeError, ValueError):
            issues.append("companion vector benchmark CSV contains malformed summary fields")
    if any(len(grouped.get(count, [])) != repeats for count in expected_counts):
        issues.append("companion vector benchmark CSV does not contain the declared repeated grid")
    elif grouped:
        medians = {count: median(item[0] for item in grouped[count]) for count in expected_counts}
        peak = max(medians.values())
        selected = min(count for count, throughput in medians.items() if throughput >= peak * fraction)
        if int(value.get("selected_num_envs", -1)) != selected:
            issues.append("selected_num_envs does not follow the declared 95%-of-peak rule")
        summaries = value.get("summary_by_env_count", [])
        if not isinstance(summaries, list) or len(summaries) != len(expected_counts):
            issues.append("vector benchmark metadata summary grid is incomplete")
        else:
            by_count = {
                int(row.get("num_envs", -1)): row for row in summaries if isinstance(row, dict)
            }
            if sorted(by_count) != expected_counts:
                issues.append("vector benchmark metadata summary environment counts are inconsistent")
            for count in expected_counts:
                summary = by_count.get(count, {})
                if int(summary.get("repeats", -1)) != repeats:
                    issues.append(f"metadata summary repeat count is wrong for {count} env(s)")
                    continue
                expected_values = {
                    "median_samples_per_second": medians[count],
                    "min_samples_per_second": min(item[0] for item in grouped[count]),
                    "max_samples_per_second": max(item[0] for item in grouped[count]),
                    "max_peak_rss_mib": max(item[1] for item in grouped[count]),
                }
                for field, expected_value in expected_values.items():
                    try:
                        actual_value = float(summary.get(field))
                    except (TypeError, ValueError):
                        issues.append(f"metadata summary has malformed {field} for {count} env(s)")
                        continue
                    if not math.isclose(actual_value, expected_value, rel_tol=1e-9, abs_tol=1e-9):
                        issues.append(f"metadata summary {field} does not match the CSV for {count} env(s)")

    provenance = value.get("data_provenance", {})
    if not isinstance(provenance, dict) or provenance.get("synthetic") is not False:
        issues.append("vector benchmark metadata must identify observed, non-synthetic data")
    elif str(provenance.get("source", "")).strip().lower() in {"", "synthetic", "generated"}:
        issues.append("vector benchmark metadata has an invalid observed-data source")
    for field in ("data_sha256", "config_sha256", "config_file_sha256"):
        if not isinstance(provenance, dict) or not _is_sha256(provenance.get(field, "")):
            issues.append(f"vector benchmark metadata contains an invalid {field}")
    code = value.get("code_provenance", {})
    if not isinstance(code, dict) or not _is_sha256(code.get("sha256", "")):
        issues.append("vector benchmark metadata contains invalid code provenance")
    runtime = value.get("runtime", {})
    for field in ("python", "numpy", "pandas", "gymnasium", "stable_baselines3", "torch"):
        if not isinstance(runtime, dict) or not str(runtime.get(field, "")).strip() or runtime.get(field) is None:
            issues.append(f"vector benchmark metadata is missing runtime version {field}")
    if Path(str(value.get("results_file", ""))).name != csv_path.name:
        issues.append("vector benchmark metadata points to a different results file")
    return sorted(set(issues))


def _determinism_validator(path: Path, options: dict[str, Any]) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read determinism result: {exc}"]
    first = str(value.get("first_sha256", ""))
    second = str(value.get("second_sha256", ""))
    issues = []
    if value.get("deterministic") is not True:
        issues.append("determinism result is not true")
    if len(first) != 64 or first != second:
        issues.append("determinism trajectory hashes do not match")
    if int(value.get("steps", 0)) <= 0:
        issues.append("determinism trajectory has no steps")
    if options.get("observed"):
        if value.get("schema_version") != 2:
            issues.append("determinism result schema must equal 2")
        if int(value.get("seed", -1)) != int(options.get("seed", 42)):
            issues.append(f"determinism seed must equal {options.get('seed', 42)}")
        if value.get("comparison_method") != "two_independent_subprocess_random_action_trajectories":
            issues.append("determinism comparison did not use two independent subprocesses")
        if value.get("random_action_distribution") != "numpy.default_rng(seed).normal":
            issues.append("determinism action distribution does not match the declared protocol")
        if int(value.get("independent_subprocess_runs", 0)) != 2:
            issues.append("determinism result must contain exactly two subprocess runs")
        process_ids = value.get("process_ids", [])
        try:
            valid_process_ids = (
                isinstance(process_ids, list)
                and len(process_ids) == 2
                and all(int(item) > 0 for item in process_ids)
                and len({int(item) for item in process_ids}) == 2
            )
        except (TypeError, ValueError):
            valid_process_ids = False
        if not valid_process_ids:
            issues.append("determinism subprocess identities are missing or not independent")
        if (
            int(value.get("first_steps", -1)) != int(value.get("steps", 0))
            or int(value.get("second_steps", -1)) != int(value.get("steps", 0))
        ):
            issues.append("determinism subprocess step counts do not match")
        if value.get("provenance_matches") is not True or value.get("runtime_matches") is not True:
            issues.append("determinism subprocess provenance or runtime differs")
        provenance = value.get("data_provenance", {})
        if not isinstance(provenance, dict) or provenance.get("synthetic") is not False:
            issues.append("determinism result must use observed, non-synthetic data")
        elif str(provenance.get("source", "")).strip().lower() in {"", "synthetic", "generated"}:
            issues.append("determinism result has an invalid observed-data source")
        elif options.get("data_source") is not None and provenance.get("source") != options.get(
            "data_source"
        ):
            issues.append(f"determinism data source must equal {options.get('data_source')}")
        for field in ("data_sha256", "config_sha256", "config_file_sha256", "code_sha256"):
            if not isinstance(provenance, dict) or not _is_sha256(provenance.get(field, "")):
                issues.append(f"determinism result contains an invalid {field}")
        runs = value.get("run_provenance", [])
        if not isinstance(runs, list) or len(runs) != 2:
            issues.append("determinism result is missing two run-provenance records")
        else:
            for index, run in enumerate(runs):
                if not isinstance(run, dict) or run.get("trajectory_sha256") != first:
                    issues.append("determinism run provenance does not match the trajectory hash")
                if not isinstance(run, dict) or int(run.get("steps", 0)) != int(value.get("steps", 0)):
                    issues.append("determinism run provenance has inconsistent step counts")
                if isinstance(run, dict):
                    if int(run.get("seed", -1)) != int(options.get("seed", 42)):
                        issues.append("determinism run provenance has the wrong seed")
                    if valid_process_ids and int(run.get("process_id", -1)) != int(process_ids[index]):
                        issues.append("determinism run provenance has the wrong process id")
                    for field in (
                        "data_sha256",
                        "config_sha256",
                        "config_file_sha256",
                        "code_sha256",
                    ):
                        if run.get(field) != provenance.get(field):
                            issues.append(f"determinism run provenance has inconsistent {field}")
        runtime = value.get("runtime", {})
        for field in ("python", "numpy", "pandas", "gymnasium", "stable_baselines3", "torch"):
            if not isinstance(runtime, dict) or not str(runtime.get(field, "")).strip() or runtime.get(field) is None:
                issues.append(f"determinism result is missing runtime version {field}")
    return issues


def _selection_validator(path: Path, options: dict[str, Any]) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read selection result: {exc}"]
    algorithms = value.get("selected_algorithms", [])
    maximum = int(options.get("maximum", 2))
    issues = []
    if value.get("selection_split") != "validation":
        issues.append("algorithm selection is not validation-only")
    if not isinstance(algorithms, list) or not algorithms or len(algorithms) > maximum:
        issues.append(f"algorithm selection must contain between 1 and {maximum} algorithms")
    if len(algorithms) != len(set(algorithms)):
        issues.append("algorithm selection contains duplicates")
    return issues


def _budget_validator(path: Path, options: dict[str, Any]) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read budget selection: {exc}"]
    allowed = {int(item) for item in options.get("allowed", [20000, 100000, 250000, 500000])}
    issues = []
    if value.get("selection_split") != "validation":
        issues.append("training-budget selection is not validation-only")
    if int(value.get("selected_training_budget", -1)) not in allowed:
        issues.append(f"selected training budget is not one of {sorted(allowed)}")
    return issues


def _stress_validator(path: Path, _: dict[str, Any]) -> list[str]:
    try:
        rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    except (OSError, csv.Error) as exc:
        return [f"cannot read stress readiness: {exc}"]
    if not rows:
        return ["stress test has no locked models"]
    if any(str(row.get("ready", "")).strip().lower() != "true" for row in rows):
        return ["one or more stress-test models are not ready"]
    for field in (
        "hash_matches",
        "feature_schema_matches",
        "feature_data_matches",
        "protocol_matches",
        "source_curve_matches",
    ):
        if any(str(row.get(field, "")).strip().lower() != "true" for row in rows):
            return [f"one or more stress-test models failed {field}"]
    return []


def _stress_completeness_validator(path: Path, _: dict[str, Any]) -> list[str]:
    try:
        rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    except (OSError, csv.Error) as exc:
        return [f"cannot read stress completeness evidence: {exc}"]
    if not rows:
        return ["stress completeness evidence has no locked-model rows"]
    identities = [
        (row.get("model_id"), row.get("algorithm"), row.get("window"), row.get("seed"), row.get("scenario"))
        for row in rows
    ]
    issues = []
    if len(identities) != len(set(identities)):
        issues.append("stress completeness evidence contains duplicate model-window-scenario rows")
    if any(str(row.get("present", "")).strip().lower() != "true" for row in rows):
        issues.append("stress completeness evidence contains a missing evaluation")
    return issues


def _stress_replay_validator(path: Path, _: dict[str, Any]) -> list[str]:
    try:
        rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    except (OSError, csv.Error) as exc:
        return [f"cannot read base-scenario replay evidence: {exc}"]
    if not rows:
        return ["base-scenario replay evidence has no locked-model rows"]
    issues = []
    identities = [(row.get("model_id"), row.get("window"), row.get("seed")) for row in rows]
    if len(identities) != len(set(identities)):
        issues.append("base-scenario replay evidence contains duplicate model-window-seed rows")
    for row in rows:
        try:
            difference = float(row["max_abs_portfolio_difference"])
            tolerance = float(row["tolerance"])
        except (KeyError, TypeError, ValueError):
            issues.append("base-scenario replay evidence contains malformed numeric fields")
            continue
        if (
            str(row.get("dates_match", "")).strip().lower() != "true"
            or str(row.get("curve_match", "")).strip().lower() != "true"
            or not math.isfinite(difference)
            or not math.isfinite(tolerance)
            or tolerance < 0
            or difference > tolerance
        ):
            issues.append("one or more base-scenario replays failed exact-window verification")
    return sorted(set(issues))


def _stress_config_validator(path: Path, _: dict[str, Any]) -> list[str]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot read locked stress configuration: {exc}"]
    stress = value.get("execution_stress", {})
    models = stress.get("model_paths", [])
    issues = []
    if stress.get("require_models") is not True:
        issues.append("locked stress configuration must set require_models: true")
    if not isinstance(models, list) or not models:
        issues.append("locked stress configuration must contain at least one model")
    if not stress.get("model_registry"):
        issues.append("locked stress configuration must reference its model registry")
    if not value.get("walk_forward"):
        issues.append("locked stress configuration must contain walk_forward boundaries")
    if not _is_sha256(stress.get("protocol_lock_hash", "")):
        issues.append("locked stress configuration must contain a valid protocol hash")
    for model in models if isinstance(models, list) else []:
        if not isinstance(model, dict):
            issues.append("locked stress configuration contains a malformed model entry")
            continue
        for field in (
            "sha256",
            "source_test_curve_sha256",
            "feature_schema_sha256",
            "feature_data_sha256",
            "protocol_lock_hash",
        ):
            if not _is_sha256(model.get(field, "")):
                issues.append(f"locked stress model is missing a valid {field}")
        if not model.get("source_test_curve_path"):
            issues.append("locked stress model is missing source robustness evidence")
    return issues


def _model_registry_validator(
    definition: CampaignDefinition,
    path: Path,
) -> tuple[list[str], list[Path]]:
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read model registry: {exc}"], []
    if not isinstance(registry, dict):
        return ["model registry must be a JSON object"], []
    lock_valid, lock_hash, lock_issues = protocol_lock_status(definition)
    if not lock_valid or not lock_hash:
        return list(lock_issues), []
    issues: list[str] = []
    if registry.get("campaign_id") != definition.campaign_id:
        issues.append("model registry campaign id does not match")
    if registry.get("protocol_lock_hash") != lock_hash:
        issues.append("model registry protocol hash does not match")
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        issues.append("model registry SHA-256 sidecar does not match")
    models = registry.get("models", [])
    if not isinstance(models, list) or not models:
        return [*issues, "model registry contains no models"], []
    referenced: list[Path] = []
    identities: list[tuple[str, int, int]] = []
    schema_hashes: set[str] = set()
    data_hashes: set[str] = set()
    expected_seed = int(definition.decisions.get("stress_model_seed", 42))
    try:
        selected_seed = int(registry.get("selected_seed", -1))
    except (TypeError, ValueError):
        selected_seed = -1
        issues.append("model registry selected_seed is malformed")
    if selected_seed != expected_seed:
        issues.append("model registry seed does not match the predeclared campaign stress seed")
    for entry in models:
        if not isinstance(entry, dict):
            issues.append("model registry contains a malformed model entry")
            continue
        try:
            identity = (
                str(entry["algorithm"]).lower(),
                int(entry["window"]),
                int(entry["seed"]),
            )
        except (KeyError, TypeError, ValueError):
            issues.append("model registry contains a malformed identity")
            continue
        identities.append(identity)
        if identity[2] != selected_seed:
            issues.append(f"model registry seed drift for {identity}")
        if entry.get("protocol_lock_hash") != lock_hash:
            issues.append(f"model registry entry protocol mismatch for {identity}")
        model_path = (definition.project_root / str(entry.get("path", ""))).resolve()
        manifest_path = (definition.project_root / str(entry.get("manifest_path", ""))).resolve()
        for referenced_path, label, expected_hash in (
            (model_path, "model", entry.get("sha256")),
            (manifest_path, "manifest", entry.get("manifest_sha256")),
            (
                (definition.project_root / str(entry.get("resolved_config_path", ""))).resolve(),
                "resolved config",
                entry.get("resolved_config_sha256"),
            ),
            (
                (definition.project_root / str(entry.get("source_test_curve_path", ""))).resolve(),
                "source test curve",
                entry.get("source_test_curve_sha256"),
            ),
            (
                (definition.project_root / str(entry.get("source_metrics_path", ""))).resolve(),
                "source metrics",
                entry.get("source_metrics_sha256"),
            ),
        ):
            if not referenced_path.is_relative_to(definition.project_root) or not referenced_path.is_file():
                issues.append(f"registered {label} is missing or outside the project: {referenced_path}")
            else:
                referenced.append(referenced_path)
                if not _is_sha256(expected_hash) or sha256_file(referenced_path) != expected_hash:
                    issues.append(f"registered {label} SHA-256 mismatch: {referenced_path}")
        if manifest_path.is_file() and manifest_path.is_relative_to(definition.project_root):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise ValueError("manifest must be a JSON object")
                manifest_identity = manifest.get("identity", {})
                if not isinstance(manifest_identity, dict):
                    raise ValueError("manifest identity must be a JSON object")
                actual = (
                    str(manifest_identity.get("algorithm", "")).lower(),
                    int(manifest_identity.get("window", -1)),
                    int(manifest_identity.get("seed", -1)),
                )
                if manifest.get("status") != "complete" or actual != identity:
                    issues.append(f"registered manifest status or identity mismatch: {manifest_path}")
                if manifest.get("campaign_id") != definition.campaign_id:
                    issues.append(f"registered manifest campaign mismatch: {manifest_path}")
                if manifest.get("protocol_lock_hash") != lock_hash:
                    issues.append(f"registered manifest protocol mismatch: {manifest_path}")
                data_files = manifest.get("data_files", {})
                if not isinstance(data_files, dict) or not isinstance(data_files.get("features", {}), dict):
                    raise ValueError("manifest feature-data record is malformed")
                manifest_data_hash = str(data_files.get("features", {}).get("sha256", ""))
                if manifest_data_hash != entry.get("feature_data_sha256"):
                    issues.append(f"registered feature data does not match its manifest: {manifest_path}")
            except (json.JSONDecodeError, TypeError, ValueError):
                issues.append(f"registered manifest is malformed: {manifest_path}")
        schema_hash = str(entry.get("feature_schema_sha256", ""))
        data_hash = str(entry.get("feature_data_sha256", ""))
        if not _is_sha256(schema_hash):
            issues.append(f"registered feature schema hash is invalid for {identity}")
        if not _is_sha256(data_hash):
            issues.append(f"registered feature data hash is invalid for {identity}")
        schema_hashes.add(schema_hash)
        data_hashes.add(data_hash)
    if len(identities) != len(set(identities)):
        issues.append("model registry contains duplicate algorithm/window/seed identities")
    try:
        lock = json.loads(definition.protocol_lock_path.read_text(encoding="utf-8"))
        if not isinstance(lock, dict) or not isinstance(lock.get("inputs", []), list):
            raise ValueError("protocol lock does not contain an inputs list")
        locked_records = [
            entry
            for entry in lock.get("inputs", [])
            if isinstance(entry, dict) and entry.get("name") == "screening_selection"
        ]
        if len(locked_records) != 1:
            raise ValueError("protocol lock must contain exactly one screening_selection input")
        locked_record = locked_records[0]
        selection_path = (definition.project_root / str(locked_record.get("path", ""))).resolve()
        selection_hash = str(locked_record.get("sha256", ""))
        if (
            not selection_path.is_relative_to(definition.project_root)
            or not selection_path.is_file()
            or not _is_sha256(selection_hash)
            or sha256_file(selection_path) != selection_hash
        ):
            raise ValueError("locked screening-selection source is missing or changed")
        if registry.get("screening_selection_path") != str(selection_path.relative_to(definition.project_root)):
            issues.append("model registry does not reference the protocol-locked screening selection")
        if registry.get("screening_selection_sha256") != selection_hash:
            issues.append("model registry screening-selection hash does not match the protocol lock")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if not isinstance(selection, dict):
            raise ValueError("locked screening selection must be a JSON object")
        algorithms = selection.get("selected_algorithms", [])
        if selection.get("selection_split") != "validation" or not isinstance(algorithms, list):
            raise ValueError("locked screening selection is not a validation-only algorithm list")
        expected = {
            (str(algorithm).lower(), int(window), expected_seed)
            for algorithm in algorithms
            for window in definition.decisions.get("window_ids", [0, 1])
        }
        if set(identities) != expected:
            issues.append("model registry does not cover the exact locked algorithm/window/seed grid")
        referenced.append(selection_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        issues.append(f"cannot validate locked screening selection: {exc}")
    if len(schema_hashes) != 1 or registry.get("feature_schema_sha256") not in schema_hashes:
        issues.append("model registry feature schema is inconsistent")
    if len(data_hashes) != 1 or registry.get("feature_data_sha256") not in data_hashes:
        issues.append("model registry feature data is inconsistent")
    return issues, referenced


def _market_data_readiness_validator(
    definition: CampaignDefinition,
    path: Path,
) -> tuple[list[str], list[Path]]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read market-data readiness receipt: {exc}"], []
    if not isinstance(receipt, dict):
        return ["market-data readiness receipt must be a JSON object"], []
    issues = []
    if receipt.get("status") != "pass" or receipt.get("execution_eligible") is not True:
        issues.append("market-data bundle did not pass its execution-readiness checks")
    if receipt.get("data_origin") != "observed_market":
        issues.append("generated or unknown-origin data cannot satisfy the real-market campaign gate")
    if receipt.get("evidence_class") != "restricted_private_research_pilot":
        issues.append("market-data evidence class is missing or unexpected")
    if receipt.get("publication_claim_eligible") is not False:
        issues.append("restricted public-web data must not be marked publication-claim eligible")
    checks = receipt.get("checks", [])
    if not isinstance(checks, list) or not checks:
        issues.append("market-data receipt contains no quality checks")
    else:
        required_checks = {
            "stock_required_columns",
            "stock_ticker_contract",
            "stock_unique_daily_grain",
            "stock_ohlc_envelope",
            "stock_ohlc_repair_disclosure",
            "stock_temporal_coverage",
            "stock_balanced_panel",
            "index_contract",
            "index_stock_date_overlap",
            "configured_feature_contract",
            "feature_values_valid",
            "feature_stock_key_integrity",
            "feature_leakage_checks",
            "source_manifest_records",
        }
        actual_checks = {
            str(row.get("check")) for row in checks if isinstance(row, dict)
        }
        missing_checks = sorted(required_checks - actual_checks)
        if missing_checks:
            issues.append(f"market-data receipt is missing required checks: {missing_checks}")
        if any(
            not isinstance(row, dict) or row.get("status") == "fail"
            for row in checks
        ):
            issues.append("market-data receipt contains failed or malformed quality checks")
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        issues.append("market-data readiness SHA-256 sidecar does not match")
    artifacts = receipt.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return [*issues, "market-data receipt artifacts must be an object"], []
    referenced: list[Path] = []
    for name in ("stocks", "market_index", "features", "source_manifest"):
        entry = artifacts.get(name, {})
        if not isinstance(entry, dict):
            issues.append(f"market-data receipt has a malformed {name} artifact")
            continue
        artifact_path = (definition.project_root / str(entry.get("path", ""))).resolve()
        expected_hash = entry.get("sha256", "")
        if not artifact_path.is_relative_to(definition.project_root) or not artifact_path.is_file():
            issues.append(f"market-data artifact is missing or outside the project: {artifact_path}")
            continue
        referenced.append(artifact_path)
        if not _is_sha256(expected_hash) or sha256_file(artifact_path) != expected_hash:
            issues.append(f"market-data artifact SHA-256 mismatch: {artifact_path}")
    return issues, referenced


def _archive_validator(path: Path, _: dict[str, Any]) -> list[str]:
    try:
        with tarfile.open(path, "r:gz") as archive:
            member = archive.getmember("REPRODUCIBILITY_MANIFEST.json")
            manifest = json.load(archive.extractfile(member))
            if int(manifest.get("file_count", 0)) <= 0:
                return ["archive manifest contains no files"]
    except (OSError, KeyError, tarfile.TarError, json.JSONDecodeError, TypeError) as exc:
        return [f"reproducibility archive verification failed: {exc}"]
    return []


def _sha256_sidecar_validator(path: Path, _: dict[str, Any]) -> list[str]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        return [f"SHA-256 sidecar is missing: {sidecar.name}"]
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        return [f"SHA-256 sidecar does not match: {sidecar.name}"]
    return []


def _screening_smoke_summary_validator(
    path: Path,
    options: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read screening smoke summary: {exc}"]
    if not isinstance(summary, dict):
        return ["screening smoke summary must be a JSON object"]
    issues.extend(_sha256_sidecar_validator(path, {}))

    expected_algorithms = [
        str(value).lower()
        for value in options.get("algorithms", ["ppo", "a2c", "ddpg", "td3", "sac"])
    ]
    expected_timesteps = int(options.get("timesteps_per_algorithm", 2048))
    expected_trials = {
        str(name).lower(): int(value)
        for name, value in dict(options.get("representative_trials", {})).items()
    }
    required_packages = [
        str(value) for value in options.get("required_runtime_packages", [])
    ]
    runtime_limit_hours = float(options.get("runtime_limit_hours", 72.0))

    if summary.get("schema_version") != 1 or summary.get("status") != "complete":
        issues.append("screening smoke summary is not a complete schema-v1 result")
    if summary.get("smoke_kind") != "observed_data_five_algorithm_screening":
        issues.append("screening smoke summary has the wrong smoke kind")
    if summary.get("algorithms") != expected_algorithms:
        issues.append("screening smoke algorithms do not match the locked five-algorithm grid")
    if summary.get("run_count") != len(expected_algorithms):
        issues.append("screening smoke run count is incomplete")
    if summary.get("complete_manifest_count") != len(expected_algorithms):
        issues.append("screening smoke complete-manifest count is incomplete")
    if summary.get("timesteps_per_algorithm") != expected_timesteps:
        issues.append("screening smoke timesteps do not match the declared fixed budget")
    if summary.get("total_requested_timesteps") != expected_timesteps * len(
        expected_algorithms
    ):
        issues.append("screening smoke total requested timesteps are inconsistent")
    if summary.get("full_screening_eligible") is not True:
        issues.append("screening smoke runtime projection does not permit full screening")
    if not math.isclose(
        float(summary.get("runtime_projection_limit_hours", -1)),
        runtime_limit_hours,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        issues.append("screening smoke runtime limit does not match the campaign gate")

    readiness = summary.get("readiness", {})
    if (
        not isinstance(readiness, dict)
        or readiness.get("data_origin") != "observed_market"
        or readiness.get("execution_eligible") is not True
        or not _is_sha256(readiness.get("sha256"))
    ):
        issues.append("screening smoke readiness is not execution-eligible observed data")
    features = summary.get("observed_features", {})
    if (
        not isinstance(features, dict)
        or str(features.get("data_source", "")).lower() in {"", "synthetic", "generated"}
        or not _is_sha256(features.get("sha256"))
        or int(features.get("bytes", 0)) <= 0
    ):
        issues.append("screening smoke observed-feature provenance is invalid")

    runs = summary.get("runs", [])
    if not isinstance(runs, list) or len(runs) != len(expected_algorithms):
        issues.append("screening smoke summary does not contain exactly five run records")
        return sorted(set(issues))

    first_run_dir = runs[0].get("run_dir") if isinstance(runs[0], dict) else None
    try:
        first_relative = Path(str(first_run_dir))
        if first_relative.is_absolute() or ".." in first_relative.parts:
            raise ValueError
        project_root = path.parent
        for _ in first_relative.parent.parts:
            project_root = project_root.parent
        if (project_root / first_relative).parent != path.parent:
            raise ValueError
    except (TypeError, ValueError):
        issues.append("screening smoke run paths cannot be resolved safely")
        return sorted(set(issues))

    observed_algorithms: list[str] = []
    observed_package_versions: dict[str, set[str]] = {
        name: set() for name in required_packages
    }
    for run in runs:
        if not isinstance(run, dict):
            issues.append("screening smoke contains a malformed run record")
            continue
        algorithm = str(run.get("algorithm", "")).lower()
        observed_algorithms.append(algorithm)
        if algorithm not in expected_algorithms:
            issues.append(f"screening smoke contains unexpected algorithm {algorithm!r}")
            continue
        if expected_trials and run.get("trial_number") != expected_trials.get(algorithm):
            issues.append(
                f"screening smoke {algorithm} representative trial does not match"
            )
        if not isinstance(run.get("sampled_params"), dict) or not run["sampled_params"]:
            issues.append(f"screening smoke {algorithm} has no sampled hyperparameters")

        projection = run.get("runtime_projection", {})
        try:
            elapsed = float(projection["elapsed_seconds"])
            projected_hours = float(projection["projected_task_hours"])
            limit_hours = float(projection["limit_hours"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"screening smoke {algorithm} runtime projection is malformed")
        else:
            if (
                not math.isfinite(elapsed)
                or not math.isfinite(projected_hours)
                or elapsed <= 0
                or projected_hours <= 0
                or not math.isclose(
                    limit_hours,
                    runtime_limit_hours,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or projection.get("within_limit") is not True
                or projected_hours > runtime_limit_hours
            ):
                issues.append(
                    f"screening smoke {algorithm} runtime projection exceeds its limit"
                )

        run_relative = Path(str(run.get("run_dir", "")))
        actual_run_dir = (project_root / run_relative).resolve()
        if (
            run_relative.is_absolute()
            or ".." in run_relative.parts
            or not actual_run_dir.is_relative_to(project_root)
            or actual_run_dir != path.parent / f"algorithm_{algorithm}"
        ):
            issues.append(f"screening smoke {algorithm} run path is unsafe or mismatched")
            continue

        manifest_record = run.get("manifest", {})
        manifest_path = actual_run_dir / "experiment_manifest.json"
        if (
            not isinstance(manifest_record, dict)
            or not manifest_path.is_file()
            or manifest_record.get("bytes") != manifest_path.stat().st_size
            or manifest_record.get("sha256") != sha256_file(manifest_path)
            or manifest_record.get("status") != "complete"
        ):
            issues.append(f"screening smoke {algorithm} manifest record is invalid")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            issues.append(f"screening smoke {algorithm} manifest cannot be read")
            continue
        identity = manifest.get("identity", {}) if isinstance(manifest, dict) else {}
        if (
            manifest.get("status") != "complete"
            or not isinstance(identity, dict)
            or identity.get("algorithm") != algorithm
            or identity.get("smoke_test") is not True
            or identity.get("trial_id") != run.get("trial_id")
            or identity.get("trial_number") != run.get("trial_number")
            or identity.get("seed") != run.get("seed")
        ):
            issues.append(f"screening smoke {algorithm} manifest identity is invalid")
        runtime = manifest.get("runtime", {})
        runtime_packages = runtime.get("packages", {}) if isinstance(runtime, dict) else {}
        for package_name in required_packages:
            version = (
                runtime_packages.get(package_name)
                if isinstance(runtime_packages, dict)
                else None
            )
            if not isinstance(version, str) or not version:
                issues.append(
                    f"screening smoke {algorithm} is missing runtime package "
                    f"{package_name}"
                )
            else:
                observed_package_versions[package_name].add(version)

        artifacts = run.get("artifacts", [])
        if not isinstance(artifacts, list) or not artifacts:
            issues.append(f"screening smoke {algorithm} artifact inventory is empty")
            continue
        for record in artifacts:
            if not isinstance(record, dict):
                issues.append(f"screening smoke {algorithm} has a malformed artifact record")
                continue
            try:
                relative_to_run = Path(str(record.get("path", ""))).relative_to(
                    run_relative
                )
            except ValueError:
                issues.append(f"screening smoke {algorithm} artifact path is unsafe")
                continue
            artifact_path = (actual_run_dir / relative_to_run).resolve()
            if (
                not artifact_path.is_relative_to(actual_run_dir)
                or not artifact_path.is_file()
                or record.get("bytes") != artifact_path.stat().st_size
                or record.get("sha256") != sha256_file(artifact_path)
            ):
                issues.append(
                    f"screening smoke {algorithm} artifact hash or size mismatch"
                )

    if observed_algorithms != expected_algorithms:
        issues.append("screening smoke run order or algorithm coverage is invalid")
    for package_name, versions in observed_package_versions.items():
        if len(versions) != 1:
            issues.append(
                f"screening smoke runtime package {package_name} differs across runs"
            )
    return sorted(set(issues))


def _current_calibration_provenance(
    definition: CampaignDefinition,
    stage: CampaignStage,
) -> dict[str, Any]:
    if not stage.config_path:
        raise ValueError(f"calibration stage {stage.id} has no config_path")
    from src.campaigns.calibration import (
        code_provenance,
        load_observed_inputs,
        training_feature_subset,
    )

    config_path = (
        definition.project_root / definition.expand_project_relative(stage.config_path)
    ).resolve()
    inputs = load_observed_inputs(config_path)
    features, split = training_feature_subset(inputs)
    code_paths = [path for path in stage.fingerprint_paths if Path(path).suffix == ".py"]
    code = code_provenance(definition.project_root, code_paths)
    return {
        "source": inputs.data_source,
        "data_sha256": inputs.data_sha256,
        "config_sha256": inputs.config_sha256,
        "config_file_sha256": inputs.config_file_sha256,
        "code_sha256": code["sha256"],
        "code_files": code["files"],
        "feature_rows": int(len(features)),
        "feature_dates": int(features["date"].nunique()),
        "feature_tickers": int(features["ticker"].nunique()),
        "training_split": split,
    }


def _calibration_receipt_validator(
    definition: CampaignDefinition,
    stage: CampaignStage,
    path: Path,
    options: dict[str, Any],
) -> tuple[list[str], list[Path]]:
    """Validate a returned calibration receipt against the latest rendered attempt.

    Remote absolute paths are intentionally not trusted. Output records are matched
    by basename to files copied into the same local report directory as the receipt,
    while every rendered attempt input is re-hashed from the local source tree.
    """

    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read calibration receipt: {exc}"], []
    if not isinstance(receipt, dict):
        return ["calibration receipt must be a JSON object"], []

    issues: list[str] = []
    referenced: list[Path] = []
    remote_project_root: Path | None = None
    expected_kind = str(options.get("kind", ""))
    if receipt.get("schema_version") != 1:
        issues.append("calibration receipt schema must equal 1")
    if receipt.get("status") != "success" or receipt.get("error"):
        issues.append("calibration receipt does not record an error-free success")
    if expected_kind and receipt.get("calibration_kind") != expected_kind:
        issues.append(f"calibration receipt kind must equal {expected_kind}")
    if receipt.get("campaign_id") != definition.campaign_id:
        issues.append("calibration receipt campaign id does not match")
    job_id = str(receipt.get("slurm_job_id", ""))
    if not job_id.isdigit() or int(job_id) <= 0:
        issues.append("calibration receipt must contain a positive numeric SLURM job id")
    hostname_prefix = str(options.get("hostname_prefix", "bistka-"))
    if hostname_prefix and not str(receipt.get("hostname", "")).startswith(hostname_prefix):
        issues.append(f"calibration receipt hostname must begin with {hostname_prefix!r}")
    try:
        if int(receipt.get("process_id", 0)) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        issues.append("calibration receipt must contain a positive process id")
    try:
        elapsed = float(receipt.get("elapsed_seconds", 0))
    except (TypeError, ValueError):
        elapsed = 0
    if not math.isfinite(elapsed) or elapsed <= 0:
        issues.append("calibration receipt elapsed time must be finite and positive")
    parsed_times: dict[str, datetime] = {}
    for field in ("started_at_utc", "finished_at_utc"):
        try:
            parsed = datetime.fromisoformat(str(receipt.get(field, "")))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError
            parsed_times[field] = parsed
        except ValueError:
            issues.append(f"calibration receipt {field} must be a timezone-aware timestamp")
    if (
        "started_at_utc" in parsed_times
        and "finished_at_utc" in parsed_times
        and parsed_times["finished_at_utc"] < parsed_times["started_at_utc"]
    ):
        issues.append("calibration receipt finished before it started")

    parameters = receipt.get("parameters", {})
    if not isinstance(parameters, dict):
        issues.append("calibration receipt parameters must be an object")
        parameters = {}
    expected_parameters = options.get("parameters", {})
    if not isinstance(expected_parameters, dict):
        issues.append("calibration receipt validator parameters option must be an object")
        expected_parameters = {}
    for key, expected_value in expected_parameters.items():
        if parameters.get(key) != expected_value:
            issues.append(f"calibration parameter {key} must equal {expected_value!r}")

    runtime = receipt.get("runtime", {})
    for field in options.get(
        "runtime_packages",
        ["python", "numpy", "pandas", "gymnasium", "stable_baselines3", "torch"],
    ):
        if not isinstance(runtime, dict) or runtime.get(str(field)) in {None, "", "missing"}:
            issues.append(f"calibration receipt is missing runtime package {field}")

    latest_attempt = definition.campaign_dir / "attempts" / f"{stage.id}_latest.json"
    attempt_id = str(receipt.get("campaign_attempt_id", ""))
    attempt: dict[str, Any] = {}
    if not attempt_id:
        issues.append("calibration receipt is not bound to a campaign attempt")
    if not latest_attempt.is_file():
        issues.append(f"latest rendered attempt is missing for stage {stage.id}")
    else:
        referenced.append(latest_attempt)
        try:
            loaded_attempt = json.loads(latest_attempt.read_text(encoding="utf-8"))
            if isinstance(loaded_attempt, dict):
                attempt = loaded_attempt
            else:
                issues.append("latest rendered attempt must be a JSON object")
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"cannot read latest rendered attempt: {exc}")
    if attempt:
        if attempt.get("status") != "rendered":
            issues.append("latest calibration attempt is not in rendered state")
        try:
            rendered_at = datetime.fromisoformat(str(attempt.get("rendered_at_utc", "")))
            if rendered_at.tzinfo is None or rendered_at.utcoffset() is None:
                raise ValueError
        except ValueError:
            issues.append("latest calibration attempt rendered_at_utc is not timezone-aware")
        if attempt.get("campaign_id") != definition.campaign_id or attempt.get("stage_id") != stage.id:
            issues.append("latest calibration attempt identity does not match the stage")
        if attempt.get("attempt_id") != attempt_id:
            issues.append("calibration receipt does not belong to the latest rendered attempt")
        stored_attempt_hash = str(attempt.get("attempt_sha256", ""))
        unhashed_attempt = dict(attempt)
        unhashed_attempt.pop("attempt_sha256", None)
        computed_attempt_hash = hashlib.sha256(
            json.dumps(unhashed_attempt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if stored_attempt_hash != computed_attempt_hash:
            issues.append("latest calibration attempt hash is invalid")
        if receipt.get("campaign_attempt_sha256") != stored_attempt_hash:
            issues.append("calibration receipt attempt hash does not match")
        if receipt.get("campaign_stage_id") != stage.id:
            issues.append("calibration receipt stage id does not match")
        deployment_release_id = str(attempt.get("deployment_release_id", ""))
        if not _is_sha256(deployment_release_id):
            issues.append("latest calibration attempt has no valid deployment release id")
        if receipt.get("deployment_release_id") != deployment_release_id:
            issues.append("calibration receipt deployment release id does not match")
        preflight_hash = str(attempt.get("preflight_sha256", ""))
        if not _is_sha256(preflight_hash):
            issues.append("latest calibration attempt has no valid BistKA preflight hash")
        if receipt.get("bistka_preflight_sha256") != preflight_hash:
            issues.append("calibration receipt BistKA preflight hash does not match")
        immutable_attempt = definition.campaign_dir / "attempts" / f"{attempt_id}.json"
        if not immutable_attempt.is_file():
            issues.append("immutable calibration attempt record is missing")
        else:
            referenced.append(immutable_attempt)
            try:
                immutable_value = json.loads(immutable_attempt.read_text(encoding="utf-8"))
                if immutable_value != attempt:
                    issues.append("latest and immutable calibration attempt records differ")
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"cannot read immutable calibration attempt: {exc}")

        attempt_inputs = attempt.get("inputs", [])
        if not isinstance(attempt_inputs, list) or not attempt_inputs:
            issues.append("latest calibration attempt contains no input inventory")
            attempt_inputs = []
        attempt_paths = [
            str(record.get("path", "")) for record in attempt_inputs if isinstance(record, dict)
        ]
        if len(attempt_paths) != len(set(attempt_paths)):
            issues.append("latest calibration attempt contains duplicate input paths")
        for record in attempt_inputs:
            if not isinstance(record, dict):
                issues.append("latest calibration attempt contains a malformed input record")
                continue
            relative = Path(str(record.get("path", "")))
            source = (definition.project_root / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not source.is_relative_to(definition.project_root):
                issues.append(f"unsafe calibration attempt input: {relative}")
                continue
            if not source.is_file():
                issues.append(f"calibration attempt input is missing: {relative}")
                continue
            referenced.append(source)
            if sha256_file(source) != record.get("sha256") or source.stat().st_size != record.get("bytes"):
                issues.append(f"calibration attempt input changed after command rendering: {relative}")
        raw_manifest_path = str(attempt.get("deployment_manifest_path", ""))
        raw_preflight_path = str(attempt.get("preflight_receipt_path", ""))
        if not raw_manifest_path or not raw_preflight_path:
            try:
                declared_manifest, declared_preflight = definition.resolve_deployment_pair(stage)
            except ValueError:
                declared_manifest = (
                    stage.deployment_manifest
                    or "{campaign_dir}/deployment/bistka_deployment_manifest.json"
                )
                declared_preflight = (
                    stage.preflight_receipt
                    or "reports/bistka_calibration_preflight.json"
                )
            raw_manifest_path = raw_manifest_path or declared_manifest
            raw_preflight_path = raw_preflight_path or declared_preflight
        deployment_manifest_path = (
            definition.project_root
            / definition.expand_project_relative(raw_manifest_path)
        ).resolve()
        if not deployment_manifest_path.is_relative_to(definition.project_root):
            issues.append("calibration deployment manifest path is unsafe")
            deployment_manifest_path = definition.project_root / ".invalid-calibration-manifest"
        if not deployment_manifest_path.is_file():
            issues.append("calibration deployment manifest is missing")
        else:
            referenced.append(deployment_manifest_path)
            try:
                deployment_manifest = json.loads(
                    deployment_manifest_path.read_text(encoding="utf-8")
                )
                if not isinstance(deployment_manifest, dict) or deployment_manifest.get(
                    "release_id"
                ) != attempt.get("deployment_release_id"):
                    issues.append("calibration attempt deployment release does not match the manifest")
                elif not str(deployment_manifest.get("target_project_root", "")):
                    issues.append("calibration deployment manifest has no target project root")
                else:
                    remote_project_root = Path(str(deployment_manifest["target_project_root"]))
                    if not remote_project_root.is_absolute():
                        issues.append("calibration deployment target project root must be absolute")
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"cannot read calibration deployment manifest: {exc}")
        preflight_path = (
            definition.project_root
            / definition.expand_project_relative(raw_preflight_path)
        ).resolve()
        if not preflight_path.is_relative_to(definition.project_root):
            issues.append("BistKA calibration preflight receipt path is unsafe")
            preflight_path = definition.project_root / ".invalid-calibration-preflight"
        if not preflight_path.is_file():
            issues.append("BistKA calibration preflight receipt is missing")
        else:
            referenced.append(preflight_path)
            if sha256_file(preflight_path) != attempt.get("preflight_sha256"):
                issues.append("calibration attempt preflight hash does not match")

    receipt_inputs = receipt.get("inputs", [])
    required_input_names = {str(value) for value in options.get("required_input_names", [])}
    seen_input_names: set[str] = set()
    if not isinstance(receipt_inputs, list) or not receipt_inputs:
        issues.append("calibration receipt contains no input inventory")
        receipt_inputs = []
    attempt_records = attempt.get("inputs", []) if isinstance(attempt.get("inputs", []), list) else []
    for record in receipt_inputs:
        if not isinstance(record, dict):
            issues.append("calibration receipt contains a malformed input record")
            continue
        name = str(record.get("name", ""))
        seen_input_names.add(name)
        digest = str(record.get("sha256", ""))
        if record.get("exists") is not True or not _is_sha256(digest):
            issues.append(f"calibration receipt input is missing or unhashed: {name or '<unnamed>'}")
            continue
        # Canonical resolved configurations are virtual records; all file-backed
        # records must match an input frozen by the controller at render time.
        if record.get("bytes") is None:
            continue
        remote_path = str(record.get("path", ""))
        matches_attempt = any(
            isinstance(candidate, dict)
            and remote_path.replace("\\", "/").endswith("/" + str(candidate.get("path", "")).replace("\\", "/"))
            and candidate.get("sha256") == digest
            and candidate.get("bytes") == record.get("bytes")
            for candidate in attempt_records
        )
        if not matches_attempt:
            issues.append(f"calibration receipt input was not frozen in the rendered attempt: {name}")
    if len(seen_input_names) != len(receipt_inputs):
        issues.append("calibration receipt contains duplicate input names")
    missing_names = sorted(required_input_names - seen_input_names)
    if missing_names:
        issues.append(f"calibration receipt is missing required inputs: {missing_names}")

    outputs = receipt.get("outputs", [])
    required_output_names = {str(value) for value in options.get("output_names", [])}
    seen_output_names: set[str] = set()
    local_outputs: dict[str, Path] = {}
    if not isinstance(outputs, list) or not outputs:
        issues.append("calibration receipt contains no output inventory")
        outputs = []
    for record in outputs:
        if not isinstance(record, dict):
            issues.append("calibration receipt contains a malformed output record")
            continue
        name = str(record.get("name", ""))
        seen_output_names.add(name)
        remote_output = Path(str(record.get("path", "")))
        remote_name = remote_output.name
        try:
            if remote_project_root is None or not remote_project_root.is_absolute():
                raise ValueError("deployment target root is unavailable")
            relative_output = remote_output.relative_to(remote_project_root)
            if ".." in relative_output.parts:
                raise ValueError("output escapes the deployment target")
            local_output = (definition.project_root / relative_output).resolve()
        except ValueError as exc:
            issues.append(f"unsafe returned calibration output path {remote_output}: {exc}")
            continue
        if not local_output.is_relative_to(definition.project_root) or not local_output.is_file():
            issues.append(f"returned calibration output is missing: {remote_name}")
            continue
        referenced.append(local_output)
        local_outputs[name] = local_output
        if (
            record.get("exists") is not True
            or not _is_sha256(record.get("sha256", ""))
            or sha256_file(local_output) != record.get("sha256")
            or local_output.stat().st_size != record.get("bytes")
        ):
            issues.append(f"returned calibration output hash or size does not match: {remote_name}")
        if job_id and job_id not in remote_name:
            issues.append(f"returned calibration output is not named for SLURM job {job_id}: {remote_name}")
    if len(seen_output_names) != len(outputs):
        issues.append("calibration receipt contains duplicate output names")
    if required_output_names and seen_output_names != required_output_names:
        issues.append(
            f"calibration receipt output names must equal {sorted(required_output_names)}"
        )

    contract = str(options.get("artifact_contract", ""))
    if contract == "vector_benchmark":
        benchmark_options = dict(options.get("benchmark_options", {}))
        csv_output = local_outputs.get("benchmark_csv")
        metadata_output = local_outputs.get("benchmark_metadata")
        expected_csv = f"vectorized_training_benchmark_{job_id}.csv"
        expected_metadata = f"vectorized_training_benchmark_{job_id}.json"
        expected_receipt = f"{expected_csv}.receipt.json"
        if csv_output is None or csv_output.name != expected_csv:
            issues.append("vector receipt does not reference the exact job-named benchmark CSV")
        else:
            issues.extend(_benchmark_validator(csv_output, benchmark_options))
        if metadata_output is None or metadata_output.name != expected_metadata:
            issues.append("vector receipt does not reference the exact job-named benchmark metadata")
        else:
            issues.extend(_vector_metadata_validator(metadata_output, benchmark_options))
        if path.name != expected_receipt:
            issues.append("vector receipt filename does not match its SLURM job id")
    elif contract == "determinism":
        determinism_options = dict(options.get("determinism_options", {}))
        result_output = local_outputs.get("determinism_result")
        expected_result = f"determinism_{job_id}.json"
        if result_output is None or result_output.name != expected_result:
            issues.append("determinism receipt does not reference the exact job-named result")
        else:
            issues.extend(_determinism_validator(result_output, determinism_options))
        if path.name != f"{expected_result}.receipt.json":
            issues.append("determinism receipt filename does not match its SLURM job id")

    if contract in {"vector_benchmark", "determinism"}:
        try:
            current = _current_calibration_provenance(definition, stage)
        except (OSError, TypeError, ValueError) as exc:
            current = {}
            issues.append(f"cannot recompute current observed calibration provenance: {exc}")
        if contract == "vector_benchmark" and "benchmark_csv" in local_outputs and current:
            try:
                benchmark_rows = list(
                    csv.DictReader(local_outputs["benchmark_csv"].open("r", encoding="utf-8"))
                )
            except (OSError, csv.Error) as exc:
                benchmark_rows = []
                issues.append(f"cannot cross-check current vector benchmark provenance: {exc}")
            for row in benchmark_rows:
                for field in (
                    "source",
                    "data_sha256",
                    "config_sha256",
                    "config_file_sha256",
                    "code_sha256",
                    "feature_rows",
                    "feature_dates",
                    "feature_tickers",
                    "training_split",
                ):
                    csv_field = "data_source" if field == "source" else field
                    if str(row.get(csv_field, "")) != str(current[field]):
                        issues.append(f"vector benchmark {csv_field} does not match current inputs")
                if row.get("hostname") != receipt.get("hostname"):
                    issues.append("vector benchmark trial hostname differs from its receipt")
        if contract == "vector_benchmark" and "benchmark_metadata" in local_outputs and current:
            try:
                metadata_value = json.loads(
                    local_outputs["benchmark_metadata"].read_text(encoding="utf-8")
                )
                if not isinstance(metadata_value, dict):
                    raise TypeError("metadata must be an object")
                provenance_value = metadata_value.get("data_provenance", {})
                code_value = metadata_value.get("code_provenance", {})
                if not isinstance(provenance_value, dict) or not isinstance(code_value, dict):
                    raise TypeError("metadata provenance must be objects")
                for field in ("source", "data_sha256", "config_sha256", "config_file_sha256"):
                    if provenance_value.get(field) != current[field]:
                        issues.append(f"vector metadata {field} does not match current inputs")
                if code_value.get("sha256") != current["code_sha256"]:
                    issues.append("vector metadata code hash does not match current code")
                if code_value.get("files") != current["code_files"]:
                    issues.append("vector metadata code inventory does not match current code")
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                issues.append(f"cannot cross-check current vector metadata provenance: {exc}")
        if contract == "determinism" and "determinism_result" in local_outputs and current:
            try:
                determinism_value = json.loads(
                    local_outputs["determinism_result"].read_text(encoding="utf-8")
                )
                if not isinstance(determinism_value, dict):
                    raise TypeError("determinism result must be an object")
                provenance_value = determinism_value.get("data_provenance", {})
                if not isinstance(provenance_value, dict):
                    raise TypeError("determinism provenance must be an object")
                for field in (
                    "source",
                    "data_sha256",
                    "config_sha256",
                    "config_file_sha256",
                    "code_sha256",
                    "feature_rows",
                    "feature_dates",
                    "feature_tickers",
                    "training_split",
                ):
                    if provenance_value.get(field) != current[field]:
                        issues.append(f"determinism {field} does not match current inputs")
                for run_value in determinism_value.get("run_provenance", []):
                    if not isinstance(run_value, dict) or run_value.get("hostname") != receipt.get(
                        "hostname"
                    ):
                        issues.append("determinism subprocess hostname differs from its receipt")
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                issues.append(f"cannot cross-check current determinism provenance: {exc}")

    result = receipt.get("result", {})
    if not isinstance(result, dict):
        issues.append("calibration receipt result must be an object")
    else:
        for field in options.get("result_sha256_fields", []):
            if not _is_sha256(result.get(str(field), "")):
                issues.append(f"calibration receipt result contains an invalid {field}")
        for field, expected_value in options.get("result_values", {}).items():
            if result.get(field) != expected_value:
                issues.append(f"calibration receipt result {field} must equal {expected_value!r}")
        if contract == "vector_benchmark" and "benchmark_metadata" in local_outputs:
            try:
                metadata = json.loads(
                    local_outputs["benchmark_metadata"].read_text(encoding="utf-8")
                )
                if not isinstance(metadata, dict):
                    raise TypeError("metadata must be an object")
                for field in (
                    "selected_num_envs",
                    "trial_count",
                ):
                    if result.get(field) != metadata.get(field):
                        issues.append(f"vector receipt result {field} does not match metadata")
                provenance = metadata.get("data_provenance", {})
                code = metadata.get("code_provenance", {})
                for result_field, metadata_value in (
                    ("data_sha256", provenance.get("data_sha256") if isinstance(provenance, dict) else None),
                    ("config_sha256", provenance.get("config_sha256") if isinstance(provenance, dict) else None),
                    ("code_sha256", code.get("sha256") if isinstance(code, dict) else None),
                ):
                    if result.get(result_field) != metadata_value:
                        issues.append(
                            f"vector receipt result {result_field} does not match metadata"
                        )
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                issues.append(f"cannot cross-check vector receipt metadata: {exc}")
        if contract == "determinism" and "determinism_result" in local_outputs:
            try:
                determinism_result = json.loads(
                    local_outputs["determinism_result"].read_text(encoding="utf-8")
                )
                if not isinstance(determinism_result, dict):
                    raise TypeError("determinism result must be an object")
                provenance = determinism_result.get("data_provenance", {})
                for result_field, expected_value in (
                    ("trajectory_sha256", determinism_result.get("first_sha256")),
                    ("data_sha256", provenance.get("data_sha256") if isinstance(provenance, dict) else None),
                    ("config_sha256", provenance.get("config_sha256") if isinstance(provenance, dict) else None),
                    ("code_sha256", provenance.get("code_sha256") if isinstance(provenance, dict) else None),
                ):
                    if result.get(result_field) != expected_value:
                        issues.append(
                            f"determinism receipt result {result_field} does not match its result"
                        )
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                issues.append(f"cannot cross-check determinism receipt result: {exc}")
    return sorted(set(issues)), referenced


VALIDATORS = {
    "capacity_snapshot": _capacity_validator,
    "vector_benchmark": _benchmark_validator,
    "vector_benchmark_metadata": _vector_metadata_validator,
    "determinism": _determinism_validator,
    "algorithm_selection": _selection_validator,
    "budget_selection": _budget_validator,
    "stress_readiness": _stress_validator,
    "stress_completeness": _stress_completeness_validator,
    "stress_replay": _stress_replay_validator,
    "stress_config": _stress_config_validator,
    "archive": _archive_validator,
    "sha256_sidecar": _sha256_sidecar_validator,
    "screening_smoke_summary": _screening_smoke_summary_validator,
}


def protocol_lock_status(definition: CampaignDefinition) -> tuple[bool, str | None, list[str]]:
    path = definition.protocol_lock_path
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        return False, None, ["protocol lock and SHA-256 sidecar are required"]
    lock_hash = sha256_file(path)
    if sidecar.read_text(encoding="utf-8").strip() != lock_hash:
        return False, lock_hash, ["protocol lock SHA-256 sidecar does not match"]
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, lock_hash, [f"protocol lock JSON is invalid: {exc}"]
    if not isinstance(lock, dict) or not isinstance(lock.get("inputs", []), list):
        return False, lock_hash, ["protocol lock must contain a JSON object with an inputs list"]
    issues = []
    for entry in lock.get("inputs", []):
        if not isinstance(entry, dict):
            issues.append("protocol lock contains a malformed input record")
            continue
        input_path = (definition.project_root / str(entry.get("path", ""))).resolve()
        if not input_path.is_relative_to(definition.project_root) or not input_path.is_file():
            issues.append(f"locked input is missing: {entry.get('path')}")
        elif sha256_file(input_path) != entry.get("sha256"):
            issues.append(f"locked input changed: {entry.get('path')}")
    return not issues, lock_hash, issues


def _strict_run_parent_job_id(
    definition: CampaignDefinition,
    stage: CampaignStage,
    run_dir: Path,
) -> tuple[str | None, list[str]]:
    if not stage.run_dir_template:
        return None, [
            f"strict SLURM stage {stage.id} has run evidence but no run_dir_template "
            "for parent-job binding"
        ]
    expanded_template = definition.expand_project_relative(stage.run_dir_template)
    if expanded_template.count("{job_id}") != 1:
        return None, [
            f"strict SLURM stage {stage.id} run_dir_template must contain exactly one "
            "{job_id} placeholder"
        ]
    relative_run_dir = run_dir.relative_to(definition.project_root).as_posix()
    escaped_template = re.escape(Path(expanded_template).as_posix())
    pattern = escaped_template.replace(
        re.escape("{job_id}"),
        r"(?P<job_id>[1-9][0-9]*)",
    )
    match = re.fullmatch(pattern, relative_run_dir)
    if match is None:
        return None, [
            f"strict SLURM run directory does not match its job-bound template: "
            f"{relative_run_dir}"
        ]
    return match.group("job_id"), []


def _strict_attempt_context(
    definition: CampaignDefinition,
    stage: CampaignStage,
) -> tuple[dict[str, Any], dict[str, str], list[str], list[Path]]:
    """Validate and return the latest immutable strict-SLURM attempt binding."""

    issues: list[str] = []
    referenced: list[Path] = []
    latest_path = definition.campaign_dir / "attempts" / f"{stage.id}_latest.json"
    if not latest_path.is_file():
        return {}, {}, [f"latest rendered attempt is missing for strict stage {stage.id}"], []
    referenced.append(latest_path)
    try:
        attempt = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, {}, [f"cannot read latest rendered attempt for strict stage {stage.id}: {exc}"], referenced
    if not isinstance(attempt, dict):
        return {}, {}, [f"latest rendered attempt for strict stage {stage.id} must be a JSON object"], referenced

    if attempt.get("schema_version") != 2:
        issues.append("latest strict SLURM attempt schema must equal 2")
    if attempt.get("status") != "rendered":
        issues.append("latest strict SLURM attempt is not in rendered state")
    if attempt.get("campaign_id") != definition.campaign_id:
        issues.append("latest strict SLURM attempt campaign id does not match")
    if attempt.get("stage_id") != stage.id:
        issues.append("latest strict SLURM attempt stage id does not match")
    try:
        rendered_at = datetime.fromisoformat(str(attempt.get("rendered_at_utc", "")))
        if rendered_at.tzinfo is None or rendered_at.utcoffset() is None:
            raise ValueError
    except ValueError:
        issues.append("latest strict SLURM attempt rendered_at_utc is not timezone-aware")

    attempt_id = attempt.get("attempt_id")
    safe_attempt_id = (
        isinstance(attempt_id, str)
        and bool(attempt_id)
        and Path(attempt_id).name == attempt_id
        and "/" not in attempt_id
        and "\\" not in attempt_id
    )
    if not safe_attempt_id:
        issues.append("latest strict SLURM attempt has an invalid attempt id")
    stored_attempt_hash = str(attempt.get("attempt_sha256", ""))
    unhashed_attempt = dict(attempt)
    unhashed_attempt.pop("attempt_sha256", None)
    computed_attempt_hash = hashlib.sha256(
        json.dumps(unhashed_attempt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not _is_sha256(stored_attempt_hash) or stored_attempt_hash != computed_attempt_hash:
        issues.append("latest strict SLURM attempt hash is invalid")

    if safe_attempt_id:
        immutable_path = definition.campaign_dir / "attempts" / f"{attempt_id}.json"
        if not immutable_path.is_file():
            issues.append("immutable strict SLURM attempt record is missing")
        else:
            referenced.append(immutable_path)
            try:
                immutable_attempt = json.loads(immutable_path.read_text(encoding="utf-8"))
                if immutable_attempt != attempt:
                    issues.append("latest and immutable strict SLURM attempt records differ")
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"cannot read immutable strict SLURM attempt: {exc}")

    raw_inputs = attempt.get("inputs", [])
    if not isinstance(raw_inputs, list) or not raw_inputs:
        issues.append("latest strict SLURM attempt contains no input inventory")
        raw_inputs = []
    input_records: dict[str, dict[str, Any]] = {}
    for record in raw_inputs:
        if not isinstance(record, dict):
            issues.append("latest strict SLURM attempt contains a malformed input record")
            continue
        raw_relative = str(record.get("path", ""))
        if raw_relative in input_records:
            issues.append(
                f"latest strict SLURM attempt contains duplicate input path: {raw_relative}"
            )
            continue
        input_records[raw_relative] = record
        relative = Path(raw_relative)
        source = (definition.project_root / relative).resolve()
        if (
            not raw_relative
            or relative.is_absolute()
            or ".." in relative.parts
            or not source.is_relative_to(definition.project_root)
        ):
            issues.append(f"unsafe strict SLURM attempt input: {raw_relative or '<empty>'}")
            continue
        expected_hash = record.get("sha256")
        expected_bytes = record.get("bytes")
        if not _is_sha256(expected_hash):
            issues.append(f"strict SLURM attempt input has an invalid SHA-256: {raw_relative}")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
        ):
            issues.append(f"strict SLURM attempt input has an invalid byte count: {raw_relative}")
        if not source.is_file():
            issues.append(f"strict SLURM attempt input is missing: {raw_relative}")
            continue
        referenced.append(source)
        if (
            sha256_file(source) != expected_hash
            or source.stat().st_size != expected_bytes
        ):
            issues.append(
                f"strict SLURM attempt input changed after command rendering: {raw_relative}"
            )

    release_id = str(attempt.get("deployment_release_id", ""))
    if not _is_sha256(release_id):
        issues.append("latest strict SLURM attempt has no valid deployment release id")
    preflight_hash = str(attempt.get("preflight_sha256", ""))
    if not _is_sha256(preflight_hash):
        issues.append("latest strict SLURM attempt has no valid BistKA preflight hash")

    manifest_relative = str(attempt.get("deployment_manifest_path", ""))
    manifest_hash = str(attempt.get("deployment_manifest_sha256", ""))
    manifest_record = input_records.get(manifest_relative)
    required_packages: list[str] = []
    if (
        not manifest_relative
        or not _is_sha256(manifest_hash)
        or not isinstance(manifest_record, dict)
        or manifest_record.get("sha256") != manifest_hash
    ):
        issues.append(
            "latest strict SLURM attempt deployment manifest is not bound to its input inventory"
        )
    else:
        manifest_path = (definition.project_root / manifest_relative).resolve()
        if manifest_path.is_file() and manifest_path.is_relative_to(definition.project_root):
            try:
                deployment_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                if (
                    not isinstance(deployment_manifest, dict)
                    or deployment_manifest.get("release_id") != release_id
                ):
                    issues.append(
                        "latest strict SLURM attempt release does not match its deployment manifest"
                    )
                elif isinstance(deployment_manifest.get("required_packages"), list):
                    required_packages = [
                        str(name) for name in deployment_manifest["required_packages"]
                    ]
                else:
                    issues.append(
                        "latest strict SLURM deployment manifest has invalid required packages"
                    )
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"cannot read strict SLURM deployment manifest: {exc}")

    preflight_relative = str(attempt.get("preflight_receipt_path", ""))
    preflight_record = input_records.get(preflight_relative)
    expected_runtime_packages: dict[str, str] = {}
    if (
        not preflight_relative
        or not isinstance(preflight_record, dict)
        or preflight_record.get("sha256") != preflight_hash
    ):
        issues.append(
            "latest strict SLURM attempt preflight is not bound to its input inventory"
        )
    else:
        preflight_path = (definition.project_root / preflight_relative).resolve()
        if preflight_path.is_file() and preflight_path.is_relative_to(definition.project_root):
            try:
                preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(preflight, dict)
                    or preflight.get("status") != "pass"
                    or preflight.get("release_id") != release_id
                    or preflight.get("manifest_sha256") != manifest_hash
                ):
                    issues.append(
                        "latest strict SLURM attempt preflight does not match its deployment"
                    )
                else:
                    packages = preflight.get("packages")
                    if not isinstance(packages, dict):
                        issues.append(
                            "latest strict SLURM preflight has invalid runtime packages"
                        )
                    else:
                        for name in required_packages:
                            version = packages.get(name)
                            if not isinstance(version, str) or version in {"", "missing"}:
                                issues.append(
                                    f"latest strict SLURM preflight is missing runtime "
                                    f"package {name}"
                                )
                            else:
                                expected_runtime_packages[name] = version
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"cannot read strict SLURM preflight receipt: {exc}")

    return {
        "campaign_id": definition.campaign_id,
        "campaign_attempt_id": attempt_id,
        "campaign_attempt_sha256": stored_attempt_hash,
        "campaign_stage_id": stage.id,
        "deployment_release_id": release_id,
        "bistka_preflight_sha256": preflight_hash,
    }, expected_runtime_packages, sorted(set(issues)), referenced


def evaluate_stage_gate(definition: CampaignDefinition, stage: CampaignStage) -> GateResult:
    run_dir = latest_run_dir(definition, stage)
    base = run_dir or definition.project_root
    requirement_results = []
    evidence_files: list[Path] = []
    issues: list[str] = []
    for requirement in stage.requirements:
        expanded_patterns = tuple(definition.expand_project_relative(pattern) for pattern in requirement.patterns)
        matches = _safe_matches(definition.project_root, base, expanded_patterns)
        evidence_files.extend(matches)
        requirement_issues = []
        if len(matches) < requirement.minimum:
            requirement_issues.append(
                f"requires {requirement.minimum} match(es), found {len(matches)}"
            )
        elif requirement.validator:
            if requirement.validator == "protocol_lock":
                valid, _, protocol_issues = protocol_lock_status(definition)
                if not valid:
                    requirement_issues.extend(protocol_issues)
                validator = None
            elif requirement.validator == "model_registry":
                registry_issues, registry_files = _model_registry_validator(
                    definition,
                    max(matches, key=lambda path: path.stat().st_mtime_ns),
                )
                requirement_issues.extend(registry_issues)
                evidence_files.extend(registry_files)
                validator = None
            elif requirement.validator == "market_data_readiness":
                data_issues, data_files = _market_data_readiness_validator(
                    definition,
                    max(matches, key=lambda path: path.stat().st_mtime_ns),
                )
                requirement_issues.extend(data_issues)
                evidence_files.extend(data_files)
                validator = None
            elif requirement.validator == "research_preregistration":
                from src.campaigns.preregistration import preregistration_gate_validator

                registration_issues, registration_files = preregistration_gate_validator(
                    definition,
                    max(matches, key=lambda path: path.stat().st_mtime_ns),
                )
                requirement_issues.extend(registration_issues)
                evidence_files.extend(registration_files)
                validator = None
            elif requirement.validator == "bistka_deployment_manifest":
                from src.campaigns.bistka_deployment import validate_deployment_manifest

                deployment_issues, deployment_files = validate_deployment_manifest(
                    max(matches, key=lambda path: path.stat().st_mtime_ns),
                    project_root=definition.project_root,
                )
                requirement_issues.extend(deployment_issues)
                evidence_files.extend(deployment_files)
                validator = None
            elif requirement.validator == "bistka_deployment_bundle":
                from src.campaigns.bistka_deployment import validate_deployment_bundle

                raw_manifest = str(requirement.options.get("manifest", ""))
                manifest_path = (
                    definition.project_root / definition.expand_project_relative(raw_manifest)
                ).resolve()
                if not raw_manifest or not manifest_path.is_relative_to(definition.project_root):
                    requirement_issues.append(
                        "BistKA bundle validator requires a safe deployment manifest path"
                    )
                    bundle_files = []
                else:
                    bundle_issues, bundle_files = validate_deployment_bundle(
                        max(matches, key=lambda path: path.stat().st_mtime_ns),
                        manifest_path=manifest_path,
                        project_root=definition.project_root,
                    )
                    requirement_issues.extend(bundle_issues)
                evidence_files.extend(bundle_files)
                validator = None
            elif requirement.validator == "bistka_preflight":
                from src.campaigns.bistka_deployment import validate_preflight_receipt

                raw_manifest = str(requirement.options.get("manifest", ""))
                manifest_path = (
                    definition.project_root / definition.expand_project_relative(raw_manifest)
                ).resolve()
                if not raw_manifest or not manifest_path.is_relative_to(definition.project_root):
                    requirement_issues.append("BistKA preflight validator requires a safe manifest path")
                    preflight_files = []
                else:
                    preflight_issues, preflight_files = validate_preflight_receipt(
                        max(matches, key=lambda path: path.stat().st_mtime_ns),
                        expected_manifest_path=manifest_path,
                    )
                    requirement_issues.extend(preflight_issues)
                evidence_files.extend(preflight_files)
                validator = None
            elif requirement.validator in {"calibration_receipt", "calibration_evidence"}:
                latest_attempt_path = (
                    definition.campaign_dir / "attempts" / f"{stage.id}_latest.json"
                )
                current_attempt_id = None
                try:
                    latest_attempt_value = json.loads(
                        latest_attempt_path.read_text(encoding="utf-8")
                    )
                    if isinstance(latest_attempt_value, dict):
                        current_attempt_id = latest_attempt_value.get("attempt_id")
                except (OSError, json.JSONDecodeError):
                    pass
                current_receipts: list[Path] = []
                for candidate in matches:
                    try:
                        candidate_value = json.loads(candidate.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if (
                        isinstance(candidate_value, dict)
                        and current_attempt_id
                        and candidate_value.get("campaign_attempt_id") == current_attempt_id
                    ):
                        current_receipts.append(candidate)
                import_path = (
                    definition.campaign_dir / "imports" / "v1_calibration_evidence.json"
                ).resolve()
                import_receipts = [
                    candidate
                    for candidate in matches
                    if candidate.resolve() == import_path
                ]
                if requirement.validator == "calibration_receipt":
                    import_receipts = []
                if current_receipts and import_receipts:
                    requirement_issues.append(
                        "native calibration receipt and migrated calibration evidence are "
                        "both present; choose exactly one evidence mode"
                    )
                elif len(current_receipts) == 1:
                    receipt_issues, receipt_files = _calibration_receipt_validator(
                        definition,
                        stage,
                        current_receipts[0],
                        requirement.options,
                    )
                    requirement_issues.extend(receipt_issues)
                    evidence_files.extend(receipt_files)
                elif len(current_receipts) > 1:
                    requirement_issues.append(
                        "expected exactly one receipt for the latest rendered calibration attempt, "
                        f"found {len(current_receipts)}"
                    )
                elif len(import_receipts) == 1:
                    from src.campaigns.evidence_migration import (
                        validate_calibration_import,
                    )

                    import_issues, import_files = validate_calibration_import(
                        definition,
                        stage,
                        import_receipts[0],
                        requirement.options,
                    )
                    requirement_issues.extend(import_issues)
                    evidence_files.extend(import_files)
                elif len(import_receipts) > 1:
                    requirement_issues.append(
                        "expected exactly one migrated calibration evidence receipt, "
                        f"found {len(import_receipts)}"
                    )
                elif requirement.validator == "calibration_receipt":
                    requirement_issues.append(
                        "expected exactly one receipt for the latest rendered calibration "
                        "attempt, found 0"
                    )
                else:
                    requirement_issues.append(
                        "expected exactly one native receipt for the latest rendered "
                        "calibration attempt or one migrated calibration evidence receipt"
                    )
                validator = None
            else:
                validator = VALIDATORS.get(requirement.validator)
            if validator is None and requirement.validator not in {
                "protocol_lock",
                "model_registry",
                "market_data_readiness",
                "research_preregistration",
                "bistka_deployment_manifest",
                "bistka_deployment_bundle",
                "bistka_preflight",
                "calibration_receipt",
                "calibration_evidence",
            }:
                requirement_issues.append(f"unknown validator: {requirement.validator}")
            elif validator is not None:
                requirement_issues.extend(validator(max(matches, key=lambda path: path.stat().st_mtime_ns), requirement.options))
        requirement_results.append(
            RequirementResult(
                name=requirement.name,
                passed=not requirement_issues,
                matches=[str(path.relative_to(definition.project_root)) for path in matches],
                issues=requirement_issues,
            )
        )
        issues.extend(f"{requirement.name}: {issue}" for issue in requirement_issues)

    manifest_files: list[Path] = []
    complete = running = failed = 0
    if stage.expected_manifests:
        if run_dir is None:
            issues.append("no stage run directory exists")
        elif not stage.manifest_pattern:
            issues.append("manifest pattern is not configured")
        else:
            manifest_files = _safe_matches(
                definition.project_root,
                run_dir,
                (stage.manifest_pattern,),
            )
            evidence_files.extend(manifest_files)
            if len(manifest_files) != stage.expected_manifests:
                issues.append(
                    f"expected exactly {stage.expected_manifests} manifests, found {len(manifest_files)}"
                )
            strict_bindings: dict[str, Any] = {}
            strict_runtime_packages: dict[str, str] = {}
            expected_parent_job_id: str | None = None
            parent_job_ids: list[str] = []
            if definition.strict_deployment_resolution and stage.kind == "slurm":
                (
                    strict_bindings,
                    strict_runtime_packages,
                    strict_issues,
                    strict_files,
                ) = _strict_attempt_context(definition, stage)
                issues.extend(strict_issues)
                evidence_files.extend(strict_files)
                expected_parent_job_id, parent_issues = _strict_run_parent_job_id(
                    definition,
                    stage,
                    run_dir,
                )
                issues.extend(parent_issues)
            experiment_ids: list[str] = []
            lock_valid, lock_hash, _ = protocol_lock_status(definition)
            for manifest_path in manifest_files:
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    issues.append(f"malformed manifest {manifest_path.name}: {exc}")
                    continue
                if not isinstance(manifest, dict):
                    issues.append(f"manifest must be a JSON object: {manifest_path}")
                    continue
                experiment_id = manifest.get("experiment_id")
                if not isinstance(experiment_id, str) or not experiment_id:
                    issues.append(f"manifest is missing a nonempty experiment id: {manifest_path}")
                else:
                    experiment_ids.append(experiment_id)
                status = manifest.get("status")
                complete += int(status == "complete")
                running += int(status == "running")
                failed += int(status == "failed")
                if status not in {"complete", "running", "failed"}:
                    issues.append(f"manifest has unsupported status {status!r}: {manifest_path}")
                if stage.require_protocol_hash:
                    if not lock_valid or manifest.get("protocol_lock_hash") != lock_hash:
                        issues.append(f"manifest protocol hash mismatch: {manifest_path}")
                if definition.strict_deployment_resolution and stage.kind == "slurm":
                    for field, expected_value in strict_bindings.items():
                        actual_value = manifest.get(field)
                        if actual_value is None or actual_value == "":
                            issues.append(
                                f"strict manifest is missing {field}: {manifest_path}"
                            )
                        elif actual_value != expected_value:
                            issues.append(
                                f"strict manifest {field} mismatch: {manifest_path}"
                            )
                    runtime = manifest.get("runtime", {})
                    if not isinstance(runtime, dict):
                        issues.append(
                            f"strict manifest runtime must be an object: {manifest_path}"
                        )
                    else:
                        runtime_packages = runtime.get("packages", {})
                        if not isinstance(runtime_packages, dict):
                            issues.append(
                                f"strict manifest runtime packages must be an object: "
                                f"{manifest_path}"
                            )
                        else:
                            for package_name, expected_version in (
                                strict_runtime_packages.items()
                            ):
                                if runtime_packages.get(package_name) != expected_version:
                                    issues.append(
                                        f"strict manifest runtime package {package_name} "
                                        f"does not match the bound preflight: {manifest_path}"
                                    )
                        parent_job_id = runtime.get("slurm_array_job_id") or runtime.get(
                            "slurm_job_id"
                        )
                        parent_job_id = str(parent_job_id or "")
                        if (
                            not parent_job_id.isdigit()
                            or int(parent_job_id) <= 0
                        ):
                            issues.append(
                                f"strict manifest has no positive parent SLURM job id: "
                                f"{manifest_path}"
                            )
                        else:
                            parent_job_ids.append(parent_job_id)
                            if (
                                expected_parent_job_id is not None
                                and parent_job_id != expected_parent_job_id
                            ):
                                issues.append(
                                    f"strict manifest parent SLURM job does not match its "
                                    f"run directory: {manifest_path}"
                                )
            if len(experiment_ids) != len(set(experiment_ids)):
                issues.append("experiment manifests contain duplicate experiment ids")
            if len(set(parent_job_ids)) > 1:
                issues.append("strict experiment manifests contain mixed parent SLURM jobs")
            if complete != stage.expected_manifests:
                issues.append(
                    f"expected {stage.expected_manifests} complete manifests, found {complete}"
                )
            if running:
                issues.append(f"{running} experiment manifest(s) are still running")
            if failed:
                issues.append(f"{failed} experiment manifest(s) failed")

    fingerprint_files = set(evidence_files)
    fingerprint_files.add(definition.spec_path)
    # A not-yet-started stage has no success evidence to invalidate. Defer its
    # potentially cloud-backed config/script reads until evidence exists.
    if evidence_files or run_dir:
        for raw_path in (stage.job_script, stage.config_path, *stage.fingerprint_paths):
            if raw_path:
                path = (definition.project_root / definition.expand_project_relative(raw_path)).resolve()
                if path.is_file():
                    fingerprint_files.add(path)
    fingerprint_entries = []
    for path in sorted(fingerprint_files):
        fingerprint_entries.append(
            {"path": str(path.relative_to(definition.project_root)), "sha256": sha256_file(path)}
        )
    lock_valid, lock_hash, lock_issues = protocol_lock_status(definition)
    if stage.require_protocol_hash and not lock_valid:
        issues.extend(lock_issues)
    fingerprint_payload = {
        "stage": stage.id,
        "files": fingerprint_entries,
        "protocol_hash": lock_hash if stage.require_protocol_hash else None,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return GateResult(
        stage_id=stage.id,
        passed=not issues,
        base_dir=str(base.relative_to(definition.project_root)),
        run_dir=str(run_dir.relative_to(definition.project_root)) if run_dir else None,
        requirements=requirement_results,
        manifest_expected=stage.expected_manifests,
        manifest_found=len(manifest_files),
        manifest_complete=complete,
        manifest_running=running,
        manifest_failed=failed,
        issues=issues,
        evidence_files=sorted({str(path.relative_to(definition.project_root)) for path in evidence_files}),
        evidence_fingerprint=fingerprint,
        evaluated_at=_utc_now(),
    )


def success_marker_path(definition: CampaignDefinition, stage_id: str) -> Path:
    return definition.campaign_dir / "stages" / stage_id / "_SUCCESS.json"


def gate_report_path(definition: CampaignDefinition, stage_id: str) -> Path:
    return definition.campaign_dir / "stages" / stage_id / "gate_report.json"


def evaluate_campaign(definition: CampaignDefinition) -> list[StageStatus]:
    statuses: dict[str, StageStatus] = {}
    lock_valid, _, lock_issues = protocol_lock_status(definition)
    for stage in definition.topological_stages():
        gate = evaluate_stage_gate(definition, stage)
        dependencies = [statuses[parent] for parent in stage.depends_on]
        marker_path = success_marker_path(definition, stage.id)
        marker = None
        if marker_path.is_file():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                marker = {}
        dependency_invalid = any(
            parent.state in {StageState.FAILED, StageState.STALE} for parent in dependencies
        )
        dependencies_verified = all(parent.state == StageState.VERIFIED for parent in dependencies)
        if marker is not None:
            if gate.passed and marker.get("evidence_fingerprint") == gate.evidence_fingerprint and dependencies_verified:
                state, reason = StageState.VERIFIED, "artifact gate verified and fingerprint is unchanged"
            else:
                state, reason = StageState.STALE, "verified evidence or one of its verified dependencies changed"
        elif gate.manifest_failed:
            state, reason = StageState.FAILED, f"{gate.manifest_failed} experiment manifest(s) failed"
        elif gate.manifest_running:
            state, reason = StageState.RUNNING, f"{gate.manifest_running} experiment manifest(s) are running"
        elif dependency_invalid:
            state, reason = StageState.BLOCKED, "one or more dependencies failed or became stale"
        elif not dependencies_verified:
            state, reason = StageState.PENDING, "waiting for dependencies to be explicitly verified"
        elif gate.passed:
            state, reason = StageState.VERIFY_PENDING, "completion evidence passes; explicit verification is required"
        elif stage.require_protocol_hash and not lock_valid:
            state, reason = StageState.BLOCKED, "; ".join(lock_issues)
        elif gate.evidence_files or gate.run_dir:
            state, reason = StageState.BLOCKED, gate.issues[0] if gate.issues else "partial evidence requires attention"
        else:
            state, reason = stage.missing_state, (
                "required external inputs are missing" if stage.missing_state == StageState.BLOCKED else "stage is ready"
            )
        progress = (
            f"{gate.manifest_complete}/{gate.manifest_expected} complete manifests"
            if gate.manifest_expected
            else f"{sum(item.passed for item in gate.requirements)}/{len(gate.requirements)} artifact gates"
        )
        statuses[stage.id] = StageStatus(
            stage_id=stage.id,
            title=stage.title,
            state=state,
            reason=reason,
            dependencies=list(stage.depends_on),
            run_dir=gate.run_dir,
            progress=progress,
            gate=gate,
        )
    return [statuses[stage.id] for stage in definition.topological_stages()]


def verify_stage(definition: CampaignDefinition, stage_id: str) -> GateResult:
    stage = definition.stage_map[stage_id]
    current = {status.stage_id: status for status in evaluate_campaign(definition)}
    unverified = [parent for parent in stage.depends_on if current[parent].state != StageState.VERIFIED]
    if unverified:
        raise ValueError(f"stage {stage_id} cannot verify before dependencies: {unverified}")
    gate = evaluate_stage_gate(definition, stage)
    atomic_json(gate_report_path(definition, stage_id), gate.to_dict())
    if not gate.passed:
        raise ValueError(f"stage {stage_id} failed verification: {'; '.join(gate.issues)}")
    marker = {
        "stage_id": stage_id,
        "verified_at": _utc_now(),
        "evidence_fingerprint": gate.evidence_fingerprint,
        "protocol_lock_hash": protocol_lock_status(definition)[1] if stage.require_protocol_hash else None,
    }
    atomic_json(success_marker_path(definition, stage_id), marker)
    return gate
