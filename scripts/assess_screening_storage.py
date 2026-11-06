from __future__ import annotations

"""Fail-closed capacity assessment for checkpoint-free screening runs.

The quota input is deliberately machine-readable.  Schema version 1 is::

    {
      "schema_version": 1,
      "captured_at": "2026-07-17T00:00:00Z",
      "filesystem": "/lustrefs/project/25sfcs03",
      "group": {
        "name": "s00658",
        "gid": 1039,
        "bytes": {
          "used": 1,
          "soft_limit": 2,
          "hard_limit": 3,
          "grace_expired": false
        },
        "inodes": {
          "used": 1,
          "soft_limit": 2,
          "hard_limit": 3,
          "grace_expired": false
        }
      },
      "project": {
        "id": 25110303,
        "bytes": {
          "used": 1,
          "soft_limit": 2,
          "hard_limit": 3,
          "grace_expired": false
        },
        "inodes": {
          "used": 1,
          "soft_limit": 2,
          "hard_limit": 3,
          "grace_expired": false
        }
      },
      "write_probe": {
        "completed_at": "2026-07-17T00:00:00Z",
        "filesystem": "/lustrefs/project/25sfcs03",
        "gid": 1039,
        "path": "/lustrefs/project/25sfcs03/results/.capacity-probe",
        "bytes_written": 4096,
        "fsync_succeeded": true,
        "removed": true
      }
    }

Both regular-file input JSON files require a regular-file sibling ``.sha256``
containing exactly the lower-case SHA-256 digest and one trailing newline.
Symlinks are rejected. Capacity is recomputed from the numeric evidence;
claimed eligibility booleans are never used as the decision.
"""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence


EXPECTED_ALGORITHMS: tuple[str, ...] = ("ppo", "a2c", "ddpg", "td3", "sac")
SMOKE_SCHEMA_VERSION = 1
QUOTA_SCHEMA_VERSION = 1
DECISION_SCHEMA_VERSION = 1
DEFAULT_SAFETY_FACTOR = Decimal("1.5")
DEFAULT_FIXED_RESERVE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_QUOTA_AGE_SECONDS = 15 * 60
DEFAULT_MAX_CLOCK_SKEW_SECONDS = 60
RETENTION_RECEIPT_BUDGET_BYTES_PER_TRIAL = 4096
RETENTION_SIDECAR_BYTES_PER_TRIAL = 65
TASK_SUMMARY_BUDGET_BYTES_PER_ALGORITHM = 1024 * 1024
RETENTION_FILES_PER_TRIAL = 2
TASK_SUMMARY_FILES_PER_ALGORITHM = 1
SMOKE_ONLY_RESULT_NAME = "screening_smoke_result.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_decision(path: str | Path, decision: dict[str, Any]) -> dict[str, str]:
    """Atomically write a canonical decision JSON and its digest sidecar."""

    destination = Path(path)
    payload = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    _atomic_text(destination, payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    sidecar = Path(str(destination) + ".sha256")
    _atomic_text(sidecar, digest + "\n")
    return {
        "path": str(destination),
        "sha256": digest,
        "sidecar_path": str(sidecar),
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def _ceil_scaled(value: int, factor: Decimal) -> int:
    return int((Decimal(value) * factor).to_integral_value(rounding=ROUND_CEILING))


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _verify_sidecar(
    path: Path,
    evidence_name: str,
    add_reason: Callable[[str, str], None],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "sidecar_path": str(Path(str(path) + ".sha256")),
        "sha256": None,
        "sidecar_verified": False,
    }
    if path.is_symlink():
        add_reason(
            f"{evidence_name}_symlink_rejected",
            f"input evidence must be a regular file, not a symlink: {path}",
        )
        return record
    if not path.is_file():
        add_reason(f"{evidence_name}_missing", f"missing input JSON: {path}")
        return record
    actual = _sha256_file(path)
    record["sha256"] = actual
    sidecar = Path(record["sidecar_path"])
    if sidecar.is_symlink():
        add_reason(
            f"{evidence_name}_sidecar_symlink_rejected",
            f"SHA-256 sidecar must be a regular file, not a symlink: {sidecar}",
        )
        return record
    if not sidecar.is_file():
        add_reason(
            f"{evidence_name}_sidecar_missing",
            f"missing SHA-256 sidecar: {sidecar}",
        )
        return record
    try:
        sidecar_bytes = sidecar.read_bytes()
    except OSError as exc:
        add_reason(
            f"{evidence_name}_sidecar_invalid",
            f"could not read SHA-256 sidecar: {exc}",
        )
        return record
    digest_bytes = sidecar_bytes[:64]
    exact_format = (
        len(sidecar_bytes) == 65
        and sidecar_bytes[64:] == b"\n"
        and all(character in b"0123456789abcdef" for character in digest_bytes)
    )
    if not exact_format:
        add_reason(
            f"{evidence_name}_sidecar_invalid",
            "SHA-256 sidecar must equal '<lower-case digest>\\n' exactly",
        )
    elif digest_bytes.decode("ascii") != actual:
        add_reason(
            f"{evidence_name}_sidecar_mismatch",
            f"SHA-256 sidecar does not match {path}",
        )
    else:
        record["sidecar_verified"] = True
    return record


def _load_object(
    path: Path,
    evidence_name: str,
    add_reason: Callable[[str, str], None],
) -> dict[str, Any] | None:
    if path.is_symlink():
        return None
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        add_reason(f"{evidence_name}_json_invalid", f"invalid JSON: {exc}")
        return None
    if not isinstance(value, dict):
        add_reason(f"{evidence_name}_schema_invalid", "JSON root must be an object")
        return None
    return value


def _has_checkpoint_component(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return "checkpoints" in PurePosixPath(normalized).parts


def _project_smoke(
    summary: dict[str, Any],
    *,
    expected_algorithms: tuple[str, ...],
    safety_factor: Decimal,
    fixed_reserve_bytes: int,
    add_reason: Callable[[str, str], None],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    valid = True

    def invalid(code: str, message: str) -> None:
        nonlocal valid
        valid = False
        add_reason(code, message)

    if summary.get("schema_version") != SMOKE_SCHEMA_VERSION:
        invalid("smoke_summary_schema_invalid", "unsupported smoke schema_version")
    if summary.get("status") != "complete":
        invalid("smoke_summary_status_incomplete", "smoke status must be complete")
    if summary.get("full_screening_eligible") is not True:
        invalid(
            "smoke_summary_claims_ineligible",
            "full_screening_eligible must be true",
        )

    algorithms = summary.get("algorithms")
    if (
        not isinstance(algorithms, list)
        or len(algorithms) != len(expected_algorithms)
        or len(set(algorithms)) != len(expected_algorithms)
        or set(algorithms) != set(expected_algorithms)
    ):
        invalid(
            "smoke_summary_algorithms_invalid",
            "smoke algorithms must be the five unique expected algorithms",
        )

    runs = summary.get("runs")
    if not isinstance(runs, list) or len(runs) != len(expected_algorithms):
        invalid(
            "smoke_summary_runs_invalid",
            "smoke summary must contain exactly five run records",
        )
        runs = []
    if summary.get("run_count") != len(expected_algorithms):
        invalid("smoke_summary_run_count_invalid", "run_count must equal five")
    if summary.get("complete_manifest_count") != len(expected_algorithms):
        invalid(
            "smoke_summary_manifest_count_invalid",
            "complete_manifest_count must equal five",
        )

    trials = summary.get("full_screening_trials_per_algorithm")
    if not _is_int(trials) or trials <= 0:
        invalid(
            "smoke_summary_trials_invalid",
            "full_screening_trials_per_algorithm must be a positive integer",
        )
        trials = None

    runtime_limit = _decimal(summary.get("runtime_projection_limit_hours"))
    if runtime_limit is None or runtime_limit <= 0:
        invalid(
            "runtime_projection_limit_invalid",
            "runtime_projection_limit_hours must be positive and finite",
        )
        runtime_limit = None

    run_by_algorithm: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            invalid("smoke_summary_runs_invalid", "every run record must be an object")
            continue
        algorithm = run.get("algorithm")
        if not isinstance(algorithm, str) or algorithm not in expected_algorithms:
            invalid("smoke_summary_runs_invalid", "run has an unexpected algorithm")
            continue
        if algorithm in run_by_algorithm:
            invalid("smoke_summary_runs_invalid", f"duplicate run for {algorithm}")
            continue
        run_by_algorithm[algorithm] = run
    if set(run_by_algorithm) != set(expected_algorithms):
        invalid(
            "smoke_summary_runs_invalid",
            "run algorithms do not match the expected algorithms",
        )

    runtime_records: list[dict[str, Any]] = []
    projection_records: list[dict[str, Any]] = []
    total_projected_retained_bytes = 0
    total_projected_retained_files = 0
    total_checkpoint_bytes = 0
    total_smoke_original_bytes = 0
    all_runtime_within_limit = runtime_limit is not None

    for algorithm in expected_algorithms:
        run = run_by_algorithm.get(algorithm)
        if run is None:
            all_runtime_within_limit = False
            continue

        manifest = run.get("manifest")
        if not isinstance(manifest, dict) or manifest.get("status") != "complete":
            invalid(
                "smoke_summary_manifest_invalid",
                f"{algorithm} manifest must have complete status",
            )

        runtime = run.get("runtime_projection")
        projected_hours: Decimal | None = None
        claimed_within_limit: Any = None
        run_limit: Decimal | None = None
        if isinstance(runtime, dict):
            projected_hours = _decimal(runtime.get("projected_task_hours"))
            run_limit = _decimal(runtime.get("limit_hours"))
            claimed_within_limit = runtime.get("within_limit")
        if projected_hours is None or projected_hours < 0:
            invalid(
                "runtime_projection_invalid",
                f"{algorithm} projected_task_hours must be finite and nonnegative",
            )
            calculated_within = False
        elif runtime_limit is None:
            calculated_within = False
        else:
            calculated_within = projected_hours <= runtime_limit
            if run_limit != runtime_limit:
                invalid(
                    "runtime_projection_limit_mismatch",
                    f"{algorithm} run limit differs from the summary limit",
                )
            if claimed_within_limit is not calculated_within:
                invalid(
                    "runtime_projection_claim_mismatch",
                    f"{algorithm} within_limit does not match recomputed runtime",
                )
        if not calculated_within:
            all_runtime_within_limit = False
            invalid(
                "runtime_projection_exceeds_limit",
                f"{algorithm} projected runtime exceeds the declared limit",
            )
        runtime_records.append(
            {
                "algorithm": algorithm,
                "projected_task_hours": (
                    float(projected_hours) if projected_hours is not None else None
                ),
                "within_limit_recomputed": calculated_within,
            }
        )

        artifacts = run.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            invalid(
                "smoke_summary_artifacts_invalid",
                f"{algorithm} artifacts must be a nonempty list",
            )
            continue
        retained_bytes = 0
        retained_files = 0
        checkpoint_bytes = 0
        checkpoint_files = 0
        smoke_only_result_bytes = 0
        smoke_only_result_files = 0
        seen_paths: set[str] = set()
        artifact_valid = True
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                artifact_valid = False
                continue
            artifact_path = artifact.get("path")
            artifact_bytes = artifact.get("bytes")
            if (
                not isinstance(artifact_path, str)
                or not artifact_path
                or not _is_int(artifact_bytes)
                or artifact_bytes < 0
                or artifact_path in seen_paths
            ):
                artifact_valid = False
                continue
            seen_paths.add(artifact_path)
            if _has_checkpoint_component(artifact_path):
                checkpoint_bytes += artifact_bytes
                checkpoint_files += 1
            else:
                retained_bytes += artifact_bytes
                retained_files += 1
                if PurePosixPath(artifact_path.replace("\\", "/")).name == SMOKE_ONLY_RESULT_NAME:
                    smoke_only_result_bytes += artifact_bytes
                    smoke_only_result_files += 1
        if not artifact_valid:
            invalid(
                "smoke_summary_artifacts_invalid",
                f"{algorithm} has a malformed or duplicate artifact record",
            )
            continue
        if smoke_only_result_files != 1:
            invalid(
                "smoke_summary_result_artifact_invalid",
                f"{algorithm} must inventory exactly one {SMOKE_ONLY_RESULT_NAME}",
            )
            continue
        candidate_bytes_per_trial = (
            retained_bytes
            - smoke_only_result_bytes
            + RETENTION_RECEIPT_BUDGET_BYTES_PER_TRIAL
            + RETENTION_SIDECAR_BYTES_PER_TRIAL
        )
        candidate_files_per_trial = (
            retained_files - smoke_only_result_files + RETENTION_FILES_PER_TRIAL
        )
        projected_candidate_bytes = (
            candidate_bytes_per_trial * trials
            + TASK_SUMMARY_BUDGET_BYTES_PER_ALGORITHM
            if trials is not None
            else None
        )
        projected_candidate_files = (
            candidate_files_per_trial * trials + TASK_SUMMARY_FILES_PER_ALGORITHM
            if trials is not None
            else None
        )
        if projected_candidate_bytes is not None and projected_candidate_files is not None:
            total_projected_retained_bytes += projected_candidate_bytes
            total_projected_retained_files += projected_candidate_files
            total_checkpoint_bytes += checkpoint_bytes * trials
            total_smoke_original_bytes += (retained_bytes + checkpoint_bytes) * trials
        projection_records.append(
            {
                "algorithm": algorithm,
                "checkpoint_bytes_excluded_per_trial": checkpoint_bytes,
                "checkpoint_files_excluded_per_trial": checkpoint_files,
                "smoke_retained_bytes_per_trial": retained_bytes,
                "smoke_retained_files_per_trial": retained_files,
                "smoke_only_result_bytes_per_trial": smoke_only_result_bytes,
                "smoke_only_result_files_per_trial": smoke_only_result_files,
                "candidate_receipt_budget_bytes_per_trial": (
                    RETENTION_RECEIPT_BUDGET_BYTES_PER_TRIAL
                    + RETENTION_SIDECAR_BYTES_PER_TRIAL
                ),
                "candidate_receipt_files_per_trial": RETENTION_FILES_PER_TRIAL,
                "retained_bytes_per_trial": candidate_bytes_per_trial,
                "retained_files_per_trial": candidate_files_per_trial,
                "task_summary_budget_bytes": TASK_SUMMARY_BUDGET_BYTES_PER_ALGORITHM,
                "task_summary_files": TASK_SUMMARY_FILES_PER_ALGORITHM,
                "projected_retained_bytes": projected_candidate_bytes,
                "projected_retained_files": projected_candidate_files,
            }
        )

    runtime_view = {
        "algorithms": runtime_records,
        "eligible_recomputed": all_runtime_within_limit,
        "limit_hours": float(runtime_limit) if runtime_limit is not None else None,
        "maximum_projected_task_hours": (
            max(
                (
                    record["projected_task_hours"]
                    for record in runtime_records
                    if record["projected_task_hours"] is not None
                ),
                default=None,
            )
        ),
    }
    if not valid or trials is None or len(projection_records) != len(expected_algorithms):
        return None, runtime_view

    projected_bytes = total_projected_retained_bytes
    projected_inodes = total_projected_retained_files
    safety_adjusted_bytes = _ceil_scaled(projected_bytes, safety_factor)
    safety_adjusted_inodes = _ceil_scaled(projected_inodes, safety_factor)
    projection = {
        "checkpoint_policy": "suppress_writes_with_bounded_candidate_evidence",
        "candidate_overhead_policy": {
            "retention_receipt_budget_bytes_per_trial": (
                RETENTION_RECEIPT_BUDGET_BYTES_PER_TRIAL
            ),
            "retention_sidecar_bytes_per_trial": RETENTION_SIDECAR_BYTES_PER_TRIAL,
            "retention_files_per_trial": RETENTION_FILES_PER_TRIAL,
            "smoke_only_result_name": SMOKE_ONLY_RESULT_NAME,
            "task_summary_budget_bytes_per_algorithm": (
                TASK_SUMMARY_BUDGET_BYTES_PER_ALGORITHM
            ),
            "task_summary_files_per_algorithm": TASK_SUMMARY_FILES_PER_ALGORITHM,
        },
        "fixed_reserve_bytes": fixed_reserve_bytes,
        "full_screening_trials_per_algorithm": trials,
        "per_algorithm": projection_records,
        "projected_checkpoint_bytes_avoided": total_checkpoint_bytes,
        "projected_retained_bytes": projected_bytes,
        "projected_retained_inodes": projected_inodes,
        "required_bytes": safety_adjusted_bytes + fixed_reserve_bytes,
        "required_inodes": safety_adjusted_inodes,
        "safety_adjusted_retained_bytes": safety_adjusted_bytes,
        "safety_adjusted_retained_inodes": safety_adjusted_inodes,
        "safety_factor": float(safety_factor),
        "smoke_projected_original_bytes": total_smoke_original_bytes,
    }
    return projection, runtime_view


def _quota_dimension(
    raw: Any,
    owner: str,
    dimension: str,
    add_reason: Callable[[str, str], None],
) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(raw, dict):
        add_reason(
            f"quota_{owner}_{dimension}_invalid",
            f"{owner}.{dimension} must be an object",
        )
        return None, False
    used = raw.get("used")
    soft = raw.get("soft_limit")
    hard = raw.get("hard_limit")
    grace_expired = raw.get("grace_expired")
    if (
        not _is_int(used)
        or not _is_int(soft)
        or not _is_int(hard)
        or used < 0
        or soft <= 0
        or hard <= 0
        or hard < soft
    ):
        add_reason(
            f"quota_{owner}_{dimension}_invalid",
            f"{owner}.{dimension} requires nonnegative usage and positive ordered limits",
        )
        return None, False
    valid = True
    if grace_expired is not False:
        add_reason(
            f"quota_{owner}_{dimension}_grace_expired",
            f"{owner}.{dimension} grace must be explicitly unexpired",
        )
        valid = False
    return (
        {
            "used": used,
            "soft_limit": soft,
            "hard_limit": hard,
            "soft_headroom": max(soft - used, 0),
            "hard_headroom": max(hard - used, 0),
            "grace_expired": grace_expired,
        },
        valid,
    )


def _validate_fresh_timestamp(
    raw: Any,
    *,
    name: str,
    now: datetime,
    max_age: timedelta,
    max_clock_skew: timedelta,
    add_reason: Callable[[str, str], None],
) -> tuple[datetime | None, bool]:
    parsed = _parse_timestamp(raw)
    if parsed is None:
        add_reason(f"{name}_timestamp_invalid", f"{name} timestamp must include a timezone")
        return None, False
    if parsed > now + max_clock_skew:
        add_reason(f"{name}_timestamp_future", f"{name} timestamp is in the future")
        return parsed, False
    if now - parsed > max_age:
        add_reason(f"{name}_stale", f"{name} timestamp is older than the freshness limit")
        return parsed, False
    return parsed, True


def _validate_quota(
    snapshot: dict[str, Any],
    *,
    expected_group: str,
    expected_gid: int,
    expected_project_id: int,
    expected_filesystem: str,
    expected_output_parent: str,
    now: datetime,
    max_age: timedelta,
    max_clock_skew: timedelta,
    add_reason: Callable[[str, str], None],
) -> tuple[dict[str, Any] | None, bool]:
    valid = True

    def invalid(code: str, message: str) -> None:
        nonlocal valid
        valid = False
        add_reason(code, message)

    if snapshot.get("schema_version") != QUOTA_SCHEMA_VERSION:
        invalid("quota_snapshot_schema_invalid", "unsupported quota schema_version")

    filesystem = snapshot.get("filesystem")
    if filesystem != expected_filesystem:
        invalid(
            "quota_filesystem_mismatch",
            "quota filesystem does not match the required filesystem",
        )

    captured_at, captured_fresh = _validate_fresh_timestamp(
        snapshot.get("captured_at"),
        name="quota_snapshot",
        now=now,
        max_age=max_age,
        max_clock_skew=max_clock_skew,
        add_reason=add_reason,
    )
    valid = valid and captured_fresh

    group = snapshot.get("group")
    if not isinstance(group, dict):
        invalid("quota_group_invalid", "group must be an object")
        group = {}
    if group.get("name") != expected_group:
        invalid("quota_group_name_mismatch", "quota group name does not match")
    if group.get("gid") != expected_gid or not _is_int(group.get("gid")):
        invalid("quota_group_gid_mismatch", "quota group GID does not match")

    group_byte_view, group_bytes_valid = _quota_dimension(
        group.get("bytes"), "group", "bytes", add_reason
    )
    group_inode_view, group_inodes_valid = _quota_dimension(
        group.get("inodes"), "group", "inodes", add_reason
    )
    valid = valid and group_bytes_valid and group_inodes_valid

    project = snapshot.get("project")
    if not isinstance(project, dict):
        invalid("quota_project_invalid", "project must be an object")
        project = {}
    if project.get("id") != expected_project_id or not _is_int(project.get("id")):
        invalid("quota_project_id_mismatch", "quota project ID does not match")
    project_byte_view, project_bytes_valid = _quota_dimension(
        project.get("bytes"), "project", "bytes", add_reason
    )
    project_inode_view, project_inodes_valid = _quota_dimension(
        project.get("inodes"), "project", "inodes", add_reason
    )
    valid = valid and project_bytes_valid and project_inodes_valid

    probe = snapshot.get("write_probe")
    probe_view: dict[str, Any] | None = None
    if not isinstance(probe, dict):
        invalid("quota_write_probe_invalid", "write_probe must be an object")
    else:
        probe_valid = True
        completed_at, completed_fresh = _validate_fresh_timestamp(
            probe.get("completed_at"),
            name="quota_write_probe",
            now=now,
            max_age=max_age,
            max_clock_skew=max_clock_skew,
            add_reason=add_reason,
        )
        probe_valid = probe_valid and completed_fresh
        if probe.get("filesystem") != expected_filesystem:
            add_reason(
                "quota_write_probe_filesystem_mismatch",
                "write probe filesystem does not match",
            )
            probe_valid = False
        if probe.get("gid") != expected_gid or not _is_int(probe.get("gid")):
            add_reason(
                "quota_write_probe_gid_mismatch",
                "write probe GID does not match the expected output group",
            )
            probe_valid = False
        probe_path = probe.get("path")
        if not isinstance(probe_path, str) or not probe_path:
            add_reason("quota_write_probe_path_invalid", "write probe path is missing")
            probe_valid = False
        else:
            expected_parent = PurePosixPath(expected_output_parent)
            candidate = PurePosixPath(probe_path)
            if (
                not candidate.is_absolute()
                or candidate.parent != expected_parent
                or candidate.name in {"", ".", ".."}
            ):
                add_reason(
                    "quota_write_probe_path_invalid",
                    "write probe path must be a direct child of expected_output_parent",
                )
                probe_valid = False
        bytes_written = probe.get("bytes_written")
        if not _is_int(bytes_written) or bytes_written <= 0:
            add_reason(
                "quota_write_probe_write_failed",
                "write probe must report a positive integer byte count",
            )
            probe_valid = False
        if probe.get("fsync_succeeded") is not True:
            add_reason(
                "quota_write_probe_fsync_failed",
                "write probe must report fsync_succeeded=true",
            )
            probe_valid = False
        if probe.get("removed") is not True:
            add_reason(
                "quota_write_probe_not_removed",
                "write probe must report removed=true after fsync",
            )
            probe_valid = False
        if (
            captured_at is not None
            and completed_at is not None
            and completed_at > captured_at + max_clock_skew
        ):
            add_reason(
                "quota_write_probe_after_snapshot",
                "write probe completed after the quota capture window",
            )
            probe_valid = False
        probe_view = {
            "bytes_written": bytes_written if _is_int(bytes_written) else None,
            "completed_at": (
                _timestamp_text(completed_at) if completed_at is not None else None
            ),
            "filesystem": probe.get("filesystem"),
            "fsync_succeeded": probe.get("fsync_succeeded") is True,
            "gid": probe.get("gid") if _is_int(probe.get("gid")) else None,
            "path": probe_path if isinstance(probe_path, str) else None,
            "removed": probe.get("removed") is True,
        }
        valid = valid and probe_valid

    if (
        group_byte_view is None
        or group_inode_view is None
        or project_byte_view is None
        or project_inode_view is None
    ):
        return None, False
    return (
        {
            "captured_at": (
                _timestamp_text(captured_at) if captured_at is not None else None
            ),
            "filesystem": filesystem,
            "group": {
                "bytes": group_byte_view,
                "gid": group.get("gid"),
                "inodes": group_inode_view,
                "name": group.get("name"),
            },
            "project": {
                "bytes": project_byte_view,
                "id": project.get("id"),
                "inodes": project_inode_view,
            },
            "write_probe": probe_view,
        },
        valid,
    )


def assess_screening_storage(
    smoke_summary_path: str | Path,
    quota_snapshot_path: str | Path,
    *,
    expected_group: str,
    expected_gid: int,
    expected_project_id: int,
    expected_filesystem: str,
    expected_output_parent: str,
    now: datetime | None = None,
    max_quota_age_seconds: int = DEFAULT_MAX_QUOTA_AGE_SECONDS,
    max_clock_skew_seconds: int = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
    safety_factor: float | str | Decimal = DEFAULT_SAFETY_FACTOR,
    fixed_reserve_bytes: int = DEFAULT_FIXED_RESERVE_BYTES,
    expected_algorithms: Sequence[str] = EXPECTED_ALGORITHMS,
) -> dict[str, Any]:
    """Assess capacity from signed smoke and quota evidence without side effects.

    The returned dictionary is deterministic for fixed input bytes and ``now``.
    Any invalid, stale, inconsistent, or insufficient evidence produces a hold.
    """

    algorithms = tuple(expected_algorithms)
    if (
        len(algorithms) != 5
        or len(set(algorithms)) != 5
        or any(not isinstance(value, str) or not value for value in algorithms)
    ):
        raise ValueError("expected_algorithms must contain exactly five unique names")
    if not isinstance(expected_group, str) or not expected_group:
        raise ValueError("expected_group must be nonempty")
    if not _is_int(expected_gid) or expected_gid < 0:
        raise ValueError("expected_gid must be a nonnegative integer")
    if not _is_int(expected_project_id) or expected_project_id < 0:
        raise ValueError("expected_project_id must be a nonnegative integer")
    if (
        not isinstance(expected_filesystem, str)
        or not expected_filesystem.startswith("/")
        or expected_filesystem == "/"
    ):
        raise ValueError("expected_filesystem must be a non-root absolute path")
    if not isinstance(expected_output_parent, str):
        raise ValueError("expected_output_parent must be an absolute path")
    output_parent = PurePosixPath(expected_output_parent)
    filesystem_root = PurePosixPath(expected_filesystem)
    if (
        not expected_output_parent.startswith("/")
        or expected_output_parent == "/"
        or ".." in output_parent.parts
        or not output_parent.is_relative_to(filesystem_root)
    ):
        raise ValueError(
            "expected_output_parent must be an absolute canonical child of expected_filesystem"
        )
    expected_output_parent = str(output_parent)
    if not _is_int(max_quota_age_seconds) or max_quota_age_seconds <= 0:
        raise ValueError("max_quota_age_seconds must be positive")
    if not _is_int(max_clock_skew_seconds) or max_clock_skew_seconds < 0:
        raise ValueError("max_clock_skew_seconds must be nonnegative")
    factor = _decimal(safety_factor)
    if factor is None or factor < 1:
        raise ValueError("safety_factor must be finite and at least 1")
    if not _is_int(fixed_reserve_bytes) or fixed_reserve_bytes < 0:
        raise ValueError("fixed_reserve_bytes must be a nonnegative integer")
    assessment_time = now or datetime.now(timezone.utc)
    if assessment_time.tzinfo is None or assessment_time.utcoffset() is None:
        raise ValueError("now must include a timezone")
    assessment_time = assessment_time.astimezone(timezone.utc)

    problems: list[dict[str, str]] = []

    def add_reason(code: str, message: str) -> None:
        item = {"code": code, "message": message}
        if item not in problems:
            problems.append(item)

    smoke_path = Path(smoke_summary_path)
    quota_path = Path(quota_snapshot_path)
    smoke_evidence = _verify_sidecar(smoke_path, "smoke_summary", add_reason)
    quota_evidence = _verify_sidecar(quota_path, "quota_snapshot", add_reason)
    smoke = _load_object(smoke_path, "smoke_summary", add_reason)
    quota = _load_object(quota_path, "quota_snapshot", add_reason)

    if smoke is None:
        projection = None
        runtime = {
            "algorithms": [],
            "eligible_recomputed": False,
            "limit_hours": None,
            "maximum_projected_task_hours": None,
        }
    else:
        projection, runtime = _project_smoke(
            smoke,
            expected_algorithms=algorithms,
            safety_factor=factor,
            fixed_reserve_bytes=fixed_reserve_bytes,
            add_reason=add_reason,
        )

    if quota is None:
        quota_view = None
        quota_valid = False
    else:
        quota_view, quota_valid = _validate_quota(
            quota,
            expected_group=expected_group,
            expected_gid=expected_gid,
            expected_project_id=expected_project_id,
            expected_filesystem=expected_filesystem,
            expected_output_parent=expected_output_parent,
            now=assessment_time,
            max_age=timedelta(seconds=max_quota_age_seconds),
            max_clock_skew=timedelta(seconds=max_clock_skew_seconds),
            add_reason=add_reason,
        )

    capacity_checks: dict[str, bool | None] = {
        f"{owner}_{dimension}_fit_{limit}_headroom": None
        for owner in ("group", "project")
        for dimension in ("bytes", "inodes")
        for limit in ("soft", "hard")
    }
    if projection is not None and quota_view is not None:
        required_bytes = projection["required_bytes"]
        required_inodes = projection["required_inodes"]
        capacity_checks = {}
        for owner in ("group", "project"):
            for dimension, required in (
                ("bytes", required_bytes),
                ("inodes", required_inodes),
            ):
                dimension_quota = quota_view[owner][dimension]
                for limit in ("soft", "hard"):
                    key = f"{owner}_{dimension}_fit_{limit}_headroom"
                    capacity_checks[key] = required <= dimension_quota[f"{limit}_headroom"]
        for key, passed in capacity_checks.items():
            if not passed:
                add_reason(f"insufficient_{key.removesuffix('_headroom')}", key)

    reason_codes = sorted({problem["code"] for problem in problems})
    candidate_eligible = (
        projection is not None
        and quota_view is not None
        and quota_valid
        and smoke_evidence["sidecar_verified"]
        and quota_evidence["sidecar_verified"]
        and runtime["eligible_recomputed"]
        and all(value is True for value in capacity_checks.values())
        and not reason_codes
    )
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "assessed_at": _timestamp_text(assessment_time),
        "candidate_capacity_eligible": candidate_eligible,
        "decision": "pass" if candidate_eligible else "hold",
        "reason_codes": reason_codes,
        "validation_errors": sorted(
            problems,
            key=lambda item: (item["code"], item["message"]),
        ),
        "policy": {
            "expected_algorithms": list(algorithms),
            "expected_filesystem": expected_filesystem,
            "expected_gid": expected_gid,
            "expected_group": expected_group,
            "expected_output_parent": expected_output_parent,
            "expected_project_id": expected_project_id,
            "fixed_reserve_bytes": fixed_reserve_bytes,
            "max_clock_skew_seconds": max_clock_skew_seconds,
            "max_quota_age_seconds": max_quota_age_seconds,
            "safety_factor": float(factor),
        },
        "inputs": {
            "quota_snapshot": quota_evidence,
            "smoke_summary": smoke_evidence,
        },
        "runtime": runtime,
        "projection": projection,
        "quota": quota_view,
        "capacity_checks": capacity_checks,
    }


# Short aliases keep the public API discoverable without changing semantics.
assess_capacity = assess_screening_storage
assess_screening_capacity = assess_screening_storage


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assess checkpoint-free full-screening storage capacity"
    )
    parser.add_argument("--smoke-summary", required=True)
    parser.add_argument("--quota-snapshot", required=True)
    parser.add_argument("--expected-group", required=True)
    parser.add_argument("--expected-gid", required=True, type=int)
    parser.add_argument("--expected-project-id", required=True, type=int)
    parser.add_argument("--expected-filesystem", required=True)
    parser.add_argument("--expected-output-parent", required=True)
    parser.add_argument("--output")
    parser.add_argument("--as-of", help="Timezone-aware ISO-8601 assessment time")
    parser.add_argument(
        "--max-quota-age-seconds",
        type=int,
        default=DEFAULT_MAX_QUOTA_AGE_SECONDS,
    )
    parser.add_argument(
        "--max-clock-skew-seconds",
        type=int,
        default=DEFAULT_MAX_CLOCK_SKEW_SECONDS,
    )
    parser.add_argument("--safety-factor", default=str(DEFAULT_SAFETY_FACTOR))
    parser.add_argument(
        "--fixed-reserve-bytes",
        type=int,
        default=DEFAULT_FIXED_RESERVE_BYTES,
    )
    parser.add_argument(
        "--expected-algorithm",
        action="append",
        dest="expected_algorithms",
        help="Repeat exactly five times; defaults to ppo,a2c,ddpg,td3,sac",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    as_of = _parse_timestamp(args.as_of) if args.as_of is not None else None
    if args.as_of is not None and as_of is None:
        parser.error("--as-of must be a timezone-aware ISO-8601 timestamp")
    decision = assess_screening_storage(
        args.smoke_summary,
        args.quota_snapshot,
        expected_group=args.expected_group,
        expected_gid=args.expected_gid,
        expected_project_id=args.expected_project_id,
        expected_filesystem=args.expected_filesystem,
        expected_output_parent=args.expected_output_parent,
        now=as_of,
        max_quota_age_seconds=args.max_quota_age_seconds,
        max_clock_skew_seconds=args.max_clock_skew_seconds,
        safety_factor=args.safety_factor,
        fixed_reserve_bytes=args.fixed_reserve_bytes,
        expected_algorithms=args.expected_algorithms or EXPECTED_ALGORITHMS,
    )
    if args.output:
        write_decision(args.output, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["candidate_capacity_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
