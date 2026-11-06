from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import tarfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence

from scripts.assess_screening_storage import assess_screening_storage


SCHEMA_VERSION = 1
DEFAULT_MAX_AGE_SECONDS = 900
DEFAULT_MAX_CLOCK_SKEW_SECONDS = 60
DEFAULT_MINIMUM_REMAINING_SECONDS = 120
DEFAULT_MINIMUM_GROUP_SOFT_MARGIN_BYTES = 1_073_741_824
DEFAULT_FIXED_RESERVE_BYTES = 268_435_456
DEFAULT_EXPECTED_ALGORITHMS = ("ppo", "a2c", "ddpg", "td3", "sac")

EXPECTED_ACCOUNT = "25sfcs03"
EXPECTED_USER = "s00658"
EXPECTED_GROUP = "s00658"
EXPECTED_GID = 1039
EXPECTED_PROJECT_ID = 25110303
EXPECTED_FILESYSTEM = "/lustrefs/project/25sfcs03"
EXPECTED_PROJECT_ROOT = "/lustrefs/project/25sfcs03/drl_thai_stock"
EXPECTED_OUTPUT_PARENT = f"{EXPECTED_PROJECT_ROOT}/results"

EXPECTED_CAMPAIGN_COUNTS = {"PENDING": 25, "READY": 1, "VERIFIED": 9}
EXPECTED_SCREENING_PROGRESS = "0/150 complete manifests"
EXPECTED_SCREENING_MANIFESTS = 150

DEFAULT_SPEC_PATH = "config/research_campaign_storage_safe.yaml"
DEFAULT_SMOKE_SUMMARY_PATH = (
    "results/screening_storage_safe_smoke_3550/screening_smoke_summary.json"
)
EXPECTED_SMOKE_SUMMARY_SHA256 = (
    "e3095e1149a6cefe9d12cef97ca38b0b1ef1585bfa3e4d20bf66a4ce977b88c3"
)
DEFAULT_MANIFEST_PATH = (
    "campaigns/large_scale_thai_drl_v2/deployment/"
    "bistka_screening_storage_safe_cutover_v2_manifest.json"
)
DEFAULT_BUNDLE_PATH = (
    "campaigns/large_scale_thai_drl_v2/deployment/"
    "bistka_screening_storage_safe_cutover_v2.tar.gz"
)
DEFAULT_PREFLIGHT_PATH = "reports/bistka_screening_storage_safe_cutover_v2_preflight.json"
DEFAULT_LOCKED_CONFIG_PATH = (
    "campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml"
)
DEFAULT_FEATURE_PATH = "data/processed/features_real_ohlcv_market.csv"
DEFAULT_JOB_SCRIPT_PATH = "slurm/large_scale_screening_storage_safe.sbatch"

CONTROLLER_RENDER_COMMAND = (
    "python -m scripts.render_authorized_screening --project-root . "
    "--receipt <receipt-path>"
)
AUTHORIZATION_OUTPUT_DIRECTORY = "reports/screening_execution_windows"
AUTHORIZATION_NAME = re.compile(
    r"^screening_execution_window_(?P<stamp>[0-9]{8}T[0-9]{6}Z)\.json$"
)
OPERATIONAL_EVIDENCE_DIRECTORY = (
    "campaigns/large_scale_thai_drl_v2/operational/screening_capacity"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEADER = re.compile(r"^(captured_at|hostname|user)=(.*)$")
_SECTION = re.compile(r"^##\s+(.+?)\s*$")
_SBATCH = re.compile(r"^#SBATCH\s+--([A-Za-z0-9-]+)(?:=|\s+)(.+?)\s*$")
_AUTHORIZATION_CAPABILITY_KEY = os.urandom(32)
_ISSUED_AUTHORIZATION_CAPABILITIES: dict[int, str] = {}


def _authorization_capability_tag(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hmac.new(
        _AUTHORIZATION_CAPABILITY_KEY,
        payload,
        hashlib.sha256,
    ).hexdigest()


class _PreparedExecutionWindow(dict[str, Any]):
    """In-process capability created only after the full gate is evaluated."""

    __slots__ = ("_capability_tag",)

    def __init__(self, value: dict[str, Any]) -> None:
        super().__init__(value)
        self._capability_tag = _authorization_capability_tag(self)

    def capability_is_current(self) -> bool:
        return hmac.compare_digest(
            self._capability_tag,
            _authorization_capability_tag(self),
        )


def _issue_execution_window(value: dict[str, Any]) -> _PreparedExecutionWindow:
    capability = _PreparedExecutionWindow(value)
    _ISSUED_AUTHORIZATION_CAPABILITIES[id(capability)] = capability._capability_tag
    return capability


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class ReleaseExpectations:
    release_id: str
    manifest_sha256: str
    bundle_sha256: str
    preflight_sha256: str
    locked_config_sha256: str
    feature_sha256: str
    job_script_sha256: str
    spec_sha256: str
    file_count: int = 248
    campaign_id: str = "large_scale_thai_drl_v2"
    target_project_root: str = EXPECTED_PROJECT_ROOT
    critical_paths: tuple[str, ...] = (
        DEFAULT_SPEC_PATH,
        DEFAULT_JOB_SCRIPT_PATH,
        "scripts/assess_screening_storage.py",
        "scripts/capture_bistka_capacity.sh",
        "scripts/capture_screening_quota_snapshot.py",
        "src/campaigns/control.py",
        "src/campaigns/gates.py",
        "src/campaigns/slurm.py",
    )


LOCKED_RELEASE = ReleaseExpectations(
    release_id="56358f2a4996f654426814d88e622b7f217ac89bfc02f8bbdacc27e66873b55e",
    manifest_sha256="439b590bc4a85f03a6d98723ee3d28e40acdc00d95bdfbf48d310f07d867e25c",
    bundle_sha256="a5135665dd14fdfcd0a5dcdbfab70a7e6f88fd78fc75ff946df9851c01a4072e",
    preflight_sha256="1ccac9827e2f018927d711c0fb0918119c79f8721bd6d82d3d08ed026ab59f0c",
    locked_config_sha256=(
        "d7b6ab71ede33ae4c4106fded3078b9a4eb3c56af345e67960061feea1897c1e"
    ),
    feature_sha256="b877b08ed7a022db47739423291df0232ce377869cb7e0de6162ab82291ac6cc",
    job_script_sha256=(
        "8fef23d9aaf0b3a8472f74e01f8a64c8b1cefc2828963967f6945f92a58402b6"
    ),
    spec_sha256="2a7934bb3c3256cee012372c25e204585436b1a4239de4a4ad2983366419fadd",
)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_project_file(project_root: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    candidate = project_root / relative
    if candidate.is_symlink():
        raise ValueError(f"project input must not be a symlink: {raw_path}")
    resolved = candidate.resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not resolved.is_relative_to(project_root)
    ):
        raise ValueError(f"unsafe project path: {raw_path}")
    return resolved


def _load_json_object(
    path: Path,
    *,
    label: str,
    add_reason: Callable[[str, str], None],
) -> dict[str, Any] | None:
    if path.is_symlink():
        add_reason(f"{label}_symlink_rejected", f"{label} must not be a symlink")
        return None
    if not path.is_file():
        add_reason(f"{label}_missing", f"missing {label}: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        add_reason(f"{label}_invalid", f"cannot parse {label}: {exc}")
        return None
    if not isinstance(value, dict):
        add_reason(f"{label}_invalid", f"{label} must contain a JSON object")
        return None
    return value


def _verify_exact_hash(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    add_reason: Callable[[str, str], None],
    require_sidecar: bool = False,
) -> dict[str, Any]:
    record = {
        "path": str(path),
        "sha256": None,
        "expected_sha256": expected_sha256,
        "verified": False,
        "sidecar_path": str(Path(str(path) + ".sha256")) if require_sidecar else None,
        "sidecar_verified": None if not require_sidecar else False,
    }
    if not _SHA256.fullmatch(expected_sha256):
        add_reason(f"{label}_expectation_invalid", f"invalid expected hash for {label}")
        return record
    if path.is_symlink():
        add_reason(f"{label}_symlink_rejected", f"{label} must not be a symlink")
        return record
    if not path.is_file():
        add_reason(f"{label}_missing", f"missing {label}: {path}")
        return record
    actual = _sha256_file(path)
    record["sha256"] = actual
    if actual != expected_sha256:
        add_reason(f"{label}_sha256_mismatch", f"{label} SHA-256 does not match")
        return record
    if require_sidecar:
        sidecar = Path(str(path) + ".sha256")
        if sidecar.is_symlink():
            add_reason(
                f"{label}_sidecar_symlink_rejected",
                f"{label} SHA-256 sidecar must not be a symlink",
            )
            return record
        if not sidecar.is_file():
            add_reason(f"{label}_sidecar_missing", f"missing sidecar for {label}")
            return record
        try:
            sidecar_bytes = sidecar.read_bytes()
        except OSError as exc:
            add_reason(f"{label}_sidecar_invalid", f"cannot read sidecar for {label}: {exc}")
            return record
        if sidecar_bytes != (expected_sha256 + "\n").encode("ascii"):
            add_reason(
                f"{label}_sidecar_invalid",
                f"{label} sidecar must be lowercase SHA-256 followed by one newline",
            )
            return record
        record["sidecar_verified"] = True
    record["verified"] = True
    return record


def _verify_checksum_bound_snapshot(
    path: Path,
    *,
    label: str,
    add_reason: Callable[[str, str], None],
    maximum_bytes: int = 2_000_000,
) -> dict[str, Any]:
    record = {
        "path": str(path),
        "sha256": None,
        "sidecar_path": str(Path(str(path) + ".sha256")),
        "sidecar_verified": False,
    }
    if path.is_symlink():
        add_reason(f"{label}_symlink_rejected", f"{label} must not be a symlink")
        return record
    if not path.is_file():
        add_reason(f"{label}_missing", f"missing {label}: {path}")
        return record
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        add_reason(
            f"{label}_size_invalid",
            f"{label} size must be between 1 and {maximum_bytes} bytes",
        )
        return record
    actual = _sha256_file(path)
    record["sha256"] = actual
    sidecar = Path(str(path) + ".sha256")
    if sidecar.is_symlink():
        add_reason(
            f"{label}_sidecar_symlink_rejected",
            f"{label} sidecar must not be a symlink",
        )
        return record
    if not sidecar.is_file():
        add_reason(f"{label}_sidecar_missing", f"missing sidecar for {label}")
        return record
    try:
        sidecar_bytes = sidecar.read_bytes()
    except OSError as exc:
        add_reason(f"{label}_sidecar_invalid", f"cannot read sidecar for {label}: {exc}")
        return record
    if sidecar_bytes != (actual + "\n").encode("ascii"):
        add_reason(
            f"{label}_sidecar_invalid",
            f"{label} sidecar must exactly match the input SHA-256",
        )
        return record
    record["sidecar_verified"] = True
    return record


def _sections(
    lines: list[str],
    *,
    add_reason: Callable[[str, str], None],
) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        match = _SECTION.fullmatch(line)
        if match:
            name = match.group(1).strip()
            if name in sections:
                add_reason(
                    "scheduler_section_duplicate",
                    f"scheduler snapshot repeats section {name!r}",
                )
            else:
                sections[name] = []
            current = name
        elif current is not None:
            sections[current].append(line)
    return sections


def _parse_positive_decimal(value: str) -> Decimal | None:
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    if not parsed.is_finite() or parsed <= 0:
        return None
    return parsed


def parse_scheduler_snapshot(
    path: str | Path,
    *,
    now: datetime,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    max_clock_skew_seconds: int = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
    expected_account: str = EXPECTED_ACCOUNT,
    expected_user: str = EXPECTED_USER,
    expected_bundle_sha256: str = LOCKED_RELEASE.bundle_sha256,
    expected_manifest_sha256: str = LOCKED_RELEASE.manifest_sha256,
) -> dict[str, Any]:
    problems: list[dict[str, str]] = []

    def add_reason(code: str, message: str) -> None:
        item = {"code": code, "message": message}
        if item not in problems:
            problems.append(item)

    snapshot_path = Path(path)
    evidence = _verify_checksum_bound_snapshot(
        snapshot_path,
        label="scheduler_snapshot",
        add_reason=add_reason,
    )
    if not evidence["sidecar_verified"]:
        text = ""
    else:
        try:
            text = snapshot_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            add_reason("scheduler_snapshot_invalid", f"cannot read scheduler snapshot: {exc}")
            text = ""
    lines = text.splitlines()
    if not lines or lines[0] != "# BistKA Capacity Snapshot":
        add_reason(
            "scheduler_header_invalid",
            "scheduler snapshot must start with the exact BistKA header",
        )

    header_values: dict[str, str] = {}
    header_counts: Counter[str] = Counter()
    for line in lines:
        match = _HEADER.fullmatch(line)
        if match:
            key, value = match.groups()
            header_counts[key] += 1
            header_values[key] = value.strip()
    for key in ("captured_at", "hostname", "user"):
        if header_counts[key] != 1:
            add_reason(
                "scheduler_header_invalid",
                f"scheduler snapshot must contain exactly one {key}= header",
            )
    if header_values.get("hostname") != "bistka-master":
        add_reason("scheduler_hostname_mismatch", "scheduler hostname must be bistka-master")
    if header_values.get("user") != expected_user:
        add_reason("scheduler_user_mismatch", "scheduler user does not match")

    captured_at = _parse_timestamp(header_values.get("captured_at"))
    if captured_at is None:
        add_reason(
            "scheduler_timestamp_invalid",
            "scheduler captured_at must be timezone-aware ISO-8601",
        )
    else:
        future_skew = timedelta(seconds=max_clock_skew_seconds)
        max_age = timedelta(seconds=max_age_seconds)
        if captured_at > now + future_skew:
            add_reason("scheduler_timestamp_future", "scheduler snapshot is too far in the future")
        if now - captured_at > max_age:
            add_reason("scheduler_snapshot_stale", "scheduler snapshot is older than the limit")

    parsed_sections = _sections(lines, add_reason=add_reason)
    required_sections = (
        "mycredit",
        "myquota",
        "sinfo",
        f"squeue -u {expected_user}",
        f"squeue -A {expected_account}",
        (
            f"sacctmgr -nP show assoc where user={expected_user} "
            f"account={expected_account} format=User,Account,Partition,QOS"
        ),
        (
            f"sacctmgr -nP show qos where name={expected_account} "
            "format=Name,GrpJobs,GrpSubmitJobs,GrpTRES,Flags"
        ),
        "scontrol show partition -o",
        "deployed release identity",
        "live scheduler policy",
    )
    for name in required_sections:
        if name not in parsed_sections:
            add_reason(
                "scheduler_section_missing",
                f"scheduler snapshot is missing section {name!r}",
            )

    error_patterns = (
        re.compile(r"^\s*unavailable\s*$", re.IGNORECASE),
        re.compile(r"command not found", re.IGNORECASE),
        re.compile(r"permission denied", re.IGNORECASE),
        re.compile(r"slurm_load_.*error", re.IGNORECASE),
    )
    critical_scheduler_lines = [
        line
        for name in required_sections
        for line in parsed_sections.get(name, [])
    ]
    for line in critical_scheduler_lines:
        if any(pattern.search(line) for pattern in error_patterns):
            add_reason(
                "scheduler_command_error",
                f"required scheduler evidence contains a command failure: {line.strip()}",
            )
            break

    credit_rows: dict[str, dict[str, Any]] = {}
    for line in parsed_sections.get("mycredit", []):
        fields = line.split()
        if len(fields) != 6 or fields[0] not in {expected_account, expected_user}:
            continue
        if fields[0] in credit_rows:
            add_reason(
                "scheduler_credit_duplicate",
                f"duplicate credit row for {fields[0]}",
            )
            continue
        allocation = _parse_positive_decimal(fields[1])
        remaining = _parse_positive_decimal(fields[3])
        if allocation is None or remaining is None:
            add_reason(
                "scheduler_credit_invalid",
                f"credit row for {fields[0]} has invalid allocation or remaining credit",
            )
        credit_rows[fields[0]] = {
            "allocation_service_hours": float(allocation) if allocation is not None else None,
            "remaining_service_hours": float(remaining) if remaining is not None else None,
            "expiry": fields[5],
        }
    for identity in (expected_account, expected_user):
        if identity not in credit_rows:
            add_reason(
                "scheduler_credit_missing",
                f"missing numeric credit row for {identity}",
            )
    account_credit = credit_rows.get(expected_account)
    if account_credit:
        expiry_text = account_credit["expiry"]
        try:
            expiry = date.fromisoformat(expiry_text)
        except (TypeError, ValueError):
            add_reason(
                "scheduler_credit_expiry_invalid",
                "project credit expiry must be an ISO calendar date",
            )
        else:
            if expiry < now.date():
                add_reason("scheduler_credit_expired", "project credit has expired")

    compute_long_rows: list[dict[str, Any]] = []
    for line in parsed_sections.get("sinfo", []):
        fields = line.split()
        if not fields or fields[0].rstrip("*") != "compute-long":
            continue
        if len(fields) < 5:
            add_reason(
                "scheduler_partition_invalid",
                "compute-long row is truncated",
            )
            continue
        try:
            nodes = int(fields[3])
        except ValueError:
            nodes = 0
        row = {
            "available": fields[1],
            "time_limit": fields[2],
            "nodes": nodes,
            "state": fields[4].rstrip("*+~#$@"),
        }
        compute_long_rows.append(row)
    acceptable_partition_rows = [
        row
        for row in compute_long_rows
        if row["available"] == "up"
        and row["time_limit"] == "4-00:00:00"
        and row["nodes"] >= 1
        and row["state"] not in {"down", "drain", "drng", "fail", "maint", "unk"}
    ]
    if not acceptable_partition_rows:
        add_reason(
            "scheduler_compute_long_unavailable",
            "compute-long must be up with a four-day limit and at least one usable node",
        )

    queue_lines = [
        line.strip()
        for line in parsed_sections.get(f"squeue -u {expected_user}", [])
        if line.strip()
    ]
    if not queue_lines or not queue_lines[0].startswith("JOBID"):
        add_reason(
            "scheduler_queue_header_invalid",
            "squeue section must contain its JOBID header",
        )
        active_queue_lines = queue_lines
    else:
        active_queue_lines = [
            line
            for line in queue_lines[1:]
            if line.lower() not in {"(no active jobs)", "no active jobs"}
        ]
    if active_queue_lines:
        add_reason(
            "scheduler_queue_not_empty",
            "the screening user must have zero queued or running jobs",
        )

    account_queue_lines = [
        line.strip()
        for line in parsed_sections.get(f"squeue -A {expected_account}", [])
        if line.strip()
    ]
    if not account_queue_lines or not account_queue_lines[0].startswith("JOBID"):
        add_reason(
            "scheduler_account_queue_header_invalid",
            "account-level squeue section must contain its JOBID header",
        )
        active_account_queue_lines = account_queue_lines
    else:
        active_account_queue_lines = [
            line
            for line in account_queue_lines[1:]
            if line.lower() not in {"(no active jobs)", "no active jobs"}
        ]
    if active_account_queue_lines:
        add_reason(
            "scheduler_account_queue_not_empty",
            "the project account must have zero queued or running jobs",
        )

    association_section = parsed_sections.get(
        (
            f"sacctmgr -nP show assoc where user={expected_user} "
            f"account={expected_account} format=User,Account,Partition,QOS"
        ),
        [],
    )
    association_rows = [line.strip() for line in association_section if line.strip()]
    expected_association = f"{expected_user}|{expected_account}||{expected_account}"
    if association_rows != [expected_association]:
        add_reason(
            "scheduler_association_mismatch",
            "scheduler association must exactly bind the user, account, and QOS",
        )

    qos_section = parsed_sections.get(
        (
            f"sacctmgr -nP show qos where name={expected_account} "
            "format=Name,GrpJobs,GrpSubmitJobs,GrpTRES,Flags"
        ),
        [],
    )
    qos_rows = [line.strip() for line in qos_section if line.strip()]
    qos_valid = False
    if len(qos_rows) == 1:
        fields = qos_rows[0].split("|")
        if len(fields) == 5:
            tres_items = [item for item in fields[3].split(",") if item]
            tres_pairs = [item.split("=", 1) for item in tres_items if item.count("=") == 1]
            tres = {key: value for key, value in tres_pairs}
            qos_valid = (
                fields[0] == expected_account
                and fields[1] == "2"
                and fields[2] == "5"
                and len(tres_pairs) == len(tres_items) == len(tres)
                and tres == {"cpu": "32", "gres/gpu": "2"}
                and fields[4] == "NoDecay"
            )
    if not qos_valid:
        add_reason(
            "scheduler_qos_mismatch",
            "QOS must allow exactly 2 running, 5 submitted, 32 CPU, and 2 GPU",
        )

    raw_partition_rows = [
        line.strip()
        for line in parsed_sections.get("scontrol show partition -o", [])
        if line.strip().startswith("PartitionName=compute-long")
    ]
    partition_control: dict[str, str] = {}
    if len(raw_partition_rows) == 1:
        for token in raw_partition_rows[0].split():
            if token.count("=") == 1:
                key, value = token.split("=", 1)
                if key in partition_control:
                    add_reason(
                        "scheduler_partition_control_invalid",
                        f"compute-long control row repeats {key}",
                    )
                partition_control[key] = value
    if (
        len(raw_partition_rows) != 1
        or partition_control.get("PartitionName") != "compute-long"
        or partition_control.get("State") != "UP"
        or partition_control.get("MaxTime") != "4-00:00:00"
    ):
        add_reason(
            "scheduler_partition_control_mismatch",
            "scontrol must report compute-long State=UP and MaxTime=4-00:00:00",
        )

    remote_release_lines = [
        line.strip()
        for line in parsed_sections.get("deployed release identity", [])
        if line.strip()
    ]
    remote_release: dict[str, str] = {}
    for line in remote_release_lines:
        if line.count("=") != 1:
            add_reason(
                "scheduler_remote_release_invalid",
                "remote release identity contains a malformed line",
            )
            continue
        key, value = line.split("=", 1)
        if key in remote_release:
            add_reason(
                "scheduler_remote_release_invalid",
                f"remote release identity repeats {key}",
            )
            continue
        remote_release[key] = value
    expected_remote_release = {
        "deployed_bundle_sha256": expected_bundle_sha256,
        "deployment_manifest_sha256": expected_manifest_sha256,
    }
    if remote_release != expected_remote_release:
        add_reason(
            "scheduler_remote_release_mismatch",
            "live BistKA release marker does not match the locked screening release",
        )

    policy_text = "\n".join(parsed_sections.get("live scheduler policy", []))
    required_policy = (
        r"(?m)^Maximum Job Submission: 5 jobs/group$",
        r"(?m)^Maximum Job Running: 2 jobs/group$",
        r"(?m)^Maximum CPU-cores = 32 CPU-cores/group$",
        r"(?m)^Maximum GPUs = 2 GPUs/group$",
        r"(?m)^compute-long: 4 days$",
    )
    if any(re.search(pattern, policy_text) is None for pattern in required_policy):
        add_reason(
            "scheduler_policy_mismatch",
            "live scheduler policy does not match the bounded campaign policy",
        )

    reason_codes = sorted({item["code"] for item in problems})
    return {
        "valid": not reason_codes,
        "reason_codes": reason_codes,
        "validation_errors": sorted(
            problems,
            key=lambda item: (item["code"], item["message"]),
        ),
        "evidence": evidence,
        "captured_at": _utc_text(captured_at) if captured_at is not None else None,
        "hostname": header_values.get("hostname"),
        "user": header_values.get("user"),
        "credit": credit_rows,
        "compute_long": acceptable_partition_rows,
        "compute_long_control": partition_control,
        "active_queue_rows": active_queue_lines,
        "active_account_queue_rows": active_account_queue_lines,
        "qos": {
            "account": expected_account,
            "maximum_running_jobs": 2,
            "maximum_submitted_jobs": 5,
            "maximum_cpu_cores": 32,
            "maximum_gpus": 2,
            "verified": qos_valid,
        },
        "remote_release": remote_release,
    }


def _manifest_records(
    manifest: dict[str, Any],
    *,
    expected: ReleaseExpectations,
    add_reason: Callable[[str, str], None],
) -> dict[str, dict[str, Any]]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != expected.file_count:
        add_reason(
            "release_manifest_inventory_invalid",
            f"release manifest must declare exactly {expected.file_count} files",
        )
        return {}
    records: dict[str, dict[str, Any]] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            add_reason("release_manifest_inventory_invalid", "manifest file record is invalid")
            continue
        raw_path = item.get("path")
        sha256 = item.get("sha256")
        size = item.get("bytes")
        if (
            not isinstance(raw_path, str)
            or PurePosixPath(raw_path).is_absolute()
            or ".." in PurePosixPath(raw_path).parts
            or not _SHA256.fullmatch(str(sha256))
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            add_reason(
                "release_manifest_inventory_invalid",
                "manifest contains an unsafe or malformed file record",
            )
            continue
        if raw_path in records:
            add_reason(
                "release_manifest_inventory_invalid",
                f"manifest repeats file path {raw_path}",
            )
            continue
        records[raw_path] = item
    return records


def _verify_bundle_inventory(
    bundle_path: Path,
    manifest_path_in_bundle: str,
    manifest_bytes: bytes,
    records: dict[str, dict[str, Any]],
    *,
    add_reason: Callable[[str, str], None],
) -> dict[str, Any]:
    verified = 0
    member_names: set[str] = set()
    expected_extras = {".bistka/deployment_manifest.json", manifest_path_in_bundle}
    if not bundle_path.is_file():
        return {"verified_files": 0, "member_count": 0, "inventory_verified": False}
    try:
        with tarfile.open(bundle_path, mode="r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                name = member.name
                if (
                    not member.isfile()
                    or PurePosixPath(name).is_absolute()
                    or ".." in PurePosixPath(name).parts
                    or name in member_names
                ):
                    add_reason(
                        "release_bundle_inventory_invalid",
                        f"bundle contains unsafe, duplicate, or non-file member {name!r}",
                    )
                    continue
                member_names.add(name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    add_reason(
                        "release_bundle_inventory_invalid",
                        f"cannot read bundle member {name!r}",
                    )
                    continue
                payload = extracted.read()
                if name in records:
                    record = records[name]
                    if len(payload) != record["bytes"] or _sha256_bytes(payload) != record["sha256"]:
                        add_reason(
                            "release_bundle_member_mismatch",
                            f"bundle member does not match manifest: {name}",
                        )
                    else:
                        verified += 1
                elif name in expected_extras:
                    if payload != manifest_bytes:
                        add_reason(
                            "release_bundle_manifest_copy_mismatch",
                            f"bundle manifest copy does not match: {name}",
                        )
                else:
                    add_reason(
                        "release_bundle_inventory_invalid",
                        f"bundle contains undeclared member {name}",
                    )
    except (OSError, tarfile.TarError) as exc:
        add_reason("release_bundle_invalid", f"cannot verify release bundle: {exc}")
        return {"verified_files": verified, "member_count": 0, "inventory_verified": False}
    expected_names = set(records) | expected_extras
    if member_names != expected_names:
        add_reason(
            "release_bundle_inventory_invalid",
            "bundle member set does not exactly match the manifest and two manifest copies",
        )
    return {
        "verified_files": verified,
        "member_count": len(member_names),
        "inventory_verified": verified == len(records) and member_names == expected_names,
    }


def validate_screening_job_script(
    path: str | Path,
    *,
    expected_account: str = EXPECTED_ACCOUNT,
) -> dict[str, Any]:
    problems: list[dict[str, str]] = []

    def add_reason(code: str, message: str) -> None:
        item = {"code": code, "message": message}
        if item not in problems:
            problems.append(item)

    job_script_file = Path(path)
    directives: dict[str, str] = {}
    job_text = ""
    if job_script_file.is_symlink():
        add_reason(
            "screening_job_symlink_rejected",
            "screening job script must not be a symlink",
        )
    elif not job_script_file.is_file():
        add_reason(
            "screening_job_missing",
            f"missing screening job script: {job_script_file}",
        )
    else:
        try:
            job_text = job_script_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            add_reason(
                "screening_job_invalid",
                f"cannot read screening job script: {exc}",
            )
        for line in job_text.splitlines():
            match = _SBATCH.fullmatch(line)
            if match:
                if match.group(1) in directives:
                    add_reason(
                        "screening_job_directive_duplicate",
                        f"job script repeats --{match.group(1)}",
                    )
                directives[match.group(1)] = match.group(2).strip()

    exact_directives = {
        "nodes": "1",
        "ntasks": "1",
        "cpus-per-task": "2",
        "mem": "12GB",
        "time": "4-00:00:00",
        "partition": "compute-long",
        "array": "0-4%2",
        "account": expected_account,
    }
    if any(directives.get(key) != value for key, value in exact_directives.items()):
        add_reason(
            "screening_job_resources_mismatch",
            "screening job must be 0-4%2, 2 CPU, 12GB, four days, compute-long",
        )
    executable_lines = [
        line
        for line in job_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    nested_sbatch_count = sum(
        1
        for line in executable_lines
        if re.search(r"(^|\s)sbatch(\s|$)", line)
    )
    if nested_sbatch_count:
        add_reason(
            "screening_job_nested_submission",
            "screening job script must never execute sbatch",
        )
    if job_text.count("src.experiments.run_screening_storage_safe") != 1:
        add_reason(
            "screening_job_entrypoint_mismatch",
            "screening job must invoke the storage-safe runner exactly once",
        )
    reason_codes = sorted({item["code"] for item in problems})
    return {
        "valid": not reason_codes,
        "reason_codes": reason_codes,
        "validation_errors": sorted(
            problems,
            key=lambda item: (item["code"], item["message"]),
        ),
        "path": str(job_script_file),
        "directives": directives,
        "expected_directives": exact_directives,
        "nested_sbatch_count": nested_sbatch_count,
        "controller_sbatch_invocations": 1,
    }


def verify_locked_release(
    project_root: str | Path,
    *,
    expected: ReleaseExpectations = LOCKED_RELEASE,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    bundle_path: str = DEFAULT_BUNDLE_PATH,
    preflight_path: str = DEFAULT_PREFLIGHT_PATH,
    locked_config_path: str = DEFAULT_LOCKED_CONFIG_PATH,
    feature_path: str = DEFAULT_FEATURE_PATH,
    job_script_path: str = DEFAULT_JOB_SCRIPT_PATH,
    spec_path: str = DEFAULT_SPEC_PATH,
) -> dict[str, Any]:
    problems: list[dict[str, str]] = []

    def add_reason(code: str, message: str) -> None:
        item = {"code": code, "message": message}
        if item not in problems:
            problems.append(item)

    root = Path(project_root).resolve()
    manifest_file = _safe_project_file(root, manifest_path)
    bundle_file = _safe_project_file(root, bundle_path)
    preflight_file = _safe_project_file(root, preflight_path)
    locked_config_file = _safe_project_file(root, locked_config_path)
    feature_file = _safe_project_file(root, feature_path)
    job_script_file = _safe_project_file(root, job_script_path)
    spec_file = _safe_project_file(root, spec_path)

    evidence = {
        "manifest": _verify_exact_hash(
            manifest_file,
            label="release_manifest",
            expected_sha256=expected.manifest_sha256,
            add_reason=add_reason,
        ),
        "bundle": _verify_exact_hash(
            bundle_file,
            label="release_bundle",
            expected_sha256=expected.bundle_sha256,
            add_reason=add_reason,
            require_sidecar=True,
        ),
        "preflight": _verify_exact_hash(
            preflight_file,
            label="release_preflight",
            expected_sha256=expected.preflight_sha256,
            add_reason=add_reason,
        ),
        "locked_config": _verify_exact_hash(
            locked_config_file,
            label="locked_screening_config",
            expected_sha256=expected.locked_config_sha256,
            add_reason=add_reason,
            require_sidecar=True,
        ),
        "feature": _verify_exact_hash(
            feature_file,
            label="observed_feature_table",
            expected_sha256=expected.feature_sha256,
            add_reason=add_reason,
        ),
        "job_script": _verify_exact_hash(
            job_script_file,
            label="screening_job_script",
            expected_sha256=expected.job_script_sha256,
            add_reason=add_reason,
        ),
        "spec": _verify_exact_hash(
            spec_file,
            label="storage_safe_campaign_spec",
            expected_sha256=expected.spec_sha256,
            add_reason=add_reason,
        ),
    }

    manifest = _load_json_object(
        manifest_file,
        label="release_manifest",
        add_reason=add_reason,
    )
    records: dict[str, dict[str, Any]] = {}
    manifest_bytes = b""
    if manifest is not None:
        manifest_bytes = manifest_file.read_bytes()
        if manifest.get("schema_version") != 1:
            add_reason("release_manifest_content_mismatch", "manifest schema_version must be 1")
        if manifest.get("campaign_id") != expected.campaign_id:
            add_reason("release_manifest_content_mismatch", "manifest campaign ID changed")
        if manifest.get("release_id") != expected.release_id:
            add_reason("release_manifest_content_mismatch", "manifest release ID changed")
        if manifest.get("content_addressed") is not True:
            add_reason("release_manifest_content_mismatch", "manifest is not content-addressed")
        if manifest.get("file_count") != expected.file_count:
            add_reason("release_manifest_content_mismatch", "manifest file count changed")
        if manifest.get("target_project_root") != expected.target_project_root:
            add_reason("release_manifest_content_mismatch", "manifest target root changed")
        if manifest.get("required_hostname_prefix") != "bistka-":
            add_reason("release_manifest_content_mismatch", "manifest hostname prefix changed")
        records = _manifest_records(manifest, expected=expected, add_reason=add_reason)

    required_hashes = {
        locked_config_path: expected.locked_config_sha256,
        feature_path: expected.feature_sha256,
        job_script_path: expected.job_script_sha256,
        spec_path: expected.spec_sha256,
    }
    for raw_path, expected_hash in required_hashes.items():
        record = records.get(raw_path)
        if record is None or record.get("sha256") != expected_hash:
            add_reason(
                "release_manifest_binding_mismatch",
                f"manifest does not bind the expected {raw_path}",
            )
    critical_evidence: list[dict[str, Any]] = []
    for raw_path in expected.critical_paths:
        record = records.get(raw_path)
        if record is None:
            add_reason(
                "release_manifest_binding_mismatch",
                f"manifest omits critical file {raw_path}",
            )
            continue
        critical_evidence.append(
            _verify_exact_hash(
                _safe_project_file(root, raw_path),
                label="release_critical_file",
                expected_sha256=str(record["sha256"]),
                add_reason=add_reason,
            )
        )

    local_inventory_verified = 0
    for raw_path, record in records.items():
        try:
            local_path = _safe_project_file(root, raw_path)
        except ValueError as exc:
            add_reason(
                "release_local_inventory_mismatch",
                f"unsafe local release file {raw_path}: {exc}",
            )
            continue
        if (
            not local_path.is_file()
            or local_path.stat().st_size != record.get("bytes")
            or _sha256_file(local_path) != record.get("sha256")
        ):
            add_reason(
                "release_local_inventory_mismatch",
                f"local release file differs from the locked manifest: {raw_path}",
            )
            continue
        local_inventory_verified += 1

    bundle_inventory = _verify_bundle_inventory(
        bundle_file,
        manifest_path,
        manifest_bytes,
        records,
        add_reason=add_reason,
    )

    preflight = _load_json_object(
        preflight_file,
        label="release_preflight",
        add_reason=add_reason,
    )
    if preflight is not None:
        required_preflight = {
            "schema_version": 1,
            "stage": "cluster_preflight",
            "status": "pass",
            "execution_eligible": True,
            "release_id": expected.release_id,
            "manifest_sha256": expected.manifest_sha256,
            "declared_file_count": expected.file_count,
            "verified_file_count": expected.file_count,
            "project_root": expected.target_project_root,
            "hostname": "bistka-master",
            "issues": [],
        }
        for key, value in required_preflight.items():
            if preflight.get(key) != value:
                add_reason(
                    "release_preflight_content_mismatch",
                    f"preflight field {key} changed",
                )
        expected_binding = {
            "config_path": locked_config_path,
            "config_sha256": expected.locked_config_sha256,
            "feature_path": feature_path,
            "feature_sha256": expected.feature_sha256,
        }
        bindings = preflight.get("runtime_feature_bindings")
        if not isinstance(bindings, list) or not any(
            isinstance(item, dict)
            and all(item.get(key) == value for key, value in expected_binding.items())
            for item in bindings
        ):
            add_reason(
                "release_preflight_binding_mismatch",
                "preflight does not bind the exact locked config and observed feature table",
            )

    job_validation = validate_screening_job_script(job_script_file)
    for item in job_validation["validation_errors"]:
        add_reason(str(item["code"]), str(item["message"]))

    reason_codes = sorted({item["code"] for item in problems})
    return {
        "valid": not reason_codes,
        "reason_codes": reason_codes,
        "validation_errors": sorted(
            problems,
            key=lambda item: (item["code"], item["message"]),
        ),
        "release_id": expected.release_id,
        "manifest_file_count": len(records),
        "local_inventory": {
            "verified_files": local_inventory_verified,
            "inventory_verified": local_inventory_verified == len(records),
        },
        "bundle_inventory": bundle_inventory,
        "evidence": evidence,
        "critical_files": critical_evidence,
        "job": job_validation,
    }


def verify_live_remote_preflight(
    path: str | Path,
    *,
    project_root: str | Path,
    now: datetime,
    expected: ReleaseExpectations = LOCKED_RELEASE,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    max_clock_skew_seconds: int = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    problems: list[dict[str, str]] = []

    def add_reason(code: str, message: str) -> None:
        item = {"code": code, "message": message}
        if item not in problems:
            problems.append(item)

    receipt_path = Path(path)
    evidence = _verify_checksum_bound_snapshot(
        receipt_path,
        label="live_remote_preflight",
        add_reason=add_reason,
    )
    receipt = _load_json_object(
        receipt_path,
        label="live_remote_preflight",
        add_reason=add_reason,
    )
    root = Path(project_root).resolve()
    manifest_file = _safe_project_file(root, manifest_path)
    manifest_evidence = _verify_exact_hash(
        manifest_file,
        label="live_remote_preflight_manifest",
        expected_sha256=expected.manifest_sha256,
        add_reason=add_reason,
    )
    manifest = _load_json_object(
        manifest_file,
        label="live_remote_preflight_manifest",
        add_reason=add_reason,
    )
    records = (
        _manifest_records(manifest, expected=expected, add_reason=add_reason)
        if manifest is not None
        else {}
    )

    completed_at: datetime | None = None
    started_at: datetime | None = None
    if receipt is not None:
        started_at = _parse_timestamp(receipt.get("started_at_utc"))
        completed_at = _parse_timestamp(receipt.get("completed_at_utc"))
        if started_at is None or completed_at is None:
            add_reason(
                "live_remote_preflight_timestamp_invalid",
                "live remote preflight timestamps must be timezone-aware ISO-8601",
            )
        else:
            if completed_at < started_at:
                add_reason(
                    "live_remote_preflight_timestamp_invalid",
                    "live remote preflight completed before it started",
                )
            if completed_at > now + timedelta(seconds=max_clock_skew_seconds):
                add_reason(
                    "live_remote_preflight_timestamp_future",
                    "live remote preflight is too far in the future",
                )
            if now - completed_at > timedelta(seconds=max_age_seconds):
                add_reason(
                    "live_remote_preflight_stale",
                    "live remote preflight is older than the execution-window limit",
                )

        required_fields = {
            "schema_version": 1,
            "status": "pass",
            "stage": "cluster_preflight",
            "hostname": "bistka-master",
            "project_root": expected.target_project_root,
            "release_id": expected.release_id,
            "manifest_sha256": expected.manifest_sha256,
            "declared_file_count": expected.file_count,
            "verified_file_count": expected.file_count,
            "data_origin": "observed_market",
            "execution_eligible": True,
            "issues": [],
        }
        for key, value in required_fields.items():
            if receipt.get(key) != value:
                add_reason(
                    "live_remote_preflight_content_mismatch",
                    f"live remote preflight field {key} changed",
                )

        required_checks = {
            "bistka_hostname",
            "project_root",
            "deployment_files",
            "runtime_feature_bindings",
            "observed_market_data",
            "runtime_packages",
        }
        raw_checks = receipt.get("checks")
        passing_checks = {
            str(item.get("check"))
            for item in raw_checks
            if isinstance(item, dict) and item.get("passed") is True
        } if isinstance(raw_checks, list) else set()
        if not required_checks.issubset(passing_checks):
            add_reason(
                "live_remote_preflight_checks_incomplete",
                "live remote preflight did not pass every required check",
            )

        expected_bindings = (
            manifest.get("runtime_feature_bindings")
            if isinstance(manifest, dict)
            else None
        )
        if receipt.get("runtime_feature_bindings") != expected_bindings:
            add_reason(
                "live_remote_preflight_binding_mismatch",
                "live remote runtime feature bindings differ from the locked manifest",
            )

        required_packages = (
            manifest.get("required_packages")
            if isinstance(manifest, dict)
            else None
        )
        packages = receipt.get("packages")
        if (
            not isinstance(required_packages, list)
            or not isinstance(packages, dict)
            or any(packages.get(str(name)) in {None, "missing"} for name in required_packages)
        ):
            add_reason(
                "live_remote_preflight_packages_incomplete",
                "live remote preflight is missing a required runtime package",
            )

        raw_verified_files = receipt.get("verified_files")
        remote_records: dict[str, dict[str, Any]] = {}
        if not isinstance(raw_verified_files, list):
            add_reason(
                "live_remote_preflight_inventory_invalid",
                "live remote preflight has no verified file inventory",
            )
        else:
            for item in raw_verified_files:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    add_reason(
                        "live_remote_preflight_inventory_invalid",
                        "live remote preflight contains a malformed file record",
                    )
                    continue
                item_path = str(item["path"])
                if item_path in remote_records:
                    add_reason(
                        "live_remote_preflight_inventory_invalid",
                        f"live remote preflight repeats {item_path}",
                    )
                    continue
                remote_records[item_path] = item
        if set(remote_records) != set(records):
            add_reason(
                "live_remote_preflight_inventory_mismatch",
                "live remote preflight file set differs from the locked manifest",
            )
        else:
            for item_path, record in records.items():
                observed = remote_records[item_path]
                if (
                    observed.get("sha256") != record.get("sha256")
                    or observed.get("bytes") != record.get("bytes")
                ):
                    add_reason(
                        "live_remote_preflight_inventory_mismatch",
                        f"live remote file differs from the locked manifest: {item_path}",
                    )
                    break

    reason_codes = sorted({item["code"] for item in problems})
    return {
        "valid": not reason_codes,
        "reason_codes": reason_codes,
        "validation_errors": sorted(
            problems,
            key=lambda item: (item["code"], item["message"]),
        ),
        "evidence": evidence,
        "manifest_evidence": manifest_evidence,
        "started_at": _utc_text(started_at) if started_at is not None else None,
        "completed_at": _utc_text(completed_at) if completed_at is not None else None,
        "release_id": receipt.get("release_id") if isinstance(receipt, dict) else None,
        "verified_file_count": (
            receipt.get("verified_file_count") if isinstance(receipt, dict) else None
        ),
        "inventory_matches_manifest": not any(
            item["code"] in {
                "live_remote_preflight_inventory_invalid",
                "live_remote_preflight_inventory_mismatch",
            }
            for item in problems
        ),
    }


def _recompute_campaign_view(project_root: Path, spec_path: Path) -> dict[str, Any]:
    from src.campaigns.control import load_definition
    from src.campaigns.gates import evaluate_campaign

    definition, _ = load_definition(
        spec_path,
        project_root=project_root,
    )
    statuses = evaluate_campaign(definition)
    counts = dict(Counter(status.state.value for status in statuses))
    screening = next(
        (status for status in statuses if status.stage_id == "screening"),
        None,
    )
    if screening is None:
        return {"counts": counts, "screening": None}
    return {
        "counts": counts,
        "screening": {
            "state": screening.state.value,
            "progress": screening.progress,
            "manifest_expected": screening.gate.manifest_expected,
            "manifest_found": screening.gate.manifest_found,
            "manifest_complete": screening.gate.manifest_complete,
            "manifest_running": screening.gate.manifest_running,
            "manifest_failed": screening.gate.manifest_failed,
        },
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor = os.open(path.parent, directory_flags)
    descriptor: int | None = None
    created = False
    try:
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            path.name,
            file_flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.fsync(directory_descriptor)
        except OSError:
            # The file itself is already fsynced. Some filesystems do not permit
            # directory fsync; consumers still require the exact sibling sidecar.
            pass
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                os.unlink(path.name, dir_fd=directory_descriptor)
            except OSError:
                pass
        raise
    finally:
        os.close(directory_descriptor)


def write_authorization_receipt(
    path: str | Path,
    receipt: dict[str, Any],
    *,
    project_root: str | Path,
    now: datetime | None = None,
) -> dict[str, str]:
    if not isinstance(receipt, _PreparedExecutionWindow):
        raise ValueError(
            "authorization receipt must be the unmodified result of the full gate"
        )
    issued_tag = _ISSUED_AUTHORIZATION_CAPABILITIES.get(id(receipt))
    if issued_tag is None or not hmac.compare_digest(
        issued_tag,
        receipt._capability_tag,
    ):
        raise ValueError("authorization capability was not issued by the full gate")
    if not receipt.capability_is_current():
        raise ValueError("authorization receipt changed after gate evaluation")
    if receipt.get("authorized") is not True:
        raise ValueError("refusing to write a non-authorizing execution-window receipt")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("gate") != "screening_execution_window"
        or receipt.get("decision") != "authorize_render_only"
        or receipt.get("reason_codes") != []
        or receipt.get("validation_errors") != []
    ):
        raise ValueError("authorization receipt invariants are invalid")
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ValueError("receipt write time must include a timezone")
    current_time = current_time.astimezone(timezone.utc)
    expires_at = _parse_timestamp(receipt.get("expires_at"))
    minimum_remaining = receipt.get("policy", {}).get("minimum_remaining_seconds")
    if (
        expires_at is None
        or not isinstance(minimum_remaining, int)
        or isinstance(minimum_remaining, bool)
        or (expires_at - current_time).total_seconds() < minimum_remaining
    ):
        raise ValueError("refusing to write an expired or nearly expired authorization")

    root = Path(project_root).resolve()
    reports_root = root / "reports"
    output_root_raw = root / AUTHORIZATION_OUTPUT_DIRECTORY
    if reports_root.is_symlink() or output_root_raw.is_symlink():
        raise ValueError("authorization output path must not contain a symlink")
    output_root = output_root_raw.resolve()
    if not output_root.is_relative_to(root):
        raise ValueError("authorization output directory must remain inside the project")
    raw_destination = Path(path)
    if raw_destination.is_absolute():
        candidate = raw_destination
    else:
        if ".." in raw_destination.parts:
            raise ValueError("authorization output path must not contain '..'")
        candidate = root / raw_destination
    if not candidate.is_relative_to(root):
        raise ValueError("authorization output must remain inside the project")
    if candidate.is_symlink():
        raise ValueError("authorization output must not be a symlink")
    destination = candidate.resolve()
    if destination.parent != output_root or AUTHORIZATION_NAME.fullmatch(destination.name) is None:
        raise ValueError(
            "authorization output must be a timestamped JSON directly under "
            f"{AUTHORIZATION_OUTPUT_DIRECTORY}"
        )
    sidecar = Path(str(destination) + ".sha256")
    if destination.exists() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError("authorization receipts are immutable and must not be overwritten")
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if not receipt.capability_is_current():
        raise ValueError("authorization receipt changed during write validation")
    digest = _sha256_bytes(payload.encode("utf-8"))
    _atomic_text(destination, payload)
    _ISSUED_AUTHORIZATION_CAPABILITIES.pop(id(receipt), None)
    _atomic_text(sidecar, digest + "\n")
    return {
        "path": str(destination),
        "sha256": digest,
        "sidecar_path": str(Path(str(destination) + ".sha256")),
    }


def prepare_screening_execution_window(
    *,
    project_root: str | Path,
    quota_snapshot_path: str | Path,
    scheduler_snapshot_path: str | Path,
    live_remote_preflight_path: str | Path,
    smoke_summary_path: str | Path = DEFAULT_SMOKE_SUMMARY_PATH,
    spec_path: str | Path = DEFAULT_SPEC_PATH,
    now: datetime | None = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    max_clock_skew_seconds: int = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
    minimum_remaining_seconds: int = DEFAULT_MINIMUM_REMAINING_SECONDS,
    minimum_group_soft_margin_bytes: int = DEFAULT_MINIMUM_GROUP_SOFT_MARGIN_BYTES,
    fixed_reserve_bytes: int = DEFAULT_FIXED_RESERVE_BYTES,
    expected_release: ReleaseExpectations = LOCKED_RELEASE,
    campaign_view_provider: Callable[[Path, Path], dict[str, Any]] = _recompute_campaign_view,
    release_view_provider: Callable[..., dict[str, Any]] = verify_locked_release,
    live_remote_preflight_provider: Callable[..., dict[str, Any]] = (
        verify_live_remote_preflight
    ),
) -> dict[str, Any]:
    injected_clock = now is not None
    if now is None:
        assessed_at = datetime.now(timezone.utc)
    elif now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must include a timezone")
    else:
        assessed_at = now.astimezone(timezone.utc)
    if (
        not _is_integer(max_age_seconds)
        or max_age_seconds <= 0
        or max_age_seconds > DEFAULT_MAX_AGE_SECONDS
    ):
        raise ValueError(
            f"max_age_seconds must be within 1..{DEFAULT_MAX_AGE_SECONDS}; "
            "the execution policy may only be made stricter"
        )
    if (
        not _is_integer(max_clock_skew_seconds)
        or max_clock_skew_seconds < 0
        or max_clock_skew_seconds > DEFAULT_MAX_CLOCK_SKEW_SECONDS
    ):
        raise ValueError(
            "max_clock_skew_seconds must be within "
            f"0..{DEFAULT_MAX_CLOCK_SKEW_SECONDS}; the execution policy may only "
            "be made stricter"
        )
    if (
        not _is_integer(minimum_remaining_seconds)
        or minimum_remaining_seconds < DEFAULT_MINIMUM_REMAINING_SECONDS
        or minimum_remaining_seconds > max_age_seconds
    ):
        raise ValueError(
            "minimum_remaining_seconds must be at least "
            f"{DEFAULT_MINIMUM_REMAINING_SECONDS} and no greater than max_age_seconds"
        )
    if (
        not _is_integer(minimum_group_soft_margin_bytes)
        or minimum_group_soft_margin_bytes < DEFAULT_MINIMUM_GROUP_SOFT_MARGIN_BYTES
    ):
        raise ValueError(
            "minimum_group_soft_margin_bytes must be at least "
            f"{DEFAULT_MINIMUM_GROUP_SOFT_MARGIN_BYTES}"
        )
    if (
        not _is_integer(fixed_reserve_bytes)
        or fixed_reserve_bytes < DEFAULT_FIXED_RESERVE_BYTES
    ):
        raise ValueError(
            f"fixed_reserve_bytes must be at least {DEFAULT_FIXED_RESERVE_BYTES}"
        )

    problems: list[dict[str, str]] = []

    def add_reason(code: str, message: str) -> None:
        item = {"code": code, "message": message}
        if item not in problems:
            problems.append(item)

    root = Path(project_root).resolve()
    smoke = Path(smoke_summary_path)
    if not smoke.is_absolute():
        smoke = _safe_project_file(root, str(smoke))
    canonical_smoke = _safe_project_file(root, DEFAULT_SMOKE_SUMMARY_PATH)
    if smoke != canonical_smoke:
        add_reason(
            "smoke_summary_path_mismatch",
            "execution authorization requires the canonical verified job-3550 smoke summary",
        )
    smoke_evidence = _verify_exact_hash(
        smoke,
        label="canonical_smoke_summary",
        expected_sha256=EXPECTED_SMOKE_SUMMARY_SHA256,
        add_reason=add_reason,
        require_sidecar=True,
    )
    quota = Path(quota_snapshot_path)
    if not quota.is_absolute():
        quota = _safe_project_file(root, str(quota))
    scheduler = Path(scheduler_snapshot_path)
    if not scheduler.is_absolute():
        scheduler = _safe_project_file(root, str(scheduler))
    live_remote_preflight_path_value = Path(live_remote_preflight_path)
    if not live_remote_preflight_path_value.is_absolute():
        live_remote_preflight_path_value = _safe_project_file(
            root,
            str(live_remote_preflight_path_value),
        )
    spec = Path(spec_path)
    if not spec.is_absolute():
        spec = _safe_project_file(root, str(spec))

    storage = assess_screening_storage(
        smoke,
        quota,
        expected_group=EXPECTED_GROUP,
        expected_gid=EXPECTED_GID,
        expected_project_id=EXPECTED_PROJECT_ID,
        expected_filesystem=EXPECTED_FILESYSTEM,
        expected_output_parent=EXPECTED_OUTPUT_PARENT,
        now=assessed_at,
        max_quota_age_seconds=max_age_seconds,
        max_clock_skew_seconds=max_clock_skew_seconds,
        safety_factor="1.5",
        fixed_reserve_bytes=fixed_reserve_bytes,
        expected_algorithms=DEFAULT_EXPECTED_ALGORITHMS,
    )
    if storage.get("candidate_capacity_eligible") is not True:
        add_reason(
            "storage_capacity_not_eligible",
            "fresh quota/write evidence does not authorize the full screening",
        )
        for item in storage.get("validation_errors", []):
            if isinstance(item, dict):
                add_reason(
                    f"storage_{item.get('code', 'invalid')}",
                    str(item.get("message", "storage validation failed")),
                )

    required_bytes = (
        storage.get("projection", {}).get("required_bytes")
        if isinstance(storage.get("projection"), dict)
        else None
    )
    group_soft_headroom = None
    quota_view = storage.get("quota")
    if isinstance(quota_view, dict):
        group = quota_view.get("group")
        if isinstance(group, dict):
            dimensions = group.get("bytes")
            if isinstance(dimensions, dict):
                group_soft_headroom = dimensions.get("soft_headroom")
    if (
        not isinstance(required_bytes, int)
        or isinstance(required_bytes, bool)
        or not isinstance(group_soft_headroom, int)
        or isinstance(group_soft_headroom, bool)
    ):
        group_soft_margin = None
        add_reason(
            "storage_margin_unavailable",
            "cannot recompute personal-group soft-quota margin",
        )
    else:
        group_soft_margin = group_soft_headroom - required_bytes
        if group_soft_margin < minimum_group_soft_margin_bytes:
            add_reason(
                "storage_group_soft_margin_below_minimum",
                "personal-group soft-quota margin is below the execution-window minimum",
            )

    scheduler_view = parse_scheduler_snapshot(
        scheduler,
        now=assessed_at,
        max_age_seconds=max_age_seconds,
        max_clock_skew_seconds=max_clock_skew_seconds,
    )
    if not scheduler_view["valid"]:
        for item in scheduler_view["validation_errors"]:
            add_reason(str(item["code"]), str(item["message"]))

    runtime_algorithms = storage.get("runtime", {}).get("algorithms")
    if isinstance(runtime_algorithms, list) and runtime_algorithms:
        projected_task_hours = [
            item.get("projected_task_hours")
            for item in runtime_algorithms
            if isinstance(item, dict)
        ]
    else:
        projected_task_hours = []
    if (
        len(projected_task_hours) != 5
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in projected_task_hours
        )
    ):
        estimated_service_hours = None
        add_reason(
            "scheduler_credit_estimate_unavailable",
            "cannot derive a five-task service-hour estimate from the smoke evidence",
        )
    else:
        # BistKA currently bills CPU at 1 and memory at 0.1 per GiB. The
        # locked task requests 2 CPUs and 12 GiB, then adds a 25% reserve.
        estimated_service_hours = (
            sum(float(value) for value in projected_task_hours)
            * (2.0 + 12.0 * 0.1)
            * 1.25
        )
        credit = scheduler_view.get("credit")
        project_remaining = (
            credit.get(EXPECTED_ACCOUNT, {}).get("remaining_service_hours")
            if isinstance(credit, dict)
            else None
        )
        user_remaining = (
            credit.get(EXPECTED_USER, {}).get("remaining_service_hours")
            if isinstance(credit, dict)
            else None
        )
        if (
            not isinstance(project_remaining, (int, float))
            or not isinstance(user_remaining, (int, float))
            or float(project_remaining) < estimated_service_hours
            or float(user_remaining) < estimated_service_hours
        ):
            add_reason(
                "scheduler_credit_below_screening_estimate",
                "project and user credit must each cover the conservative screening estimate",
            )
    scheduler_view["screening_credit_estimate"] = {
        "projected_service_hours_with_25_percent_reserve": estimated_service_hours,
        "billing_assumption": "2 CPU + 12 GiB x 0.1 per task-hour",
        "project_and_user_credit_required": True,
    }

    release = release_view_provider(
        root,
        expected=expected_release,
        spec_path=str(spec.relative_to(root)),
    )
    if not release["valid"]:
        for item in release["validation_errors"]:
            add_reason(str(item["code"]), str(item["message"]))

    live_remote_preflight = live_remote_preflight_provider(
        live_remote_preflight_path_value,
        project_root=root,
        now=assessed_at,
        expected=expected_release,
        max_age_seconds=max_age_seconds,
        max_clock_skew_seconds=max_clock_skew_seconds,
    )
    if not live_remote_preflight["valid"]:
        for item in live_remote_preflight["validation_errors"]:
            add_reason(str(item["code"]), str(item["message"]))

    try:
        campaign = campaign_view_provider(root, spec)
    except Exception as exc:
        campaign = {"counts": {}, "screening": None}
        add_reason(
            "campaign_recompute_failed",
            f"cannot recompute campaign readiness: {type(exc).__name__}: {exc}",
        )
    if campaign.get("counts") != EXPECTED_CAMPAIGN_COUNTS:
        add_reason(
            "campaign_state_mismatch",
            "campaign must recompute to exactly 9 VERIFIED, 1 READY, and 25 PENDING",
        )
    screening_view = campaign.get("screening")
    expected_screening = {
        "state": "READY",
        "progress": EXPECTED_SCREENING_PROGRESS,
        "manifest_expected": EXPECTED_SCREENING_MANIFESTS,
        "manifest_found": 0,
        "manifest_complete": 0,
        "manifest_running": 0,
        "manifest_failed": 0,
    }
    if screening_view != expected_screening:
        add_reason(
            "screening_stage_not_pristine_ready",
            "screening must be READY with zero of 150 manifests and no active evidence",
        )

    live_timestamps: list[datetime] = []
    scheduler_timestamp = _parse_timestamp(scheduler_view.get("captured_at"))
    if scheduler_timestamp is not None:
        live_timestamps.append(scheduler_timestamp)
    live_preflight_timestamp = _parse_timestamp(
        live_remote_preflight.get("completed_at")
    )
    if live_preflight_timestamp is not None:
        live_timestamps.append(live_preflight_timestamp)
    if isinstance(quota_view, dict):
        quota_timestamp = _parse_timestamp(quota_view.get("captured_at"))
        probe = quota_view.get("write_probe")
        probe_timestamp = (
            _parse_timestamp(probe.get("completed_at"))
            if isinstance(probe, dict)
            else None
        )
        if quota_timestamp is not None:
            live_timestamps.append(quota_timestamp)
        if probe_timestamp is not None:
            live_timestamps.append(probe_timestamp)
    finalized_at = assessed_at if injected_clock else datetime.now(timezone.utc)
    if len(live_timestamps) != 4:
        expires_at = None
        remaining_seconds = None
        add_reason(
            "execution_window_expiry_unavailable",
            "scheduler, quota, write-probe, and remote-preflight timestamps are required",
        )
    else:
        expires_at = min(
            timestamp + timedelta(seconds=max_age_seconds)
            for timestamp in live_timestamps
        )
        remaining_seconds = max(
            0,
            math.floor((expires_at - finalized_at).total_seconds()),
        )
        if remaining_seconds < minimum_remaining_seconds:
            add_reason(
                "execution_window_ttl_below_minimum",
                "fresh evidence will expire too soon to safely render and inspect a command",
            )

    reason_codes = sorted({item["code"] for item in problems})
    authorized = not reason_codes
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "gate": "screening_execution_window",
        "assessed_at": _utc_text(assessed_at),
        "finalized_at": _utc_text(finalized_at),
        "authorized": authorized,
        "decision": "authorize_render_only" if authorized else "hold",
        "expires_at": _utc_text(expires_at) if expires_at is not None else None,
        "remaining_seconds": remaining_seconds,
        "reason_codes": reason_codes,
        "validation_errors": sorted(
            problems,
            key=lambda item: (item["code"], item["message"]),
        ),
        "policy": {
            "jobs_submitted_by_gate": 0,
            "max_age_seconds": max_age_seconds,
            "max_clock_skew_seconds": max_clock_skew_seconds,
            "minimum_remaining_seconds": minimum_remaining_seconds,
            "minimum_group_soft_margin_bytes": minimum_group_soft_margin_bytes,
            "required_empty_user_queue": True,
            "maximum_array_elements": 5,
            "maximum_array_concurrency": 2,
        },
        "storage": {
            "canonical_smoke_evidence": smoke_evidence,
            "candidate_capacity_eligible": storage.get("candidate_capacity_eligible"),
            "assessment": storage,
            "required_bytes": required_bytes,
            "group_soft_headroom_bytes": group_soft_headroom,
            "group_soft_margin_bytes": group_soft_margin,
        },
        "scheduler": scheduler_view,
        "live_remote_preflight": live_remote_preflight,
        "release": release,
        "campaign": campaign,
        "render": {
            "performed": False,
            "next_command": CONTROLLER_RENDER_COMMAND,
            "warning": (
                "Run once within this receipt's TTL, inspect the generated command, "
                "and never reuse an older rendered attempt."
            ),
        },
        "submission": {
            "performed": False,
            "sbatch_calls": 0,
            "array_shape": "0-4%2",
            "scheduler_jobs_if_submitted": 5,
        },
    }
    return _issue_execution_window(receipt)


def _validate_operational_evidence_triplet(
    *,
    project_root: str | Path,
    quota_snapshot_path: str | Path,
    scheduler_snapshot_path: str | Path,
    live_remote_preflight_path: str | Path,
) -> tuple[Path, Path, Path]:
    """Restrict production CLI evidence to one same-stamp operational capture."""

    root = Path(project_root).resolve()
    operational_raw = root / OPERATIONAL_EVIDENCE_DIRECTORY
    current = root
    for part in Path(OPERATIONAL_EVIDENCE_DIRECTORY).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("operational evidence path must not contain a symlink")
    operational = operational_raw.resolve()
    if not operational.is_relative_to(root) or not operational.is_dir():
        raise ValueError("operational evidence directory is unavailable")

    def resolve_input(
        value: str | Path,
        *,
        label: str,
        pattern: re.Pattern[str],
    ) -> tuple[Path, str]:
        raw = Path(value)
        if raw.is_absolute():
            candidate = raw
        else:
            if ".." in raw.parts:
                raise ValueError(f"{label} path must not contain '..'")
            candidate = root / raw
        if not candidate.is_relative_to(root):
            raise ValueError(f"{label} must remain inside the project")
        if candidate.is_symlink():
            raise ValueError(f"{label} must not be a symlink")
        resolved = candidate.resolve()
        if resolved.parent != operational:
            raise ValueError(
                f"{label} must be directly under {OPERATIONAL_EVIDENCE_DIRECTORY}"
            )
        match = pattern.fullmatch(resolved.name)
        if match is None:
            raise ValueError(f"{label} filename is invalid")
        return resolved, match.group("stamp")

    quota, quota_stamp = resolve_input(
        quota_snapshot_path,
        label="quota snapshot",
        pattern=re.compile(
            r"^quota_snapshot_(?P<stamp>[0-9]{8}T[0-9]{6}Z)\.json$"
        ),
    )
    scheduler, scheduler_stamp = resolve_input(
        scheduler_snapshot_path,
        label="scheduler snapshot",
        pattern=re.compile(
            r"^scheduler_snapshot_(?P<stamp>[0-9]{8}T[0-9]{6}Z)\.txt$"
        ),
    )
    live_preflight, preflight_stamp = resolve_input(
        live_remote_preflight_path,
        label="live remote preflight",
        pattern=re.compile(
            r"^live_preflight_(?P<stamp>[0-9]{8}T[0-9]{6}Z)\.json$"
        ),
    )
    if len({quota_stamp, scheduler_stamp, preflight_stamp}) != 1:
        raise ValueError("scheduler, quota, and preflight evidence stamps must match")
    return quota, scheduler, live_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authorize one render-only full-screening window from fresh, "
            "checksum-bound BistKA evidence. This command never runs sbatch."
        ),
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--spec", default=DEFAULT_SPEC_PATH)
    parser.add_argument("--quota-snapshot", required=True)
    parser.add_argument("--scheduler-snapshot", required=True)
    parser.add_argument("--live-remote-preflight", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        quota_snapshot, scheduler_snapshot, live_remote_preflight = (
            _validate_operational_evidence_triplet(
                project_root=args.project_root,
                quota_snapshot_path=args.quota_snapshot,
                scheduler_snapshot_path=args.scheduler_snapshot,
                live_remote_preflight_path=args.live_remote_preflight,
            )
        )
        evidence_stamp = quota_snapshot.name.removeprefix(
            "quota_snapshot_"
        ).removesuffix(".json")
        output_match = AUTHORIZATION_NAME.fullmatch(Path(args.output).name)
        if output_match is None or output_match.group("stamp") != evidence_stamp:
            raise ValueError(
                "authorization receipt filename stamp must match the evidence capture"
            )
        receipt = prepare_screening_execution_window(
            project_root=args.project_root,
            spec_path=args.spec,
            quota_snapshot_path=quota_snapshot,
            scheduler_snapshot_path=scheduler_snapshot,
            live_remote_preflight_path=live_remote_preflight,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if receipt["authorized"]:
        write_authorization_receipt(
            args.output,
            receipt,
            project_root=args.project_root,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
