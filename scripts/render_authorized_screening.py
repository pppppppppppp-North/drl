from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scripts.prepare_screening_execution_window import (
    AUTHORIZATION_NAME,
    AUTHORIZATION_OUTPUT_DIRECTORY,
    CONTROLLER_RENDER_COMMAND,
    DEFAULT_MAX_AGE_SECONDS,
    DEFAULT_MAX_CLOCK_SKEW_SECONDS,
    DEFAULT_MINIMUM_GROUP_SOFT_MARGIN_BYTES,
    DEFAULT_MINIMUM_REMAINING_SECONDS,
    DEFAULT_SPEC_PATH,
    EXPECTED_CAMPAIGN_COUNTS,
    LOCKED_RELEASE,
    _atomic_text,
    _is_integer,
    _parse_timestamp,
    _validate_operational_evidence_triplet,
    prepare_screening_execution_window,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_ID = re.compile(r"^screening_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_receipt(
    project_root: Path,
    raw_path: str | Path,
) -> tuple[Path, dict[str, Any], str, str]:
    reports_root = project_root / "reports"
    receipt_root_raw = project_root / AUTHORIZATION_OUTPUT_DIRECTORY
    if reports_root.is_symlink() or receipt_root_raw.is_symlink():
        raise ValueError("authorization receipt path must not contain a symlink")
    receipt_root = receipt_root_raw.resolve()
    if not receipt_root.is_relative_to(project_root):
        raise ValueError("authorization receipt directory escapes the project")

    raw = Path(raw_path)
    if raw.is_absolute():
        candidate = raw
    else:
        if ".." in raw.parts:
            raise ValueError("authorization receipt path must not contain '..'")
        candidate = project_root / raw
    if not candidate.is_relative_to(project_root) or candidate.is_symlink():
        raise ValueError("authorization receipt must be a regular project file")
    path = candidate.resolve()
    name_match = AUTHORIZATION_NAME.fullmatch(path.name)
    if path.parent != receipt_root or name_match is None:
        raise ValueError(
            "authorization receipt must be a timestamped JSON directly under "
            f"{AUTHORIZATION_OUTPUT_DIRECTORY}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"authorization receipt is missing: {path}")
    sidecar = Path(str(path) + ".sha256")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError("authorization receipt sidecar is missing or unsafe")
    digest = _sha256_file(path)
    if sidecar.read_bytes() != (digest + "\n").encode("ascii"):
        raise ValueError("authorization receipt sidecar does not match")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"authorization receipt is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("authorization receipt must contain a JSON object")
    return path, value, digest, name_match.group("stamp")


def _require_receipt_invariants(receipt: dict[str, Any], now: datetime) -> None:
    if (
        receipt.get("schema_version") != 1
        or receipt.get("gate") != "screening_execution_window"
        or receipt.get("authorized") is not True
        or receipt.get("decision") != "authorize_render_only"
        or receipt.get("reason_codes") != []
        or receipt.get("validation_errors") != []
    ):
        raise ValueError("receipt is not a successful screening execution window")
    policy = receipt.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("receipt policy is missing")
    if (
        policy.get("jobs_submitted_by_gate") != 0
        or not _is_integer(policy.get("max_age_seconds"))
        or not 0 < policy["max_age_seconds"] <= DEFAULT_MAX_AGE_SECONDS
        or not _is_integer(policy.get("max_clock_skew_seconds"))
        or not 0
        <= policy["max_clock_skew_seconds"]
        <= DEFAULT_MAX_CLOCK_SKEW_SECONDS
        or not _is_integer(policy.get("minimum_remaining_seconds"))
        or policy["minimum_remaining_seconds"] < DEFAULT_MINIMUM_REMAINING_SECONDS
        or not _is_integer(policy.get("minimum_group_soft_margin_bytes"))
        or policy["minimum_group_soft_margin_bytes"]
        < DEFAULT_MINIMUM_GROUP_SOFT_MARGIN_BYTES
        or policy.get("maximum_array_elements") != 5
        or policy.get("maximum_array_concurrency") != 2
    ):
        raise ValueError("receipt safety policy is weaker than the production policy")
    expires_at = _parse_timestamp(receipt.get("expires_at"))
    if expires_at is None:
        raise ValueError("receipt expiration is invalid")
    if (expires_at - now).total_seconds() < policy["minimum_remaining_seconds"]:
        raise ValueError("receipt has expired or has too little time remaining")
    if receipt.get("campaign", {}).get("counts") != EXPECTED_CAMPAIGN_COUNTS:
        raise ValueError("receipt campaign state is not the locked READY boundary")
    if receipt.get("release", {}).get("release_id") != LOCKED_RELEASE.release_id:
        raise ValueError("receipt release identity does not match the locked release")
    if receipt.get("render", {}).get("performed") is not False:
        raise ValueError("receipt is not render-only")
    if receipt.get("render", {}).get("next_command") != CONTROLLER_RENDER_COMMAND:
        raise ValueError("receipt render command is not the locked controller command")
    if receipt.get("submission") != {
        "performed": False,
        "sbatch_calls": 0,
        "array_shape": "0-4%2",
        "scheduler_jobs_if_submitted": 5,
    }:
        raise ValueError("receipt submission boundary is invalid")


def _evidence_paths(receipt: dict[str, Any]) -> tuple[str, str, str]:
    try:
        quota = receipt["storage"]["assessment"]["inputs"]["quota_snapshot"]["path"]
        scheduler = receipt["scheduler"]["evidence"]["path"]
        preflight = receipt["live_remote_preflight"]["evidence"]["path"]
    except (KeyError, TypeError) as exc:
        raise ValueError("receipt does not bind all live evidence paths") from exc
    if not all(isinstance(path, str) and path for path in (quota, scheduler, preflight)):
        raise ValueError("receipt live evidence paths are invalid")
    return quota, scheduler, preflight


def _evidence_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    try:
        return {
            "quota_sha256": receipt["storage"]["assessment"]["inputs"]
            ["quota_snapshot"]["sha256"],
            "scheduler_sha256": receipt["scheduler"]["evidence"]["sha256"],
            "preflight_sha256": receipt["live_remote_preflight"]["evidence"]["sha256"],
            "expires_at": receipt["expires_at"],
            "release_id": receipt["release"]["release_id"],
            "required_bytes": receipt["storage"]["required_bytes"],
            "group_soft_margin_bytes": receipt["storage"]["group_soft_margin_bytes"],
            "campaign": receipt["campaign"],
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("receipt evidence identity is incomplete") from exc


def _authorization_identity(receipt: dict[str, Any]) -> str:
    payload = json.dumps(
        _evidence_identity(receipt),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _default_renderer(project_root: Path) -> dict[str, Any]:
    from src.campaigns.control import _render_command, load_definition

    campaigns_raw = project_root / "campaigns"
    campaign_raw = campaigns_raw / "large_scale_thai_drl_v2"
    attempts_raw = campaign_raw / "attempts"
    commands_raw = campaign_raw / "commands"
    reports_raw = campaign_raw / "reports"
    latest_raw = attempts_raw / "screening_latest.json"
    command_raw = commands_raw / "screening.sh"
    status_raw = reports_raw / "campaign_status.md"
    events_raw = campaign_raw / "events.jsonl"
    for label, candidate in (
        ("campaigns", campaigns_raw),
        ("campaign", campaign_raw),
        ("attempts", attempts_raw),
        ("commands", commands_raw),
        ("reports", reports_raw),
    ):
        if candidate.is_symlink():
            raise ValueError(f"{label} output directory must not be a symlink")
    resolved_campaign = campaign_raw.resolve()
    if not resolved_campaign.is_relative_to(project_root):
        raise ValueError("screening campaign output directory escapes the project")
    for label, candidate in (
        ("latest attempt", latest_raw),
        ("screening command", command_raw),
        ("campaign status", status_raw),
        ("campaign event log", events_raw),
    ):
        if candidate.is_symlink():
            raise ValueError(f"{label} output path must not be a symlink")
        if not candidate.resolve().is_relative_to(resolved_campaign):
            raise ValueError(f"{label} output path escapes the campaign")

    definition, _ = load_definition(
        project_root / DEFAULT_SPEC_PATH,
        project_root=project_root,
        campaign_dir=None,
    )
    if definition.campaign_dir.resolve() != resolved_campaign:
        raise ValueError("resolved screening campaign directory is unexpected")
    attempt_dir = definition.campaign_dir / "attempts"
    before_attempts = (
        {
            path.resolve()
            for path in attempt_dir.glob("screening_*.json")
            if path.name != "screening_latest.json"
        }
        if attempt_dir.is_dir()
        else set()
    )
    stage, command, script = _render_command(definition, "screening")
    latest_path = attempt_dir / "screening_latest.json"
    if (
        stage != "screening"
        or latest_path.is_symlink()
        or not latest_path.resolve().is_relative_to(resolved_campaign)
        or not latest_path.is_file()
    ):
        raise ValueError("renderer did not create the screening attempt record")
    attempt = json.loads(latest_path.read_text(encoding="utf-8"))
    if not isinstance(attempt, dict):
        raise ValueError("screening attempt record is invalid")
    attempt_id = attempt.get("attempt_id")
    attempt_sha256 = attempt.get("attempt_sha256")
    if (
        not isinstance(attempt_id, str)
        or _ATTEMPT_ID.fullmatch(attempt_id) is None
        or not isinstance(attempt_sha256, str)
        or _SHA256.fullmatch(attempt_sha256) is None
        or attempt.get("schema_version") != 2
        or attempt.get("status") != "rendered"
        or attempt.get("campaign_id") != "large_scale_thai_drl_v2"
        or attempt.get("stage_id") != "screening"
    ):
        raise ValueError("screening attempt identity or schema is invalid")
    attempt_basis = dict(attempt)
    attempt_basis.pop("attempt_sha256", None)
    recomputed_attempt_sha256 = hashlib.sha256(
        json.dumps(
            attempt_basis,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if recomputed_attempt_sha256 != attempt_sha256:
        raise ValueError("screening attempt self-hash is invalid")
    immutable_attempt_path = attempt_dir / f"{attempt_id}.json"
    after_attempts = {
        path.resolve()
        for path in attempt_dir.glob("screening_*.json")
        if path.name != "screening_latest.json"
    }
    new_attempts = after_attempts - before_attempts
    if (
        immutable_attempt_path.is_symlink()
        or not immutable_attempt_path.resolve().is_relative_to(resolved_campaign)
        or not immutable_attempt_path.is_file()
        or immutable_attempt_path.resolve() not in new_attempts
        or len(new_attempts) != 1
        or immutable_attempt_path.read_bytes() != latest_path.read_bytes()
    ):
        raise ValueError("renderer did not create exactly one matching immutable attempt")
    if (
        command.count("sbatch ") != 1
        or f"CAMPAIGN_ATTEMPT_ID={attempt_id}" not in command
        or f"CAMPAIGN_ATTEMPT_SHA256={attempt_sha256}" not in command
    ):
        raise ValueError("rendered command is not bound to the immutable attempt")
    expected_script = definition.campaign_dir / "commands/screening.sh"
    if (
        script.is_symlink()
        or not script.resolve().is_relative_to(resolved_campaign)
        or script.resolve() != expected_script.resolve()
        or not script.is_file()
    ):
        raise ValueError("renderer returned an unexpected screening command path")
    return {
        "stage": stage,
        "command": command,
        "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "script": str(script),
        "script_sha256": _sha256_file(script),
        "attempt_path": str(immutable_attempt_path),
        "attempt_file_sha256": _sha256_file(immutable_attempt_path),
        "attempt_id": attempt_id,
        "attempt_sha256": attempt_sha256,
    }


def _write_exclusive_record(path: Path, value: dict[str, Any]) -> dict[str, str]:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    _atomic_text(path, payload)
    _atomic_text(Path(str(path) + ".sha256"), digest + "\n")
    return {"path": str(path), "sha256": digest}


def _run_live_gate(
    project_root: Path,
    quota: Path,
    scheduler: Path,
    preflight: Path,
) -> dict[str, Any]:
    return prepare_screening_execution_window(
        project_root=project_root,
        quota_snapshot_path=quota,
        scheduler_snapshot_path=scheduler,
        live_remote_preflight_path=preflight,
    )


def render_authorized_screening(
    *,
    project_root: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    current = _utc_now()
    root = Path(project_root).resolve()
    path, receipt, receipt_sha256, receipt_stamp = _load_receipt(root, receipt_path)
    _require_receipt_invariants(receipt, current)
    quota_raw, scheduler_raw, preflight_raw = _evidence_paths(receipt)
    quota, scheduler, preflight = _validate_operational_evidence_triplet(
        project_root=root,
        quota_snapshot_path=quota_raw,
        scheduler_snapshot_path=scheduler_raw,
        live_remote_preflight_path=preflight_raw,
    )
    capture_stamp = quota.name.removeprefix("quota_snapshot_").removesuffix(".json")
    if receipt_stamp != capture_stamp:
        raise ValueError("authorization receipt stamp must match its evidence capture stamp")

    fresh = _run_live_gate(root, quota, scheduler, preflight)
    if fresh.get("authorized") is not True:
        raise ValueError(
            "live gate revalidation failed: "
            + ",".join(str(code) for code in fresh.get("reason_codes", []))
        )
    if _evidence_identity(receipt) != _evidence_identity(fresh):
        raise ValueError("receipt no longer matches the freshly recomputed gate")

    authorization_identity = _authorization_identity(fresh)
    claim_path = path.with_name(
        f"screening_execution_identity_{authorization_identity}.claim.json"
    )
    consumed_path = path.with_name(
        f"screening_execution_identity_{authorization_identity}.consumed.json"
    )
    failed_path = path.with_name(
        f"screening_execution_identity_{authorization_identity}.failed.json"
    )
    stage_lock_path = path.with_name("screening_stage_render.lock.json")
    claim_sidecar = Path(str(claim_path) + ".sha256")
    if consumed_path.exists() or Path(str(consumed_path) + ".sha256").exists():
        raise FileExistsError("authorization receipt was already consumed")
    if claim_path.exists() or claim_sidecar.exists() or failed_path.exists():
        raise FileExistsError("authorization identity is already claimed or in doubt")

    campaign_dir = root / "campaigns/large_scale_thai_drl_v2"
    attempt_dir = campaign_dir / "attempts"
    command_dir = campaign_dir / "commands"
    reports_dir = campaign_dir / "reports"
    command_path = command_dir / "screening.sh"
    status_path = reports_dir / "campaign_status.md"
    events_path = campaign_dir / "events.jsonl"
    unsafe_directories = (
        root / "campaigns",
        campaign_dir,
        attempt_dir,
        command_dir,
        reports_dir,
    )
    unsafe_leaves = (
        attempt_dir / "screening_latest.json",
        command_path,
        status_path,
        events_path,
    )
    if any(candidate.is_symlink() for candidate in unsafe_directories + unsafe_leaves):
        raise FileExistsError(
            "screening has an unsafe symlinked output path; reconcile it before any new window"
        )
    existing_attempts = (
        [
            candidate
            for candidate in attempt_dir.glob("screening_*.json")
            if candidate.name == "screening_latest.json"
            or _ATTEMPT_ID.fullmatch(candidate.stem) is not None
        ]
        if attempt_dir.is_dir()
        else []
    )
    if command_path.exists() or command_path.is_symlink() or existing_attempts:
        raise FileExistsError(
            "screening already has a rendered command or attempt; reconcile it before any new window"
        )
    stage_lock = {
        "schema_version": 1,
        "status": "locked_for_single_render",
        "locked_at": current.isoformat().replace("+00:00", "Z"),
        "authorization_identity": authorization_identity,
        "receipt_sha256": receipt_sha256,
        "jobs_submitted": 0,
    }
    stage_lock_record = _write_exclusive_record(stage_lock_path, stage_lock)
    claim = {
        "schema_version": 1,
        "status": "claimed",
        "claimed_at": current.isoformat().replace("+00:00", "Z"),
        "receipt_path": str(path),
        "receipt_sha256": receipt_sha256,
        "authorization_identity": authorization_identity,
        "stage_lock": stage_lock_record,
        "jobs_submitted": 0,
    }
    claim_record = _write_exclusive_record(claim_path, claim)

    try:
        pre_render_time = _utc_now()
        _require_receipt_invariants(receipt, pre_render_time)
        _require_receipt_invariants(fresh, pre_render_time)
        rendered = _default_renderer(root)
        if (
            rendered.get("stage") != "screening"
            or not isinstance(rendered.get("script"), str)
            or not isinstance(rendered.get("attempt_id"), str)
            or _ATTEMPT_ID.fullmatch(rendered["attempt_id"]) is None
            or not isinstance(rendered.get("attempt_sha256"), str)
            or _SHA256.fullmatch(rendered["attempt_sha256"]) is None
            or not isinstance(rendered.get("attempt_file_sha256"), str)
            or _SHA256.fullmatch(rendered["attempt_file_sha256"]) is None
            or not isinstance(rendered.get("command_sha256"), str)
            or _SHA256.fullmatch(rendered["command_sha256"]) is None
            or not isinstance(rendered.get("script_sha256"), str)
            or _SHA256.fullmatch(rendered["script_sha256"]) is None
        ):
            raise ValueError("renderer did not return a bound screening attempt")
        completed_time = _utc_now()
        _require_receipt_invariants(receipt, completed_time)
        _require_receipt_invariants(fresh, completed_time)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "render_failed_or_in_doubt",
            "failed_at": _utc_now().isoformat().replace("+00:00", "Z"),
            "authorization_identity": authorization_identity,
            "receipt_sha256": receipt_sha256,
            "claim": claim_record,
            "stage_lock": stage_lock_record,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "jobs_submitted": 0,
        }
        try:
            _write_exclusive_record(failed_path, failure)
        except (FileExistsError, OSError):
            pass
        raise
    consumed = {
        "schema_version": 1,
        "status": "consumed",
        "consumed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "receipt_path": str(path),
        "receipt_sha256": receipt_sha256,
        "authorization_identity": authorization_identity,
        "claim": claim_record,
        "stage_lock": stage_lock_record,
        "stage": "screening",
        "attempt_id": rendered["attempt_id"],
        "attempt_sha256": rendered["attempt_sha256"],
        "attempt_path": rendered.get("attempt_path"),
        "attempt_file_sha256": rendered.get("attempt_file_sha256"),
        "rendered_command_sha256": rendered.get("command_sha256"),
        "rendered_script": rendered["script"],
        "rendered_script_sha256": rendered.get("script_sha256"),
        "jobs_submitted": 0,
    }
    consumed_record = _write_exclusive_record(consumed_path, consumed)
    return {
        "decision": "rendered_once",
        "stage": "screening",
        "receipt_sha256": receipt_sha256,
        "authorization_identity": authorization_identity,
        "claim": claim_record,
        "stage_lock": stage_lock_record,
        "consumption": consumed_record,
        "attempt_id": rendered["attempt_id"],
        "attempt_sha256": rendered["attempt_sha256"],
        "script": rendered["script"],
        "jobs_submitted": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate and consume one fresh screening authorization receipt, "
            "then render exactly once. This command never submits the rendered job."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = render_authorized_screening(
            project_root=args.project_root,
            receipt_path=args.receipt,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"render authorization rejected: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
