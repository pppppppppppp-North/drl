from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.prepare_screening_execution_window import (
    DEFAULT_FIXED_RESERVE_BYTES,
    DEFAULT_MINIMUM_GROUP_SOFT_MARGIN_BYTES,
    DEFAULT_SMOKE_SUMMARY_PATH,
    EXPECTED_CAMPAIGN_COUNTS,
    EXPECTED_FILESYSTEM,
    EXPECTED_GID,
    EXPECTED_OUTPUT_PARENT,
    EXPECTED_PROJECT_ID,
    LOCKED_RELEASE,
    _PreparedExecutionWindow,
    _validate_operational_evidence_triplet,
    build_parser,
    main as prepare_main,
    prepare_screening_execution_window,
    validate_screening_job_script,
    verify_live_remote_preflight,
    verify_locked_release,
    write_authorization_receipt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
GIB = 1024**3


def _write_signed(path: Path, value: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    else:
        payload = value
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".sha256").write_text(digest + "\n", encoding="ascii")


def _quota_snapshot(
    *,
    timestamp: datetime = NOW,
    group_soft_headroom: int = 20 * GIB,
) -> dict:
    group_used = 5 * GIB
    project_used = 20 * GIB
    return {
        "schema_version": 1,
        "captured_at": timestamp.isoformat(),
        "filesystem": EXPECTED_FILESYSTEM,
        "group": {
            "name": "s00658",
            "gid": EXPECTED_GID,
            "bytes": {
                "used": group_used,
                "soft_limit": group_used + group_soft_headroom,
                "hard_limit": group_used + group_soft_headroom + 20 * GIB,
                "grace_expired": False,
            },
            "inodes": {
                "used": 100_000,
                "soft_limit": 1_000_000,
                "hard_limit": 1_100_000,
                "grace_expired": False,
            },
        },
        "project": {
            "id": EXPECTED_PROJECT_ID,
            "bytes": {
                "used": project_used,
                "soft_limit": project_used + 200 * GIB,
                "hard_limit": project_used + 250 * GIB,
                "grace_expired": False,
            },
            "inodes": {
                "used": 200_000,
                "soft_limit": 10_000_000,
                "hard_limit": 11_000_000,
                "grace_expired": False,
            },
        },
        "write_probe": {
            "completed_at": timestamp.isoformat(),
            "filesystem": EXPECTED_FILESYSTEM,
            "gid": EXPECTED_GID,
            "path": f"{EXPECTED_OUTPUT_PARENT}/.screening-window-probe",
            "bytes_written": 1_048_576,
            "fsync_succeeded": True,
            "removed": True,
        },
    }


def _scheduler_snapshot(
    *,
    timestamp: datetime = NOW,
    queue_row: str | None = None,
    account_queue_row: str | None = None,
    project_remaining: str = "9984.12",
    user_remaining: str = "496.25",
) -> str:
    queue = (
        "JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)\n"
        + (f"{queue_row}\n" if queue_row else "")
    )
    account_queue = (
        "JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)\n"
        + (f"{account_queue_row}\n" if account_queue_row else "")
    )
    return f"""# BistKA Capacity Snapshot
captured_at={timestamp.isoformat()}
hostname=bistka-master
user=s00658

## mycredit
Account Allocation(SHr) Used(SHr) Remaining(SHr) Remaining(%) Expiry Date
25sfcs03 10000.00 15.88 {project_remaining} 99.84% 2027-02-28
s00658 500.00 3.75 {user_remaining} 99.25% -

## myquota
/lustrefs/project/25sfcs03 39G 300G 280000 11000000

## sinfo
PARTITION AVAIL TIMELIMIT NODES STATE NODELIST
compute-long up 4-00:00:00 1 idle bistka-compute-01

## squeue -u s00658
{queue}
## squeue -A 25sfcs03
{account_queue}
## sacctmgr -nP show assoc where user=s00658 account=25sfcs03 format=User,Account,Partition,QOS
s00658|25sfcs03||25sfcs03

## sacctmgr -nP show qos where name=25sfcs03 format=Name,GrpJobs,GrpSubmitJobs,GrpTRES,Flags
25sfcs03|2|5|cpu=32,gres/gpu=2|NoDecay

## scontrol show partition -o
PartitionName=compute-long State=UP MaxTime=4-00:00:00

## deployed release identity
deployed_bundle_sha256={LOCKED_RELEASE.bundle_sha256}
deployment_manifest_sha256={LOCKED_RELEASE.manifest_sha256}

## live scheduler policy
Maximum Job Submission: 5 jobs/group
Maximum Job Running: 2 jobs/group
Maximum CPU-cores = 32 CPU-cores/group
Maximum GPUs = 2 GPUs/group
compute-long: 4 days
"""


def _campaign_view(_root: Path, _spec: Path) -> dict:
    return {
        "counts": EXPECTED_CAMPAIGN_COUNTS,
        "screening": {
            "state": "READY",
            "progress": "0/150 complete manifests",
            "manifest_expected": 150,
            "manifest_found": 0,
            "manifest_complete": 0,
            "manifest_running": 0,
            "manifest_failed": 0,
        },
    }


def _valid_release(*_args, **_kwargs) -> dict:
    return {
        "valid": True,
        "reason_codes": [],
        "validation_errors": [],
        "release_id": LOCKED_RELEASE.release_id,
        "job": {
            "directives": {
                "array": "0-4%2",
                "cpus-per-task": "2",
                "mem": "12GB",
                "partition": "compute-long",
                "account": "25sfcs03",
            },
            "controller_sbatch_invocations": 1,
            "nested_sbatch_count": 0,
        },
    }


def _valid_live_preflight(*_args, **_kwargs) -> dict:
    return {
        "valid": True,
        "reason_codes": [],
        "validation_errors": [],
        "completed_at": NOW.isoformat().replace("+00:00", "Z"),
        "release_id": LOCKED_RELEASE.release_id,
        "verified_file_count": 248,
        "inventory_matches_manifest": True,
    }


def _prepare(
    tmp_path: Path,
    *,
    quota: dict | None = None,
    scheduler_text: str | None = None,
    now: datetime = NOW,
    release_provider=_valid_release,
    live_preflight_provider=_valid_live_preflight,
    smoke_summary_path: str | Path = DEFAULT_SMOKE_SUMMARY_PATH,
    policy_overrides: dict | None = None,
) -> dict:
    quota_path = tmp_path / "quota.json"
    scheduler_path = tmp_path / "scheduler.txt"
    live_preflight_path = tmp_path / "live-preflight.json"
    _write_signed(quota_path, quota or _quota_snapshot())
    _write_signed(
        scheduler_path,
        scheduler_text if scheduler_text is not None else _scheduler_snapshot(),
    )
    return prepare_screening_execution_window(
        project_root=PROJECT_ROOT,
        smoke_summary_path=smoke_summary_path,
        quota_snapshot_path=quota_path,
        scheduler_snapshot_path=scheduler_path,
        live_remote_preflight_path=live_preflight_path,
        now=now,
        campaign_view_provider=_campaign_view,
        release_view_provider=release_provider,
        live_remote_preflight_provider=live_preflight_provider,
        **(policy_overrides or {}),
    )


def test_fresh_exact_window_authorizes_render_only(tmp_path: Path) -> None:
    receipt = _prepare(tmp_path)

    assert receipt["authorized"] is True
    assert receipt["decision"] == "authorize_render_only"
    assert receipt["remaining_seconds"] == 900
    assert receipt["storage"]["group_soft_margin_bytes"] >= (
        DEFAULT_MINIMUM_GROUP_SOFT_MARGIN_BYTES
    )
    assert receipt["scheduler"]["active_queue_rows"] == []
    assert receipt["render"]["performed"] is False
    assert "scripts.render_authorized_screening" in receipt["render"]["next_command"]
    assert receipt["submission"] == {
        "performed": False,
        "sbatch_calls": 0,
        "array_shape": "0-4%2",
        "scheduler_jobs_if_submitted": 5,
    }


def test_expired_quota_and_write_probe_hold(tmp_path: Path) -> None:
    old = NOW - timedelta(seconds=901)
    receipt = _prepare(tmp_path, quota=_quota_snapshot(timestamp=old))

    assert receipt["authorized"] is False
    assert "storage_capacity_not_eligible" in receipt["reason_codes"]
    assert "storage_quota_snapshot_stale" in receipt["reason_codes"]
    assert "storage_quota_write_probe_stale" in receipt["reason_codes"]


def test_bad_scheduler_sidecar_holds(tmp_path: Path) -> None:
    quota_path = tmp_path / "quota.json"
    scheduler_path = tmp_path / "scheduler.txt"
    _write_signed(quota_path, _quota_snapshot())
    _write_signed(scheduler_path, _scheduler_snapshot())
    Path(str(scheduler_path) + ".sha256").write_text("0" * 64 + "\n", encoding="ascii")

    receipt = prepare_screening_execution_window(
        project_root=PROJECT_ROOT,
        quota_snapshot_path=quota_path,
        scheduler_snapshot_path=scheduler_path,
        live_remote_preflight_path=tmp_path / "live-preflight.json",
        now=NOW,
        campaign_view_provider=_campaign_view,
        release_view_provider=_valid_release,
        live_remote_preflight_provider=_valid_live_preflight,
    )

    assert receipt["authorized"] is False
    assert "scheduler_snapshot_sidecar_invalid" in receipt["reason_codes"]


def test_less_than_one_gib_group_soft_margin_holds(tmp_path: Path) -> None:
    receipt = _prepare(
        tmp_path,
        quota=_quota_snapshot(group_soft_headroom=7_000_000_000),
    )

    assert receipt["storage"]["candidate_capacity_eligible"] is True
    assert 0 <= receipt["storage"]["group_soft_margin_bytes"] < GIB
    assert receipt["authorized"] is False
    assert "storage_group_soft_margin_below_minimum" in receipt["reason_codes"]


def test_one_active_queue_row_holds(tmp_path: Path) -> None:
    receipt = _prepare(
        tmp_path,
        scheduler_text=_scheduler_snapshot(
            queue_row="4012 compute-long other s00658 R 0:02 1 bistka-compute-01",
        ),
    )

    assert receipt["authorized"] is False
    assert "scheduler_queue_not_empty" in receipt["reason_codes"]


def test_one_account_queue_row_holds_even_when_user_queue_is_empty(
    tmp_path: Path,
) -> None:
    receipt = _prepare(
        tmp_path,
        scheduler_text=_scheduler_snapshot(
            account_queue_row=(
                "4013 compute-long peer otheruser R 0:02 1 bistka-compute-01"
            ),
        ),
    )

    assert receipt["scheduler"]["active_queue_rows"] == []
    assert receipt["authorized"] is False
    assert "scheduler_account_queue_not_empty" in receipt["reason_codes"]


def test_credit_must_cover_conservative_screening_estimate(
    tmp_path: Path,
) -> None:
    receipt = _prepare(
        tmp_path,
        scheduler_text=_scheduler_snapshot(
            project_remaining="100.00",
            user_remaining="100.00",
        ),
    )

    assert receipt["authorized"] is False
    assert "scheduler_credit_below_screening_estimate" in receipt["reason_codes"]


@pytest.mark.parametrize(
    "policy_overrides",
    [
        {"max_age_seconds": 901},
        {"max_age_seconds": float("nan")},
        {"max_age_seconds": True},
        {"max_clock_skew_seconds": 61},
        {"max_clock_skew_seconds": float("nan")},
        {"max_clock_skew_seconds": False},
        {"minimum_remaining_seconds": 119},
        {"minimum_remaining_seconds": float("nan")},
        {"minimum_remaining_seconds": True},
        {
            "minimum_group_soft_margin_bytes": (
                DEFAULT_MINIMUM_GROUP_SOFT_MARGIN_BYTES - 1
            )
        },
        {"minimum_group_soft_margin_bytes": float("nan")},
        {"minimum_group_soft_margin_bytes": True},
        {"fixed_reserve_bytes": DEFAULT_FIXED_RESERVE_BYTES - 1},
        {"fixed_reserve_bytes": float("nan")},
        {"fixed_reserve_bytes": True},
    ],
)
def test_execution_policy_can_only_be_made_stricter(
    tmp_path: Path,
    policy_overrides: dict,
) -> None:
    with pytest.raises(ValueError, match="must be"):
        _prepare(
            tmp_path,
            policy_overrides=policy_overrides,
        )


@pytest.mark.parametrize(
    ("original", "replacement", "reason_code"),
    [
        (
            "25sfcs03|2|5|cpu=32,gres/gpu=2|NoDecay",
            "25sfcs03|2|5|cpu=32,gres/gpu=2|",
            "scheduler_qos_mismatch",
        ),
        (
            "25sfcs03|2|5|cpu=32,gres/gpu=2|NoDecay",
            "25sfcs03|2|5|cpu=32,cpu=32,gres/gpu=2|NoDecay",
            "scheduler_qos_mismatch",
        ),
        (
            "PartitionName=compute-long State=UP MaxTime=4-00:00:00",
            "PartitionName=compute-long State=DOWN MaxTime=4-00:00:00",
            "scheduler_partition_control_mismatch",
        ),
        (
            "PartitionName=compute-long State=UP MaxTime=4-00:00:00",
            "PartitionName=compute-long State=UP MaxTime=2-00:00:00",
            "scheduler_partition_control_mismatch",
        ),
    ],
)
def test_qos_and_partition_control_are_semantically_enforced(
    tmp_path: Path,
    original: str,
    replacement: str,
    reason_code: str,
) -> None:
    scheduler = _scheduler_snapshot().replace(original, replacement)

    receipt = _prepare(tmp_path, scheduler_text=scheduler)

    assert receipt["authorized"] is False
    assert reason_code in receipt["reason_codes"]


def test_release_drift_holds(tmp_path: Path) -> None:
    def drifted_release(*_args, **_kwargs) -> dict:
        return {
            "valid": False,
            "reason_codes": ["observed_feature_table_sha256_mismatch"],
            "validation_errors": [
                {
                    "code": "observed_feature_table_sha256_mismatch",
                    "message": "observed feature table SHA-256 does not match",
                }
            ],
        }

    receipt = _prepare(tmp_path, release_provider=drifted_release)

    assert receipt["authorized"] is False
    assert "observed_feature_table_sha256_mismatch" in receipt["reason_codes"]


def test_noncanonical_self_signed_smoke_summary_cannot_authorize(
    tmp_path: Path,
) -> None:
    canonical = PROJECT_ROOT / DEFAULT_SMOKE_SUMMARY_PATH
    alternate = tmp_path / "alternate-smoke.json"
    alternate.write_bytes(canonical.read_bytes())
    Path(str(alternate) + ".sha256").write_text(
        hashlib.sha256(alternate.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )

    receipt = _prepare(
        tmp_path / "inputs",
        smoke_summary_path=alternate,
    )

    assert receipt["storage"]["candidate_capacity_eligible"] is True
    assert receipt["authorized"] is False
    assert "smoke_summary_path_mismatch" in receipt["reason_codes"]


def test_live_remote_preflight_failure_holds(tmp_path: Path) -> None:
    def failed_preflight(*_args, **_kwargs) -> dict:
        return {
            "valid": False,
            "reason_codes": ["live_remote_preflight_inventory_mismatch"],
            "validation_errors": [
                {
                    "code": "live_remote_preflight_inventory_mismatch",
                    "message": "one deployed file differs",
                }
            ],
            "completed_at": NOW.isoformat().replace("+00:00", "Z"),
        }

    receipt = _prepare(
        tmp_path,
        live_preflight_provider=failed_preflight,
    )

    assert receipt["authorized"] is False
    assert "live_remote_preflight_inventory_mismatch" in receipt["reason_codes"]


def test_live_remote_release_marker_drift_holds(tmp_path: Path) -> None:
    scheduler = _scheduler_snapshot().replace(
        f"deployed_bundle_sha256={LOCKED_RELEASE.bundle_sha256}",
        f"deployed_bundle_sha256={'0' * 64}",
    )
    receipt = _prepare(tmp_path, scheduler_text=scheduler)

    assert receipt["authorized"] is False
    assert "scheduler_remote_release_mismatch" in receipt["reason_codes"]


def test_optional_module_listing_failure_does_not_block_scheduler_gate(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler_snapshot() + """
## loaded modules
module: command not found
"""
    receipt = _prepare(tmp_path, scheduler_text=scheduler)

    assert receipt["authorized"] is True


def test_remote_capture_producer_contains_every_gate_input() -> None:
    path = PROJECT_ROOT / "scripts/capture_screening_execution_window_evidence.sh"
    text = path.read_text(encoding="utf-8")

    assert "squeue -A \"$ACCOUNT\"" in text
    assert "## deployed release identity" in text
    assert "capture_screening_quota_snapshot" in text
    assert "check_bistka_calibration_preflight" in text
    assert text.count("sha256sum") >= 3
    assert "sbatch" not in text


@pytest.mark.parametrize(
    ("directive", "value"),
    [
        ("array", "0-5%3"),
        ("cpus-per-task", "4"),
        ("mem", "16GB"),
        ("partition", "compute-normal"),
        ("account", "wrong-account"),
    ],
)
def test_wrong_array_or_resources_are_rejected(
    tmp_path: Path,
    directive: str,
    value: str,
) -> None:
    directives = {
        "nodes": "1",
        "ntasks": "1",
        "cpus-per-task": "2",
        "mem": "12GB",
        "time": "4-00:00:00",
        "partition": "compute-long",
        "array": "0-4%2",
        "account": "25sfcs03",
    }
    directives[directive] = value
    lines = ["#!/bin/bash"] + [
        f"#SBATCH --{name}={item}" for name, item in directives.items()
    ]
    lines.extend(
        [
            'python -m src.experiments.run_screening_storage_safe \\',
            "  --task-id \"$SLURM_ARRAY_TASK_ID\"",
        ]
    )
    path = tmp_path / "screening.sbatch"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate_screening_job_script(path)

    assert result["valid"] is False
    assert "screening_job_resources_mismatch" in result["reason_codes"]


def test_hold_does_not_touch_campaign_or_existing_output(tmp_path: Path) -> None:
    watched = [
        PROJECT_ROOT / "campaigns/large_scale_thai_drl_v2/state.json",
        PROJECT_ROOT / "campaigns/large_scale_thai_drl_v2/reports/campaign_status.md",
        PROJECT_ROOT / "campaigns/large_scale_thai_drl_v2/events.jsonl",
        PROJECT_ROOT / "campaigns/large_scale_thai_drl_v2/commands/screening.sh",
        PROJECT_ROOT / "campaigns/large_scale_thai_drl_v2/attempts/screening_latest.json",
    ]
    before = {
        path: path.read_bytes() if path.exists() else None
        for path in watched
    }
    output = tmp_path / "window.json"
    output.write_bytes(b"existing receipt must survive\n")
    receipt = _prepare(
        tmp_path,
        scheduler_text=_scheduler_snapshot(
            queue_row="4012 compute-long other s00658 PD 0:00 1 (Resources)",
        ),
    )

    with pytest.raises(ValueError):
        write_authorization_receipt(output, receipt, project_root=PROJECT_ROOT)

    assert output.read_bytes() == b"existing receipt must survive\n"
    assert not Path(str(output) + ".sha256").exists()
    assert {
        path: path.read_bytes() if path.exists() else None
        for path in watched
    } == before


def test_authorized_receipt_is_atomic_and_checksum_bound(tmp_path: Path) -> None:
    receipt = _prepare(tmp_path)
    output = (
        tmp_path
        / "reports/screening_execution_windows"
        / "screening_execution_window_20260725T120000Z.json"
    )

    written = write_authorization_receipt(
        output,
        receipt,
        project_root=tmp_path,
        now=NOW,
    )

    assert output.is_file()
    assert Path(written["sidecar_path"]).read_text(encoding="ascii") == (
        hashlib.sha256(output.read_bytes()).hexdigest() + "\n"
    )

    second_receipt = _prepare(tmp_path / "second-inputs")
    with pytest.raises(FileExistsError):
        write_authorization_receipt(
            output,
            second_receipt,
            project_root=tmp_path,
            now=NOW,
        )


def test_fabricated_dictionary_cannot_be_written_as_authorization(
    tmp_path: Path,
) -> None:
    output = (
        tmp_path
        / "reports/screening_execution_windows"
        / "screening_execution_window_20260725T120000Z.json"
    )
    fabricated = {
        "authorized": True,
        "expires_at": (NOW + timedelta(hours=1)).isoformat(),
        "policy": {"minimum_remaining_seconds": 0},
    }

    with pytest.raises(ValueError, match="unmodified result of the full gate"):
        write_authorization_receipt(
            output,
            fabricated,
            project_root=tmp_path,
            now=NOW,
        )

    assert not output.exists()


def test_directly_constructed_private_capability_is_not_gate_issued(
    tmp_path: Path,
) -> None:
    output = (
        tmp_path
        / "reports/screening_execution_windows"
        / "screening_execution_window_20260725T120000Z.json"
    )
    fabricated = _PreparedExecutionWindow(
        {
            "authorized": True,
            "expires_at": (NOW + timedelta(hours=1)).isoformat(),
            "policy": {"minimum_remaining_seconds": 120},
        }
    )

    with pytest.raises(ValueError, match="not issued by the full gate"):
        write_authorization_receipt(
            output,
            fabricated,
            project_root=tmp_path,
            now=NOW,
        )

    assert not output.exists()


def test_authorization_output_rejects_intermediate_symlink(
    tmp_path: Path,
) -> None:
    receipt = _prepare(tmp_path / "inputs")
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (project / "reports").symlink_to(outside, target_is_directory=True)
    output = (
        project
        / "reports/screening_execution_windows"
        / "screening_execution_window_20260725T120000Z.json"
    )

    with pytest.raises(ValueError, match="symlink"):
        write_authorization_receipt(
            output,
            receipt,
            project_root=project,
            now=NOW,
        )

    assert not (outside / "screening_execution_windows").exists()


def test_authorized_receipt_cannot_overwrite_protected_project_paths(
    tmp_path: Path,
) -> None:
    receipt = _prepare(tmp_path / "inputs")
    protected = tmp_path / "campaigns/state.json"

    with pytest.raises(ValueError):
        write_authorization_receipt(
            protected,
            receipt,
            project_root=tmp_path,
            now=NOW,
        )

    assert not protected.exists()


def test_authorized_receipt_is_rechecked_against_wall_time(
    tmp_path: Path,
) -> None:
    receipt = _prepare(tmp_path / "inputs")
    output = (
        tmp_path
        / "reports/screening_execution_windows"
        / "screening_execution_window_20260725T120000Z.json"
    )

    with pytest.raises(ValueError):
        write_authorization_receipt(
            output,
            receipt,
            project_root=tmp_path,
            now=NOW + timedelta(seconds=901),
        )

    assert not output.exists()


def test_exact_locked_release_and_bundle_are_current() -> None:
    release = verify_locked_release(PROJECT_ROOT)

    assert release["valid"] is True
    assert release["reason_codes"] == []
    assert release["manifest_file_count"] == 248
    assert release["bundle_inventory"] == {
        "verified_files": 248,
        "member_count": 250,
        "inventory_verified": True,
    }
    assert release["job"]["directives"]["array"] == "0-4%2"
    assert release["job"]["nested_sbatch_count"] == 0


def test_fetched_live_remote_preflight_verifies_all_248_files() -> None:
    path = (
        PROJECT_ROOT
        / "campaigns/large_scale_thai_drl_v2/operational/screening_capacity"
        / "live_preflight_20260726T143716Z.json"
    )
    completed = datetime.fromisoformat(
        json.loads(path.read_text(encoding="utf-8"))["completed_at_utc"]
    )

    result = verify_live_remote_preflight(
        path,
        project_root=PROJECT_ROOT,
        now=completed + timedelta(seconds=1),
    )

    assert result["valid"] is True
    assert result["verified_file_count"] == 248
    assert result["inventory_matches_manifest"] is True


def test_fetched_live_remote_preflight_expires_after_900_seconds() -> None:
    path = (
        PROJECT_ROOT
        / "campaigns/large_scale_thai_drl_v2/operational/screening_capacity"
        / "live_preflight_20260726T143716Z.json"
    )
    completed = datetime.fromisoformat(
        json.loads(path.read_text(encoding="utf-8"))["completed_at_utc"]
    )

    result = verify_live_remote_preflight(
        path,
        project_root=PROJECT_ROOT,
        now=completed + timedelta(seconds=901),
    )

    assert result["valid"] is False
    assert "live_remote_preflight_stale" in result["reason_codes"]


def test_scheduler_freshness_boundary_900_passes_901_holds(tmp_path: Path) -> None:
    at_limit = NOW - timedelta(seconds=900)
    pass_receipt = _prepare(
        tmp_path / "pass",
        quota=_quota_snapshot(timestamp=at_limit),
        scheduler_text=_scheduler_snapshot(timestamp=at_limit),
        now=NOW,
    )
    hold_receipt = _prepare(
        tmp_path / "hold",
        quota=_quota_snapshot(timestamp=at_limit - timedelta(seconds=1)),
        scheduler_text=_scheduler_snapshot(
            timestamp=at_limit - timedelta(seconds=1),
        ),
        now=NOW,
    )

    assert pass_receipt["authorized"] is False
    assert pass_receipt["remaining_seconds"] == 0
    assert "execution_window_ttl_below_minimum" in pass_receipt["reason_codes"]
    assert "scheduler_snapshot_stale" not in pass_receipt["reason_codes"]
    assert hold_receipt["authorized"] is False
    assert "scheduler_snapshot_stale" in hold_receipt["reason_codes"]


def test_production_cli_has_no_policy_weakening_flags() -> None:
    help_text = build_parser().format_help()

    assert "--max-age-seconds" not in help_text
    assert "--max-clock-skew-seconds" not in help_text
    assert "--minimum-remaining-seconds" not in help_text
    assert "--minimum-group-soft-margin-bytes" not in help_text


def test_production_evidence_requires_one_operational_capture_stamp() -> None:
    directory = (
        PROJECT_ROOT
        / "campaigns/large_scale_thai_drl_v2/operational/screening_capacity"
    )
    quota, scheduler, preflight = _validate_operational_evidence_triplet(
        project_root=PROJECT_ROOT,
        quota_snapshot_path=directory / "quota_snapshot_20260726T143716Z.json",
        scheduler_snapshot_path=directory / "scheduler_snapshot_20260726T143716Z.txt",
        live_remote_preflight_path=directory / "live_preflight_20260726T143716Z.json",
    )

    assert quota.name == "quota_snapshot_20260726T143716Z.json"
    assert scheduler.name == "scheduler_snapshot_20260726T143716Z.txt"
    assert preflight.name == "live_preflight_20260726T143716Z.json"

    with pytest.raises(ValueError, match="stamps must match"):
        _validate_operational_evidence_triplet(
            project_root=PROJECT_ROOT,
            quota_snapshot_path=directory / "quota_snapshot_20260726T142444Z.json",
            scheduler_snapshot_path=directory / "scheduler_snapshot_20260726T143716Z.txt",
            live_remote_preflight_path=directory / "live_preflight_20260726T143716Z.json",
        )


def test_production_receipt_filename_must_match_evidence_stamp() -> None:
    directory = (
        PROJECT_ROOT
        / "campaigns/large_scale_thai_drl_v2/operational/screening_capacity"
    )

    with pytest.raises(SystemExit) as raised:
        prepare_main(
            [
                "--project-root",
                str(PROJECT_ROOT),
                "--quota-snapshot",
                str(directory / "quota_snapshot_20260726T143716Z.json"),
                "--scheduler-snapshot",
                str(directory / "scheduler_snapshot_20260726T143716Z.txt"),
                "--live-remote-preflight",
                str(directory / "live_preflight_20260726T143716Z.json"),
                "--output",
                "reports/screening_execution_windows/"
                "screening_execution_window_20260726T143717Z.json",
            ]
        )

    assert raised.value.code == 2
