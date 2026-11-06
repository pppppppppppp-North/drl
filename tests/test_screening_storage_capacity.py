from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.assess_screening_storage import (
    EXPECTED_ALGORITHMS,
    assess_screening_storage,
    write_decision,
)


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
FILESYSTEM = "/lustrefs/project/25sfcs03"
GROUP = "s00658"
GID = 1039
PROJECT_ID = 25110303
OUTPUT_PARENT = f"{FILESYSTEM}/results/screening"


def _write_signed(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".sha256").write_text(digest + "\n", encoding="utf-8")


def _smoke_summary() -> dict:
    runs = []
    for algorithm in EXPECTED_ALGORITHMS:
        runs.append(
            {
                "algorithm": algorithm,
                "manifest": {"status": "complete"},
                "runtime_projection": {
                    "limit_hours": 72.0,
                    "projected_task_hours": 12.0,
                    "within_limit": True,
                },
                "artifacts": [
                    {
                        "path": f"results/{algorithm}/{algorithm}_model.zip",
                        "bytes": 10,
                    },
                    {
                        "path": f"results/{algorithm}/screening_smoke_result.json",
                        "bytes": 5,
                    },
                    {
                        "path": (
                            f"results/{algorithm}/checkpoints/{algorithm}/"
                            f"{algorithm}_checkpoint.zip"
                        ),
                        "bytes": 1_000,
                    },
                ],
            }
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "full_screening_eligible": True,
        "algorithms": list(EXPECTED_ALGORITHMS),
        "run_count": 5,
        "complete_manifest_count": 5,
        "full_screening_trials_per_algorithm": 3,
        "runtime_projection_limit_hours": 72.0,
        "runs": runs,
    }


def _quota_snapshot(*, byte_headroom: int = 10_000, inode_headroom: int = 10_000) -> dict:
    return {
        "schema_version": 1,
        "captured_at": NOW.isoformat(),
        "filesystem": FILESYSTEM,
        "group": {
            "name": GROUP,
            "gid": GID,
            "bytes": {
                "used": 1_000,
                "soft_limit": 1_000 + byte_headroom,
                "hard_limit": 1_000 + byte_headroom,
                "grace_expired": False,
            },
            "inodes": {
                "used": 100,
                "soft_limit": 100 + inode_headroom,
                "hard_limit": 100 + inode_headroom,
                "grace_expired": False,
            },
        },
        "project": {
            "id": PROJECT_ID,
            "bytes": {
                "used": 2_000,
                "soft_limit": 2_000 + byte_headroom,
                "hard_limit": 2_000 + byte_headroom,
                "grace_expired": False,
            },
            "inodes": {
                "used": 200,
                "soft_limit": 200 + inode_headroom,
                "hard_limit": 200 + inode_headroom,
                "grace_expired": False,
            },
        },
        "write_probe": {
            "completed_at": NOW.isoformat(),
            "filesystem": FILESYSTEM,
            "gid": GID,
            "path": f"{OUTPUT_PARENT}/.capacity-probe",
            "bytes_written": 4096,
            "fsync_succeeded": True,
            "removed": True,
        },
    }


def _assess(
    tmp_path: Path,
    smoke: dict | None = None,
    quota: dict | None = None,
    *,
    reserve: int = 256,
) -> dict:
    smoke_path = tmp_path / "smoke.json"
    quota_path = tmp_path / "quota.json"
    _write_signed(smoke_path, smoke or _smoke_summary())
    _write_signed(quota_path, quota or _quota_snapshot())
    return assess_screening_storage(
        smoke_path,
        quota_path,
        expected_group=GROUP,
        expected_gid=GID,
        expected_project_id=PROJECT_ID,
        expected_filesystem=FILESYSTEM,
        expected_output_parent=OUTPUT_PARENT,
        now=NOW,
        safety_factor="1.5",
        fixed_reserve_bytes=reserve,
    )


def test_exact_projection_excludes_checkpoint_components_and_counts_files(
    tmp_path: Path,
) -> None:
    # Each candidate trial retains the 10-byte model and budgets 4,161 bytes
    # for its receipt/sidecar; each algorithm also budgets a 1 MiB task summary.
    # Five algorithms * (3 * 4,171 + 1,048,576) = 5,305,445 bytes.
    # Bytes: ceil(5,305,445 * 1.5) + 256 = 7,958,424.
    # Inodes: five * (3 trials * 3 files + 1 summary) * 1.5 = 75.
    decision = _assess(
        tmp_path,
        quota=_quota_snapshot(byte_headroom=7_958_424, inode_headroom=75),
    )

    assert decision["candidate_capacity_eligible"] is True
    assert decision["reason_codes"] == []
    assert decision["projection"] == {
        "checkpoint_policy": "suppress_writes_with_bounded_candidate_evidence",
        "candidate_overhead_policy": {
            "retention_receipt_budget_bytes_per_trial": 4096,
            "retention_sidecar_bytes_per_trial": 65,
            "retention_files_per_trial": 2,
            "smoke_only_result_name": "screening_smoke_result.json",
            "task_summary_budget_bytes_per_algorithm": 1_048_576,
            "task_summary_files_per_algorithm": 1,
        },
        "fixed_reserve_bytes": 256,
        "full_screening_trials_per_algorithm": 3,
        "per_algorithm": [
            {
                "algorithm": algorithm,
                "checkpoint_bytes_excluded_per_trial": 1_000,
                "checkpoint_files_excluded_per_trial": 1,
                "smoke_retained_bytes_per_trial": 15,
                "smoke_retained_files_per_trial": 2,
                "smoke_only_result_bytes_per_trial": 5,
                "smoke_only_result_files_per_trial": 1,
                "candidate_receipt_budget_bytes_per_trial": 4_161,
                "candidate_receipt_files_per_trial": 2,
                "retained_bytes_per_trial": 4_171,
                "retained_files_per_trial": 3,
                "task_summary_budget_bytes": 1_048_576,
                "task_summary_files": 1,
                "projected_retained_bytes": 1_061_089,
                "projected_retained_files": 10,
            }
            for algorithm in EXPECTED_ALGORITHMS
        ],
        "projected_checkpoint_bytes_avoided": 15_000,
        "projected_retained_bytes": 5_305_445,
        "projected_retained_inodes": 50,
        "required_bytes": 7_958_424,
        "required_inodes": 75,
        "safety_adjusted_retained_bytes": 7_958_168,
        "safety_adjusted_retained_inodes": 75,
        "safety_factor": 1.5,
        "smoke_projected_original_bytes": 15_225,
    }
    assert all(decision["capacity_checks"].values())


def test_one_byte_below_soft_byte_threshold_fails(tmp_path: Path) -> None:
    quota = _quota_snapshot(byte_headroom=7_958_424, inode_headroom=75)
    quota["group"]["bytes"]["soft_limit"] -= 1
    decision = _assess(tmp_path, quota=quota)

    assert decision["candidate_capacity_eligible"] is False
    assert decision["capacity_checks"]["group_bytes_fit_soft_headroom"] is False
    assert "insufficient_group_bytes_fit_soft" in decision["reason_codes"]


def test_runtime_is_recomputed_instead_of_trusting_eligibility_booleans(
    tmp_path: Path,
) -> None:
    smoke = _smoke_summary()
    smoke["runs"][0]["runtime_projection"]["projected_task_hours"] = 72.000001
    smoke["runs"][0]["runtime_projection"]["within_limit"] = True
    decision = _assess(tmp_path, smoke=smoke)

    assert decision["candidate_capacity_eligible"] is False
    assert decision["runtime"]["eligible_recomputed"] is False
    assert "runtime_projection_exceeds_limit" in decision["reason_codes"]
    assert "runtime_projection_claim_mismatch" in decision["reason_codes"]


def test_tampered_smoke_with_stale_sidecar_fails_closed(tmp_path: Path) -> None:
    smoke_path = tmp_path / "smoke.json"
    quota_path = tmp_path / "quota.json"
    _write_signed(smoke_path, _smoke_summary())
    _write_signed(quota_path, _quota_snapshot())
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["runs"][0]["artifacts"][0]["bytes"] = 1
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")

    decision = assess_screening_storage(
        smoke_path,
        quota_path,
        expected_group=GROUP,
        expected_gid=GID,
        expected_project_id=PROJECT_ID,
        expected_filesystem=FILESYSTEM,
        expected_output_parent=OUTPUT_PARENT,
        now=NOW,
        fixed_reserve_bytes=256,
    )

    assert decision["candidate_capacity_eligible"] is False
    assert "smoke_summary_sidecar_mismatch" in decision["reason_codes"]


def test_stale_quota_snapshot_fails_closed(tmp_path: Path) -> None:
    quota = _quota_snapshot()
    old = NOW - timedelta(seconds=901)
    quota["captured_at"] = old.isoformat()
    quota["write_probe"]["completed_at"] = old.isoformat()
    decision = _assess(tmp_path, quota=quota)

    assert decision["candidate_capacity_eligible"] is False
    assert "quota_snapshot_stale" in decision["reason_codes"]
    assert "quota_write_probe_stale" in decision["reason_codes"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("name", "wrong-group", "quota_group_name_mismatch"),
        ("gid", 9999, "quota_group_gid_mismatch"),
    ],
)
def test_group_identity_mismatch_fails_closed(
    tmp_path: Path,
    field: str,
    value: str | int,
    reason: str,
) -> None:
    quota = _quota_snapshot()
    quota["group"][field] = value
    decision = _assess(tmp_path, quota=quota)

    assert decision["candidate_capacity_eligible"] is False
    assert reason in decision["reason_codes"]


def test_filesystem_and_fsync_probe_are_bound_and_required(tmp_path: Path) -> None:
    quota = _quota_snapshot()
    quota["write_probe"]["filesystem"] = "/wrong"
    quota["write_probe"]["fsync_succeeded"] = False
    decision = _assess(tmp_path, quota=quota)

    assert decision["candidate_capacity_eligible"] is False
    assert "quota_write_probe_filesystem_mismatch" in decision["reason_codes"]
    assert "quota_write_probe_fsync_failed" in decision["reason_codes"]


def test_write_probe_gid_must_match_expected_output_group(tmp_path: Path) -> None:
    quota = _quota_snapshot()
    quota["write_probe"]["gid"] = GID + 1
    decision = _assess(tmp_path, quota=quota)

    assert decision["candidate_capacity_eligible"] is False
    assert "quota_write_probe_gid_mismatch" in decision["reason_codes"]


def test_write_probe_must_be_direct_child_of_exact_output_parent(tmp_path: Path) -> None:
    quota = _quota_snapshot()
    quota["write_probe"]["path"] = f"{OUTPUT_PARENT}/nested/.capacity-probe"
    decision = _assess(tmp_path, quota=quota)

    assert decision["candidate_capacity_eligible"] is False
    assert "quota_write_probe_path_invalid" in decision["reason_codes"]


def test_write_probe_must_prove_cleanup(tmp_path: Path) -> None:
    quota = _quota_snapshot()
    quota["write_probe"]["removed"] = False
    decision = _assess(tmp_path, quota=quota)

    assert decision["candidate_capacity_eligible"] is False
    assert "quota_write_probe_not_removed" in decision["reason_codes"]


def test_quota_filesystem_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    quota = _quota_snapshot()
    quota["filesystem"] = "/lustrefs/project/wrong"
    decision = _assess(tmp_path, quota=quota)

    assert decision["candidate_capacity_eligible"] is False
    assert "quota_filesystem_mismatch" in decision["reason_codes"]


@pytest.mark.parametrize(
    ("owner", "dimension"),
    [
        ("group", "bytes"),
        ("group", "inodes"),
        ("project", "bytes"),
        ("project", "inodes"),
    ],
)
def test_expired_byte_or_inode_grace_fails_closed(
    tmp_path: Path,
    owner: str,
    dimension: str,
) -> None:
    quota = _quota_snapshot()
    quota[owner][dimension]["grace_expired"] = True
    decision = _assess(tmp_path, quota=quota)

    assert decision["candidate_capacity_eligible"] is False
    assert f"quota_{owner}_{dimension}_grace_expired" in decision["reason_codes"]


@pytest.mark.parametrize(
    ("dimension", "limit", "reason"),
    [
        ("bytes", "soft_limit", "insufficient_group_bytes_fit_soft"),
        ("bytes", "hard_limit", "insufficient_group_bytes_fit_hard"),
        ("inodes", "soft_limit", "insufficient_group_inodes_fit_soft"),
        ("inodes", "hard_limit", "insufficient_group_inodes_fit_hard"),
    ],
)
def test_each_soft_and_hard_byte_inode_headroom_is_enforced(
    tmp_path: Path,
    dimension: str,
    limit: str,
    reason: str,
) -> None:
    quota = _quota_snapshot(byte_headroom=7_958_424, inode_headroom=75)
    used = quota["group"][dimension]["used"]
    quota["group"][dimension][limit] = used
    # Keep the limits structurally ordered while isolating each failed check.
    if limit == "hard_limit":
        quota["group"][dimension]["soft_limit"] = used
    decision = _assess(tmp_path, quota=quota)

    assert decision["candidate_capacity_eligible"] is False
    assert reason in decision["reason_codes"]


@pytest.mark.parametrize("dimension", ["bytes", "inodes"])
def test_project_quota_shortage_blocks_admission(
    tmp_path: Path,
    dimension: str,
) -> None:
    quota = _quota_snapshot(byte_headroom=7_958_424, inode_headroom=75)
    used = quota["project"][dimension]["used"]
    quota["project"][dimension]["soft_limit"] = used
    decision = _assess(tmp_path, quota=quota)

    assert decision["candidate_capacity_eligible"] is False
    assert decision["capacity_checks"][f"project_{dimension}_fit_soft_headroom"] is False
    assert f"insufficient_project_{dimension}_fit_soft" in decision["reason_codes"]


def test_project_id_must_match_exactly(tmp_path: Path) -> None:
    quota = _quota_snapshot()
    quota["project"]["id"] = PROJECT_ID + 1
    decision = _assess(tmp_path, quota=quota)

    assert decision["candidate_capacity_eligible"] is False
    assert "quota_project_id_mismatch" in decision["reason_codes"]


@pytest.mark.parametrize("sidecar_text", ["digest_without_newline", "uppercase"])
def test_sidecar_format_must_be_exact(tmp_path: Path, sidecar_text: str) -> None:
    smoke_path = tmp_path / "smoke.json"
    quota_path = tmp_path / "quota.json"
    _write_signed(smoke_path, _smoke_summary())
    _write_signed(quota_path, _quota_snapshot())
    digest = hashlib.sha256(smoke_path.read_bytes()).hexdigest()
    value = digest if sidecar_text == "digest_without_newline" else digest.upper() + "\n"
    Path(str(smoke_path) + ".sha256").write_text(value, encoding="ascii")

    decision = assess_screening_storage(
        smoke_path,
        quota_path,
        expected_group=GROUP,
        expected_gid=GID,
        expected_project_id=PROJECT_ID,
        expected_filesystem=FILESYSTEM,
        expected_output_parent=OUTPUT_PARENT,
        now=NOW,
        fixed_reserve_bytes=256,
    )

    assert decision["candidate_capacity_eligible"] is False
    assert "smoke_summary_sidecar_invalid" in decision["reason_codes"]


def test_symlink_input_evidence_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target-smoke.json"
    smoke_path = tmp_path / "smoke.json"
    quota_path = tmp_path / "quota.json"
    _write_signed(target, _smoke_summary())
    smoke_path.symlink_to(target)
    Path(str(smoke_path) + ".sha256").write_text(
        hashlib.sha256(target.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    _write_signed(quota_path, _quota_snapshot())

    decision = assess_screening_storage(
        smoke_path,
        quota_path,
        expected_group=GROUP,
        expected_gid=GID,
        expected_project_id=PROJECT_ID,
        expected_filesystem=FILESYSTEM,
        expected_output_parent=OUTPUT_PARENT,
        now=NOW,
        fixed_reserve_bytes=256,
    )

    assert decision["candidate_capacity_eligible"] is False
    assert "smoke_summary_symlink_rejected" in decision["reason_codes"]


def test_symlink_sidecar_is_rejected(tmp_path: Path) -> None:
    smoke_path = tmp_path / "smoke.json"
    quota_path = tmp_path / "quota.json"
    _write_signed(smoke_path, _smoke_summary())
    _write_signed(quota_path, _quota_snapshot())
    sidecar = Path(str(smoke_path) + ".sha256")
    sidecar_target = tmp_path / "sidecar-target"
    sidecar_target.write_bytes(sidecar.read_bytes())
    sidecar.unlink()
    sidecar.symlink_to(sidecar_target)

    decision = assess_screening_storage(
        smoke_path,
        quota_path,
        expected_group=GROUP,
        expected_gid=GID,
        expected_project_id=PROJECT_ID,
        expected_filesystem=FILESYSTEM,
        expected_output_parent=OUTPUT_PARENT,
        now=NOW,
        fixed_reserve_bytes=256,
    )

    assert decision["candidate_capacity_eligible"] is False
    assert "smoke_summary_sidecar_symlink_rejected" in decision["reason_codes"]


def test_decision_writer_is_deterministic_and_sidecar_matches(tmp_path: Path) -> None:
    decision = _assess(tmp_path / "inputs")
    output = tmp_path / "decision.json"
    first = write_decision(output, decision)
    first_bytes = output.read_bytes()
    second = write_decision(output, copy.deepcopy(decision))

    assert output.read_bytes() == first_bytes
    assert first["sha256"] == second["sha256"]
    assert Path(second["sidecar_path"]).read_text(encoding="utf-8").strip() == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
