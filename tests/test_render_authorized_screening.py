from __future__ import annotations

import hashlib
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import scripts.render_authorized_screening as render_module

from scripts.prepare_screening_execution_window import (
    CONTROLLER_RENDER_COMMAND,
    EXPECTED_CAMPAIGN_COUNTS,
    LOCKED_RELEASE,
)
from scripts.render_authorized_screening import (
    _default_renderer,
    render_authorized_screening,
)


NOW = datetime(2026, 7, 26, 15, 0, tzinfo=timezone.utc)
STAMP = "20260726T150000Z"


def _write_json_sidecar(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    Path(str(path) + ".sha256").write_text(
        hashlib.sha256(payload.encode("utf-8")).hexdigest() + "\n",
        encoding="ascii",
    )


def _receipt(
    project_root: Path,
    *,
    expires_at: datetime | None = None,
    stamp: str = STAMP,
) -> dict:
    evidence = (
        project_root
        / "campaigns/large_scale_thai_drl_v2/operational/screening_capacity"
    )
    quota = evidence / f"quota_snapshot_{stamp}.json"
    scheduler = evidence / f"scheduler_snapshot_{stamp}.txt"
    preflight = evidence / f"live_preflight_{stamp}.json"
    evidence.mkdir(parents=True, exist_ok=True)
    for path in (quota, scheduler, preflight):
        path.write_text("fixture\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "gate": "screening_execution_window",
        "authorized": True,
        "decision": "authorize_render_only",
        "assessed_at": NOW.isoformat().replace("+00:00", "Z"),
        "finalized_at": NOW.isoformat().replace("+00:00", "Z"),
        "expires_at": (expires_at or NOW + timedelta(minutes=10))
        .isoformat()
        .replace("+00:00", "Z"),
        "remaining_seconds": 600,
        "reason_codes": [],
        "validation_errors": [],
        "policy": {
            "jobs_submitted_by_gate": 0,
            "max_age_seconds": 900,
            "max_clock_skew_seconds": 60,
            "minimum_remaining_seconds": 120,
            "minimum_group_soft_margin_bytes": 1_073_741_824,
            "required_empty_user_queue": True,
            "maximum_array_elements": 5,
            "maximum_array_concurrency": 2,
        },
        "campaign": {
            "counts": EXPECTED_CAMPAIGN_COUNTS,
            "screening": {"state": "READY"},
        },
        "release": {"release_id": LOCKED_RELEASE.release_id},
        "storage": {
            "required_bytes": 6_000_000_000,
            "group_soft_margin_bytes": 2_000_000_000,
            "assessment": {
                "inputs": {
                    "quota_snapshot": {
                        "path": str(quota),
                        "sha256": "1" * 64,
                    }
                }
            },
        },
        "scheduler": {
            "evidence": {"path": str(scheduler), "sha256": "2" * 64}
        },
        "live_remote_preflight": {
            "evidence": {"path": str(preflight), "sha256": "3" * 64}
        },
        "render": {
            "performed": False,
            "next_command": CONTROLLER_RENDER_COMMAND,
        },
        "submission": {
            "performed": False,
            "sbatch_calls": 0,
            "array_shape": "0-4%2",
            "scheduler_jobs_if_submitted": 5,
        },
    }


def _receipt_path(project_root: Path, *, stamp: str = STAMP) -> Path:
    return (
        project_root
        / "reports/screening_execution_windows"
        / f"screening_execution_window_{stamp}.json"
    )


def _renderer(_root: Path) -> dict:
    return {
        "stage": "screening",
        "command": "fixture",
        "command_sha256": "4" * 64,
        "script": "campaigns/large_scale_thai_drl_v2/commands/screening.sh",
        "script_sha256": "5" * 64,
        "attempt_path": "campaigns/large_scale_thai_drl_v2/attempts/screening.json",
        "attempt_file_sha256": "6" * 64,
        "attempt_id": "screening_20260726T150000Z_deadbeef",
        "attempt_sha256": "7" * 64,
    }


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gate_result: dict,
    renderer=_renderer,
    times: list[datetime] | None = None,
) -> None:
    monkeypatch.setattr(render_module, "_run_live_gate", lambda *_args: gate_result)
    monkeypatch.setattr(render_module, "_default_renderer", renderer)
    if times is None:
        monkeypatch.setattr(render_module, "_utc_now", lambda: NOW)
    else:
        values = iter(times)
        monkeypatch.setattr(render_module, "_utc_now", lambda: next(values))


def test_valid_receipt_is_revalidated_and_consumed_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(tmp_path)
    path = _receipt_path(tmp_path)
    _write_json_sidecar(path, receipt)
    _install_runtime(monkeypatch, gate_result=receipt)

    result = render_authorized_screening(
        project_root=tmp_path,
        receipt_path=path,
    )

    assert result["decision"] == "rendered_once"
    assert result["jobs_submitted"] == 0
    assert Path(result["claim"]["path"]).is_file()
    assert Path(result["consumption"]["path"]).is_file()

    with pytest.raises(FileExistsError, match="already consumed"):
        render_authorized_screening(
            project_root=tmp_path,
            receipt_path=path,
        )

    copied = path.with_name("screening_execution_window_20260726T150001Z.json")
    copied.write_bytes(path.read_bytes())
    Path(str(copied) + ".sha256").write_bytes(
        Path(str(path) + ".sha256").read_bytes()
    )
    with pytest.raises(ValueError, match="stamp must match"):
        render_authorized_screening(
            project_root=tmp_path,
            receipt_path=copied,
        )


def test_expired_receipt_is_rejected_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(tmp_path, expires_at=NOW + timedelta(seconds=119))
    path = _receipt_path(tmp_path)
    _write_json_sidecar(path, receipt)
    _install_runtime(monkeypatch, gate_result=receipt)

    with pytest.raises(ValueError, match="too little time"):
        render_authorized_screening(
            project_root=tmp_path,
            receipt_path=path,
        )

    assert list(path.parent.glob("screening_execution_identity_*.claim.json")) == []


def test_fresh_gate_hold_prevents_claim_and_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(tmp_path)
    path = _receipt_path(tmp_path)
    _write_json_sidecar(path, receipt)
    rendered = False

    def fake_renderer(_root: Path) -> dict:
        nonlocal rendered
        rendered = True
        return _renderer(_root)

    _install_runtime(
        monkeypatch,
        gate_result={
            "authorized": False,
            "reason_codes": ["scheduler_queue_not_empty"],
        },
        renderer=fake_renderer,
    )

    with pytest.raises(ValueError, match="live gate revalidation failed"):
        render_authorized_screening(
            project_root=tmp_path,
            receipt_path=path,
        )

    assert rendered is False
    assert list(path.parent.glob("screening_execution_identity_*.claim.json")) == []


def test_boolean_policy_value_cannot_be_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(tmp_path)
    receipt["policy"]["max_age_seconds"] = True
    path = _receipt_path(tmp_path)
    _write_json_sidecar(path, receipt)
    _install_runtime(monkeypatch, gate_result=receipt)

    with pytest.raises(ValueError, match="weaker than the production policy"):
        render_authorized_screening(
            project_root=tmp_path,
            receipt_path=path,
        )


def test_recomputed_evidence_mismatch_prevents_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(tmp_path)
    path = _receipt_path(tmp_path)
    _write_json_sidecar(path, receipt)
    recomputed = json.loads(json.dumps(receipt))
    recomputed["storage"]["group_soft_margin_bytes"] += 1
    _install_runtime(monkeypatch, gate_result=recomputed)

    with pytest.raises(ValueError, match="no longer matches"):
        render_authorized_screening(
            project_root=tmp_path,
            receipt_path=path,
        )


def test_public_consumer_has_no_clock_gate_or_renderer_override() -> None:
    parameters = inspect.signature(render_authorized_screening).parameters

    assert set(parameters) == {"project_root", "receipt_path"}


def test_stage_lock_blocks_a_second_distinct_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _receipt(tmp_path)
    first_path = _receipt_path(tmp_path)
    _write_json_sidecar(first_path, first)
    _install_runtime(monkeypatch, gate_result=first)
    render_authorized_screening(project_root=tmp_path, receipt_path=first_path)

    second_stamp = "20260726T150001Z"
    second = _receipt(tmp_path, stamp=second_stamp)
    second["storage"]["assessment"]["inputs"]["quota_snapshot"]["sha256"] = (
        "8" * 64
    )
    second_path = _receipt_path(tmp_path, stamp=second_stamp)
    _write_json_sidecar(second_path, second)
    _install_runtime(monkeypatch, gate_result=second)

    with pytest.raises(FileExistsError):
        render_authorized_screening(project_root=tmp_path, receipt_path=second_path)


def test_stage_lock_allows_only_one_concurrent_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _receipt(tmp_path, stamp="20260726T150000Z")
    second = _receipt(tmp_path, stamp="20260726T150001Z")
    second["storage"]["assessment"]["inputs"]["quota_snapshot"]["sha256"] = (
        "8" * 64
    )
    first_path = _receipt_path(tmp_path, stamp="20260726T150000Z")
    second_path = _receipt_path(tmp_path, stamp="20260726T150001Z")
    _write_json_sidecar(first_path, first)
    _write_json_sidecar(second_path, second)

    def live_gate(_root: Path, quota: Path, _scheduler: Path, _preflight: Path):
        return first if "150000Z" in quota.name else second

    monkeypatch.setattr(render_module, "_run_live_gate", live_gate)
    monkeypatch.setattr(render_module, "_default_renderer", _renderer)
    monkeypatch.setattr(render_module, "_utc_now", lambda: NOW)

    def run(path: Path):
        try:
            return render_authorized_screening(
                project_root=tmp_path,
                receipt_path=path,
            )
        except Exception as exc:  # result inspection is the purpose of this test
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, (first_path, second_path)))

    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, FileExistsError) for result in results) == 1


def test_ttl_is_rechecked_at_actual_pre_render_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(tmp_path)
    path = _receipt_path(tmp_path)
    _write_json_sidecar(path, receipt)
    rendered = False

    def fake_renderer(_root: Path) -> dict:
        nonlocal rendered
        rendered = True
        return _renderer(_root)

    _install_runtime(
        monkeypatch,
        gate_result=receipt,
        renderer=fake_renderer,
        times=[NOW, NOW + timedelta(minutes=9), NOW + timedelta(minutes=9)],
    )

    with pytest.raises(ValueError, match="too little time"):
        render_authorized_screening(project_root=tmp_path, receipt_path=path)

    assert rendered is False
    assert len(list(path.parent.glob("screening_execution_identity_*.failed.json"))) == 1


def test_renderer_failure_is_recorded_as_in_doubt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(tmp_path)
    path = _receipt_path(tmp_path)
    _write_json_sidecar(path, receipt)

    def failed_renderer(_root: Path) -> dict:
        raise RuntimeError("fixture render failure")

    _install_runtime(
        monkeypatch,
        gate_result=receipt,
        renderer=failed_renderer,
        times=[NOW, NOW, NOW],
    )

    with pytest.raises(RuntimeError, match="fixture render failure"):
        render_authorized_screening(project_root=tmp_path, receipt_path=path)

    failed = list(path.parent.glob("screening_execution_identity_*.failed.json"))
    assert len(failed) == 1
    assert json.loads(failed[0].read_text(encoding="utf-8"))["status"] == (
        "render_failed_or_in_doubt"
    )


def test_stricter_receipt_ttl_is_respected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(tmp_path, expires_at=NOW + timedelta(seconds=200))
    receipt["policy"]["minimum_remaining_seconds"] = 300
    path = _receipt_path(tmp_path)
    _write_json_sidecar(path, receipt)
    _install_runtime(monkeypatch, gate_result=receipt)

    with pytest.raises(ValueError, match="too little time"):
        render_authorized_screening(project_root=tmp_path, receipt_path=path)


def test_preexisting_screening_attempt_blocks_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(tmp_path)
    path = _receipt_path(tmp_path)
    _write_json_sidecar(path, receipt)
    attempt = (
        tmp_path
        / "campaigns/large_scale_thai_drl_v2/attempts"
        / "screening_20260726T140000Z_aaaaaaaa.json"
    )
    attempt.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text("{}\n", encoding="utf-8")
    _install_runtime(monkeypatch, gate_result=receipt)

    with pytest.raises(FileExistsError, match="reconcile"):
        render_authorized_screening(project_root=tmp_path, receipt_path=path)


def test_historical_screening_smoke_attempt_does_not_block_full_screening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(tmp_path)
    path = _receipt_path(tmp_path)
    _write_json_sidecar(path, receipt)
    smoke = (
        tmp_path
        / "campaigns/large_scale_thai_drl_v2/attempts"
        / "screening_smoke_20260717T144734Z_652d3bc9.json"
    )
    smoke.parent.mkdir(parents=True, exist_ok=True)
    smoke.write_text("{}\n", encoding="utf-8")
    _install_runtime(monkeypatch, gate_result=receipt)

    result = render_authorized_screening(project_root=tmp_path, receipt_path=path)

    assert result["decision"] == "rendered_once"


def test_default_renderer_binds_one_immutable_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.campaigns.control as control

    campaign_dir = tmp_path / "campaigns/large_scale_thai_drl_v2"
    definition = SimpleNamespace(campaign_dir=campaign_dir)
    attempt_id = "screening_20260726T150000Z_deadbeef"
    attempt = {
        "schema_version": 2,
        "status": "rendered",
        "campaign_id": "large_scale_thai_drl_v2",
        "stage_id": "screening",
        "attempt_id": attempt_id,
        "rendered_at_utc": "2026-07-26T15:00:00Z",
        "inputs": [],
        "predecessor_gates": [],
    }
    attempt["attempt_sha256"] = hashlib.sha256(
        json.dumps(attempt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    def fake_render(_definition, stage: str):
        assert stage == "screening"
        attempt_dir = campaign_dir / "attempts"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(attempt, indent=2, sort_keys=True) + "\n"
        (attempt_dir / f"{attempt_id}.json").write_text(payload, encoding="utf-8")
        (attempt_dir / "screening_latest.json").write_text(payload, encoding="utf-8")
        command = (
            "SCREENING_SUBMISSION=$(sbatch --parsable "
            f"--export=CAMPAIGN_ATTEMPT_ID={attempt_id},"
            f"CAMPAIGN_ATTEMPT_SHA256={attempt['attempt_sha256']} job.sbatch)"
        )
        script = campaign_dir / "commands/screening.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(command + "\n", encoding="utf-8")
        return "screening", command, script

    monkeypatch.setattr(
        control,
        "load_definition",
        lambda *_args, **_kwargs: (definition, {}),
    )
    monkeypatch.setattr(control, "_render_command", fake_render)

    result = _default_renderer(tmp_path)

    assert result["attempt_id"] == attempt_id
    assert result["attempt_sha256"] == attempt["attempt_sha256"]
    assert len(result["attempt_file_sha256"]) == 64
    assert len(result["command_sha256"]) == 64


@pytest.mark.parametrize("directory_name", ["attempts", "commands"])
def test_default_renderer_rejects_symlinked_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory_name: str,
) -> None:
    import src.campaigns.control as control

    campaign = tmp_path / "campaigns/large_scale_thai_drl_v2"
    outside = tmp_path / "outside" / directory_name
    campaign.mkdir(parents=True)
    outside.mkdir(parents=True)
    (campaign / directory_name).symlink_to(outside, target_is_directory=True)
    definition = SimpleNamespace(campaign_dir=campaign.resolve())
    rendered = False

    def fake_render(*_args):
        nonlocal rendered
        rendered = True
        raise AssertionError("unsafe renderer must not be called")

    monkeypatch.setattr(
        control,
        "load_definition",
        lambda *_args, **_kwargs: (definition, {}),
    )
    monkeypatch.setattr(control, "_render_command", fake_render)

    with pytest.raises(ValueError, match="must not be a symlink"):
        _default_renderer(tmp_path)

    assert rendered is False


def test_default_renderer_rejects_dangling_command_symlink_before_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.campaigns.control as control

    campaign = tmp_path / "campaigns/large_scale_thai_drl_v2"
    commands = campaign / "commands"
    commands.mkdir(parents=True)
    outside = tmp_path / "outside/screening.sh"
    (commands / "screening.sh").symlink_to(outside)
    definition = SimpleNamespace(campaign_dir=campaign.resolve())
    rendered = False

    def fake_render(*_args):
        nonlocal rendered
        rendered = True
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("unsafe\n", encoding="utf-8")
        raise AssertionError("unsafe renderer must not be called")

    monkeypatch.setattr(
        control,
        "load_definition",
        lambda *_args, **_kwargs: (definition, {}),
    )
    monkeypatch.setattr(control, "_render_command", fake_render)

    with pytest.raises(ValueError, match="must not be a symlink"):
        _default_renderer(tmp_path)

    assert rendered is False
    assert not outside.exists()


def test_consumer_rejects_dangling_command_symlink_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(tmp_path)
    path = _receipt_path(tmp_path)
    _write_json_sidecar(path, receipt)
    command = (
        tmp_path
        / "campaigns/large_scale_thai_drl_v2/commands/screening.sh"
    )
    command.parent.mkdir(parents=True)
    outside = tmp_path / "outside/screening.sh"
    command.symlink_to(outside)
    _install_runtime(monkeypatch, gate_result=receipt)

    with pytest.raises(FileExistsError, match="unsafe symlinked output path"):
        render_authorized_screening(project_root=tmp_path, receipt_path=path)

    receipt_dir = path.parent
    assert not list(receipt_dir.glob("*.claim.json"))
    assert not (receipt_dir / "screening_stage_render.lock.json").exists()
    assert not outside.exists()


@pytest.mark.parametrize(
    "relative_path",
    ["reports/campaign_status.md", "events.jsonl"],
)
def test_default_renderer_rejects_auxiliary_output_symlink_before_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    import src.campaigns.control as control

    campaign = tmp_path / "campaigns/large_scale_thai_drl_v2"
    unsafe = campaign / relative_path
    unsafe.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside" / Path(relative_path).name
    unsafe.symlink_to(outside)
    definition = SimpleNamespace(campaign_dir=campaign.resolve())
    rendered = False

    def fake_render(*_args):
        nonlocal rendered
        rendered = True
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("unsafe\n", encoding="utf-8")
        raise AssertionError("unsafe renderer must not be called")

    monkeypatch.setattr(
        control,
        "load_definition",
        lambda *_args, **_kwargs: (definition, {}),
    )
    monkeypatch.setattr(control, "_render_command", fake_render)

    with pytest.raises(ValueError, match="must not be a symlink"):
        _default_renderer(tmp_path)

    assert rendered is False
    assert not outside.exists()


@pytest.mark.parametrize(
    "relative_path",
    ["reports/campaign_status.md", "events.jsonl"],
)
def test_consumer_rejects_auxiliary_output_symlink_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    receipt = _receipt(tmp_path)
    path = _receipt_path(tmp_path)
    _write_json_sidecar(path, receipt)
    unsafe = (
        tmp_path
        / "campaigns/large_scale_thai_drl_v2"
        / relative_path
    )
    unsafe.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside" / Path(relative_path).name
    unsafe.symlink_to(outside)
    _install_runtime(monkeypatch, gate_result=receipt)

    with pytest.raises(FileExistsError, match="unsafe symlinked output path"):
        render_authorized_screening(project_root=tmp_path, receipt_path=path)

    receipt_dir = path.parent
    assert not list(receipt_dir.glob("*.claim.json"))
    assert not (receipt_dir / "screening_stage_render.lock.json").exists()
    assert not outside.exists()


def test_consumer_source_contains_no_process_or_submission_execution() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/render_authorized_screening.py"
    ).read_text(encoding="utf-8")

    assert "subprocess" not in source
    assert "os.system" not in source
    assert "jobs_submitted\": 0" in source
