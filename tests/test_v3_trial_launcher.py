from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import types

import pandas as pd
import pytest
import yaml

import src.v3.trial_launcher as launcher
from src.v3.trial_launcher import V3LaunchDenied


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_PROTOCOL = PROJECT_ROOT / "config/v3/development_protocol.yaml"
REAL_POLICY = PROJECT_ROOT / "config/v3/evaluation_policy_draft.yaml"
REAL_CANDIDATES = (
    PROJECT_ROOT / "reports/data_quality_v3/v3_candidate_ledger_20260820.csv"
)
REAL_SCHEDULE = (
    PROJECT_ROOT
    / "reports/data_quality_v3/v3_evaluation_fold_schedule_20260820.csv"
)
VALID_CANDIDATE = "ppo_stable_default_050k"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sidecar(path: Path) -> None:
    Path(f"{path}.sha256").write_text(
        f"{_sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def _descriptor(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _locked_metadata_capsule(tmp_path: Path) -> dict[str, Path]:
    policy_path = tmp_path / "config/v3/evaluation_policy_locked.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy = yaml.safe_load(REAL_POLICY.read_text(encoding="utf-8"))
    policy["registration"]["status"] = "locked"
    policy["registration"]["approved_by_authors"] = True
    policy["author_decisions_required_before_lock"] = []
    policy_path.write_text(
        yaml.safe_dump(policy, sort_keys=False),
        encoding="utf-8",
    )
    _write_sidecar(policy_path)

    candidate_path = (
        tmp_path / "reports/data_quality_v3/v3_candidate_ledger_locked.csv"
    )
    schedule_path = (
        tmp_path / "reports/data_quality_v3/v3_evaluation_fold_schedule_locked.csv"
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REAL_CANDIDATES, candidate_path)
    shutil.copy2(REAL_SCHEDULE, schedule_path)

    receipt_path = (
        tmp_path / "reports/data_quality_v3/v3_evaluation_policy_locked.json"
    )
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "evaluation_policy_eligible": True,
        "training_enabled": False,
        "test_access_enabled": False,
        "all_access_enabled": False,
        "outputs": {
            "candidate_ledger": _descriptor(tmp_path, candidate_path),
            "fold_schedule": _descriptor(tmp_path, schedule_path),
        },
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sidecar(receipt_path)

    protocol_path = tmp_path / "config/v3/development_protocol.yaml"
    protocol = {
        "schema_version": 1,
        "registration": {
            "status": "locked",
            "approved_by_authors": True,
        },
        "evaluation_policy": {
            "status": "locked",
            "path": policy_path.relative_to(tmp_path).as_posix(),
            "sha256": _sha256(policy_path),
            "receipt": {
                "status": "bound",
                "path": receipt_path.relative_to(tmp_path).as_posix(),
                "sha256": _sha256(receipt_path),
            },
        },
        "execution": {
            "training_enabled": True,
            "hpc_submission_enabled": False,
        },
    }
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False),
        encoding="utf-8",
    )
    _write_sidecar(protocol_path)
    return {
        "protocol": protocol_path,
        "policy": policy_path,
        "candidates": candidate_path,
        "schedule": schedule_path,
        "receipt": receipt_path,
    }


def _allow_gate(monkeypatch: pytest.MonkeyPatch, events: list[str] | None = None) -> None:
    def allow(*args, **kwargs):
        del args
        assert kwargs["split"] == "train"
        if events is not None:
            events.append("gate")
        return [], []

    monkeypatch.setattr(launcher, "validate_development_request", allow)


def _rebind_receipt_and_protocol(paths: dict[str, Path], root: Path) -> None:
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt["outputs"]["candidate_ledger"] = _descriptor(root, paths["candidates"])
    receipt["outputs"]["fold_schedule"] = _descriptor(root, paths["schedule"])
    paths["receipt"].write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sidecar(paths["receipt"])
    protocol = yaml.safe_load(paths["protocol"].read_text(encoding="utf-8"))
    protocol["evaluation_policy"]["receipt"]["sha256"] = _sha256(paths["receipt"])
    paths["protocol"].write_text(
        yaml.safe_dump(protocol, sort_keys=False),
        encoding="utf-8",
    )
    _write_sidecar(paths["protocol"])


def _launch_arguments(paths: dict[str, Path], root: Path) -> dict[str, object]:
    return {
        "protocol_path": paths["protocol"],
        "project_root": root,
        "candidate_id": VALID_CANDIDATE,
        "fold_id": "fold_01",
        "seed": 42,
    }


def test_current_draft_denies_before_metadata_or_backend_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("draft request reached work after the development gate")

    monkeypatch.setattr(launcher, "_bind_locked_cell", forbidden)
    monkeypatch.setattr(launcher, "import_module", forbidden)

    with pytest.raises(V3LaunchDenied) as caught:
        launcher.launch_v3_cell(
            protocol_path=REAL_PROTOCOL,
            project_root=PROJECT_ROOT,
            candidate_id=VALID_CANDIDATE,
            fold_id="fold_01",
            seed=42,
        )

    assert "V3 protocol is not locked" in caught.value.issues
    assert "V3 training is disabled" in caught.value.issues
    assert "licensed-market package receipt is not bound" in caught.value.issues


def test_launcher_surface_rejects_split_data_and_execution_overrides() -> None:
    parameters = inspect.signature(launcher.launch_v3_cell).parameters
    assert set(parameters) == {
        "protocol_path",
        "project_root",
        "candidate_id",
        "fold_id",
        "seed",
    }
    parser = launcher.build_parser()
    base = [
        "--candidate-id",
        VALID_CANDIDATE,
        "--fold-id",
        "fold_01",
        "--seed",
        "42",
    ]
    forbidden = (
        ["--split", "test"],
        ["--train-split", "all"],
        ["--eval-split", "test"],
        ["--config", "config/default.yaml"],
        ["--candidate-ledger", "candidate.csv"],
        ["--fold-schedule", "schedule.csv"],
        ["--output-dir", "output"],
        ["--backend", "src.train"],
        ["--hpc-submit"],
    )
    for override in forbidden:
        with pytest.raises(SystemExit):
            parser.parse_args([*base, *override])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--candidate-id",
                VALID_CANDIDATE,
                "--fold-id",
                "test",
                "--seed",
                "42",
            ]
        )


def test_bound_cell_is_exact_and_backend_import_occurs_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _locked_metadata_capsule(tmp_path)
    events: list[str] = []
    _allow_gate(monkeypatch, events)
    original_bind = launcher._bind_locked_cell

    def bind(**kwargs):
        events.append("bind")
        return original_bind(**kwargs)

    def run(authorization):
        events.append("run")
        return authorization.to_dict()

    def import_backend(name: str):
        events.append("import")
        assert name == launcher.BACKEND_MODULE
        return types.SimpleNamespace(run_authorized_cell=run)

    monkeypatch.setattr(launcher, "_bind_locked_cell", bind)
    monkeypatch.setattr(launcher, "import_module", import_backend)
    result = launcher.launch_v3_cell(**_launch_arguments(paths, tmp_path))

    assert events == ["gate", "bind", "import", "run"]
    assert result["candidate_id"] == VALID_CANDIDATE
    assert result["algorithm"] == "ppo"
    assert result["parameter_profile"] == "stable_default"
    assert result["timestep_budget"] == 50_000
    assert result["registry_status"] == "draft_not_executable"
    assert result["fold_id"] == "fold_01"
    assert result["seed"] == 42
    assert len(result["train_context_dates"]) == 30
    assert len(result["train_transitions"]) == 209
    assert len(result["validation_context_dates"]) == 30
    assert len(result["validation_transitions"]) == 118
    assert len(result["cell_sha256"]) == 64
    json.dumps(result)


@pytest.mark.parametrize(
    ("candidate_id", "fold_id", "seed", "message"),
    [
        ("unregistered", "fold_01", 42, "candidate_id is not registered"),
        (VALID_CANDIDATE, "fold_07", 42, "fold_id is not preregistered"),
        (VALID_CANDIDATE, "fold_01", True, "seed must be an integer"),
        (VALID_CANDIDATE, "fold_01", 45, "seed is not preregistered"),
    ],
)
def test_invalid_candidate_fold_or_seed_never_imports_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_id: str,
    fold_id: str,
    seed: object,
    message: str,
) -> None:
    paths = _locked_metadata_capsule(tmp_path)
    _allow_gate(monkeypatch)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("invalid cell imported the backend")

    monkeypatch.setattr(launcher, "import_module", forbidden)
    arguments = _launch_arguments(paths, tmp_path)
    arguments.update(candidate_id=candidate_id, fold_id=fold_id, seed=seed)
    with pytest.raises(V3LaunchDenied, match=message):
        launcher.launch_v3_cell(**arguments)


def test_candidate_byte_tampering_fails_at_point_of_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _locked_metadata_capsule(tmp_path)
    _allow_gate(monkeypatch)
    paths["candidates"].write_text(
        paths["candidates"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(V3LaunchDenied, match="point of use"):
        launcher.authorize_v3_cell(**_launch_arguments(paths, tmp_path))


def test_protocol_and_receipt_sidecars_are_rechecked_after_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _locked_metadata_capsule(tmp_path)
    _allow_gate(monkeypatch)
    paths["protocol"].write_text(
        paths["protocol"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(V3LaunchDenied, match="protocol SHA-256 sidecar"):
        launcher.authorize_v3_cell(**_launch_arguments(paths, tmp_path))

    paths = _locked_metadata_capsule(tmp_path / "receipt_case")
    _allow_gate(monkeypatch)
    paths["receipt"].write_text(
        paths["receipt"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    protocol = yaml.safe_load(paths["protocol"].read_text(encoding="utf-8"))
    protocol["evaluation_policy"]["receipt"]["sha256"] = _sha256(paths["receipt"])
    paths["protocol"].write_text(
        yaml.safe_dump(protocol, sort_keys=False),
        encoding="utf-8",
    )
    _write_sidecar(paths["protocol"])
    with pytest.raises(V3LaunchDenied, match="receipt SHA-256 sidecar"):
        launcher.authorize_v3_cell(**_launch_arguments(paths, tmp_path / "receipt_case"))


def test_fully_rebound_noncanonical_candidate_grid_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _locked_metadata_capsule(tmp_path)
    _allow_gate(monkeypatch)
    candidates = pd.read_csv(paths["candidates"], keep_default_na=False)
    candidates.loc[0, "timestep_budget"] = 300_000
    candidates.to_csv(paths["candidates"], index=False, lineterminator="\n")
    _rebind_receipt_and_protocol(paths, tmp_path)

    with pytest.raises(V3LaunchDenied, match="exact preregistered grid"):
        launcher.authorize_v3_cell(**_launch_arguments(paths, tmp_path))


def test_fully_rebound_cross_boundary_schedule_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _locked_metadata_capsule(tmp_path)
    _allow_gate(monkeypatch)
    schedule = pd.read_csv(paths["schedule"], keep_default_na=False)
    boundary_index = schedule.index[schedule["unused_boundary_transition_start"]].tolist()[0]
    schedule.loc[boundary_index, "transition_owner"] = "validation"
    schedule.loc[boundary_index, "transition_outcome_date"] = "2022-01-04"
    schedule.to_csv(paths["schedule"], index=False, lineterminator="\n")
    _rebind_receipt_and_protocol(paths, tmp_path)

    with pytest.raises(V3LaunchDenied, match="canonical policy materialization"):
        launcher.authorize_v3_cell(**_launch_arguments(paths, tmp_path))


def test_locked_metadata_still_blocks_when_reviewed_backend_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _locked_metadata_capsule(tmp_path)
    _allow_gate(monkeypatch)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    with pytest.raises(V3LaunchDenied, match="intentionally absent"):
        launcher.launch_v3_cell(**_launch_arguments(paths, tmp_path))

    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert after == before


def test_launcher_module_has_no_eager_trainer_or_data_import() -> None:
    source_path = Path(launcher.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert "src.train" not in imported
    assert not any(name.startswith("src.experiments") for name in imported)
    assert "src.v3.trading_env" not in imported
    assert "stable_baselines3" not in imported


def test_current_cli_reports_blocked_without_backend_or_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("current draft imported a backend")

    monkeypatch.setattr(launcher, "import_module", forbidden)
    exit_code = launcher.main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--protocol",
            str(REAL_PROTOCOL),
            "--candidate-id",
            VALID_CANDIDATE,
            "--fold-id",
            "fold_01",
            "--seed",
            "42",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["backend_imported"] is False
    assert payload["output_written"] is False
    assert "V3 protocol is not locked" in payload["issues"]
