"""The run-once gate for the test years: what it refuses, and the plan it would run (no data is read)."""
from __future__ import annotations

import json
import sys

import pytest
import yaml

from scripts import run_final_test_evaluation as gate

CLEAN = {"head": "abc123", "dirty": [], "lock_tracked": True}


def _setup(tmp_path, **changes):
    (tmp_path / "config").mkdir()
    cfg = tmp_path / "config" / "design.yaml"
    cfg.write_text(yaml.safe_dump({"splits": {"validation_end": "2022-12-30"}, "base": {"training": {"algo": "ppo"}}}))
    lock = {
        "stage": "test_2023_2024",
        "status": "APPROVED",
        "approved_by": ["Author One", "Author Two"],
        "approved_on": "2026-11-26",
        "design": {"candidate": "pit_x", "runner": "scripts.run_pit_walk_forward", "config": "config/design.yaml",
                   "config_sha256": gate.sha256(cfg), "variant": "pit_x", "timesteps": 100000, "seeds": [42, 43],
                   "ensemble": {"band": 0.005, "modes": ["ensemble_band"]}},
        "test_years": [2023, 2024],
        "test_end": "2024-12-31",
        "references": ["pit_equal_weight_monthly"],
        "n_trials": 22,
        "output_dir": "results/final_test",
    }
    for key, value in changes.items():
        if key.startswith("design_"):
            lock["design"][key[len("design_"):]] = value
        else:
            lock[key] = value
    return lock


def test_an_approved_lock_passes(tmp_path):
    assert gate.check_lock(_setup(tmp_path), tmp_path, CLEAN) == []


@pytest.mark.parametrize("changes, message", [
    ({"status": "DRAFT"}, "not 'APPROVED'"),
    ({"approved_by": []}, "lists no one"),
    ({"approved_on": None}, "approved_on is empty"),
    ({"design_config_sha256": "0" * 64}, "changed since the lock"),
    ({"design_seeds": []}, "design.seeds is missing"),
    ({"test_years": [2024, 2025]}, "subset of [2023, 2024]"),
    ({"test_end": "2025-03-31"}, "before 2025-01-01"),
    ({"test_end": "2024-06-30"}, "cuts the last test year short"),
    ({"n_trials": None}, "n_trials"),
    ({"test_end": None}, "before 2025-01-01"),
    ({"stage": None}, "stage is missing"),
])
def test_refusals(tmp_path, changes, message):
    problems = gate.check_lock(_setup(tmp_path, **changes), tmp_path, CLEAN)
    assert any(message in p for p in problems), problems


def test_uncommitted_code_is_refused(tmp_path):
    lock = _setup(tmp_path)
    problems = gate.check_lock(lock, tmp_path, {"head": "abc", "dirty": [" M src/envs/x.py"], "lock_tracked": True})
    assert any("uncommitted" in p for p in problems)
    problems = gate.check_lock(lock, tmp_path, {"head": "abc", "dirty": [], "lock_tracked": False})
    assert problems == ["the lock is not committed"]


def test_runs_once_output_folder_and_ledger(tmp_path):
    lock = _setup(tmp_path)
    (tmp_path / "results" / "final_test").mkdir(parents=True)
    assert any("runs once" in p for p in gate.check_lock(lock, tmp_path, CLEAN))
    (tmp_path / "results" / "final_test").rmdir()
    (tmp_path / "results" / gate.LEDGER).write_text(json.dumps({"stage": "test_2023_2024"}) + "\n")
    assert any("already has" in p for p in gate.check_lock(lock, tmp_path, CLEAN))


def test_config_copy_moves_only_the_validation_end(tmp_path):
    spec = gate.derive_test_config(_setup(tmp_path), tmp_path)
    assert spec["splits"] == {"validation_end": "2024-12-31"}
    assert spec["base"] == {"training": {"algo": "ppo"}}


def test_plan_tasks_and_commands(tmp_path):
    steps = gate.plan(_setup(tmp_path), tmp_path, "py")
    assert steps["tasks"] == [
        f"results/final_test/runs scripts.run_pit_walk_forward results/final_test/test_config.yaml 100000 {y} pit_x {s}"
        for y in (2023, 2024) for s in (42, 43)
    ]
    ensemble, aggregate = steps["after"]
    assert ensemble[:3] == ["py", "-m", "scripts.evaluate_pit_seed_ensemble"]
    assert ensemble[ensemble.index("--fold-years") + 1:ensemble.index("--band")] == ["2023", "2024"]
    assert aggregate[:3] == ["py", "-m", "scripts.aggregate_walk_forward_reward"]
    assert "--extra-run-dirs" in aggregate and aggregate[aggregate.index("--n-trials") + 1] == "22"


def test_main_refuses_a_draft_and_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    lock = _setup(tmp_path, status="DRAFT")
    path = tmp_path / "config" / "lock.yaml"
    path.write_text(yaml.safe_dump(lock))
    monkeypatch.setattr(gate, "git_state", lambda root, lock: CLEAN)
    monkeypatch.setattr(sys, "argv", ["x", "--lock", str(path), "--root", str(tmp_path), "--dry-run"])
    gate.main()
    printed = capsys.readouterr().out
    assert "not 'APPROVED'" in printed and "4 training tasks" in printed
    assert not (tmp_path / "results").exists()
    monkeypatch.setattr(sys, "argv", ["x", "--lock", str(path), "--root", str(tmp_path)])
    with pytest.raises(SystemExit, match="may not run"):
        gate.main()
    assert not (tmp_path / "results").exists()
