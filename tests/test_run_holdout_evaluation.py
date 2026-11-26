"""The run-once gate for the sealed holdout: what it refuses, how it continues the development files,
and the plan it would run. Every holdout here is synthetic, sealed by the same function as the real one."""
from __future__ import annotations

import json
import sys

import pandas as pd
import pytest
import yaml

from scripts import run_holdout_evaluation as gate
from scripts.download_extended_history import _write_holdout
from src.data.sealed_holdout import SealError

CLEAN = {"head": "abc123", "dirty": [], "lock_tracked": True}


def _prices(dates, tickers, start=10.0):
    rows = [{"date": d, "ticker": t, "open": start + i, "high": start + i, "low": start + i, "close": start + i,
             "volume": 100} for i, d in enumerate(dates) for t in tickers]
    return pd.DataFrame(rows)


def _setup(tmp_path, *, pit=True, **changes):
    root = tmp_path
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    dev_dates = pd.bdate_range("2024-12-20", "2024-12-31")
    hold_dates = pd.bdate_range("2025-01-02", "2025-01-10")
    _prices(dev_dates, ["AAA.BK", "BBB.BK"]).to_csv(root / "data" / "prices.csv", index=False)
    index_dev = _prices(dev_dates, ["TDEX.BK"], 5.0).assign(index_name="set50_etf")
    index_dev.to_csv(root / "data" / "index.csv", index=False)
    _write_holdout({"stocks": _prices(hold_dates, ["AAA.BK", "BBB.BK"], 25.0),
                    "index": _prices(hold_dates, ["TDEX.BK"], 15.0).assign(index_name="set50_etf")},
                   root / "data" / "sealed", {"source": "synthetic"})
    data = {"prices": ["data/prices.csv"], "index": "data/index.csv", "index_name": "set50_etf"}
    if pit:
        data |= {"membership": "data/membership.csv", "price_map": "data/price_map.csv"}
        pd.DataFrame({"record_id": ["l2", "l2"], "period_label": ["2024 H2"] * 2, "effective_from": ["2024-07-01"] * 2,
                      "effective_to_exclusive": ["2025-01-01"] * 2, "published_at": ["2024-06-17"] * 2,
                      "revision_status": ["original"] * 2, "ticker": ["AAA.BK", "BBB.BK"],
                      "in_set50": ["True", "False"]}).to_csv(root / "data" / "membership.csv", index=False)
        pd.DataFrame({"record_id": ["l2", "l2"], "effective_from": ["2024-07-01"] * 2,
                      "effective_to_exclusive": ["2025-01-01"] * 2, "ticker": ["AAA.BK", "BBB.BK"],
                      "in_set50": ["True", "False"], "price_ticker": ["AAA.BK", "BBB.BK"],
                      "coverage": ["own_symbol"] * 2}).to_csv(root / "data" / "price_map.csv", index=False)
    cfg = root / "config" / "design.yaml"
    cfg.write_text(yaml.safe_dump({"data": data, "splits": {"validation_end": "2022-12-30"}}))
    (root / "results").mkdir()
    (root / "results" / gate.LEDGER).write_text(json.dumps({"stage": "test_2023_2024"}) + "\n")
    lock = {
        "stage": "holdout_2025_2026",
        "status": "APPROVED",
        "approved_by": ["Author One"],
        "approved_on": "2026-11-26",
        "after_stage": "test_2023_2024",
        "design": {"candidate": "x", "runner": "scripts.run_pit_walk_forward", "config": "config/design.yaml",
                   "config_sha256": gate.sha256(cfg), "variant": "pit_x", "timesteps": 1000, "seeds": [42, 43],
                   "ensemble": {"band": 0.005, "modes": ["ensemble_band"]}},
        "holdout": {"end": "2025-01-10", "folds": [2025],
                    "extend": [{"dev": "data/prices.csv", "sealed": "data/sealed", "part": "stocks"},
                               {"dev": "data/index.csv", "sealed": "data/sealed", "part": "index"}],
                    **({"membership": "hold_last_list"} if pit else {}),
                    "stop_if_stale_share_above": None},
        "references": ["pit_equal_weight_monthly"],
        "n_trials": 22,
        "output_dir": "results/holdout",
    }
    for key, value in changes.items():
        section, _, name = key.partition("__")
        if name:
            lock[section][name] = value
        else:
            lock[key] = value
    return root, lock


def test_an_approved_lock_passes(tmp_path):
    root, lock = _setup(tmp_path)
    assert gate.check_lock(lock, root, CLEAN) == []


@pytest.mark.parametrize("changes, message", [
    ({"status": "DRAFT"}, "not 'APPROVED'"),
    ({"stage": "test_2025"}, "not a holdout stage"),
    ({"after_stage": "test_never_run"}, "is not in the ledger yet"),
    ({"after_stage": None}, "after_stage"),
    ({"holdout__end": "2024-12-31"}, "on or after 2025-01-01"),
    ({"holdout__folds": [2026]}, "folds must be years"),
    ({"holdout__membership": None}, "holdout.membership must be one of"),
    ({"holdout__stop_if_stale_share_above": 5}, "stop_if_stale_share_above"),
    ({"holdout__extend": [{"dev": "data/prices.csv", "sealed": "data/sealed", "part": "stocks"}]}, "does not continue"),
    ({"holdout__extend": [{"dev": "data/prices.csv", "sealed": "data/sealed", "part": "stocks"},
                          {"dev": "data/index.csv", "sealed": "data/sealed", "part": "macro"}]}, "no sealed part"),
    ({"design__config_sha256": "0" * 64}, "changed since the lock"),
])
def test_refusals(tmp_path, changes, message):
    root, lock = _setup(tmp_path, **changes)
    assert any(message in p for p in gate.check_lock(lock, root, CLEAN))


def test_a_broken_seal_or_used_stage_is_refused(tmp_path):
    root, lock = _setup(tmp_path)
    with open(root / "results" / gate.LEDGER, "a") as ledger:
        ledger.write(json.dumps({"stage": "holdout_2025_2026"}) + "\n")
    (root / "data" / "sealed" / "index.csv.gz").write_bytes(b"changed")
    problems = gate.check_lock(lock, root, CLEAN)
    assert any("changed since it was sealed" in p for p in problems)
    assert any("already has a 'holdout_2025_2026' run" in p for p in problems)
    assert any("uncommitted" in p for p in gate.check_lock(lock, root, {"head": "a", "dirty": ["M x"], "lock_tracked": True}))


def test_hold_last_list_keeps_the_publication_date(tmp_path):
    root, _ = _setup(tmp_path)
    membership = pd.read_csv(root / "data" / "membership.csv", dtype=str)
    price_map = pd.read_csv(root / "data" / "price_map.csv", dtype=str)
    held, held_map = gate.hold_last_list(membership, price_map, pd.Timestamp("2026-11-04"))
    new = held[held["record_id"] == "l2_held"]
    assert len(new) == 2 and set(new["effective_from"]) == {"2025-01-01"}
    assert set(new["effective_to_exclusive"]) == {"2026-11-05"} and set(new["published_at"]) == {"2024-06-17"}
    assert set(held_map.loc[held_map["record_id"] == "l2_held", "price_ticker"]) == {"AAA.BK", "BBB.BK"}
    later = membership.assign(effective_to_exclusive="2025-07-01")
    with pytest.raises(ValueError, match="as_recorded"):
        gate.hold_last_list(later, price_map, pd.Timestamp("2026-11-04"))


def test_append_holdout_refuses_overlap_and_trims_to_the_end():
    dev = _prices(pd.bdate_range("2024-12-27", "2024-12-31"), ["A"])
    sealed = _prices(pd.bdate_range("2025-01-02", "2025-01-10"), ["A"]).assign(extra=1)
    merged = gate.append_holdout(dev, sealed, pd.Timestamp("2025-01-06"))
    assert list(merged.columns) == list(dev.columns) and merged["date"].max() == pd.Timestamp("2025-01-06")
    with pytest.raises(ValueError, match="overlap"):
        gate.append_holdout(dev, _prices(pd.bdate_range("2024-12-31", "2025-01-03"), ["A"]), pd.Timestamp("2025-01-06"))
    with pytest.raises(ValueError, match="lack the columns"):
        gate.append_holdout(dev, sealed.drop(columns="volume"), pd.Timestamp("2025-01-06"))


def test_open_and_merge_writes_continued_files_and_counts_stale_rows(tmp_path):
    root, lock = _setup(tmp_path)
    receipt = gate.open_and_merge(lock, root)
    merged = pd.read_csv(root / gate.merged_path("results/holdout", "data/prices.csv"), parse_dates=["date"])
    assert merged["date"].min() == pd.Timestamp("2024-12-20") and merged["date"].max() == pd.Timestamp("2025-01-10")
    assert receipt["files"]["data/prices.csv"]["holdout_rows"] == 14
    assert receipt["files"]["data/prices.csv"]["series_with_step_above_20pct"] == ["AAA.BK", "BBB.BK"]
    assert receipt["stale_rows"]["data/index.csv"]["stale_rows"] == 0
    held = pd.read_csv(root / gate.merged_path("results/holdout", "data/membership.csv"), dtype=str)
    assert "l2_held" in set(held["record_id"])
    assert (root / "data" / "prices.csv").read_text().count("\n") == 17  # the development file is unchanged


def test_open_and_merge_refuses_a_draft_lock(tmp_path):
    root, lock = _setup(tmp_path, status="DRAFT")
    with pytest.raises(SealError, match="may not be opened"):
        gate.open_and_merge(lock, root)


def test_holdout_config_points_at_the_merged_files(tmp_path):
    root, lock = _setup(tmp_path)
    spec = gate.derive_holdout_config(lock, root)
    assert spec["data"]["prices"] == ["results/holdout/holdout_data/data/prices.csv"]
    assert spec["data"]["index"] == "results/holdout/holdout_data/data/index.csv"
    assert spec["data"]["membership"] == "results/holdout/holdout_data/data/membership.csv"
    assert spec["data"]["end"] == "2025-01-10" and spec["splits"]["validation_end"] == "2025-01-10"
    root5, lock5 = _setup(tmp_path / "five", pit=False)
    spec5 = gate.derive_holdout_config(lock5, root5)
    assert "end" not in spec5["data"] and "membership" not in spec5["data"]
    assert gate.check_lock(lock5, root5, CLEAN) == []


def test_merged_paths_stay_inside_the_output_folder():
    assert gate.merged_path("results/h", "data/x.csv") == "results/h/holdout_data/data/x.csv"
    assert gate.merged_path("results/h", "/abs/data/x.csv") == "results/h/holdout_data/abs/data/x.csv"


def test_plan_tasks_and_commands(tmp_path):
    _, lock = _setup(tmp_path, holdout__folds=[2025], holdout__end="2026-11-04")
    lock["holdout"]["folds"] = [2025, 2026]
    steps = gate.plan(lock, "python")
    assert len(steps["tasks"]) == 4
    assert steps["tasks"][0] == ("results/holdout/runs scripts.run_pit_walk_forward results/holdout/holdout_config.yaml "
                                 "1000 2025 pit_x 42")
    ensemble, aggregate = steps["after"]
    assert ensemble[2] == "scripts.evaluate_pit_seed_ensemble" and "2026" in ensemble
    assert "--references" in aggregate and "results/holdout/ensemble" in aggregate


def test_main_refuses_a_draft_and_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    root, lock = _setup(tmp_path, status="DRAFT")
    lock_path = root / "config" / "holdout.yaml"
    lock_path.write_text(yaml.safe_dump(lock))
    monkeypatch.setattr(gate, "git_state", lambda *a: CLEAN)
    monkeypatch.setattr(sys, "argv", ["x", "--lock", str(lock_path), "--root", str(root), "--dry-run"])
    gate.main()
    assert "not 'APPROVED'" in capsys.readouterr().out
    assert not (root / "results" / "holdout").exists()
    monkeypatch.setattr(sys, "argv", ["x", "--lock", str(lock_path), "--root", str(root)])
    with pytest.raises(SystemExit, match="may not run"):
        gate.main()
    assert not (root / "results" / "holdout").exists()
    assert (root / "results" / gate.LEDGER).read_text().count("\n") == 1
