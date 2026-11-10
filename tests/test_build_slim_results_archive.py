from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from scripts.build_slim_results_archive import build, collect


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "repo"
    _touch(project / "config" / "experiment.yaml", "seed: 42\n")
    _touch(project / "src" / "module.py", "x = 1\n")
    _touch(project / "src" / "__pycache__" / "module.cpython-311.pyc")
    _touch(project / "README.md", "# readme\n")
    _touch(project / "data" / "reference" / "set50_set100_membership_2012_2024.csv", "date,ticker\n")
    _touch(project / "data" / "raw" / "extended" / "prices.csv.gz.receipt.json", "{}")
    _touch(project / "data" / "raw" / "extended" / "prices.csv.gz", "binary prices")
    run = tmp_path / "outside" / "pit_run_20261111"
    task = run / "fold_2016" / "variant_seed42"
    _touch(task / "validation_equity_curves.csv", "date,policy,portfolio_value\n")
    _touch(task / "timing.csv", "seconds\n1\n")
    _touch(task / "runs" / "variant_seed42" / "ppo_model.zip", "model")
    _touch(task / "runs" / "variant_seed42" / "vecnormalize.pkl", "stats")
    _touch(run / "logs" / "worker.log", "noise")
    _touch(run / "locks" / "task.lock" / "pid", "1")
    _touch(run / "launcher.log", "all tasks finished\n")
    _touch(run / "walk_forward_chained_equity.png", "png")
    return project, run


def test_collect_keeps_outputs_and_drops_models_logs_locks_and_prices(tmp_path: Path) -> None:
    project, run = _project(tmp_path)
    names = set(collect(project, [run]))
    assert {"config/experiment.yaml", "src/module.py", "README.md",
            "data/reference/set50_set100_membership_2012_2024.csv",
            "data/raw/extended/prices.csv.gz.receipt.json",
            "results/pit_run_20261111/fold_2016/variant_seed42/validation_equity_curves.csv",
            "results/pit_run_20261111/fold_2016/variant_seed42/timing.csv",
            "results/pit_run_20261111/launcher.log",
            "results/pit_run_20261111/walk_forward_chained_equity.png"} <= names
    assert not [n for n in names if n.endswith((".zip", ".pkl", ".pyc", ".csv.gz"))]
    assert not [n for n in names if "/logs/" in n or "/locks/" in n]


def test_archive_has_a_verified_manifest(tmp_path: Path) -> None:
    project, run = _project(tmp_path)
    output = tmp_path / "slim.tar.gz"
    manifest = build(project, output, collect(project, [run]), [run])
    with tarfile.open(output, "r:gz") as archive:
        stored = json.load(archive.extractfile("REPRODUCIBILITY_MANIFEST.json"))
        members = set(archive.getnames())
    assert stored["file_count"] == manifest["file_count"] == len(members) - 1
    assert stored["run_folders"] == ["pit_run_20261111"]


def test_sealed_holdout_files_are_refused(tmp_path: Path) -> None:
    project, run = _project(tmp_path)
    _touch(run / "holdout_sealed" / "rows.csv", "date\n")
    with pytest.raises(ValueError, match="sealed"):
        collect(project, [run])
