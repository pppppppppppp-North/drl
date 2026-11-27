"""Run the single evaluation of the locked final design on the 2023-2024 test years (milestone M6).

Every walk-forward runner stops at its config's ``splits.validation_end`` (2022-12-30), so nothing so
far has read the test years. This script is the only intended way to read them, and it runs once:

1. **The lock** (``config/final_test_protocol.yaml``) names the design, its config and that config's
   SHA-256, the training budget, the seeds, the seed ensemble and band if any, the test years, the
   references, the deflated-Sharpe trial count and the output folder. The authors set
   ``status: APPROVED`` with their names and the date; a draft lock is refused.
2. **Checks before anything is read:** the config matches its recorded hash; tracked files under
   ``src/``, ``scripts/`` and ``config/`` are committed (the receipt records the commit); the test
   end is before 2025, so the sealed holdout stays sealed; the output folder does not exist; and the
   ledger beside it has no earlier entry for this stage.
3. **The run:** a copy of the config with ``splits.validation_end`` moved to the test end, then one
   walk-forward fold per test year and seed (each trained on everything before that year, exactly as
   in validation), through ``scripts/run_task_pool.sh``. Then the seed ensemble with the band, if the
   lock asks for it, and the aggregate tables with paired bootstrap intervals and the deflated Sharpe.
4. **Receipts:** ``receipt_start.json`` and ``receipt_end.json`` in the output folder, and one line
   per stage in ``FINAL_TEST_LEDGER.jsonl`` beside it.

``--dry-run`` prints the checks and the plan and writes nothing; it reads no market data.

    python -m scripts.run_final_test_evaluation --lock config/final_test_protocol.yaml --dry-run
    python -m scripts.run_final_test_evaluation --lock config/final_test_protocol.yaml --workers 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

HOLDOUT_START = pd.Timestamp("2025-01-01")
TEST_YEARS = {2023, 2024}
LEDGER = "FINAL_TEST_LEDGER.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(root: Path, lock_path: Path) -> dict:
    """The commit, uncommitted changes to tracked files under src/, scripts/ and config/, and whether
    the lock itself is committed."""

    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()

    try:
        head = run("rev-parse", "HEAD")
        dirty = run("status", "--porcelain", "--untracked-files=no", "--", "src", "scripts", "config")
        tracked = bool(run("ls-files", "--", str(lock_path.resolve().relative_to(root))))
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
        return {"head": None, "dirty": [f"git failed: {exc}"], "lock_tracked": False}
    return {"head": head, "dirty": [line for line in dirty.splitlines() if line], "lock_tracked": tracked}


def check_lock(lock: dict, root: Path, git: dict) -> list[str]:
    """Every reason the lock may not run; empty when it may."""

    problems = []
    if not lock.get("stage"):
        problems.append("stage is missing")
    if lock.get("status") != "APPROVED":
        problems.append(f"status is {lock.get('status')!r}, not 'APPROVED'")
    if not lock.get("approved_by"):
        problems.append("approved_by lists no one")
    if not lock.get("approved_on"):
        problems.append("approved_on is empty")
    design = lock.get("design") or {}
    for key in ("candidate", "runner", "config", "config_sha256", "variant", "timesteps", "seeds"):
        if not design.get(key):
            problems.append(f"design.{key} is missing")
    config = root / str(design.get("config", ""))
    if design.get("config"):
        if not config.is_file():
            problems.append(f"config {design['config']} not found")
        elif design.get("config_sha256") and sha256(config) != design["config_sha256"]:
            problems.append(f"config {design['config']} changed since the lock (SHA-256 differs)")
    years = set(lock.get("test_years") or [])
    if not years or not years <= TEST_YEARS:
        problems.append(f"test_years must be a non-empty subset of {sorted(TEST_YEARS)}")
    try:
        end = pd.Timestamp(str(lock.get("test_end") or ""))
    except ValueError:
        end = pd.NaT
    if pd.isna(end) or end >= HOLDOUT_START:
        problems.append("test_end must be a date before 2025-01-01 (the sealed holdout)")
    elif years and end < pd.Timestamp(f"{max(years)}-12-30"):
        problems.append(f"test_end {end.date()} cuts the last test year short")
    if not lock.get("n_trials"):
        problems.append("n_trials (designs tried, for the deflated Sharpe) is missing")
    out = root / str(lock.get("output_dir", ""))
    if not lock.get("output_dir"):
        problems.append("output_dir is missing")
    elif out.exists():
        problems.append(f"output folder {lock['output_dir']} already exists: the test evaluation runs once")
    ledger = out.parent / LEDGER
    if lock.get("output_dir") and ledger.is_file():
        stage = lock.get("stage")
        for line in ledger.read_text().splitlines():
            if line.strip() and json.loads(line).get("stage") == stage:
                problems.append(f"the ledger {ledger} already has a {stage!r} run")
    if git.get("head") is None or git.get("dirty"):
        problems.append(f"uncommitted changes to tracked code or configs: {git.get('dirty')}")
    if not git.get("lock_tracked"):
        problems.append("the lock is not committed")
    return problems


def derive_test_config(lock: dict, root: Path) -> dict:
    spec = yaml.safe_load((root / lock["design"]["config"]).read_text())
    spec.setdefault("splits", {})["validation_end"] = str(pd.Timestamp(str(lock["test_end"])).date())
    return spec


def plan(lock: dict, root: Path, python: str) -> dict:
    """The task lines and the commands after training, with paths relative to ``root``."""

    design, out = lock["design"], Path(lock["output_dir"])
    cfg, runs = out / "test_config.yaml", out / "runs"
    tasks = [f"{runs} {design['runner']} {cfg} {int(design['timesteps'])} {year} {design['variant']} {seed}"
             for year in sorted(lock["test_years"]) for seed in design["seeds"]]
    years = [str(y) for y in sorted(lock["test_years"])]
    seeds = [str(s) for s in design["seeds"]]
    after, extra, variants = [], [], [design["variant"]]
    ensemble = design.get("ensemble")
    if ensemble:
        ens = out / "ensemble"
        after.append([python, "-m", "scripts.evaluate_pit_seed_ensemble", "--config", str(cfg), "--source-run", str(runs),
                      "--variant", design["variant"], "--output-dir", str(ens), "--seeds", *seeds,
                      "--fold-years", *years, "--band", str(ensemble.get("band", 0.005)),
                      "--modes", *ensemble.get("modes", ["ensemble_band"])])
        extra = ["--extra-run-dirs", str(ens)]
    references = [r for r in lock.get("references", [])]
    after.append([python, "-m", "scripts.aggregate_walk_forward_reward", "--run-dir", str(runs), *extra,
                  "--variants", *variants, "--seeds", *seeds, "--n-trials", str(int(lock["n_trials"])),
                  "--output-dir", str(out / "summary"), *(["--references", *references] if references else []),
                  "--title", "Test years, chained (each year trained only on earlier data; net of costs)"])
    return {"tasks": tasks, "after": after}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    lock_path = Path(args.lock)
    lock = yaml.safe_load(lock_path.read_text())
    git = git_state(root, lock_path)
    problems = check_lock(lock, root, git)
    steps = plan(lock, root, args.python) if lock.get("design", {}).get("variant") and lock.get("output_dir") else None
    if args.dry_run:
        print("checks: " + ("all passed" if not problems else "\n  - " + "\n  - ".join(problems)))
        if steps:
            print(f"{len(steps['tasks'])} training tasks:")
            print("\n".join("  " + t for t in steps["tasks"]))
            print("then:")
            print("\n".join("  " + " ".join(c) for c in steps["after"]))
        return
    if problems:
        raise SystemExit("the test evaluation may not run:\n  - " + "\n  - ".join(problems))

    out = root / lock["output_dir"]
    out.mkdir(parents=True)
    (out / "test_config.yaml").write_text(yaml.safe_dump(derive_test_config(lock, root), sort_keys=False))
    (out / "tasks.txt").write_text("\n".join(steps["tasks"]) + "\n")
    start = {"stage": lock.get("stage"), "lock": str(lock_path), "lock_sha256": sha256(lock_path),
             "config_sha256": lock["design"]["config_sha256"], "git_head": git["head"],
             "approved_by": lock["approved_by"], "approved_on": str(lock["approved_on"]),
             "started_at": dt.datetime.now().isoformat(timespec="seconds")}
    (out / "receipt_start.json").write_text(json.dumps(start, indent=2) + "\n")
    with open(out.parent / LEDGER, "a") as ledger:
        ledger.write(json.dumps({**start, "output_dir": lock["output_dir"]}) + "\n")

    env = {**os.environ, "PYTHON": args.python, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run(["bash", "scripts/run_task_pool.sh", str(out.relative_to(root) / "tasks.txt"), str(args.workers)],
                   cwd=root, env=env, check=True)
    design = lock["design"]
    missing = [f"fold {y} seed {s}" for y in lock["test_years"] for s in design["seeds"]
               if not (out / "runs" / f"fold_{y}" / f"{design['variant']}_seed{s}" / "validation_equity_curves.csv").is_file()]
    if missing:
        raise SystemExit(f"training tasks failed: {missing}; see {out / 'runs' / 'logs'}")
    for command in steps["after"]:
        subprocess.run(command, cwd=root, env=env, check=True)
    end = {**start, "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
           "summary": str(Path(lock["output_dir"]) / "summary")}
    (out / "receipt_end.json").write_text(json.dumps(end, indent=2) + "\n")
    print(f"test evaluation finished: {out / 'summary'}")


if __name__ == "__main__":
    main()
