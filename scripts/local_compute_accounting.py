"""Local compute used by walk-forward runs (milestone M6, RQ6 on one machine).

For each run folder, this sums the time that every task recorded in ``fold_<Y>/<task>/timing.csv``
(training plus validation of one model on one CPU core) and estimates the wall time from file
modification times: from the first task's ``splits.json`` to the last task's
``validation_equity_curves.csv``. The wall time therefore includes any time the machine slept or a
run was paused, and the speed-up (task time over wall time) is the average number of tasks that ran
at once.

    python -m scripts.local_compute_accounting --run-dirs results/walk_forward_reward_20261106 \\
        results/pit_walk_forward_shared_20261110 --output results/local_compute_accounting.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def account(run_dir: Path) -> dict[str, object]:
    timings = sorted(run_dir.glob("fold_*/*/timing.csv"))
    if not timings:
        raise FileNotFoundError(f"no fold_*/*/timing.csv under {run_dir}")
    frame = pd.concat([pd.read_csv(p) for p in timings], ignore_index=True)
    starts = [p.stat().st_mtime for p in run_dir.glob("fold_*/*/splits.json")]
    ends = [p.stat().st_mtime for p in run_dir.glob("fold_*/*/validation_equity_curves.csv")]
    wall_hours = (max(ends) - min(starts)) / 3600.0 if starts and ends else float("nan")
    task_hours = float(frame["seconds"].sum()) / 3600.0
    throughput = frame["timesteps"] / frame["seconds"]
    return {
        "run": run_dir.name,
        "variants": " ".join(sorted(frame["variant"].astype(str).unique())),
        "tasks": int(len(frame)),
        "timesteps_per_task": int(frame["timesteps"].median()),
        "task_hours": round(task_hours, 2),
        "median_task_minutes": round(float(frame["seconds"].median()) / 60.0, 1),
        "min_task_minutes": round(float(frame["seconds"].min()) / 60.0, 1),
        "max_task_minutes": round(float(frame["seconds"].max()) / 60.0, 1),
        "median_timesteps_per_second": round(float(throughput.median()), 1),
        "wall_hours": round(wall_hours, 2),
        "speed_up": round(task_hours / wall_hours, 2) if wall_hours and wall_hours > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", default=None, help="CSV to write (optional)")
    args = parser.parse_args()
    table = pd.DataFrame([account(Path(d)) for d in args.run_dirs])
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 60):
        print(table.to_string(index=False))
        print(f"\ntotal: {int(table['tasks'].sum())} tasks, {table['task_hours'].sum():.1f} task-hours")


if __name__ == "__main__":
    main()
