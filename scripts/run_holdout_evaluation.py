"""Run the single evaluation of the locked final design on the sealed 2025+ holdout (milestone M6).

The holdout (``data/holdout_sealed/``) holds Yahoo prices from 2025 on, sealed on 2026-11-06 with
SHA-256 manifests (``src/data/sealed_holdout.py``). Nothing so far has read it. This script is the
only intended way to read it, and it runs once per universe, after that universe's test stage:

1. **The lock** (``config/holdout_protocol_*.yaml``) names the design (as the test lock does), the
   holdout's end date and folds, which sealed folder continues each development file the design's
   config reads, how SET50 membership continues after the 2024 archive (point-in-time only), an
   optional stop rule for stale prices, the references, the deflated-Sharpe trial count and the output
   folder. The authors set ``status: APPROVED`` with their names and the date; a draft is refused.
2. **Checks before anything is opened:** the lock is approved for a holdout stage; every seal holds
   (``verify_seal`` hashes the files without parsing them); the config matches its recorded hash; the
   tracked code and configs are committed; the test stage named in ``after_stage`` is in the ledger and
   this stage is not; the output folder does not exist.
3. **Opening:** the ledger records the stage first, so an opened holdout counts as used even if a later
   step fails. Each sealed part is appended to the development file it continues (refusing any overlap)
   and written inside the output folder; the development files are not changed. Only series already in
   the development file are continued: a series that first trades in 2025 (a new listing after a merger,
   for example) is not in the development price map, and two downloads can both carry it. Dropped series
   are listed in the receipt. For the point-in-time design, ``hold_last_list`` carries the last published
   SET50/SET100 list (2024 H2) to the holdout end.
4. **Stale prices** (``src/data/stale_rows.py``) are counted in the opened rows and written to
   ``stale_rows.json`` before any training. With ``stop_if_stale_share_above`` set, a larger share
   stops the stage there.
5. **The run:** a copy of the config pointing at the merged files, with ``splits.validation_end`` (and,
   point-in-time, ``data.end``) at the holdout end. One walk-forward fold per holdout year and seed:
   2025 trained on everything through 2024, 2026 on everything through 2025, exactly as in validation
   and the test years. Then the seed ensemble with the band, if the lock asks for it, and the aggregate
   tables with paired bootstrap intervals and the deflated Sharpe.
6. **Receipts:** ``receipt_start.json`` and ``receipt_end.json`` in the output folder, and one line per
   stage in ``FINAL_TEST_LEDGER.jsonl`` beside it (the test stages' ledger).

``--dry-run`` prints the checks and the plan; it verifies the seals but reads no market data.

**A failed attempt.** If an attempt stops before any model has been evaluated (no validation curve and
no end receipt), ``--record-failed-attempt REASON`` renames its folder to ``<output_dir>_attemptN_failed``
and adds a ``failed_before_training`` line to the ledger. A new attempt then needs ``retry_of`` in the
lock, naming that folder, and the authors' approval like any lock change. An attempt that evaluated
anything can never be retried.

    python -m scripts.run_holdout_evaluation --lock config/holdout_protocol_pit.yaml --dry-run
    python -m scripts.run_holdout_evaluation --lock config/holdout_protocol_pit.yaml --workers 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_final_test_evaluation import git_state, sha256
from src.data.sealed_holdout import SealError, check_holdout_lock, open_sealed, verify_seal
from src.data.stale_rows import stale_summary

HOLDOUT_START = pd.Timestamp("2025-01-01")
LEDGER = "FINAL_TEST_LEDGER.jsonl"
MEMBERSHIP_MODES = {"hold_last_list", "as_recorded"}
HOLDOUT_DATA = "holdout_data"


def _date(value) -> pd.Timestamp:
    try:
        return pd.Timestamp(str(value or ""))
    except ValueError:
        return pd.NaT


def config_data_files(spec: dict) -> list[str]:
    """The price and index files a design's config reads (the files the holdout must continue)."""

    data = spec.get("data") or {}
    prices = data.get("prices")
    files = list(prices) if isinstance(prices, list) else ([prices] if prices else [])
    return [*files, *([data["index"]] if data.get("index") else [])]


def _ledger(ledger: Path) -> list[dict]:
    if not ledger.is_file():
        return []
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]


def _ledger_stages(ledger: Path) -> set[str]:
    """Stages that have started (failure records are not starts)."""

    return {entry.get("stage") for entry in _ledger(ledger) if not entry.get("event")}


def evaluated_anything(folder: Path) -> bool:
    """An attempt evaluated something if any validation curve or an end receipt exists in it."""

    return (folder / "receipt_end.json").exists() or any(folder.rglob("validation_equity_curves.csv"))


def retry_problems(lock: dict, root: Path, entries: list[dict]) -> list[str]:
    """Reasons a stage already in the ledger may not run again; empty for an allowed retry."""

    stage = lock.get("stage")
    starts = [e for e in entries if e.get("stage") == stage and not e.get("event")]
    failures = [e for e in entries if e.get("stage") == stage and e.get("event") == "failed_before_training"]
    if not starts:
        return []
    retry = lock.get("retry_of")
    if not retry:
        return [f"the ledger already has a {stage!r} run"]
    problems = []
    if len(failures) != len(starts):
        problems.append(f"an earlier {stage!r} attempt is not recorded as failed before training")
    if retry not in {e.get("attempt_dir") for e in failures}:
        problems.append(f"retry_of {retry} is not a recorded failed attempt")
    folder = root / retry
    if not folder.is_dir():
        problems.append(f"the failed attempt's folder {retry} is missing")
    elif evaluated_anything(folder):
        problems.append(f"the attempt in {retry} evaluated something, so it may not be retried")
    return problems


def check_lock(lock: dict, root: Path, git: dict) -> list[str]:
    """Every reason the lock may not open the holdout; empty when it may. Opens no data."""

    problems = check_holdout_lock(lock)
    design = lock.get("design") or {}
    for key in ("candidate", "runner", "config", "config_sha256", "variant", "timesteps", "seeds"):
        if not design.get(key):
            problems.append(f"design.{key} is missing")
    config = root / str(design.get("config", ""))
    spec = None
    if design.get("config"):
        if not config.is_file():
            problems.append(f"config {design['config']} not found")
        else:
            if design.get("config_sha256") and sha256(config) != design["config_sha256"]:
                problems.append(f"config {design['config']} changed since the lock (SHA-256 differs)")
            spec = yaml.safe_load(config.read_text())

    holdout = lock.get("holdout") or {}
    end = _date(holdout.get("end"))
    if pd.isna(end) or end < HOLDOUT_START:
        problems.append("holdout.end must be a date on or after 2025-01-01")
    folds = sorted(holdout.get("folds") or [])
    if not folds:
        problems.append("holdout.folds is empty")
    elif not pd.isna(end) and any(year < HOLDOUT_START.year or year > end.year for year in folds):
        problems.append(f"holdout.folds must be years from 2025 to the holdout end ({end.year})")

    extend = holdout.get("extend") or []
    if not extend:
        problems.append("holdout.extend lists no files")
    for item in extend:
        dev, sealed, part = item.get("dev"), item.get("sealed"), item.get("part")
        if not dev or not sealed or not part:
            problems.append(f"holdout.extend entry {item} needs dev, sealed and part")
            continue
        if not (root / dev).is_file():
            problems.append(f"development file {dev} not found")
        try:
            manifest = verify_seal(root / sealed)
        except SealError as exc:
            problems.append(str(exc))
            continue
        if part not in manifest.get("files", {}):
            problems.append(f"{sealed} has no sealed part {part!r}")
        elif not pd.isna(end) and _date(manifest["files"][part].get("end")) < end - pd.Timedelta(days=7):
            problems.append(f"{sealed} {part} ends on {manifest['files'][part].get('end')}, well before holdout.end")
    if spec is not None:
        needed, given = set(config_data_files(spec)), {item.get("dev") for item in extend}
        if needed - given:
            problems.append(f"holdout.extend does not continue {sorted(needed - given)}, which the config reads")
        if given - needed:
            problems.append(f"holdout.extend continues {sorted(given - needed)}, which the config does not read")
        if (spec.get("data") or {}).get("membership") and holdout.get("membership") not in MEMBERSHIP_MODES:
            problems.append(f"holdout.membership must be one of {sorted(MEMBERSHIP_MODES)} for a point-in-time design")
    stop = holdout.get("stop_if_stale_share_above")
    if stop is not None and not (isinstance(stop, (int, float)) and 0 < stop <= 1):
        problems.append("holdout.stop_if_stale_share_above must be empty or a share in (0, 1]")

    if not lock.get("n_trials"):
        problems.append("n_trials (designs tried, for the deflated Sharpe) is missing")
    out = root / str(lock.get("output_dir", ""))
    if not lock.get("output_dir"):
        problems.append("output_dir is missing")
    elif out.exists():
        problems.append(f"output folder {lock['output_dir']} already exists: the holdout evaluation runs once")
    if lock.get("output_dir"):
        entries = _ledger(out.parent / LEDGER)
        stages = _ledger_stages(out.parent / LEDGER)
        problems.extend(retry_problems(lock, root, entries))
        if not lock.get("after_stage"):
            problems.append("after_stage (the test stage that must come first) is missing")
        elif lock["after_stage"] not in stages:
            problems.append(f"the test stage {lock['after_stage']!r} is not in the ledger yet")
    if git.get("head") is None or git.get("dirty"):
        problems.append(f"uncommitted changes to tracked code or configs: {git.get('dirty')}")
    if not git.get("lock_tracked"):
        problems.append("the lock is not committed")
    return problems


def merged_path(output_dir: str, dev: str) -> str:
    """Where the development file continued by the holdout is written, relative to the root."""

    path = Path(dev)
    return str(Path(output_dir) / HOLDOUT_DATA / (path.relative_to(path.anchor) if path.is_absolute() else path))


def derive_holdout_config(lock: dict, root: Path) -> dict:
    """The design's config, reading the merged files up to the holdout end."""

    spec = yaml.safe_load((root / lock["design"]["config"]).read_text())
    out, holdout = lock["output_dir"], lock["holdout"]
    end = str(_date(holdout["end"]).date())
    data = spec["data"]
    if isinstance(data.get("prices"), list):
        data["prices"] = [merged_path(out, p) for p in data["prices"]]
    elif data.get("prices"):
        data["prices"] = merged_path(out, data["prices"])
    if data.get("index"):
        data["index"] = merged_path(out, data["index"])
    if data.get("membership"):
        data["end"] = end
        if holdout.get("membership") == "hold_last_list":
            data["membership"] = merged_path(out, data["membership"])
            data["price_map"] = merged_path(out, data["price_map"])
    spec.setdefault("splits", {})["validation_end"] = end
    return spec


def hold_last_list(membership: pd.DataFrame, price_map: pd.DataFrame,
                   end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carry the last published list to ``end``: a copy of it from the day the archive stops.

    Its publication date stays the original one, so the mask never uses it before it was known.
    """

    last_end = membership["effective_to_exclusive"].max()
    last = membership[membership["effective_to_exclusive"] == last_end]
    if last["record_id"].nunique() != 1:
        raise ValueError("the last membership period has more than one list")
    if pd.Timestamp(last_end) > HOLDOUT_START:
        raise ValueError(f"the membership archive already runs to {last_end}; use 'as_recorded'")
    record = str(last["record_id"].iloc[0])
    held_id, until = f"{record}_held", (end + pd.Timedelta(days=1)).date().isoformat()
    held = last.assign(record_id=held_id, period_label=last["period_label"] + " (held)",
                       effective_from=last_end, effective_to_exclusive=until, revision_status="held_for_holdout")
    rows = price_map[price_map["record_id"] == record]
    held_map = rows.assign(record_id=held_id, effective_from=last_end, effective_to_exclusive=until)
    return pd.concat([membership, held], ignore_index=True), pd.concat([price_map, held_map], ignore_index=True)


def append_holdout(dev: pd.DataFrame, sealed: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """The development rows, then the sealed rows up to ``end``, with the development file's columns."""

    missing = [c for c in dev.columns if c not in sealed.columns]
    if missing:
        raise ValueError(f"the sealed rows lack the columns {missing}")
    sealed = sealed[pd.to_datetime(sealed["date"]) <= end][list(dev.columns)]
    if len(sealed) and pd.to_datetime(sealed["date"]).min() <= pd.to_datetime(dev["date"]).max():
        raise ValueError("the sealed rows overlap the development rows")
    return pd.concat([dev, sealed], ignore_index=True)


def known_series(dev: pd.DataFrame, sealed: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The sealed rows of series the development file has, and the names of the others."""

    if "ticker" not in dev.columns or "ticker" not in sealed.columns:
        return sealed, []
    known = set(dev["ticker"])
    dropped = sorted(str(t) for t in set(sealed["ticker"]) - known)
    return sealed[sealed["ticker"].isin(known)], dropped


def record_failed_attempt(lock: dict, root: Path, reason: str) -> str:
    """Rename a failed attempt's folder and record it in the ledger; refuse if it evaluated anything."""

    out = root / lock["output_dir"]
    if not out.is_dir():
        raise SystemExit(f"no attempt folder {lock['output_dir']}")
    if evaluated_anything(out):
        raise SystemExit(f"the attempt in {lock['output_dir']} evaluated something; it stays a used run")
    ledger = out.parent / LEDGER
    starts = [e for e in _ledger(ledger) if e.get("stage") == lock["stage"] and not e.get("event")]
    if not starts:
        raise SystemExit(f"the ledger has no {lock['stage']!r} attempt")
    attempt = f"{lock['output_dir']}_attempt{len(starts)}_failed"
    out.rename(root / attempt)
    with open(ledger, "a") as handle:
        handle.write(json.dumps({"stage": lock["stage"], "event": "failed_before_training", "attempt_dir": attempt,
                                 "reason": reason, "recorded_at": dt.datetime.now().isoformat(timespec="seconds")}) + "\n")
    return attempt


def join_returns(dev: pd.DataFrame, sealed: pd.DataFrame) -> pd.Series:
    """Per series, the first sealed close over the last development close, minus one.

    Two downloads on different days can be adjusted for dividends differently; a large step where the
    files meet shows that, and is recorded in the receipt.
    """

    key = "ticker" if "ticker" in dev.columns else None
    last = dev.sort_values("date").groupby(key)["close"].last() if key else pd.Series({"": dev.sort_values("date")["close"].iloc[-1]})
    first = sealed.sort_values("date").groupby(key)["close"].first() if key else pd.Series({"": sealed.sort_values("date")["close"].iloc[0]})
    return (first / last.reindex(first.index) - 1).dropna()


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.assign(date=pd.to_datetime(frame["date"]).dt.date.astype(str))
    frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)


def open_and_merge(lock: dict, root: Path) -> dict:
    """Open each sealed folder, write the merged files and the held membership; return a receipt."""

    out, holdout = lock["output_dir"], lock["holdout"]
    end = _date(holdout["end"])
    opened: dict[str, dict[str, pd.DataFrame]] = {}
    files, stale = {}, {}
    for item in holdout["extend"]:
        if item["sealed"] not in opened:
            opened[item["sealed"]] = open_sealed(root / item["sealed"], lock)
        sealed = opened[item["sealed"]][item["part"]]
        dev = pd.read_csv(root / item["dev"], parse_dates=["date"])
        sealed, dropped = known_series(dev, sealed)
        merged = append_holdout(dev, sealed, end)
        path = root / merged_path(out, item["dev"])
        _write(merged, path)
        joins = join_returns(dev, merged.iloc[len(dev):])
        files[item["dev"]] = {"sealed": item["sealed"], "part": item["part"], "development_rows": int(len(dev)),
                              "holdout_rows": int(len(merged) - len(dev)), "sha256": sha256(path),
                              "largest_step_at_join": float(joins.abs().max()) if len(joins) else None,
                              "series_with_step_above_20pct": sorted(str(k) for k in joins[joins.abs() > 0.2].index),
                              "series_not_in_development_file": dropped}
        stale[item["dev"]] = stale_summary(merged, since=HOLDOUT_START)
    spec = yaml.safe_load((root / lock["design"]["config"]).read_text())
    data = spec["data"]
    if data.get("membership") and holdout.get("membership") == "hold_last_list":
        membership = pd.read_csv(root / data["membership"], dtype=str, keep_default_na=False)
        price_map = pd.read_csv(root / data["price_map"], dtype=str, keep_default_na=False)
        membership, price_map = hold_last_list(membership, price_map, end)
        for key, frame in (("membership", membership), ("price_map", price_map)):
            path = root / merged_path(out, data[key])
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            files[data[key]] = {"held_to": str(end.date()), "sha256": sha256(path)}
    return {"files": files, "stale_rows": stale}


def plan(lock: dict, python: str) -> dict:
    """The task lines and the commands after training, with paths relative to the root."""

    design, out = lock["design"], Path(lock["output_dir"])
    cfg, runs = out / "holdout_config.yaml", out / "runs"
    folds = sorted(lock["holdout"]["folds"])
    tasks = [f"{runs} {design['runner']} {cfg} {int(design['timesteps'])} {year} {design['variant']} {seed}"
             for year in folds for seed in design["seeds"]]
    years, seeds = [str(y) for y in folds], [str(s) for s in design["seeds"]]
    after, extra = [], []
    ensemble = design.get("ensemble")
    if ensemble:
        ens = out / "ensemble"
        after.append([python, "-m", "scripts.evaluate_pit_seed_ensemble", "--config", str(cfg), "--source-run", str(runs),
                      "--variant", design["variant"], "--output-dir", str(ens), "--seeds", *seeds,
                      "--fold-years", *years, "--band", str(ensemble.get("band", 0.005)),
                      "--modes", *ensemble.get("modes", ["ensemble_band"])])
        extra = ["--extra-run-dirs", str(ens)]
    references = list(lock.get("references") or [])
    after.append([python, "-m", "scripts.aggregate_walk_forward_reward", "--run-dir", str(runs), *extra,
                  "--variants", design["variant"], "--seeds", *seeds, "--n-trials", str(int(lock["n_trials"])),
                  "--output-dir", str(out / "summary"), *(["--references", *references] if references else []),
                  "--title", "Sealed holdout, chained (each year trained only on earlier data; net of costs)"])
    return {"tasks": tasks, "after": after}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-failed-attempt", metavar="REASON",
                        help="rename an attempt that failed before evaluating anything, and record it in the ledger")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    lock_path = Path(args.lock)
    lock = yaml.safe_load(lock_path.read_text())
    if args.record_failed_attempt:
        print(f"recorded the failed attempt as {record_failed_attempt(lock, root, args.record_failed_attempt)}")
        return
    git = git_state(root, lock_path)
    problems = check_lock(lock, root, git)
    ready = lock.get("design", {}).get("variant") and lock.get("output_dir") and (lock.get("holdout") or {}).get("folds")
    steps = plan(lock, args.python) if ready else None
    if args.dry_run:
        print("checks: " + ("all passed" if not problems else "\n  - " + "\n  - ".join(problems)))
        if steps:
            print(f"{len(steps['tasks'])} training tasks:")
            print("\n".join("  " + t for t in steps["tasks"]))
            print("then:")
            print("\n".join("  " + " ".join(c) for c in steps["after"]))
        return
    if problems:
        raise SystemExit("the holdout evaluation may not run:\n  - " + "\n  - ".join(problems))

    out = root / lock["output_dir"]
    out.mkdir(parents=True)
    start = {"stage": lock["stage"], "lock": str(lock_path), "lock_sha256": sha256(lock_path),
             "config_sha256": lock["design"]["config_sha256"], "git_head": git["head"],
             "approved_by": lock["approved_by"], "approved_on": str(lock["approved_on"]),
             "started_at": dt.datetime.now().isoformat(timespec="seconds"),
             **({"retry_of": lock["retry_of"]} if lock.get("retry_of") else {})}
    (out / "receipt_start.json").write_text(json.dumps(start, indent=2) + "\n")
    with open(out.parent / LEDGER, "a") as ledger:
        ledger.write(json.dumps({**start, "output_dir": lock["output_dir"]}) + "\n")

    opened = open_and_merge(lock, root)
    (out / "holdout_data_receipt.json").write_text(json.dumps(opened["files"], indent=2) + "\n")
    (out / "stale_rows.json").write_text(json.dumps(opened["stale_rows"], indent=2) + "\n")
    shares = {path: s["stale_share"] for path, s in opened["stale_rows"].items()}
    print("stale rows from 2025 on: " + ", ".join(f"{Path(p).name} {v:.1%}" for p, v in shares.items()))
    stop = lock["holdout"].get("stop_if_stale_share_above")
    if stop is not None and max(shares.values()) > stop:
        end = {**start, "stopped_at": dt.datetime.now().isoformat(timespec="seconds"),
               "stopped_because": f"a stale-row share above {stop:.1%}", "stale_shares": shares}
        (out / "receipt_end.json").write_text(json.dumps(end, indent=2) + "\n")
        raise SystemExit(f"stopped before training: a stale-row share above {stop:.1%} ({out / 'stale_rows.json'})")

    (out / "holdout_config.yaml").write_text(yaml.safe_dump(derive_holdout_config(lock, root), sort_keys=False))
    (out / "tasks.txt").write_text("\n".join(steps["tasks"]) + "\n")
    env = {**os.environ, "PYTHON": args.python, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run(["bash", "scripts/run_task_pool.sh", str(out.relative_to(root) / "tasks.txt"), str(args.workers)],
                   cwd=root, env=env, check=True)
    design = lock["design"]
    missing = [f"fold {y} seed {s}" for y in lock["holdout"]["folds"] for s in design["seeds"]
               if not (out / "runs" / f"fold_{y}" / f"{design['variant']}_seed{s}" / "validation_equity_curves.csv").is_file()]
    if missing:
        raise SystemExit(f"training tasks failed: {missing}; see {out / 'runs' / 'logs'}")
    for command in steps["after"]:
        subprocess.run(command, cwd=root, env=env, check=True)
    end = {**start, "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
           "summary": str(Path(lock["output_dir"]) / "summary"), "stale_shares": shares}
    (out / "receipt_end.json").write_text(json.dumps(end, indent=2) + "\n")
    print(f"holdout evaluation finished: {out / 'summary'}")


if __name__ == "__main__":
    main()
