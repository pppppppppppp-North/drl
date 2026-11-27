"""Slim reproducibility archive: code, configs, documents and run outputs, without trained models (M8).

What goes in:

- the code and documents, as in ``scripts.build_reproducibility_archive``: ``config/``, ``src/``,
  ``scripts/``, ``slurm/``, ``docs/``, ``tests/`` and the root files (README, requirements, ...);
- the small public reference inputs (the official SET50/SET100 membership lists and the BOT
  policy-rate history) and the receipts of the price files that git does not track
  (``data/raw/extended/**/*.receipt.json``), which identify those files by SHA-256. The Yahoo price
  files themselves are not included;
- for every run folder given, which may live outside the repository: metrics, equity curves,
  holdings, receipts, splits, timings, launcher logs and figures (``.csv``, ``.json``, ``.md``,
  ``.txt``, ``.png``, ``.yaml``, ``.log``), under ``results/<folder name>/``.

What stays out: trained models (``*.zip``), ``VecNormalize`` statistics (``*.pkl``), TensorBoard
events, per-task worker logs (``logs/``), lock folders, the holdout stages' price copies
(``holdout_data/``), and anything under ``data/holdout_sealed``.
The models can be retrained from the configs and seeds.

The archive carries a SHA-256 manifest (``REPRODUCIBILITY_MANIFEST.json``) and is reopened and
verified before the command succeeds.

    python -m scripts.build_slim_results_archive --output results/slim_archive_20261111.tar.gz \\
        --run-dir results/walk_forward_reward_20261106 --run-dir results/pit_walk_forward_shared_20261110
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_reproducibility_archive import EXCLUDED_PARTS, ROOT_FILES, SOURCE_ROOTS, _sha256, verify_archive

RUN_SUFFIXES = {".csv", ".json", ".md", ".txt", ".png", ".yaml", ".yml", ".log"}
# holdout_data/ holds the holdout stages' copies of the Yahoo price files (scripts/run_holdout_evaluation.py),
# which stay out like every other price file; their hashes are in holdout_data_receipt.json.
RUN_EXCLUDED_DIRS = {"locks", "logs", "tensorboard", "checkpoints", "__pycache__", "holdout_data"}
REFERENCE_INPUTS = (
    "data/reference/set50_set100_membership_2012_2024.csv",
    "data/external/bot_policy_rate_history.csv",
)
# One line per run-once stage (test years and sealed holdout), the audit record of each opening.
LEDGER = "results/FINAL_TEST_LEDGER.jsonl"
RECEIPT_ROOT = "data/raw/extended"
SEALED = "holdout_sealed"


def collect(project_root: Path, run_dirs: list[Path]) -> dict[str, Path]:
    """Map archive names to source files."""
    project_root = project_root.resolve()
    files: dict[str, Path] = {}

    def add_project_file(path: Path) -> None:
        relative = path.resolve().relative_to(project_root)
        if not any(part in EXCLUDED_PARTS for part in relative.parts):
            files[relative.as_posix()] = path

    for name in SOURCE_ROOTS:
        root = project_root / name
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file():
                    add_project_file(path)
    for name in (*ROOT_FILES, *REFERENCE_INPUTS, LEDGER):
        if (project_root / name).is_file():
            add_project_file(project_root / name)
    receipts = project_root / RECEIPT_ROOT
    if receipts.is_dir():
        for path in receipts.rglob("*.receipt.json"):
            add_project_file(path)

    names = [d.resolve().name for d in run_dirs]
    if len(set(names)) != len(names):
        raise ValueError(f"run folders must have distinct names: {names}")
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        for path in run_dir.rglob("*"):
            relative = path.relative_to(run_dir)
            if (path.is_file() and path.suffix.lower() in RUN_SUFFIXES
                    and not set(relative.parts[:-1]) & (RUN_EXCLUDED_DIRS | EXCLUDED_PARTS)):
                files[f"results/{run_dir.name}/{relative.as_posix()}"] = path

    leaked = [name for name in files if SEALED in name]
    if leaked:
        raise ValueError(f"refusing to archive sealed holdout files: {leaked[:3]}")
    return dict(sorted(files.items()))


def _commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(project_root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(project_root: Path, output: Path, files: dict[str, Path], run_dirs: list[Path]) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = [{"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for name, path in files.items()]
    manifest = {
        "format": 1,
        "kind": "slim_results_archive",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": _commit(project_root),
        "run_folders": [d.resolve().name for d in run_dirs],
        "excluded": ["*.zip (trained models)", "*.pkl (VecNormalize statistics)", "logs/", "locks/",
                     "tensorboard/", "checkpoints/", "Yahoo price files (identified by their receipts)",
                     "data/holdout_sealed/"],
        "file_count": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "files": entries,
    }
    payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    with tarfile.open(output, "w:gz") as archive:
        for entry in entries:
            archive.add(files[entry["path"]], arcname=entry["path"], recursive=False)
        info = tarfile.TarInfo("REPRODUCIBILITY_MANIFEST.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    verify_archive(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", action="append", default=[], help="a run folder; repeat for each")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    run_dirs = [Path(d) for d in args.run_dir]
    files = collect(project_root, run_dirs)
    manifest = build(project_root, Path(args.output), files, run_dirs)
    print(json.dumps({"archive": args.output, "file_count": manifest["file_count"],
                      "total_bytes": manifest["total_bytes"], "archive_bytes": Path(args.output).stat().st_size,
                      "code_commit": manifest["code_commit"], "verified": True}, indent=2))


if __name__ == "__main__":
    main()
