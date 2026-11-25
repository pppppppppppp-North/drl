"""Open the sealed 2025+ holdout only when its seal holds and an approved holdout lock asks for it (M2/M6).

``scripts/download_extended_history.py`` wrote each folder under ``data/holdout_sealed/`` with:
- the data files (``stocks.csv.gz``, ``index.csv.gz``);
- ``HOLDOUT_MANIFEST.json``: status ``SEALED``, and each file's rows, first and last date and SHA-256;
- ``HOLDOUT_MANIFEST.json.sha256``: the manifest's own hash.

:func:`verify_seal` checks the manifest against its hash and every file against the manifest. It
hashes the files' bytes but does not parse them, so it reveals nothing about the data.

:func:`open_sealed` reads the files, and only after three checks:
- the seal holds;
- the lock is ``APPROVED``, with names and a date, for a stage whose name contains ``holdout``;
- each file's rows and dates match its manifest.

The manifest's paths were written as absolute paths, so files are found by name inside the folder,
wherever the folder now is.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

MANIFEST = "HOLDOUT_MANIFEST.json"


class SealError(RuntimeError):
    """The holdout's seal does not hold, or the request to open it is not approved."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal(folder: str | Path) -> dict:
    """The manifest, if it matches its hash, says ``SEALED`` and every file matches it."""

    folder = Path(folder)
    manifest_path, hash_path = folder / MANIFEST, folder / f"{MANIFEST}.sha256"
    if not manifest_path.is_file() or not hash_path.is_file():
        raise SealError(f"{folder}: {MANIFEST} or its .sha256 is missing")
    expected = hash_path.read_text().split()[0]
    if _sha256(manifest_path) != expected:
        raise SealError(f"{folder}: {MANIFEST} does not match its recorded hash")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "SEALED":
        raise SealError(f"{folder}: status is {manifest.get('status')!r}, not 'SEALED'")
    for name, entry in manifest.get("files", {}).items():
        path = folder / Path(entry["path"]).name
        if not path.is_file():
            raise SealError(f"{folder}: sealed file {path.name} ({name}) is missing")
        if _sha256(path) != entry["sha256"]:
            raise SealError(f"{folder}: {path.name} ({name}) changed since it was sealed")
    return manifest


def check_holdout_lock(lock: dict) -> list[str]:
    """Reasons the lock may not open the holdout; empty when it may."""

    problems = []
    if "holdout" not in str(lock.get("stage") or ""):
        problems.append(f"stage {lock.get('stage')!r} is not a holdout stage")
    if lock.get("status") != "APPROVED":
        problems.append(f"status is {lock.get('status')!r}, not 'APPROVED'")
    if not lock.get("approved_by"):
        problems.append("approved_by lists no one")
    if not lock.get("approved_on"):
        problems.append("approved_on is empty")
    return problems


def open_sealed(folder: str | Path, lock: dict) -> dict[str, pd.DataFrame]:
    """Each sealed file as a frame (``date`` parsed), after the seal and lock checks."""

    problems = check_holdout_lock(lock)
    if problems:
        raise SealError("the holdout may not be opened: " + "; ".join(problems))
    folder = Path(folder)
    manifest = verify_seal(folder)
    frames = {}
    for name, entry in manifest["files"].items():
        frame = pd.read_csv(folder / Path(entry["path"]).name, parse_dates=["date"])
        start = str(frame["date"].min().date()) if len(frame) else None
        end = str(frame["date"].max().date()) if len(frame) else None
        if (len(frame), start, end) != (entry["rows"], entry["start"], entry["end"]):
            raise SealError(f"{folder}: {name} has {len(frame)} rows {start}..{end}, "
                            f"the manifest says {entry['rows']} rows {entry['start']}..{entry['end']}")
        frames[name] = frame
    return frames
