from __future__ import annotations

import re
from pathlib import Path


_STEP_PATTERN = re.compile(r"_(\d+)_steps(?:\.[^.]+)?$")


def prune_checkpoints(directory: Path, *, name_prefix: str, keep_last: int) -> list[Path]:
    """Delete complete old checkpoint groups while retaining the latest steps."""
    if keep_last <= 0:
        raise ValueError("checkpoint keep_last must be positive")
    checkpoint_files: list[tuple[int, Path]] = []
    for path in directory.glob(f"{name_prefix}*"):
        if not path.is_file():
            continue
        match = _STEP_PATTERN.search(path.name)
        if match:
            checkpoint_files.append((int(match.group(1)), path))
    retained_steps = set(sorted({step for step, _ in checkpoint_files})[-keep_last:])
    removed = []
    for step, path in checkpoint_files:
        if step not in retained_steps:
            path.unlink()
            removed.append(path)
    return removed
