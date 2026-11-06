from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.campaigns.gates import atomic_json


def test_atomic_json_uses_unique_temporaries_for_parallel_writers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "state.json"
    barrier = threading.Barrier(2)
    original_replace = Path.replace

    def synchronized_replace(self: Path, target: Path) -> Path:
        if Path(target) == output:
            barrier.wait(timeout=5)
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", synchronized_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(atomic_json, output, {"writer": writer})
            for writer in (1, 2)
        ]
        for future in futures:
            future.result(timeout=5)

    assert json.loads(output.read_text(encoding="utf-8"))["writer"] in {1, 2}
    assert not list(tmp_path.glob(".state.json.*.tmp"))
