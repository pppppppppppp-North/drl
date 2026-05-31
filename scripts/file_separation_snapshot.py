from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def iter_project_files(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and not path.relative_to(root).parts[0] == "_file_separation"
    )


def create_snapshot(root: Path, uploaded_files: list[str]) -> Path:
    separation_dir = root / "_file_separation"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    snapshot_dir = separation_dir / timestamp
    uploaded_snapshot_dir = snapshot_dir / "uploaded_snapshot"
    uploaded_snapshot_dir.mkdir(parents=True, exist_ok=True)

    all_files = iter_project_files(root)
    uploaded_set = set(uploaded_files)
    generated_or_existing = [path for path in all_files if path not in uploaded_set]

    (snapshot_dir / "uploaded_files.txt").write_text("\n".join(uploaded_files) + "\n", encoding="utf-8")
    (snapshot_dir / "pre_existing_or_generated_files.txt").write_text(
        "\n".join(generated_or_existing) + "\n",
        encoding="utf-8",
    )
    (snapshot_dir / "all_project_files_after_upload.txt").write_text("\n".join(all_files) + "\n", encoding="utf-8")

    for relative_path in uploaded_files:
        source = root / relative_path
        if not source.exists():
            continue
        destination = uploaded_snapshot_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    latest = separation_dir / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(snapshot_dir.name)
    return snapshot_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--uploaded-file", action="append", required=True)
    args = parser.parse_args()

    snapshot_dir = create_snapshot(Path(args.root).resolve(), args.uploaded_file)
    print(snapshot_dir.name)


if __name__ == "__main__":
    main()
