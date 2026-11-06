from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


SOURCE_ROOTS = ("config", "src", "scripts", "slurm", "docs", "tests")
ROOT_FILES = ("README.md", "requirements.txt", "pyproject.toml", "environment.yml")
RESULT_SUFFIXES = {".csv", ".json", ".md", ".png", ".pdf", ".tex", ".zip"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "tensorboard", ".git"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_archive_files(
    project_root: Path,
    *,
    result_dirs: list[Path],
    selected_models: list[Path],
    include_checkpoints: bool = False,
) -> list[Path]:
    project_root = project_root.resolve()
    files: set[Path] = set()
    for root_name in SOURCE_ROOTS:
        root = project_root / root_name
        if root.exists():
            files.update(path for path in root.rglob("*") if path.is_file())
    files.update(path for name in ROOT_FILES if (path := project_root / name).is_file())
    for result_dir in result_dirs:
        resolved = result_dir.resolve()
        if not resolved.is_relative_to(project_root):
            raise ValueError(f"result directory must be inside the project: {result_dir}")
        for path in resolved.rglob("*"):
            if path.is_file() and path.suffix.lower() in RESULT_SUFFIXES:
                files.add(path)
    for model in selected_models:
        resolved = model.resolve()
        if not resolved.is_relative_to(project_root):
            raise ValueError(f"selected model must be inside the project: {model}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        files.add(resolved)
    return sorted(
        path
        for path in files
        if not any(part in EXCLUDED_PARTS for part in path.parts)
        and (include_checkpoints or "checkpoints" not in path.parts)
    )


def build_archive(project_root: Path, output: Path, files: list[Path]) -> dict:
    project_root = project_root.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "path": str(path.resolve().relative_to(project_root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    manifest = {"format": 1, "project_root_name": project_root.name, "file_count": len(entries), "files": entries}
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    with tarfile.open(output, "w:gz") as archive:
        for path, entry in zip(files, entries):
            archive.add(path, arcname=entry["path"], recursive=False)
        info = tarfile.TarInfo("REPRODUCIBILITY_MANIFEST.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
    verify_archive(output)
    return manifest


def verify_archive(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        manifest_member = archive.getmember("REPRODUCIBILITY_MANIFEST.json")
        manifest = json.load(archive.extractfile(manifest_member))
        for entry in manifest["files"]:
            member = archive.getmember(entry["path"])
            extracted = archive.extractfile(member)
            digest = hashlib.sha256()
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"archive hash mismatch: {entry['path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-dir", action="append", default=[])
    parser.add_argument("--selected-model", action="append", default=[])
    parser.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    files = collect_archive_files(
        project_root,
        result_dirs=[Path(value) for value in args.result_dir],
        selected_models=[Path(value) for value in args.selected_model],
        include_checkpoints=args.include_checkpoints,
    )
    manifest = build_archive(project_root, Path(args.output), files)
    print(json.dumps({"archive": args.output, "file_count": manifest["file_count"], "verified": True}, indent=2))


if __name__ == "__main__":
    main()
