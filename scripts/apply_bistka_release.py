from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.campaigns.atomic_release_apply import apply_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a BistKA release without rewriting exact files")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", default=None)
    parser.add_argument("--staging-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--quota-margin-mib", type=int, default=1)
    args = parser.parse_args()
    result = apply_release(
        bundle_path=args.bundle,
        manifest_path=args.manifest,
        target_root=args.target_root,
        expected_bundle_sha256=args.expected_bundle_sha256,
        staging_root=args.staging_root,
        quota_margin_bytes=args.quota_margin_mib * 1024 * 1024,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
