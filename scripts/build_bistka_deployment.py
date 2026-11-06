from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.campaigns.bistka_deployment import build_deployment


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a content-addressed, minimal BistKA campaign bundle")
    parser.add_argument("--config", default="config/bistka_deployment.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="campaigns/large_scale_thai_drl/deployment/bistka_deployment_manifest.json")
    parser.add_argument("--bundle", default="campaigns/large_scale_thai_drl/deployment/bistka_release.tar.gz")
    args = parser.parse_args()
    result = build_deployment(
        project_root=Path(args.project_root),
        config_path=Path(args.config),
        output_manifest=Path(args.manifest),
        output_bundle=Path(args.bundle),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
