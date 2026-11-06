from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.campaigns.bistka_deployment import build_deployment
from src.campaigns.deployment_profiles import build_cumulative_deployment_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve an exact cumulative campaign profile and optionally build its "
            "content-addressed BistKA release"
        )
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--config-output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--campaign-dir",
        help="Project-relative campaign directory used to expand {campaign_dir}",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--bundle")
    args = parser.parse_args()
    if bool(args.manifest) != bool(args.bundle):
        parser.error("--manifest and --bundle must be provided together")

    root = Path(args.project_root).resolve()
    profile_result = build_cumulative_deployment_config(
        project_root=root,
        profile_path=Path(args.profile),
        output_path=Path(args.config_output),
        campaign_dir=Path(args.campaign_dir) if args.campaign_dir else None,
    )
    result: dict[str, object] = {"profile": profile_result}
    if args.manifest and args.bundle:
        result["deployment"] = build_deployment(
            project_root=root,
            config_path=(root / args.config_output).resolve(),
            output_manifest=(root / args.manifest).resolve(),
            output_bundle=(root / args.bundle).resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
