from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.campaigns.bistka_deployment import run_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact campaign deployment and observed data on BistKA")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="campaigns/large_scale_thai_drl/deployment/bistka_deployment_manifest.json")
    parser.add_argument("--output", default="reports/bistka_calibration_preflight.json")
    parser.add_argument("--allow-non-bistka", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    receipt = run_preflight(
        project_root=Path(args.project_root),
        manifest_path=Path(args.manifest),
        output_path=Path(args.output),
        allow_non_bistka=args.allow_non_bistka,
    )
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
