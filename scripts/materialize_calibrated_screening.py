from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.campaigns.vector_calibration import BENCHMARK_GLOB, materialize_calibrated_screening


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize an attempt-bound screening config from observed vector calibration"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--spec", default="config/research_campaign.yaml")
    parser.add_argument("--base-config", default="config/large_scale_screening.yaml")
    parser.add_argument("--benchmark-glob", default=BENCHMARK_GLOB)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decision = materialize_calibrated_screening(
        project_root=Path(args.project_root),
        campaign_dir=Path(args.campaign_dir),
        base_config=Path(args.base_config),
        spec_path=Path(args.spec),
        benchmark_glob=args.benchmark_glob,
        output=Path(args.output),
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
