from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _flatten_trial_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "trial_id": metadata["trial_id"],
        "trial_number": metadata["trial_number"],
        "algorithm": metadata["algorithm"],
        "seed": metadata["seed"],
        "objective_metric": metadata["objective_metric"],
        "objective_value": metadata["objective_value"],
        "trial_dir": str(path.parent),
    }
    for key, value in metadata.get("metrics", {}).items():
        row[f"metric_{key}"] = value
    for key, value in metadata.get("sampled_params", {}).items():
        row[f"param_{key}"] = value
    return row


def aggregate_trials(run_dir: str | Path, objective_metric: str | None = None) -> pd.DataFrame:
    root = Path(run_dir)
    metadata_paths = sorted(root.glob("trial_*/trial_metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"no trial_metadata.json files found under {root}")

    rows = [_flatten_trial_metadata(path) for path in metadata_paths]
    results = pd.DataFrame(rows)
    metric = objective_metric or str(results.iloc[0]["objective_metric"])
    if metric != str(results.iloc[0]["objective_metric"]):
        metric_column = f"metric_{metric}"
        if metric_column not in results:
            raise KeyError(f"objective metric {metric!r} was not found in aggregated trial metrics")
        results["objective_metric"] = metric
        results["objective_value"] = results[metric_column]
    return results.sort_values(["objective_value", "trial_id"], ascending=[False, True]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--objective-metric", default=None)
    args = parser.parse_args()

    results = aggregate_trials(args.run_dir, objective_metric=args.objective_metric)
    output_path = Path(args.output) if args.output else Path(args.run_dir) / "trial_results.csv"
    results.to_csv(output_path, index=False)
    display_columns = [
        column
        for column in [
            "trial_id",
            "algorithm",
            "objective_value",
            "metric_cumulative_return",
            "metric_max_drawdown",
            "param_learning_rate",
            "param_gamma",
            "param_n_steps",
        ]
        if column in results.columns
    ]
    print(results[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
