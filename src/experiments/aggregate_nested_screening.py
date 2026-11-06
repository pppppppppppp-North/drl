from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.experiment_manifest import collect_experiment_manifests
from src.experiments.run_screening import screening_algorithms
from src.utils.config import load_config


def aggregate_nested_screening(
    run_dir: Path,
    *,
    config: dict[str, Any],
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = output_dir or run_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    nested = config.get("nested_screening", {})
    algorithms = screening_algorithms({"screening": {"algorithms": nested.get("algorithms", ["ppo", "a2c"])}})
    windows = [int(value) for value in nested.get("window_ids", [0, 1])]
    trials_per_window = int(nested.get("trials_per_algorithm_window", 10))
    objective_metric = str(nested.get("objective_metric", "sharpe"))
    rows = []
    for path in sorted(run_dir.rglob("nested_trial_result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("selection_split") != "validation":
            raise ValueError(f"nested selection result is not validation-only: {path}")
        row = {
            "source_path": str(path.relative_to(run_dir)),
            "trial_id": data["trial_id"],
            "trial_number": data["trial_number"],
            "algorithm": data["algorithm"],
            "window": data["window"],
            "seed": data["seed"],
            "objective_metric": objective_metric,
            "objective_value": data["metrics"][objective_metric],
        }
        row.update({f"metric_{key}": value for key, value in data.get("metrics", {}).items()})
        row.update({f"param_{key}": value for key, value in data.get("sampled_params", {}).items()})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no nested_trial_result.json files found under {run_dir}")
    results = pd.DataFrame(rows)

    expected = pd.DataFrame(
        itertools.product(algorithms, windows, range(trials_per_window)),
        columns=["algorithm", "window", "trial_number"],
    )
    actual = results[["algorithm", "window", "trial_number"]].drop_duplicates()
    completeness = expected.merge(actual.assign(present=True), on=["algorithm", "window", "trial_number"], how="left")
    completeness["present"] = completeness["present"].eq(True)
    if not completeness["present"].all():
        missing = completeness[~completeness["present"]].to_dict("records")
        raise ValueError(f"incomplete nested-screening grid: {missing[:10]}")

    manifests = pd.DataFrame(collect_experiment_manifests(run_dir))
    if not manifests.empty and not manifests["status"].eq("complete").all():
        raise ValueError("nested-screening manifests include non-complete runs")
    selected = (
        results.sort_values(["algorithm", "window", "objective_value"], ascending=[True, True, False])
        .groupby(["algorithm", "window"], as_index=False)
        .first()
        .sort_values(["window", "objective_value"], ascending=[True, False])
        .reset_index(drop=True)
    )
    results.to_csv(output_dir / "nested_validation_trials.csv", index=False)
    selected.to_csv(output_dir / "nested_selected_trials.csv", index=False)
    completeness.to_csv(output_dir / "nested_screening_completeness.csv", index=False)
    if not manifests.empty:
        manifests.to_csv(output_dir / "experiment_manifest.csv", index=False)
    return results, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="config/nested_walk_forward_screening.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results, selected = aggregate_nested_screening(
        Path(args.run_dir),
        config=config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"aggregated {len(results)} validation-only nested trials")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
