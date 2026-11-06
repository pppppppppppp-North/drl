from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.baselines import run_baselines_on_features
from src.data.splits import (
    generate_walk_forward_splits,
    select_chronological_split,
    split_config_from_mapping,
    walk_forward_config_from_mapping,
)
from src.envs.trading_env import load_feature_table
from src.evaluation.metrics import equity_metrics
from src.experiments.experiment_manifest import tracked_experiment
from src.train import _stable_baselines_algorithm, ensure_feature_inputs, evaluate_model, make_trading_env_from_features
from src.utils.config import ensure_dirs, load_config


def execution_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = [dict(item) for item in config.get("execution_stress", {}).get("scenarios", [])]
    if not scenarios:
        raise ValueError("execution_stress.scenarios must not be empty")
    names = [str(item.get("name", "")) for item in scenarios]
    if any(not name for name in names):
        raise ValueError("every execution scenario needs a name")
    if len(names) != len(set(names)):
        raise ValueError("execution scenario names must be unique")
    return scenarios


def apply_execution_scenario(config: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    scenario_config = copy.deepcopy(config)
    environment = scenario_config.setdefault("environment", {})
    for key in ("transaction_cost", "slippage", "max_turnover", "max_position_weight", "min_cash_weight"):
        if key in scenario:
            environment[key] = scenario[key]
    return scenario_config


def _plot_execution_sensitivity(metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = sorted(metrics["policy"].unique())
    scenarios = list(dict.fromkeys(metrics["scenario"]))
    plot_metrics = metrics.groupby(["scenario", "policy"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric in zip(axes, ("cumulative_return", "sharpe", "max_drawdown")):
        for policy in policies:
            part = plot_metrics[plot_metrics["policy"].eq(policy)].set_index("scenario").reindex(scenarios)
            axis.plot(scenarios, part[metric], marker="o", label=policy)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(metric.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=45)
        axis.grid(True, axis="y", alpha=0.25)
    axes[-1].legend(loc="best", fontsize=7)
    fig.suptitle("Execution-Cost and Constraint Sensitivity (Locked Walk-Forward Test Windows)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def _resolved_feature_path(config: dict[str, Any]) -> Path:
    primary = Path(str(config["paths"]["features"]))
    if primary.is_file():
        return primary
    fallback = Path(str(config["paths"].get("fallback_features_csv", primary.with_suffix(".csv"))))
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(primary)


def _feature_schema_hash(config: dict[str, Any]) -> str:
    schema = {
        "tickers": config.get("data", {}).get("tickers", []),
        "lookback": config.get("features", {}).get("lookback"),
        "columns": config.get("features", {}).get("columns", []),
    }
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def execution_evaluation_slices(
    config: dict[str, Any],
    features: pd.DataFrame,
    model_specs: list[dict[str, Any]],
) -> dict[int | str, pd.DataFrame]:
    configured_windows = [spec.get("window") for spec in model_specs]
    if configured_windows and any(value is None for value in configured_windows):
        raise ValueError("every locked stress-test model must declare a walk-forward window")
    if configured_windows:
        if "walk_forward" not in config:
            raise ValueError("window-aligned stress testing requires the locked walk_forward configuration")
        windows = generate_walk_forward_splits(features, walk_forward_config_from_mapping(config.get("walk_forward")))
        requested = sorted({int(value) for value in configured_windows})
        invalid = [value for value in requested if value < 0 or value >= len(windows)]
        if invalid:
            raise ValueError(f"stress-test model window ids are out of range: {invalid}")
        return {window: windows[window]["test"] for window in requested}
    return {
        "chronological_test": select_chronological_split(
            features,
            "test",
            split_config_from_mapping(config.get("splits")),
        )
    }


def run_execution_stress(config: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_feature_inputs(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = _resolved_feature_path(config)
    actual_feature_data_hash = _sha256(feature_path)
    features = load_feature_table(config["paths"]["features"], config["paths"].get("fallback_features_csv"))
    stress_cfg = config.get("execution_stress", {})
    seed = int(config.get("project", {}).get("seed", 42))
    model_specs = [dict(value) for value in stress_cfg.get("model_paths", [])]
    require_models = bool(stress_cfg.get("require_models", False))
    configured_protocol = str(stress_cfg.get("protocol_lock_hash", ""))
    runtime_protocol = str(os.environ.get("PROTOCOL_LOCK_HASH", ""))
    if require_models and not model_specs:
        raise ValueError("locked execution stress requires at least one registered model")
    if require_models and not _is_sha256(configured_protocol):
        raise ValueError("locked execution stress requires a 64-character protocol_lock_hash")
    if require_models and not _is_sha256(runtime_protocol):
        raise ValueError("locked execution stress requires PROTOCOL_LOCK_HASH at runtime")
    if runtime_protocol and configured_protocol != runtime_protocol:
        raise ValueError("runtime protocol lock does not match the locked execution-stress configuration")
    evaluation_slices = execution_evaluation_slices(config, features, model_specs)
    expected_schema_hash = _feature_schema_hash(config)
    readiness_rows = []
    metric_rows = []
    curve_frames = []
    action_frames = []
    completeness_rows = []
    replay_rows = []
    scenarios = execution_scenarios(config)
    if sum(str(item["name"]) == "base" for item in scenarios) != 1:
        raise ValueError("execution stress requires exactly one base scenario for replay verification")
    replay_tolerance = float(stress_cfg.get("base_replay_tolerance", 1e-6))
    if replay_tolerance < 0:
        raise ValueError("execution_stress.base_replay_tolerance must be nonnegative")

    identities = [
        (str(spec.get("policy", "")), int(spec.get("window", -1)), int(spec.get("seed", -1)))
        for spec in model_specs
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("execution_stress.model_paths contains duplicate policy/window/seed identities")
    prepared_models = []
    for spec in model_specs:
        path = Path(str(spec["path"]))
        policy = str(spec.get("policy", spec.get("algo", "model")))
        algo = str(spec.get("algo", policy)).lower()
        window = int(spec["window"]) if spec.get("window") is not None else "chronological_test"
        model_seed = int(spec.get("seed", seed))
        expected_hash = str(spec.get("sha256", ""))
        actual_hash = _sha256(path) if path.is_file() else ""
        hash_matches = _is_sha256(expected_hash) and actual_hash == expected_hash
        declared_schema_hash = str(spec.get("feature_schema_sha256", ""))
        schema_matches = _is_sha256(declared_schema_hash) and declared_schema_hash == expected_schema_hash
        declared_data_hash = str(spec.get("feature_data_sha256", ""))
        data_matches = _is_sha256(declared_data_hash) and declared_data_hash == actual_feature_data_hash
        declared_protocol = str(spec.get("protocol_lock_hash", ""))
        protocol_matches = _is_sha256(declared_protocol) and declared_protocol == configured_protocol
        source_path = Path(str(spec.get("source_test_curve_path", "")))
        expected_source_hash = str(spec.get("source_test_curve_sha256", ""))
        source_curve_matches = (
            source_path.is_file()
            and _is_sha256(expected_source_hash)
            and _sha256(source_path) == expected_source_hash
        )
        ready = (
            path.is_file()
            and hash_matches
            and schema_matches
            and data_matches
            and protocol_matches
            and source_curve_matches
        )
        readiness_rows.append(
            {
                "policy": policy,
                "algorithm": algo,
                "window": window,
                "seed": model_seed,
                "path": str(path),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "hash_matches": hash_matches,
                "feature_schema_matches": schema_matches,
                "feature_data_matches": data_matches,
                "protocol_matches": protocol_matches,
                "source_curve_matches": source_curve_matches,
                "ready": ready,
            }
        )
        if not ready:
            if require_models:
                raise ValueError(f"required stress-test model failed registry verification: {path}")
            continue
        model_cls = _stable_baselines_algorithm(algo)
        if model_cls is None:
            raise ImportError(f"stable-baselines3 algorithm is unavailable: {algo}")
        prepared_models.append((spec, model_cls.load(path)))

    for scenario in scenarios:
        scenario_name = str(scenario["name"])
        scenario_config = apply_execution_scenario(config, scenario)
        for window, test_features in evaluation_slices.items():
            curves, baseline_metrics = run_baselines_on_features(scenario_config, test_features, seed=seed)
            curves.insert(0, "window", window)
            curves.insert(0, "scenario", scenario_name)
            baseline_metrics.insert(0, "window", window)
            baseline_metrics.insert(0, "scenario", scenario_name)
            curve_frames.append(curves)
            metric_rows.extend(baseline_metrics.to_dict("records"))

            for spec, model in prepared_models:
                model_window = int(spec["window"]) if spec.get("window") is not None else "chronological_test"
                if model_window != window:
                    continue
                policy = str(spec.get("policy", spec.get("algo", "model")))
                model_seed = int(spec.get("seed", seed))
                curve, actions = evaluate_model(
                    model,
                    make_trading_env_from_features(scenario_config, test_features),
                    model_seed,
                    policy_name=policy,
                    return_actions=True,
                )
                curve.insert(0, "seed", model_seed)
                curve.insert(0, "window", window)
                curve.insert(0, "scenario", scenario_name)
                curve_frames.append(curve)
                actions.insert(0, "model_sha256", str(spec.get("sha256", "")))
                actions.insert(0, "seed", model_seed)
                actions.insert(0, "window", window)
                actions.insert(0, "scenario", scenario_name)
                action_frames.append(actions)
                row = {
                    "scenario": scenario_name,
                    "window": window,
                    "seed": model_seed,
                    "algorithm": str(spec.get("algo", "")).lower(),
                    "model_id": str(spec.get("model_id", policy)),
                    "model_sha256": str(spec.get("sha256", "")),
                    "policy": policy,
                }
                row.update(equity_metrics(curve))
                metric_rows.append(row)
                completeness_rows.append(
                    {
                        "model_id": str(spec.get("model_id", policy)),
                        "algorithm": str(spec.get("algo", "")).lower(),
                        "window": window,
                        "seed": model_seed,
                        "scenario": scenario_name,
                        "present": True,
                    }
                )
                if scenario_name == "base":
                    source_path = Path(str(spec["source_test_curve_path"]))
                    if not source_path.is_file() or _sha256(source_path) != str(
                        spec.get("source_test_curve_sha256", "")
                    ):
                        raise ValueError(f"source robustness curve failed hash verification: {source_path}")
                    source_curve = pd.read_csv(source_path)
                    source_dates = pd.to_datetime(source_curve["date"]).reset_index(drop=True)
                    replay_dates = pd.to_datetime(curve["date"]).reset_index(drop=True)
                    dates_match = source_dates.equals(replay_dates)
                    if len(source_curve) == len(curve):
                        max_difference = float(
                            np.max(
                                np.abs(
                                    pd.to_numeric(source_curve["portfolio_value"]).to_numpy(dtype=float)
                                    - pd.to_numeric(curve["portfolio_value"]).to_numpy(dtype=float)
                                )
                            )
                        )
                    else:
                        max_difference = float("inf")
                    curve_match = dates_match and max_difference <= replay_tolerance
                    replay_rows.append(
                        {
                            "model_id": str(spec.get("model_id", policy)),
                            "algorithm": str(spec.get("algo", "")).lower(),
                            "window": window,
                            "seed": model_seed,
                            "dates_match": dates_match,
                            "max_abs_portfolio_difference": max_difference,
                            "tolerance": replay_tolerance,
                            "curve_match": curve_match,
                        }
                    )
                    if require_models and not curve_match:
                        raise ValueError(
                            f"base stress replay does not reproduce the robustness curve for {policy}"
                        )

    if require_models:
        expected_completeness = {
            (
                str(spec.get("model_id", spec.get("policy", spec.get("algo", "model")))),
                str(spec.get("algo", "")).lower(),
                int(spec["window"]),
                int(spec.get("seed", seed)),
                str(scenario["name"]),
            )
            for spec in model_specs
            for scenario in scenarios
        }
        actual_completeness = {
            (row["model_id"], row["algorithm"], row["window"], row["seed"], row["scenario"])
            for row in completeness_rows
        }
        if actual_completeness != expected_completeness:
            raise ValueError("execution stress did not complete the exact registered model-window-scenario grid")
        expected_replays = {
            (
                str(spec.get("model_id", spec.get("policy", spec.get("algo", "model")))),
                int(spec["window"]),
                int(spec.get("seed", seed)),
            )
            for spec in model_specs
        }
        actual_replays = {(row["model_id"], row["window"], row["seed"]) for row in replay_rows}
        if actual_replays != expected_replays or not all(row["curve_match"] for row in replay_rows):
            raise ValueError("execution stress lacks one successful base replay per registered model")

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["scenario", "window", "sharpe"], ascending=[True, True, False]
    )
    readiness = pd.DataFrame(
        readiness_rows,
        columns=[
            "policy",
            "algorithm",
            "window",
            "seed",
            "path",
            "expected_sha256",
            "actual_sha256",
            "hash_matches",
            "feature_schema_matches",
            "feature_data_matches",
            "protocol_matches",
            "source_curve_matches",
            "ready",
        ],
    )
    metrics.to_csv(output_dir / "execution_stress_metrics.csv", index=False)
    pd.concat(curve_frames, ignore_index=True).to_csv(output_dir / "execution_stress_equity_curves.csv", index=False)
    readiness.to_csv(output_dir / "execution_stress_model_readiness.csv", index=False)
    pd.DataFrame(completeness_rows).to_csv(output_dir / "execution_stress_completeness.csv", index=False)
    pd.DataFrame(replay_rows).to_csv(output_dir / "execution_stress_base_replay.csv", index=False)
    if action_frames:
        pd.concat(action_frames, ignore_index=True).to_csv(output_dir / "execution_stress_actions.csv", index=False)
    else:
        pd.DataFrame(columns=["scenario", "window", "seed", "model_sha256", "date", "policy", "ticker", "weight"]).to_csv(
            output_dir / "execution_stress_actions.csv", index=False
        )
    window_rows = []
    for window, part in evaluation_slices.items():
        dates = pd.to_datetime(part["date"])
        window_rows.append(
            {
                "window": window,
                "start_date": dates.min(),
                "end_date": dates.max(),
                "rows": len(part),
                "unique_dates": dates.nunique(),
                "tickers": part["ticker"].nunique(),
            }
        )
    pd.DataFrame(window_rows).to_csv(output_dir / "execution_stress_window_summary.csv", index=False)
    _plot_execution_sensitivity(metrics, output_dir / "execution_stress_sensitivity.png")
    return metrics, readiness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/execution_stress_real_ohlcv_market.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config["paths"]["results_dir"])
    ensure_dirs(output_dir)
    identity = {
        "experiment_id": output_dir.name,
        "algorithm": "locked_models",
        "window": "locked_walk_forward_test_windows",
        "seed": int(config.get("project", {}).get("seed", 42)),
        "feature_group": config.get("project", {}).get("name"),
        "cost_scenario": "execution_grid",
        "claim_split": "test",
    }
    with tracked_experiment(output_dir, config=config, identity=identity):
        metrics, readiness = run_execution_stress(config, output_dir)
    print(readiness.to_string(index=False))
    print(metrics[["scenario", "policy", "cumulative_return", "sharpe", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
