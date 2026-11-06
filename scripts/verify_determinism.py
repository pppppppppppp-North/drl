from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.campaigns.calibration import (
    CalibrationInputs,
    CalibrationRun,
    artifact_record,
    atomic_json,
    calibration_inputs,
    code_provenance,
    receipt_path_for,
    runtime_versions,
    training_feature_subset,
)
from src.train import make_trading_env_from_features


CODE_FILES = (
    "scripts/verify_determinism.py",
    "src/campaigns/calibration.py",
    "src/train.py",
    "src/envs/trading_env.py",
    "src/data/splits.py",
    "src/utils/config.py",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _update_array(digest: Any, label: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())


def _trajectory(inputs: CalibrationInputs, seed: int) -> dict[str, Any]:
    features, split = training_feature_subset(inputs)
    env = make_trading_env_from_features(inputs.config, features)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    observation, info = env.reset(seed=seed)
    _update_array(digest, "initial_observation", observation)
    _update_array(digest, "initial_portfolio", [info["portfolio_value"]])
    steps = 0
    done = False
    try:
        while not done:
            action = rng.normal(size=env.action_space.shape).astype(np.float32)
            _update_array(digest, "action", action)
            observation, reward, terminated, truncated, info = env.step(action)
            _update_array(digest, "observation", observation)
            _update_array(
                digest,
                "transition",
                [reward, info["portfolio_value"], info["turnover"], terminated, truncated],
            )
            steps += 1
            done = terminated or truncated
    finally:
        env.close()
    code = code_provenance(PROJECT_ROOT, CODE_FILES)
    return {
        "trajectory_sha256": digest.hexdigest(),
        "steps": steps,
        "seed": int(seed),
        "process_id": os.getpid(),
        "hostname": socket.gethostname(),
        "training_split": split,
        "feature_rows": int(len(features)),
        "feature_dates": int(np.asarray(features["date"].unique()).size),
        "feature_tickers": int(features["ticker"].nunique()),
        "data_source": inputs.data_source,
        "data_sha256": inputs.data_sha256,
        "config_sha256": inputs.config_sha256,
        "config_file_sha256": inputs.config_file_sha256,
        "code_sha256": code["sha256"],
        "runtime": runtime_versions(),
    }


def trajectory_command(
    *,
    config_path: str | Path | None,
    synthetic: bool,
    seed: int,
    output: str | Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.verify_determinism",
        "--trajectory-worker",
        "--seed",
        str(seed),
        "--trajectory-output",
        str(Path(output).resolve()),
    ]
    if synthetic:
        command.append("--synthetic")
    elif config_path is not None:
        command.extend(["--config", str(Path(config_path).resolve())])
    return command


def _trajectory_subprocess(
    *, config_path: str | Path | None, synthetic: bool, seed: int, output: str | Path
) -> dict[str, Any]:
    command = trajectory_command(
        config_path=config_path,
        synthetic=synthetic,
        seed=seed,
        output=output,
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "worker exited without diagnostics"
        raise RuntimeError(f"independent determinism worker failed: {detail[-2000:]}")
    return json.loads(Path(output).read_text(encoding="utf-8"))


def run_determinism(
    seed: int,
    *,
    config_path: str | Path | None = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    if bool(config_path) == bool(synthetic):
        raise ValueError("choose exactly one input mode: config_path or synthetic=True")
    with tempfile.TemporaryDirectory(prefix="determinism-calibration-") as temporary_dir:
        root = Path(temporary_dir)
        first = _trajectory_subprocess(
            config_path=config_path,
            synthetic=synthetic,
            seed=seed,
            output=root / "first.json",
        )
        second = _trajectory_subprocess(
            config_path=config_path,
            synthetic=synthetic,
            seed=seed,
            output=root / "second.json",
        )
    provenance_keys = ("data_sha256", "config_sha256", "config_file_sha256", "code_sha256")
    provenance_matches = all(first.get(key) == second.get(key) for key in provenance_keys)
    runtime_matches = first.get("runtime") == second.get("runtime")
    deterministic = (
        first["trajectory_sha256"] == second["trajectory_sha256"]
        and int(first["steps"]) == int(second["steps"])
        and provenance_matches
        and runtime_matches
    )
    return {
        "schema_version": 2,
        "seed": int(seed),
        "first_sha256": first["trajectory_sha256"],
        "second_sha256": second["trajectory_sha256"],
        "steps": int(first["steps"]),
        "first_steps": int(first["steps"]),
        "second_steps": int(second["steps"]),
        "deterministic": deterministic,
        "comparison_method": "two_independent_subprocess_random_action_trajectories",
        "random_action_distribution": "numpy.default_rng(seed).normal",
        "independent_subprocess_runs": 2,
        "process_ids": [first["process_id"], second["process_id"]],
        "data_provenance": {
            "source": first["data_source"],
            "data_sha256": first["data_sha256"],
            "config_sha256": first["config_sha256"],
            "config_file_sha256": first["config_file_sha256"],
            "code_sha256": first["code_sha256"],
            "training_split": first["training_split"],
            "feature_rows": first["feature_rows"],
            "feature_dates": first["feature_dates"],
            "feature_tickers": first["feature_tickers"],
            "synthetic": synthetic,
        },
        "runtime": first["runtime"],
        "run_provenance": [first, second],
        "provenance_matches": provenance_matches,
        "runtime_matches": runtime_matches,
    }


def verify(seed: int) -> dict[str, Any]:
    """Compatibility helper: an explicit unit fixture exercised in fresh processes."""
    return run_determinism(seed, synthetic=True)


def _worker_main(args: argparse.Namespace) -> None:
    if args.trajectory_output is None:
        raise ValueError("trajectory worker requires --trajectory-output")
    inputs = calibration_inputs(config_path=args.config, synthetic=args.synthetic, seed=args.seed)
    atomic_json(args.trajectory_output, _trajectory(inputs, args.seed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="observed-data experiment config with an existing feature table")
    parser.add_argument("--synthetic", action="store_true", help="explicit unit-only synthetic input")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/determinism_verification.json")
    parser.add_argument("--trajectory-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trajectory-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.trajectory_worker:
        _worker_main(args)
        return

    output = Path(args.output)
    initial_inputs = [artifact_record("config_candidate", args.config)] if args.config else []
    run = CalibrationRun(
        kind="fixed_seed_determinism",
        receipt_path=receipt_path_for(output),
        parameters={"seed": args.seed, "synthetic": args.synthetic},
        inputs=initial_inputs,
    )
    with run:
        inputs = calibration_inputs(config_path=args.config, synthetic=args.synthetic, seed=args.seed)
        for record in inputs.receipt_inputs():
            run.add_input(record)
        for relative_path in CODE_FILES:
            run.add_input(artifact_record(f"code:{relative_path}", PROJECT_ROOT / relative_path))
        result = run_determinism(args.seed, config_path=args.config, synthetic=args.synthetic)
        result["config_path"] = str(inputs.config_path) if inputs.config_path else None
        result["feature_path"] = str(inputs.feature_path) if inputs.feature_path else None
        atomic_json(output, result)
        run.add_output("determinism_result", output)
        if not result["deterministic"]:
            raise RuntimeError("fixed-seed environment trajectory was not deterministic")
        run.succeed(
            {
                "deterministic": result["deterministic"],
                "trajectory_sha256": result["first_sha256"],
                "data_sha256": inputs.data_sha256,
                "config_sha256": inputs.config_sha256,
                "code_sha256": result["data_provenance"]["code_sha256"],
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
