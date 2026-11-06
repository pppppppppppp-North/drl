from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.campaigns.calibration import (
    CalibrationInputs,
    CalibrationRun,
    artifact_record,
    atomic_csv,
    atomic_json,
    calibration_inputs,
    code_provenance,
    receipt_path_for,
    runtime_versions,
    training_feature_subset,
)
from src.train import build_vectorized_training_env


CODE_FILES = (
    "scripts/benchmark_vectorized_training.py",
    "src/campaigns/calibration.py",
    "src/train.py",
    "src/envs/trading_env.py",
    "src/data/splits.py",
    "src/utils/config.py",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _peak_rss_mib(who: int) -> float:
    value = float(resource.getrusage(who).ru_maxrss)
    return value / (1024.0 * 1024.0) if platform.system() == "Darwin" else value / 1024.0


def _linux_rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        pass
    return 0


def _linux_descendants(pid: int) -> set[int]:
    descendants: set[int] = set()
    pending = [pid]
    while pending:
        parent = pending.pop()
        try:
            raw = Path(f"/proc/{parent}/task/{parent}/children").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for value in raw.split():
            child = int(value)
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


class _ProcessTreeRssSampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        pid = os.getpid()
        process_ids = {pid, *_linux_descendants(pid)}
        self.peak_bytes = max(self.peak_bytes, sum(_linux_rss_bytes(item) for item in process_ids))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        if Path("/proc/self/status").is_file():
            self._sample()
            self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
            self._thread.start()

    def stop(self) -> tuple[float, str]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._sample()
        if self.peak_bytes:
            return self.peak_bytes / (1024.0 * 1024.0), "sampled_concurrent_process_tree_rss"
        fallback = _peak_rss_mib(resource.RUSAGE_SELF) + _peak_rss_mib(resource.RUSAGE_CHILDREN)
        return fallback, "fresh_process_ru_maxrss_self_plus_children_fallback"


def _trial(
    inputs: CalibrationInputs,
    *,
    num_envs: int,
    timesteps: int,
    backend: str,
    seed: int,
    repeat: int,
) -> dict[str, Any]:
    from stable_baselines3 import PPO

    config = copy.deepcopy(inputs.config)
    training = config.setdefault("training", {})
    training["num_envs"] = int(num_envs)
    training["vectorization_backend"] = backend
    features, split = training_feature_subset(inputs)

    sampler = _ProcessTreeRssSampler()
    sampler.start()
    started = time.perf_counter()
    env = build_vectorized_training_env(config, features, seed)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=32,
            batch_size=32,
            verbose=0,
            seed=seed,
            device="cpu",
        )
        model.learn(total_timesteps=timesteps)
        completed = int(model.num_timesteps)
    finally:
        env.close()
    elapsed = time.perf_counter() - started
    peak_rss, rss_method = sampler.stop()
    code = code_provenance(PROJECT_ROOT, CODE_FILES)
    versions = runtime_versions()
    return {
        "num_envs": int(num_envs),
        "repeat": int(repeat),
        "backend": backend if num_envs > 1 else "monitor",
        "requested_timesteps": int(timesteps),
        "completed_timesteps": completed,
        "elapsed_seconds": elapsed,
        "samples_per_second": completed / elapsed,
        "peak_rss_mib": peak_rss,
        "peak_self_rss_mib": _peak_rss_mib(resource.RUSAGE_SELF),
        "peak_child_rss_mib": _peak_rss_mib(resource.RUSAGE_CHILDREN),
        "rss_measurement": rss_method,
        "seed": int(seed),
        "data_source": inputs.data_source,
        "data_sha256": inputs.data_sha256,
        "config_sha256": inputs.config_sha256,
        "config_file_sha256": inputs.config_file_sha256,
        "code_sha256": code["sha256"],
        "feature_rows": int(len(features)),
        "feature_dates": int(pd.to_datetime(features["date"]).nunique()),
        "feature_tickers": int(features["ticker"].nunique()),
        "training_split": split,
        "hostname": socket.gethostname(),
        "process_id": os.getpid(),
        "python_version": versions["python"],
        "numpy_version": versions["numpy"],
        "pandas_version": versions["pandas"],
        "gymnasium_version": versions["gymnasium"],
        "stable_baselines3_version": versions["stable_baselines3"],
        "torch_version": versions["torch"],
    }


def trial_command(
    *,
    config_path: str | Path | None,
    synthetic: bool,
    num_envs: int,
    timesteps: int,
    backend: str,
    seed: int,
    repeat: int,
    output: str | Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark_vectorized_training",
        "--trial-worker",
        "--num-envs",
        str(num_envs),
        "--timesteps",
        str(timesteps),
        "--backend",
        backend,
        "--seed",
        str(seed),
        "--repeat",
        str(repeat),
        "--trial-output",
        str(Path(output).resolve()),
    ]
    if synthetic:
        command.append("--synthetic")
    elif config_path is not None:
        command.extend(["--config", str(Path(config_path).resolve())])
    return command


def _run_trial_subprocess(**kwargs: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="vector-calibration-") as temporary_dir:
        output = Path(temporary_dir) / "trial.json"
        command = trial_command(output=output, **kwargs)
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "worker exited without diagnostics"
            raise RuntimeError(f"isolated vector benchmark trial failed: {detail[-2000:]}")
        return json.loads(output.read_text(encoding="utf-8"))


TrialRunner = Callable[..., dict[str, Any]]


def benchmark_vector_counts(
    env_counts: list[int],
    *,
    timesteps: int,
    backend: str,
    seed: int,
    repeats: int = 3,
    config_path: str | Path | None = None,
    synthetic: bool = False,
    trial_runner: TrialRunner | None = None,
) -> pd.DataFrame:
    if bool(config_path) == bool(synthetic):
        raise ValueError("choose exactly one input mode: config_path or synthetic=True")
    if not env_counts or any(value <= 0 for value in env_counts):
        raise ValueError("env_counts must contain positive integers")
    if len(set(env_counts)) != len(env_counts):
        raise ValueError("env_counts must not contain duplicates")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    runner = trial_runner or _run_trial_subprocess
    rows = []
    for num_envs in env_counts:
        for repeat in range(1, repeats + 1):
            row = runner(
                config_path=config_path,
                synthetic=synthetic,
                num_envs=num_envs,
                timesteps=timesteps,
                backend=backend,
                seed=seed,
                repeat=repeat,
            )
            if int(row.get("num_envs", 0)) != num_envs or int(row.get("repeat", 0)) != repeat:
                raise RuntimeError("isolated trial returned the wrong environment count or repeat index")
            rows.append(row)
    return pd.DataFrame(rows)


def throughput_summary(results: pd.DataFrame) -> pd.DataFrame:
    required = {"num_envs", "samples_per_second", "peak_rss_mib"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"benchmark results are missing columns: {missing}")
    throughputs = pd.to_numeric(results["samples_per_second"], errors="raise")
    memory = pd.to_numeric(results["peak_rss_mib"], errors="raise")
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in throughputs):
        raise ValueError("benchmark throughput values must be finite and positive")
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in memory):
        raise ValueError("benchmark memory values must be finite and positive")
    return (
        results.groupby("num_envs", as_index=False)
        .agg(
            repeats=("samples_per_second", "size"),
            median_samples_per_second=("samples_per_second", "median"),
            min_samples_per_second=("samples_per_second", "min"),
            max_samples_per_second=("samples_per_second", "max"),
            max_peak_rss_mib=("peak_rss_mib", "max"),
        )
        .sort_values("num_envs")
        .reset_index(drop=True)
    )


def select_num_envs(results: pd.DataFrame, fraction_of_max: float = 0.95) -> tuple[int, pd.DataFrame]:
    if not 0.0 < fraction_of_max <= 1.0:
        raise ValueError("fraction_of_max must be in (0, 1]")
    summary = throughput_summary(results)
    maximum = float(summary["median_samples_per_second"].max())
    threshold = maximum * fraction_of_max
    eligible = summary[summary["median_samples_per_second"] >= threshold]
    return int(eligible["num_envs"].min()), summary


def _metadata(results: pd.DataFrame, inputs: CalibrationInputs, repeats: int) -> dict[str, Any]:
    for column, expected in (
        ("data_sha256", inputs.data_sha256),
        ("config_sha256", inputs.config_sha256),
    ):
        values = set(results[column].astype(str))
        if values != {expected}:
            raise RuntimeError(f"isolated trials disagree on {column}: {sorted(values)}")
    code_hashes = set(results["code_sha256"].astype(str))
    if len(code_hashes) != 1:
        raise RuntimeError("code changed between isolated benchmark trials")
    selected, summary = select_num_envs(results)
    code = code_provenance(PROJECT_ROOT, CODE_FILES)
    if code_hashes != {code["sha256"]}:
        raise RuntimeError("code changed after the isolated benchmark trials completed")
    return {
        "schema_version": 2,
        "selected_num_envs": selected,
        "selection_rule": "smallest num_envs with median throughput at least 95% of maximum median throughput",
        "selection_fraction_of_max": 0.95,
        "throughput_statistic": "median_samples_per_second",
        "trial_isolation": "one fresh Python subprocess per environment count and repeat",
        "repeats_per_env_count": repeats,
        "trial_count": int(len(results)),
        "data_provenance": {
            "source": inputs.data_source,
            "synthetic": inputs.synthetic,
            "feature_path": str(inputs.feature_path) if inputs.feature_path else None,
            "data_sha256": inputs.data_sha256,
            "config_path": str(inputs.config_path) if inputs.config_path else None,
            "config_sha256": inputs.config_sha256,
            "config_file_sha256": inputs.config_file_sha256,
        },
        "code_provenance": code,
        "runtime": runtime_versions(),
        "summary_by_env_count": summary.to_dict(orient="records"),
    }


def _worker_main(args: argparse.Namespace) -> None:
    if args.num_envs is None or args.trial_output is None:
        raise ValueError("trial worker requires --num-envs and --trial-output")
    inputs = calibration_inputs(config_path=args.config, synthetic=args.synthetic, seed=args.seed)
    result = _trial(
        inputs,
        num_envs=args.num_envs,
        timesteps=args.timesteps,
        backend=args.backend,
        seed=args.seed,
        repeat=args.repeat,
    )
    atomic_json(args.trial_output, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="observed-data experiment config with an existing feature table")
    parser.add_argument("--synthetic", action="store_true", help="explicit unit-only synthetic input")
    parser.add_argument("--env-counts", default="1,2,4,8,16")
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/vectorized_training_benchmark.csv")
    parser.add_argument("--trial-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-envs", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--trial-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.trial_worker:
        _worker_main(args)
        return

    env_counts = [int(value) for value in args.env_counts.split(",") if value.strip()]
    output = Path(args.output)
    metadata_path = output.with_suffix(".json")
    initial_inputs = [artifact_record("config_candidate", args.config)] if args.config else []
    run = CalibrationRun(
        kind="vectorized_training_benchmark",
        receipt_path=receipt_path_for(output),
        parameters={
            "env_counts": env_counts,
            "timesteps": args.timesteps,
            "repeats": args.repeats,
            "backend": args.backend,
            "seed": args.seed,
            "synthetic": args.synthetic,
        },
        inputs=initial_inputs,
    )
    with run:
        inputs = calibration_inputs(config_path=args.config, synthetic=args.synthetic, seed=args.seed)
        for record in inputs.receipt_inputs():
            run.add_input(record)
        for relative_path in CODE_FILES:
            run.add_input(artifact_record(f"code:{relative_path}", PROJECT_ROOT / relative_path))
        results = benchmark_vector_counts(
            env_counts,
            timesteps=args.timesteps,
            backend=args.backend,
            seed=args.seed,
            repeats=args.repeats,
            config_path=args.config,
            synthetic=args.synthetic,
        )
        metadata = _metadata(results, inputs, args.repeats)
        metadata["results_file"] = str(output.resolve())
        atomic_csv(results, output)
        atomic_json(metadata_path, metadata)
        run.add_output("benchmark_csv", output)
        run.add_output("benchmark_metadata", metadata_path)
        run.succeed(
            {
                "selected_num_envs": metadata["selected_num_envs"],
                "trial_count": metadata["trial_count"],
                "data_sha256": inputs.data_sha256,
                "config_sha256": inputs.config_sha256,
                "code_sha256": metadata["code_provenance"]["sha256"],
            }
        )
    print(results.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
