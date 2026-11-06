from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.envs.trading_env import load_feature_table
from src.utils.config import load_config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_record(name: str, path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    record: dict[str, Any] = {"name": name, "path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        record.update({"sha256": sha256_file(resolved), "bytes": resolved.stat().st_size})
    return record


def virtual_input_record(name: str, identifier: str, sha256: str) -> dict[str, Any]:
    return {"name": name, "path": identifier, "exists": True, "sha256": sha256, "bytes": None}


def runtime_versions() -> dict[str, str | None]:
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "gymnasium": "gymnasium",
        "stable_baselines3": "stable-baselines3",
        "torch": "torch",
    }
    versions: dict[str, str | None] = {}
    for key, distribution in packages.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        **versions,
    }


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "src").is_dir() and (candidate / "config").is_dir():
            return candidate
    if resolved.parent.name == "config":
        return resolved.parent.parent
    return resolved.parent


def _resolve_project_path(raw_path: str | Path, project_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


@dataclass(frozen=True)
class CalibrationInputs:
    config: dict[str, Any]
    features: pd.DataFrame
    config_path: Path | None
    feature_path: Path | None
    project_root: Path
    data_source: str
    config_sha256: str
    config_file_sha256: str
    data_sha256: str
    synthetic: bool

    def receipt_inputs(self) -> list[dict[str, Any]]:
        if self.synthetic:
            return [
                virtual_input_record("resolved_config", "synthetic://unit-config", self.config_sha256),
                virtual_input_record("feature_table", "synthetic://unit-features", self.data_sha256),
            ]
        assert self.config_path is not None and self.feature_path is not None
        return [
            artifact_record("config_file", self.config_path),
            virtual_input_record("resolved_config", str(self.config_path), self.config_sha256),
            artifact_record("feature_table", self.feature_path),
        ]


def dataframe_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    order = [column for column in ("date", "ticker") if column in normalized.columns]
    if order:
        normalized = normalized.sort_values(order).reset_index(drop=True)
    return hashlib.sha256(normalized.to_csv(index=False).encode("utf-8")).hexdigest()


def load_observed_inputs(config_path: str | Path) -> CalibrationInputs:
    resolved_config = Path(config_path).expanduser().resolve()
    if not resolved_config.is_file():
        raise FileNotFoundError(f"calibration config does not exist: {resolved_config}")
    config = load_config(resolved_config)
    source = str(config.get("data", {}).get("source", "synthetic")).strip().lower()
    if source in {"", "synthetic", "generated"}:
        raise ValueError(
            "observed calibration requires a non-synthetic data.source; "
            "use the explicit --synthetic mode only for unit checks"
        )
    paths = config.get("paths")
    if not isinstance(paths, dict) or not paths.get("features"):
        raise ValueError("calibration config must declare paths.features")
    project_root = _project_root(resolved_config)
    primary = _resolve_project_path(paths["features"], project_root)
    fallback_raw = paths.get("fallback_features_csv")
    fallback = _resolve_project_path(fallback_raw, project_root) if fallback_raw else None
    if primary.is_file():
        feature_path = primary
    elif fallback is not None and fallback.is_file():
        feature_path = fallback
    else:
        declared = [str(primary), *(str(fallback) for _ in [0] if fallback is not None)]
        raise FileNotFoundError(
            "observed calibration will not synthesize missing inputs; no declared feature table exists: "
            + ", ".join(declared)
        )
    features = load_feature_table(feature_path)
    required = {"date", "ticker", "close", *config.get("features", {}).get("columns", [])}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"feature table is missing configured columns: {missing}")
    return CalibrationInputs(
        config=config,
        features=features,
        config_path=resolved_config,
        feature_path=feature_path,
        project_root=project_root,
        data_source=source,
        config_sha256=canonical_sha256(config),
        config_file_sha256=sha256_file(resolved_config),
        data_sha256=sha256_file(feature_path),
        synthetic=False,
    )


def make_synthetic_calibration_inputs(seed: int) -> CalibrationInputs:
    """Create the explicit, unit-only calibration fixture."""
    from src.data.synthetic import make_synthetic_prices
    from src.features.technical import add_technical_features

    config: dict[str, Any] = {
        "project": {"name": "calibration_unit_fixture", "seed": int(seed)},
        "data": {
            "source": "synthetic",
            "tickers": ["PTT.BK", "AOT.BK", "CPALL.BK", "KBANK.BK", "ADVANC.BK"],
            "start": "2021-01-01",
            "end": "2023-12-31",
            "initial_price": 100.0,
        },
        "features": {
            "lookback": 5,
            "columns": ["return_1d", "return_5d", "volatility_20d", "ma_ratio_10", "rsi_14"],
        },
        "environment": {
            "initial_cash": 1_000_000.0,
            "transaction_cost": 0.001,
            "reward": {},
        },
        "training": {
            "env_window_dates": 100,
            "vectorization_backend": "dummy",
            "subproc_start_method": "forkserver",
        },
    }
    features = add_technical_features(make_synthetic_prices(config))
    config_hash = canonical_sha256(config)
    return CalibrationInputs(
        config=config,
        features=features,
        config_path=None,
        feature_path=None,
        project_root=Path(__file__).resolve().parents[2],
        data_source="synthetic_unit_fixture",
        config_sha256=config_hash,
        config_file_sha256=config_hash,
        data_sha256=dataframe_sha256(features),
        synthetic=True,
    )


def calibration_inputs(
    *, config_path: str | Path | None, synthetic: bool, seed: int
) -> CalibrationInputs:
    if bool(config_path) == bool(synthetic):
        raise ValueError("choose exactly one input mode: --config <observed-config> or --synthetic")
    if synthetic:
        return make_synthetic_calibration_inputs(seed)
    assert config_path is not None
    return load_observed_inputs(config_path)


def training_feature_subset(inputs: CalibrationInputs) -> tuple[pd.DataFrame, str]:
    config = inputs.config
    if "splits" not in config:
        return inputs.features.copy(), "all"
    from src.data.splits import select_chronological_split, split_config_from_mapping

    split = str(config.get("training", {}).get("train_split", config["splits"].get("train_split", "train")))
    selected = select_chronological_split(
        inputs.features,
        split,
        split_config_from_mapping(config.get("splits")),
    )
    return selected, split


def code_provenance(project_root: str | Path, relative_paths: Iterable[str | Path]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    files = []
    for relative in relative_paths:
        path = Path(relative)
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if path.is_file():
            try:
                label = str(path.relative_to(root))
            except ValueError:
                label = str(path)
            files.append({"path": label, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    files.sort(key=lambda item: item["path"])
    return {"sha256": canonical_sha256(files), "files": files}


def receipt_path_for(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(f"{output.name}.receipt.json")


class CalibrationRun:
    """Write one atomic, scheduler-bound receipt for a calibration attempt."""

    def __init__(
        self,
        *,
        kind: str,
        receipt_path: str | Path,
        parameters: dict[str, Any],
        inputs: Iterable[dict[str, Any]] = (),
        runtime: dict[str, Any] | None = None,
    ) -> None:
        self.kind = kind
        self.receipt_path = Path(receipt_path)
        self.parameters = dict(parameters)
        self.inputs = list(inputs)
        self.runtime = dict(runtime or runtime_versions())
        self.outputs: list[dict[str, Any]] = []
        self.result: dict[str, Any] = {}
        self.started_at = utc_now()
        self.started_monotonic = time.monotonic()
        self.campaign_attempt_id = os.environ.get("CAMPAIGN_ATTEMPT_ID")
        self.campaign_attempt_sha256 = os.environ.get("CAMPAIGN_ATTEMPT_SHA256")
        self.campaign_stage_id = os.environ.get("CAMPAIGN_STAGE_ID")
        self.deployment_release_id = os.environ.get("DEPLOYMENT_RELEASE_ID")
        self.bistka_preflight_sha256 = os.environ.get("BISTKA_PREFLIGHT_SHA256")
        self.slurm_job_id = os.environ.get("SLURM_JOB_ID")
        self._finished = False

    def add_input(self, record: dict[str, Any]) -> None:
        self.inputs.append(record)

    def add_output(self, name: str, path: str | Path) -> None:
        self.outputs.append(artifact_record(name, path))

    def _payload(self, status: str, error: BaseException | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "calibration_kind": self.kind,
            "status": status,
            "campaign_attempt_id": self.campaign_attempt_id,
            "campaign_attempt_sha256": self.campaign_attempt_sha256,
            "campaign_stage_id": self.campaign_stage_id,
            "deployment_release_id": self.deployment_release_id,
            "bistka_preflight_sha256": self.bistka_preflight_sha256,
            "slurm_job_id": self.slurm_job_id,
            "campaign_id": os.environ.get("CAMPAIGN_ID"),
            "hostname": socket.gethostname(),
            "process_id": os.getpid(),
            "started_at_utc": self.started_at,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - self.started_monotonic,
            "parameters": self.parameters,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "runtime": self.runtime,
            "result": self.result,
        }
        if error is not None:
            payload["error"] = {"type": type(error).__name__, "message": str(error)}
        return payload

    def succeed(self, result: dict[str, Any] | None = None) -> None:
        if result:
            self.result.update(result)
        atomic_json(self.receipt_path, self._payload("success"))
        self._finished = True

    def fail(self, error: BaseException) -> None:
        atomic_json(self.receipt_path, self._payload("failed", error))
        self._finished = True

    def __enter__(self) -> "CalibrationRun":
        required_scheduler_bindings = {
            "CAMPAIGN_ATTEMPT_ID": self.campaign_attempt_id,
            "CAMPAIGN_ATTEMPT_SHA256": self.campaign_attempt_sha256,
            "CAMPAIGN_STAGE_ID": self.campaign_stage_id,
            "DEPLOYMENT_RELEASE_ID": self.deployment_release_id,
            "BISTKA_PREFLIGHT_SHA256": self.bistka_preflight_sha256,
        }
        missing = [name for name, value in required_scheduler_bindings.items() if not value]
        malformed_hashes = [
            name
            for name in (
                "CAMPAIGN_ATTEMPT_SHA256",
                "DEPLOYMENT_RELEASE_ID",
                "BISTKA_PREFLIGHT_SHA256",
            )
            if required_scheduler_bindings[name]
            and (
                len(str(required_scheduler_bindings[name])) != 64
                or any(character not in "0123456789abcdef" for character in str(required_scheduler_bindings[name]).lower())
            )
        ]
        if self.slurm_job_id and (missing or malformed_hashes):
            error = RuntimeError(
                "SLURM calibration requires complete attempt, stage, deployment, and preflight bindings; "
                f"missing={missing}, malformed_hashes={malformed_hashes}"
            )
            self.fail(error)
            raise error
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None and not self._finished:
            self.fail(exc)
        elif exc is None and not self._finished:
            self.succeed()
        return False
