"""Gate-first launcher for one exact V3 candidate/fold/seed cell.

This module is deliberately limited to authorization and identity binding.  It
does not import a trainer, load market data, create an output directory, or
submit work until the locked V3 development gate passes.  The production V3
runner is intentionally absent while the protocol and its data policies remain
drafts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.v3.development_gate import validate_development_request
from src.v3.evaluation_policy import (
    FOLD_IDS,
    validate_candidate_ledger,
    validate_evaluation_policy,
    validate_fold_schedule,
)


BACKEND_MODULE = "src.v3.trial_runner"
BACKEND_FUNCTION = "run_authorized_cell"
OPERATION = "train_then_validation"


class V3LaunchDenied(RuntimeError):
    """Raised before execution when any authorization or identity check fails."""

    def __init__(self, issues: Sequence[str] | str) -> None:
        values = [issues] if isinstance(issues, str) else list(issues)
        normalized = tuple(
            dict.fromkeys(str(value).strip() for value in values if str(value).strip())
        )
        self.issues = normalized or ("V3 launch was denied",)
        super().__init__("; ".join(self.issues))


@dataclass(frozen=True)
class AuthorizedV3Cell:
    """Immutable, content-bound input passed to a future V3-only runner."""

    schema_version: int
    operation: str
    project_root: Path
    protocol_path: Path
    protocol_sha256: str
    evaluation_policy_path: Path
    evaluation_policy_sha256: str
    readiness_receipt_path: Path
    readiness_receipt_sha256: str
    candidate_ledger_path: Path
    candidate_ledger_sha256: str
    fold_schedule_path: Path
    fold_schedule_sha256: str
    candidate_id: str
    candidate_sha256: str
    registry_status: str
    algorithm: str
    parameter_profile: str
    timestep_budget: int
    design_seed: int
    hyperparameters_json: str
    fold_id: str
    fold_slice_sha256: str
    seed: int
    train_context_dates: tuple[str, ...]
    train_transitions: tuple[tuple[str, str], ...]
    validation_context_dates: tuple[str, ...]
    validation_transitions: tuple[tuple[str, str], ...]
    cell_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe audit representation without changing identity."""

        value = asdict(self)
        for key in (
            "project_root",
            "protocol_path",
            "evaluation_policy_path",
            "readiness_receipt_path",
            "candidate_ledger_path",
            "fold_schedule_path",
        ):
            value[key] = str(value[key])
        return value


@dataclass(frozen=True)
class _BoundBytes:
    path: Path
    payload: bytes
    sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V3LaunchDenied(f"{label} must be a mapping")
    return value


def _requested_path(project_root: Path, raw_path: Path, label: str) -> Path:
    candidate = raw_path if raw_path.is_absolute() else project_root / raw_path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(project_root):
        raise V3LaunchDenied(f"{label} escapes the project")
    if not resolved.is_file():
        raise V3LaunchDenied(f"{label} is missing: {resolved}")
    return resolved


def _artifact_path(project_root: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise V3LaunchDenied(f"{label} path must be a nonempty project-relative string")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise V3LaunchDenied(f"{label} path must stay inside the project")
    resolved = (project_root / relative).resolve()
    if not resolved.is_relative_to(project_root):
        raise V3LaunchDenied(f"{label} path escapes the project")
    if not resolved.is_file():
        raise V3LaunchDenied(f"{label} is missing: {resolved}")
    return resolved


def _read_path(path: Path, label: str) -> _BoundBytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise V3LaunchDenied(f"cannot read {label}: {exc}") from exc
    return _BoundBytes(path=path, payload=payload, sha256=_sha256_bytes(payload))


def _read_bound_bytes(
    project_root: Path,
    descriptor: Any,
    label: str,
    *,
    require_bytes: bool,
) -> _BoundBytes:
    value = _mapping(descriptor, label)
    path = _artifact_path(project_root, value.get("path"), label)
    expected_hash = value.get("sha256")
    if not _is_sha256(expected_hash):
        raise V3LaunchDenied(f"{label} must contain a valid SHA-256")
    bound = _read_path(path, label)
    if bound.sha256 != str(expected_hash).lower():
        raise V3LaunchDenied(f"{label} SHA-256 does not match at point of use")
    if require_bytes:
        expected_bytes = value.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise V3LaunchDenied(f"{label} must contain a nonnegative integer byte count")
        if len(bound.payload) != expected_bytes:
            raise V3LaunchDenied(f"{label} byte count does not match at point of use")
    return bound


def _verify_sidecar_for_bound_bytes(bound: _BoundBytes, label: str) -> None:
    """Recheck a locked artifact's sidecar against the bytes just read."""

    sidecar = Path(f"{bound.path}.sha256")
    try:
        tokens = sidecar.read_text(encoding="utf-8").split()
    except OSError as exc:
        raise V3LaunchDenied(f"{label} SHA-256 sidecar is missing: {exc}") from exc
    if not tokens or tokens[0].lower() != bound.sha256:
        raise V3LaunchDenied(f"{label} SHA-256 sidecar does not match at point of use")


def _load_yaml(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise V3LaunchDenied(f"cannot parse {label}: {exc}") from exc
    return _mapping(value, label)


def _load_json(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3LaunchDenied(f"cannot parse {label}: {exc}") from exc
    return _mapping(value, label)


def _load_csv(payload: bytes, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(payload), keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise V3LaunchDenied(f"cannot parse {label}: {exc}") from exc


def _canonical_frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return _sha256_bytes(payload)


def _transition_pairs(
    fold: pd.DataFrame,
    *,
    flag: str,
    owner: str,
) -> tuple[tuple[str, str], ...]:
    selected = fold.loc[fold[flag].astype(bool)]
    if selected.empty:
        raise V3LaunchDenied(f"selected fold has no {owner} transitions")
    if not selected["transition_owner"].eq(owner).all():
        raise V3LaunchDenied(f"selected fold has invalid {owner} transition ownership")
    if selected["transition_outcome_date"].astype(str).eq("").any():
        raise V3LaunchDenied(f"selected fold has a missing {owner} outcome date")
    return tuple(
        (str(row.date), str(row.transition_outcome_date))
        for row in selected[["date", "transition_outcome_date"]].itertuples(index=False)
    )


def _cell_identity_payload(
    *,
    protocol_sha256: str,
    policy_sha256: str,
    receipt_sha256: str,
    candidate_ledger_sha256: str,
    fold_schedule_sha256: str,
    candidate: Mapping[str, Any],
    fold_id: str,
    fold_slice_sha256: str,
    seed: int,
    train_context_dates: tuple[str, ...],
    train_transitions: tuple[tuple[str, str], ...],
    validation_context_dates: tuple[str, ...],
    validation_transitions: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "operation": OPERATION,
        "backend_contract": f"{BACKEND_MODULE}.{BACKEND_FUNCTION}",
        "bindings": {
            "protocol_sha256": protocol_sha256,
            "evaluation_policy_sha256": policy_sha256,
            "readiness_receipt_sha256": receipt_sha256,
            "candidate_ledger_sha256": candidate_ledger_sha256,
            "fold_schedule_sha256": fold_schedule_sha256,
        },
        "candidate": {
            "candidate_id": str(candidate["candidate_id"]),
            "candidate_sha256": str(candidate["candidate_sha256"]),
            "registry_status": str(candidate["execution_status"]),
            "algorithm": str(candidate["algorithm"]),
            "parameter_profile": str(candidate["parameter_profile"]),
            "timestep_budget": int(candidate["timestep_budget"]),
            "design_seed": int(candidate["design_seed"]),
            "hyperparameters_json": str(candidate["hyperparameters_json"]),
        },
        "fold": {
            "fold_id": fold_id,
            "fold_slice_sha256": fold_slice_sha256,
            "train_context_dates": train_context_dates,
            "train_transitions": train_transitions,
            "validation_context_dates": validation_context_dates,
            "validation_transitions": validation_transitions,
        },
        "seed": seed,
        "split_contract": {
            "training": "train",
            "evaluation": "validation",
            "test_access": False,
            "all_access": False,
        },
    }


def _bind_locked_cell(
    *,
    protocol_path: Path,
    project_root: Path,
    candidate_id: str,
    fold_id: str,
    seed: int,
) -> AuthorizedV3Cell:
    """Resolve one cell from already gate-validated, bound metadata only."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise V3LaunchDenied(f"project root is missing: {root}")
    protocol_file = _requested_path(root, Path(protocol_path), "V3 protocol")
    protocol_bound = _read_path(protocol_file, "V3 protocol")
    _verify_sidecar_for_bound_bytes(protocol_bound, "V3 protocol")
    protocol = _load_yaml(protocol_bound.payload, "V3 protocol")

    registration = _mapping(protocol.get("registration"), "V3 protocol registration")
    execution = _mapping(protocol.get("execution"), "V3 protocol execution")
    if registration.get("status") != "locked":
        raise V3LaunchDenied("V3 protocol changed after the gate and is not locked")
    if registration.get("approved_by_authors") is not True:
        raise V3LaunchDenied("V3 protocol changed after the gate and is not author-approved")
    if execution.get("training_enabled") is not True:
        raise V3LaunchDenied("V3 protocol changed after the gate and training is disabled")

    evaluation = _mapping(protocol.get("evaluation_policy"), "evaluation policy binding")
    if evaluation.get("status") != "locked":
        raise V3LaunchDenied("evaluation policy binding is not locked")
    policy_bound = _read_bound_bytes(
        root,
        evaluation,
        "evaluation policy",
        require_bytes=False,
    )
    policy_issues, policy = validate_evaluation_policy(
        policy_bound.path,
        require_locked=True,
    )
    if policy_issues:
        raise V3LaunchDenied(policy_issues)
    if _read_path(policy_bound.path, "evaluation policy").sha256 != policy_bound.sha256:
        raise V3LaunchDenied("evaluation policy changed during point-of-use validation")

    receipt_binding = _mapping(evaluation.get("receipt"), "evaluation receipt binding")
    if receipt_binding.get("status") != "bound":
        raise V3LaunchDenied("evaluation readiness receipt is not bound")
    receipt_bound = _read_bound_bytes(
        root,
        receipt_binding,
        "evaluation readiness receipt",
        require_bytes=False,
    )
    _verify_sidecar_for_bound_bytes(
        receipt_bound,
        "evaluation readiness receipt",
    )
    receipt = _load_json(receipt_bound.payload, "evaluation readiness receipt")
    if receipt.get("status") != "pass":
        raise V3LaunchDenied("evaluation readiness receipt status is not pass")
    if receipt.get("evaluation_policy_eligible") is not True:
        raise V3LaunchDenied("evaluation readiness receipt is not policy-eligible")
    if receipt.get("training_enabled") is not False:
        raise V3LaunchDenied("evaluation readiness receipt cannot authorize training")
    if receipt.get("test_access_enabled") is not False:
        raise V3LaunchDenied("evaluation readiness receipt exposes test access")
    if receipt.get("all_access_enabled") is not False:
        raise V3LaunchDenied("evaluation readiness receipt exposes all-data access")

    outputs = _mapping(receipt.get("outputs"), "evaluation readiness outputs")
    candidates_bound = _read_bound_bytes(
        root,
        outputs.get("candidate_ledger"),
        "candidate ledger",
        require_bytes=True,
    )
    schedule_bound = _read_bound_bytes(
        root,
        outputs.get("fold_schedule"),
        "fold schedule",
        require_bytes=True,
    )
    candidates = _load_csv(candidates_bound.payload, "candidate ledger")
    schedule = _load_csv(schedule_bound.payload, "fold schedule")

    candidate_issues = validate_candidate_ledger(candidates, policy)
    if candidate_issues:
        raise V3LaunchDenied(candidate_issues)
    if "date" not in schedule:
        raise V3LaunchDenied("fold schedule is missing its date column")
    schedule_issues = validate_fold_schedule(
        schedule,
        schedule["date"].drop_duplicates().tolist(),
        policy,
    )
    if schedule_issues:
        raise V3LaunchDenied(schedule_issues)

    if not isinstance(candidate_id, str) or not candidate_id:
        raise V3LaunchDenied("candidate_id must be a nonempty exact registry key")
    candidate_rows = candidates.loc[candidates["candidate_id"].eq(candidate_id)]
    if len(candidate_rows) != 1:
        raise V3LaunchDenied(f"candidate_id is not registered exactly once: {candidate_id!r}")
    candidate = candidate_rows.iloc[0].to_dict()

    if not isinstance(fold_id, str) or fold_id not in FOLD_IDS:
        raise V3LaunchDenied(f"fold_id is not preregistered: {fold_id!r}")
    fold = schedule.loc[schedule["fold_id"].eq(fold_id)].copy()
    if fold.empty:
        raise V3LaunchDenied(f"fold schedule has no rows for {fold_id!r}")

    if type(seed) is not int:
        raise V3LaunchDenied("seed must be an integer registry value, not a Boolean or coercion")
    registered_seeds = tuple(policy["candidate_budget"]["model_seeds"])
    if seed not in registered_seeds:
        raise V3LaunchDenied(f"seed is not preregistered: {seed!r}")

    train_context_dates = tuple(
        fold.loc[fold["train_context"].astype(bool), "date"].astype(str)
    )
    validation_context_dates = tuple(
        fold.loc[fold["validation_context"].astype(bool), "date"].astype(str)
    )
    train_transitions = _transition_pairs(
        fold,
        flag="train_decision",
        owner="train",
    )
    validation_transitions = _transition_pairs(
        fold,
        flag="validation_decision",
        owner="validation",
    )
    fold_slice_sha256 = _canonical_frame_hash(fold)
    identity = _cell_identity_payload(
        protocol_sha256=protocol_bound.sha256,
        policy_sha256=policy_bound.sha256,
        receipt_sha256=receipt_bound.sha256,
        candidate_ledger_sha256=candidates_bound.sha256,
        fold_schedule_sha256=schedule_bound.sha256,
        candidate=candidate,
        fold_id=fold_id,
        fold_slice_sha256=fold_slice_sha256,
        seed=seed,
        train_context_dates=train_context_dates,
        train_transitions=train_transitions,
        validation_context_dates=validation_context_dates,
        validation_transitions=validation_transitions,
    )
    cell_sha256 = _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    return AuthorizedV3Cell(
        schema_version=1,
        operation=OPERATION,
        project_root=root,
        protocol_path=protocol_bound.path,
        protocol_sha256=protocol_bound.sha256,
        evaluation_policy_path=policy_bound.path,
        evaluation_policy_sha256=policy_bound.sha256,
        readiness_receipt_path=receipt_bound.path,
        readiness_receipt_sha256=receipt_bound.sha256,
        candidate_ledger_path=candidates_bound.path,
        candidate_ledger_sha256=candidates_bound.sha256,
        fold_schedule_path=schedule_bound.path,
        fold_schedule_sha256=schedule_bound.sha256,
        candidate_id=str(candidate["candidate_id"]),
        candidate_sha256=str(candidate["candidate_sha256"]),
        registry_status=str(candidate["execution_status"]),
        algorithm=str(candidate["algorithm"]),
        parameter_profile=str(candidate["parameter_profile"]),
        timestep_budget=int(candidate["timestep_budget"]),
        design_seed=int(candidate["design_seed"]),
        hyperparameters_json=str(candidate["hyperparameters_json"]),
        fold_id=fold_id,
        fold_slice_sha256=fold_slice_sha256,
        seed=seed,
        train_context_dates=train_context_dates,
        train_transitions=train_transitions,
        validation_context_dates=validation_context_dates,
        validation_transitions=validation_transitions,
        cell_sha256=cell_sha256,
    )


def authorize_v3_cell(
    *,
    protocol_path: Path,
    project_root: Path,
    candidate_id: str,
    fold_id: str,
    seed: int,
) -> AuthorizedV3Cell:
    """Run the locked gate before reading any executable cell metadata."""

    try:
        issues, _ = validate_development_request(
            Path(protocol_path),
            project_root=Path(project_root),
            split="train",
        )
    except Exception as exc:
        raise V3LaunchDenied(
            f"V3 development gate could not complete: {type(exc).__name__}: {exc}"
        ) from exc
    if issues:
        raise V3LaunchDenied(issues)
    return _bind_locked_cell(
        protocol_path=Path(protocol_path),
        project_root=Path(project_root),
        candidate_id=candidate_id,
        fold_id=fold_id,
        seed=seed,
    )


def launch_v3_cell(
    *,
    protocol_path: Path,
    project_root: Path,
    candidate_id: str,
    fold_id: str,
    seed: int,
) -> Any:
    """Authorize one exact cell, then and only then import the V3-only backend."""

    authorization = authorize_v3_cell(
        protocol_path=protocol_path,
        project_root=project_root,
        candidate_id=candidate_id,
        fold_id=fold_id,
        seed=seed,
    )
    try:
        backend = import_module(BACKEND_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != BACKEND_MODULE:
            raise
        raise V3LaunchDenied(
            "V3 trial backend is intentionally absent pending a separate reviewed release"
        ) from exc
    runner = getattr(backend, BACKEND_FUNCTION, None)
    if not callable(runner):
        raise V3LaunchDenied(
            f"V3 backend does not expose callable {BACKEND_FUNCTION!r}"
        )
    return runner(authorization)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch one checksum-bound V3 train/validation cell"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("config/v3/development_protocol.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold-id", choices=FOLD_IDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = launch_v3_cell(
            protocol_path=args.protocol,
            project_root=args.project_root,
            candidate_id=args.candidate_id,
            fold_id=args.fold_id,
            seed=args.seed,
        )
    except V3LaunchDenied as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "operation": OPERATION,
                    "candidate_id": args.candidate_id,
                    "fold_id": args.fold_id,
                    "seed": args.seed,
                    "issues": list(exc.issues),
                    "backend_imported": False,
                    "output_written": False,
                },
                indent=2,
            )
        )
        return 2
    print(json.dumps({"status": "complete", "result": result}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
