"""Fail-closed gate for the isolated V3 development capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from src.v3.evaluation_policy import validate_evaluation_policy
from src.v3.investability_policy import validate_investability_policy
from src.v3.membership_mask import REVISION_AVAILABILITY_RULE


ALLOWED_SPLITS = ("train", "validation")
FORBIDDEN_SPLITS = ("test", "all")
EXPECTED_PROTOCOL_ID = "thai_drl_point_in_time_set100_v3_development"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str, issues: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        issues.append(f"{label} must be a mapping")
        return {}
    return value


def _safe_path(
    project_root: Path,
    raw_path: Any,
    label: str,
    issues: list[str],
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        issues.append(f"{label} path must be a nonempty project-relative string")
        return None
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        issues.append(f"{label} path must stay inside the project")
        return None
    resolved = (project_root / relative).resolve()
    if not resolved.is_relative_to(project_root):
        issues.append(f"{label} path escapes the project")
        return None
    return resolved


def _valid_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _check_binding(
    project_root: Path,
    label: str,
    binding: Any,
    issues: list[str],
    referenced: list[Path],
) -> None:
    value = _mapping(binding, f"data_bindings.{label}", issues)
    path = _safe_path(project_root, value.get("path"), label, issues)
    expected_hash = value.get("sha256")
    if not _valid_sha256(expected_hash):
        issues.append(f"{label} binding must contain a valid SHA-256")
    if path is None:
        return
    if not path.is_file():
        issues.append(f"{label} binding is missing: {path}")
        return
    referenced.append(path)
    if _valid_sha256(expected_hash) and sha256_file(path) != str(expected_hash).lower():
        issues.append(f"{label} binding SHA-256 does not match")

    if value.get("require_sidecar") is True:
        sidecar = Path(f"{path}.sha256")
        if not sidecar.is_file():
            issues.append(f"{label} SHA-256 sidecar is missing")
        else:
            referenced.append(sidecar)
            tokens = sidecar.read_text(encoding="utf-8").split()
            if not tokens or tokens[0].lower() != sha256_file(path):
                issues.append(f"{label} SHA-256 sidecar does not match")


def _check_exact_descriptor(
    project_root: Path,
    path: Path,
    descriptor: Any,
    label: str,
    issues: list[str],
) -> None:
    """Check a receipt descriptor against one already-bound local file."""

    value = _mapping(descriptor, label, issues)
    if not path.is_file():
        return
    expected = {
        "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            issues.append(f"{label} {key} does not match the bound file")


def _check_receipt_descriptors(
    project_root: Path,
    value: Any,
    label: str,
    issues: list[str],
    referenced: list[Path],
) -> None:
    """Recompute every file descriptor embedded in a readiness receipt."""

    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            path = _safe_path(
                project_root,
                value.get("path"),
                label,
                issues,
            )
            if path is None:
                return
            if not path.is_file():
                issues.append(f"{label} descriptor file is missing: {path}")
                return
            referenced.append(path)
            _check_exact_descriptor(
                project_root,
                path,
                value,
                label,
                issues,
            )
            return
        for key, child in value.items():
            _check_receipt_descriptors(
                project_root,
                child,
                f"{label}.{key}",
                issues,
                referenced,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_receipt_descriptors(
                project_root,
                child,
                f"{label}[{index}]",
                issues,
                referenced,
            )


def validate_development_protocol(
    path: Path,
    *,
    project_root: Path,
    require_locked: bool = False,
) -> tuple[list[str], list[Path]]:
    """Validate the V3 protocol without opening any model or holdout data."""

    root = Path(project_root).resolve()
    protocol_path = Path(path).resolve()
    issues: list[str] = []
    referenced: list[Path] = []
    if not protocol_path.is_relative_to(root):
        return ["V3 protocol is outside the project"], []
    if not protocol_path.is_file():
        return [f"V3 protocol is missing: {protocol_path}"], []
    try:
        loaded = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot read V3 protocol: {exc}"], []
    protocol = _mapping(loaded, "protocol", issues)
    if protocol.get("schema_version") != 1:
        issues.append("protocol schema_version must be 1")

    registration = _mapping(protocol.get("registration"), "registration", issues)
    if registration.get("id") != EXPECTED_PROTOCOL_ID:
        issues.append(f"registration.id must be {EXPECTED_PROTOCOL_ID}")
    status = registration.get("status")
    if status not in {"draft_not_locked", "locked"}:
        issues.append("registration.status must be draft_not_locked or locked")
    if registration.get("replaces_v2") is not False:
        issues.append("V3 must not replace or amend V2")

    boundary = _mapping(
        protocol.get("scientific_boundary"),
        "scientific_boundary",
        issues,
    )
    if boundary.get("development_only") is not True:
        issues.append("V3 capsule must remain development-only")
    if boundary.get("allowed_splits") != list(ALLOWED_SPLITS):
        issues.append("allowed_splits must be exactly train and validation")
    if boundary.get("forbidden_splits") != list(FORBIDDEN_SPLITS):
        issues.append("forbidden_splits must be exactly test and all")
    if boundary.get("known_2021_2024_period_usable_as_new_final_holdout") is not False:
        issues.append("known 2021-2024 results cannot be relabeled as a new V3 holdout")
    final_holdout = _mapping(
        boundary.get("final_holdout"),
        "scientific_boundary.final_holdout",
        issues,
    )
    if final_holdout.get("status") != "absent":
        issues.append("the V3 final holdout must remain absent from this capsule")

    membership = _mapping(protocol.get("membership"), "membership", issues)
    if membership.get("index_name") not in {"SET50", "SET100"}:
        issues.append("membership.index_name must be SET50 or SET100")
    if membership.get("acquisition_universe_is_static_membership") is not False:
        issues.append("the acquisition union must never be static membership")
    if membership.get("same_day_revision_rule") != REVISION_AVAILABILITY_RULE:
        issues.append("same-day revisions must use the next observed session")

    eligibility = _mapping(protocol.get("eligibility"), "eligibility", issues)
    if eligibility.get("next_session_price_may_define_current_eligibility") is not False:
        issues.append("next-session prices must not define current eligibility")
    final_investability_ready = eligibility.get("final_investability_ready")
    if status == "draft_not_locked" and final_investability_ready is not False:
        issues.append("a draft V3 capsule cannot claim final investability")
    if status == "locked" and final_investability_ready is not True:
        issues.append("a locked executable V3 protocol requires final investability")
    handling_policy = _mapping(
        eligibility.get("handling_policy"),
        "eligibility.handling_policy",
        issues,
    )
    handling_status = handling_policy.get("status")
    if handling_status not in {"absent", "draft", "locked"}:
        issues.append("eligibility handling-policy status is invalid")
    if handling_status in {"draft", "locked"}:
        _check_binding(
            root,
            "investability_handling_policy",
            handling_policy,
            issues,
            referenced,
        )
        handling_path = _safe_path(
            root,
            handling_policy.get("path"),
            "investability_handling_policy",
            issues,
        )
        if handling_path is not None and handling_path.is_file():
            policy_issues, policy_value = validate_investability_policy(
                handling_path,
                require_locked=require_locked,
            )
            issues.extend(policy_issues)
            policy_registration = _mapping(
                policy_value.get("registration"),
                "investability_policy.registration",
                issues,
            )
            expected_policy_status = (
                "locked" if handling_status == "locked" else "draft_not_approved"
            )
            if policy_registration.get("status") != expected_policy_status:
                issues.append(
                    "handling-policy status does not match the bound policy"
                )

        policy_receipt = _mapping(
            handling_policy.get("receipt"),
            "eligibility.handling_policy.receipt",
            issues,
        )
        if policy_receipt.get("status") != "bound":
            issues.append("investability handling-policy receipt is not bound")
        else:
            _check_binding(
                root,
                "investability_handling_policy_receipt",
                policy_receipt,
                issues,
                referenced,
            )
            receipt_path = _safe_path(
                root,
                policy_receipt.get("path"),
                "investability_handling_policy_receipt",
                issues,
            )
            if receipt_path is not None and receipt_path.is_file():
                try:
                    receipt_value = json.loads(
                        receipt_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    issues.append(
                        f"cannot read investability policy receipt: {exc}"
                    )
                else:
                    _check_receipt_descriptors(
                        root,
                        receipt_value,
                        "investability_policy_receipt",
                        issues,
                        referenced,
                    )
                    receipt_policy = _mapping(
                        receipt_value.get("policy"),
                        "investability_policy_receipt.policy",
                        issues,
                    )
                    if receipt_policy.get("structural_status") != "pass":
                        issues.append(
                            "investability policy receipt did not pass structural validation"
                        )
                    if (
                        handling_path is not None
                        and handling_path.is_file()
                        and receipt_policy.get("sha256")
                        != sha256_file(handling_path)
                    ):
                        issues.append(
                            "investability policy receipt is bound to a different policy"
                        )
                    if handling_status == "draft":
                        if receipt_value.get("status") != "blocked":
                            issues.append(
                                "draft investability policy receipt must remain blocked"
                            )
                        if receipt_value.get("training_enabled") is not False:
                            issues.append(
                                "draft investability policy receipt cannot enable training"
                            )
                    elif require_locked:
                        if receipt_value.get("status") != "pass":
                            issues.append(
                                "locked investability policy receipt is not pass"
                            )
                        if (
                            receipt_value.get("final_investability_eligible")
                            is not True
                        ):
                            issues.append(
                                "locked investability policy receipt is not final-investability eligible"
                            )

    bindings = _mapping(protocol.get("data_bindings"), "data_bindings", issues)
    for label in (
        "constituent_history",
        "acquisition_universe",
        "constituent_receipt",
    ):
        _check_binding(root, label, bindings.get(label), issues, referenced)
    market_receipt = _mapping(
        bindings.get("licensed_market_package_receipt"),
        "data_bindings.licensed_market_package_receipt",
        issues,
    )
    market_receipt_status = market_receipt.get("status")
    if market_receipt_status not in {"absent", "bound"}:
        issues.append(
            "licensed-market package receipt status must be absent or bound"
        )
    if market_receipt_status == "bound":
        _check_binding(
            root,
            "licensed_market_package_receipt",
            market_receipt,
            issues,
            referenced,
        )
        market_path = _safe_path(
            root,
            market_receipt.get("path"),
            "licensed_market_package_receipt",
            issues,
        )
        if market_path is not None and market_path.is_file():
            try:
                market_value = json.loads(
                    market_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(
                    f"cannot read licensed-market package receipt: {exc}"
                )
            else:
                _check_receipt_descriptors(
                    root,
                    market_value,
                    "licensed_market_package_receipt",
                    issues,
                    referenced,
                )
                if market_value.get("status") != "pass":
                    issues.append(
                        "licensed-market package receipt status is not pass"
                    )
                if (
                    market_value.get("licensed_market_package_eligible")
                    is not True
                ):
                    issues.append(
                        "licensed-market package receipt is not eligible"
                    )

    evaluation_policy = _mapping(
        protocol.get("evaluation_policy"),
        "evaluation_policy",
        issues,
    )
    evaluation_status = evaluation_policy.get("status")
    if evaluation_status not in {"absent", "draft", "locked"}:
        issues.append("evaluation_policy.status must be absent, draft, or locked")
    if evaluation_status in {"draft", "locked"}:
        if evaluation_policy.get("require_sidecar") is not True:
            issues.append("evaluation policy must require its SHA-256 sidecar")
        _check_binding(
            root,
            "evaluation_policy",
            evaluation_policy,
            issues,
            referenced,
        )
        evaluation_path = _safe_path(
            root,
            evaluation_policy.get("path"),
            "evaluation_policy",
            issues,
        )
        if evaluation_path is not None and evaluation_path.is_file():
            policy_issues, policy_value = validate_evaluation_policy(
                evaluation_path,
                require_locked=evaluation_status == "locked",
            )
            issues.extend(policy_issues)
            evaluation_registration = _mapping(
                policy_value.get("registration"),
                "evaluation_policy.registration",
                issues,
            )
            expected_registration_status = (
                "locked"
                if evaluation_status == "locked"
                else "draft_not_approved"
            )
            if (
                evaluation_registration.get("status")
                != expected_registration_status
            ):
                issues.append(
                    "evaluation-policy status does not match the bound policy"
                )

        evaluation_receipt = _mapping(
            evaluation_policy.get("receipt"),
            "evaluation_policy.receipt",
            issues,
        )
        if evaluation_receipt.get("status") != "bound":
            issues.append("evaluation-policy readiness receipt is not bound")
        else:
            if evaluation_receipt.get("require_sidecar") is not True:
                issues.append(
                    "evaluation-policy readiness receipt must require its SHA-256 sidecar"
                )
            _check_binding(
                root,
                "evaluation_policy_readiness_receipt",
                evaluation_receipt,
                issues,
                referenced,
            )
            evaluation_receipt_path = _safe_path(
                root,
                evaluation_receipt.get("path"),
                "evaluation_policy_readiness_receipt",
                issues,
            )
            if (
                evaluation_receipt_path is not None
                and evaluation_receipt_path.is_file()
            ):
                try:
                    evaluation_receipt_value = json.loads(
                        evaluation_receipt_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    issues.append(
                        f"cannot read evaluation-policy readiness receipt: {exc}"
                    )
                else:
                    _check_receipt_descriptors(
                        root,
                        evaluation_receipt_value,
                        "evaluation_policy_receipt",
                        issues,
                        referenced,
                    )
                    if (
                        evaluation_receipt_value.get("evidence_class")
                        != "v3_evaluation_policy_structural_readiness"
                    ):
                        issues.append(
                            "evaluation-policy receipt evidence class is invalid"
                        )
                    receipt_policy = _mapping(
                        evaluation_receipt_value.get("policy"),
                        "evaluation_policy_receipt.policy",
                        issues,
                    )
                    if receipt_policy.get("structural_status") != "pass":
                        issues.append(
                            "evaluation-policy receipt did not pass structural validation"
                        )
                    if evaluation_path is not None and evaluation_path.is_file():
                        _check_exact_descriptor(
                            root,
                            evaluation_path,
                            receipt_policy,
                            "evaluation_policy_receipt.policy",
                            issues,
                        )
                        policy_sidecar = Path(f"{evaluation_path}.sha256")
                        if policy_sidecar.is_file():
                            _check_exact_descriptor(
                                root,
                                policy_sidecar,
                                receipt_policy.get("sidecar"),
                                "evaluation_policy_receipt.policy.sidecar",
                                issues,
                            )
                        else:
                            issues.append(
                                "evaluation-policy bound policy sidecar is missing"
                            )

                    expected_receipt_policy_status = (
                        "locked"
                        if evaluation_status == "locked"
                        else "draft_not_approved"
                    )
                    if (
                        receipt_policy.get("registration_status")
                        != expected_receipt_policy_status
                    ):
                        issues.append(
                            "evaluation-policy receipt registration status does not match"
                        )
                    expected_approved = evaluation_status == "locked"
                    if (
                        receipt_policy.get("approved_by_authors")
                        is not expected_approved
                    ):
                        issues.append(
                            "evaluation-policy receipt author-approval flag does not match"
                        )
                    if receipt_policy.get("locked") is not expected_approved:
                        issues.append(
                            "evaluation-policy receipt lock flag does not match"
                        )

                    for flag, expected in {
                        "schedule_structural_eligible": True,
                        "candidate_registry_eligible": True,
                        "training_enabled": False,
                        "test_access_enabled": False,
                        "all_access_enabled": False,
                        "hpc_jobs_submitted": 0,
                    }.items():
                        actual = evaluation_receipt_value.get(flag)
                        mismatch = (
                            actual is not expected
                            if isinstance(expected, bool)
                            else type(actual) is not type(expected)
                            or actual != expected
                        )
                        if mismatch:
                            issues.append(
                                f"evaluation-policy receipt {flag} must be {expected!r}"
                            )
                    receipt_boundary = _mapping(
                        evaluation_receipt_value.get("scientific_boundary"),
                        "evaluation_policy_receipt.scientific_boundary",
                        issues,
                    )
                    for flag, expected in {
                        "known_period_is_not_final_holdout": True,
                        "no_performance_metrics_generated": True,
                        "shared_v2_splitter_not_used": True,
                        "development_protocol_not_an_input": True,
                    }.items():
                        if receipt_boundary.get(flag) is not expected:
                            issues.append(
                                "evaluation-policy receipt scientific boundary "
                                f"{flag} must be {expected!r}"
                            )

                    if evaluation_status == "draft":
                        if evaluation_receipt_value.get("status") != "blocked":
                            issues.append(
                                "draft evaluation-policy receipt must remain blocked"
                            )
                        if (
                            evaluation_receipt_value.get(
                                "evaluation_policy_eligible"
                            )
                            is not False
                        ):
                            issues.append(
                                "draft evaluation-policy receipt cannot be eligible"
                            )
                        if (
                            receipt_boundary.get("current_schedule_is_provisional")
                            is not True
                        ):
                            issues.append(
                                "draft evaluation-policy schedule must remain provisional"
                            )
                    else:
                        if market_receipt_status != "bound":
                            issues.append(
                                "locked evaluation policy requires a bound licensed-market package receipt"
                            )
                        if evaluation_receipt_value.get("status") != "pass":
                            issues.append(
                                "locked evaluation-policy readiness receipt is not pass"
                            )
                        if (
                            evaluation_receipt_value.get(
                                "evaluation_policy_eligible"
                            )
                            is not True
                        ):
                            issues.append(
                                "locked evaluation-policy receipt is not eligible"
                            )
                        if (
                            receipt_boundary.get("current_schedule_is_provisional")
                            is not False
                        ):
                            issues.append(
                                "locked evaluation-policy schedule cannot be provisional"
                            )
                        receipt_inputs = _mapping(
                            evaluation_receipt_value.get("inputs"),
                            "evaluation_policy_receipt.inputs",
                            issues,
                        )
                        if (
                            receipt_inputs.get("licensed_market_package_eligible")
                            is not True
                        ):
                            issues.append(
                                "locked evaluation-policy receipt requires an eligible licensed market package"
                            )

    execution = _mapping(protocol.get("execution"), "execution", issues)
    if execution.get("hpc_submission_enabled") is not False:
        issues.append("HPC submission must remain disabled in the development capsule")

    if require_locked:
        if status != "locked":
            issues.append("V3 protocol is not locked")
        if registration.get("approved_by_authors") is not True:
            issues.append("V3 protocol is not approved by both authors")
        if execution.get("training_enabled") is not True:
            issues.append("V3 training is disabled")
        if market_receipt_status != "bound":
            issues.append("licensed-market package receipt is not bound")
        if handling_status != "locked":
            issues.append("V3 investability handling policy is not locked")
        if evaluation_status != "locked":
            issues.append("V3 evaluation and selection policy is not locked")
        sidecar = Path(f"{protocol_path}.sha256")
        if not sidecar.is_file():
            issues.append("locked V3 protocol SHA-256 sidecar is missing")
        else:
            referenced.append(sidecar)
            tokens = sidecar.read_text(encoding="utf-8").split()
            if not tokens or tokens[0].lower() != sha256_file(protocol_path):
                issues.append("locked V3 protocol SHA-256 sidecar does not match")

    referenced.append(protocol_path)
    return list(dict.fromkeys(issues)), sorted(set(referenced))


def validate_development_request(
    path: Path,
    *,
    project_root: Path,
    split: str,
) -> tuple[list[str], list[Path]]:
    """Reject any executable request unless the protocol and split are safe."""

    issues: list[str] = []
    if split in FORBIDDEN_SPLITS:
        issues.append(f"V3 development request forbids split {split!r}")
    elif split not in ALLOWED_SPLITS:
        issues.append(f"unsupported V3 development split: {split!r}")
    protocol_issues, referenced = validate_development_protocol(
        path,
        project_root=project_root,
        require_locked=True,
    )
    return list(dict.fromkeys([*issues, *protocol_issues])), referenced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the isolated V3 development gate")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("config/v3/development_protocol.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--split", choices=("structural", *ALLOWED_SPLITS, *FORBIDDEN_SPLITS), default="structural")
    args = parser.parse_args(argv)
    if args.split == "structural":
        issues, referenced = validate_development_protocol(
            args.protocol,
            project_root=args.project_root,
            require_locked=False,
        )
    else:
        issues, referenced = validate_development_request(
            args.protocol,
            project_root=args.project_root,
            split=args.split,
        )
    print(
        json.dumps(
            {
                "status": "pass" if not issues else "fail",
                "mode": args.split,
                "issues": issues,
                "referenced_files": [str(value) for value in referenced],
            },
            indent=2,
        )
    )
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
