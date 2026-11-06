from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

from src.v3.development_gate import (
    sha256_file,
    validate_development_protocol,
    validate_development_request,
)
from scripts.build_v3_membership_readiness import V2_IMMUTABILITY_BINDINGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "config/v3/development_protocol.yaml"


def _descriptor_paths(value) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            paths.append(value["path"])
        else:
            for child in value.values():
                paths.extend(_descriptor_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.extend(_descriptor_paths(child))
    return paths


def test_v3_draft_passes_structural_validation_without_enabling_training() -> None:
    issues, referenced = validate_development_protocol(
        PROTOCOL_PATH,
        project_root=PROJECT_ROOT,
        require_locked=False,
    )

    assert issues == []
    assert PROTOCOL_PATH in referenced
    assert (
        PROJECT_ROOT / "data/reference/set50_set100_membership_2021_2024.csv"
    ) in referenced


def test_v3_executable_request_fails_while_protocol_is_draft() -> None:
    issues, _ = validate_development_request(
        PROTOCOL_PATH,
        project_root=PROJECT_ROOT,
        split="train",
    )

    assert "V3 protocol is not locked" in issues
    assert "V3 protocol is not approved by both authors" in issues
    assert "V3 training is disabled" in issues
    assert "licensed-market package receipt is not bound" in issues
    assert "V3 investability handling policy is not locked" in issues
    assert "V3 evaluation and selection policy is not locked" in issues
    assert "locked V3 protocol SHA-256 sidecar is missing" in issues


def test_v3_gate_rejects_test_and_all_before_execution() -> None:
    for split in ("test", "all"):
        issues, _ = validate_development_request(
            PROTOCOL_PATH,
            project_root=PROJECT_ROOT,
            split=split,
        )
        assert f"V3 development request forbids split {split!r}" in issues


def test_v3_capsule_leaves_locked_v2_hashes_unchanged() -> None:
    for binding in V2_IMMUTABILITY_BINDINGS.values():
        assert sha256_file(PROJECT_ROOT / binding["path"]) == binding["sha256"]


def _copy_protocol_capsule(tmp_path: Path) -> Path:
    relative_paths = [
        "config/v3/development_protocol.yaml",
        "config/v3/investability_policy_draft.yaml",
        "config/v3/investability_policy_draft.yaml.sha256",
        "data/reference/set50_set100_membership_2021_2024.csv",
        "data/reference/set50_or_set100_universe.csv",
        "reports/data_sources/set_constituent_archive_20260731.json",
        "reports/data_sources/set_constituent_archive_20260731.json.sha256",
        (
            "reports/data_quality_v3/"
            "v3_investability_policy_readiness_20260731.json"
        ),
        (
            "reports/data_quality_v3/"
            "v3_investability_policy_readiness_20260731.json.sha256"
        ),
    ]
    protocol_value = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    evaluation = protocol_value["evaluation_policy"]
    if evaluation.get("status") in {"draft", "locked"}:
        relative_paths.extend(
            [
                evaluation["path"],
                f"{evaluation['path']}.sha256",
                evaluation["receipt"]["path"],
                f"{evaluation['receipt']['path']}.sha256",
            ]
        )
    receipt_paths = [
        protocol_value["eligibility"]["handling_policy"]["receipt"]["path"]
    ]
    if evaluation.get("status") in {"draft", "locked"}:
        receipt_paths.append(evaluation["receipt"]["path"])
    for relative in receipt_paths:
        receipt = json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
        relative_paths.extend(_descriptor_paths(receipt))
    for relative in dict.fromkeys(relative_paths):
        source = PROJECT_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return tmp_path / "config/v3/development_protocol.yaml"


def _evaluation_binding(protocol: Path) -> dict:
    value = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    evaluation = value["evaluation_policy"]
    if evaluation.get("status") not in {"draft", "locked"}:
        pytest.skip("evaluation policy is not bound in the protocol yet")
    return evaluation


def _write_sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )


def test_policy_byte_tampering_fails_protocol_binding(tmp_path) -> None:
    protocol = _copy_protocol_capsule(tmp_path)
    policy = tmp_path / "config/v3/investability_policy_draft.yaml"
    policy.write_text(
        policy.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    issues, _ = validate_development_protocol(
        protocol,
        project_root=tmp_path,
        require_locked=False,
    )

    assert "investability_handling_policy binding SHA-256 does not match" in issues
    assert "investability_handling_policy SHA-256 sidecar does not match" in issues
    assert "investability policy receipt is bound to a different policy" in issues


def test_evaluation_policy_byte_tampering_fails_all_exact_bindings(
    tmp_path: Path,
) -> None:
    protocol = _copy_protocol_capsule(tmp_path)
    evaluation = _evaluation_binding(protocol)
    policy = tmp_path / evaluation["path"]
    policy.write_text(
        policy.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    issues, _ = validate_development_protocol(
        protocol,
        project_root=tmp_path,
        require_locked=False,
    )

    assert "evaluation_policy binding SHA-256 does not match" in issues
    assert "evaluation_policy SHA-256 sidecar does not match" in issues
    assert (
        "evaluation_policy_receipt.policy bytes does not match the bound file"
        in issues
    )
    assert (
        "evaluation_policy_receipt.policy sha256 does not match the bound file"
        in issues
    )


def test_evaluation_receipt_byte_tampering_fails_exact_binding(
    tmp_path: Path,
) -> None:
    protocol = _copy_protocol_capsule(tmp_path)
    evaluation = _evaluation_binding(protocol)
    receipt = tmp_path / evaluation["receipt"]["path"]
    receipt.write_text(
        receipt.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    issues, _ = validate_development_protocol(
        protocol,
        project_root=tmp_path,
        require_locked=False,
    )

    assert (
        "evaluation_policy_readiness_receipt binding SHA-256 does not match"
        in issues
    )
    assert (
        "evaluation_policy_readiness_receipt SHA-256 sidecar does not match"
        in issues
    )


def test_evaluation_output_tampering_fails_transitive_receipt_binding(
    tmp_path: Path,
) -> None:
    protocol = _copy_protocol_capsule(tmp_path)
    evaluation = _evaluation_binding(protocol)
    receipt = json.loads(
        (tmp_path / evaluation["receipt"]["path"]).read_text(encoding="utf-8")
    )
    candidate_path = tmp_path / receipt["outputs"]["candidate_ledger"]["path"]
    candidate_path.write_text(
        candidate_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    issues, _ = validate_development_protocol(
        protocol,
        project_root=tmp_path,
        require_locked=False,
    )

    assert (
        "evaluation_policy_receipt.outputs.candidate_ledger bytes does not "
        "match the bound file"
    ) in issues
    assert (
        "evaluation_policy_receipt.outputs.candidate_ledger sha256 does not "
        "match the bound file"
    ) in issues


def test_rebound_evaluation_receipt_cannot_enable_training_or_holdout(
    tmp_path: Path,
) -> None:
    protocol = _copy_protocol_capsule(tmp_path)
    evaluation = _evaluation_binding(protocol)
    receipt = tmp_path / evaluation["receipt"]["path"]
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_value["training_enabled"] = True
    receipt_value["test_access_enabled"] = True
    receipt_value["all_access_enabled"] = True
    receipt_value["scientific_boundary"]["no_performance_metrics_generated"] = False
    receipt.write_text(
        json.dumps(receipt_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sidecar(receipt)

    protocol_value = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    protocol_value["evaluation_policy"]["receipt"]["sha256"] = sha256_file(
        receipt
    )
    protocol_value["evaluation_policy"]["require_sidecar"] = False
    protocol_value["evaluation_policy"]["receipt"]["require_sidecar"] = False
    protocol.write_text(
        yaml.safe_dump(protocol_value, sort_keys=False),
        encoding="utf-8",
    )

    issues, _ = validate_development_protocol(
        protocol,
        project_root=tmp_path,
        require_locked=False,
    )

    assert "evaluation-policy receipt training_enabled must be False" in issues
    assert "evaluation-policy receipt test_access_enabled must be False" in issues
    assert "evaluation-policy receipt all_access_enabled must be False" in issues
    assert (
        "evaluation-policy receipt scientific boundary "
        "no_performance_metrics_generated must be True"
    ) in issues
    assert "evaluation policy must require its SHA-256 sidecar" in issues
    assert (
        "evaluation-policy readiness receipt must require its SHA-256 sidecar"
        in issues
    )


def test_locked_evaluation_binding_requires_strict_policy_and_passing_receipt(
    tmp_path: Path,
) -> None:
    protocol = _copy_protocol_capsule(tmp_path)
    _evaluation_binding(protocol)
    protocol_value = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    protocol_value["evaluation_policy"]["status"] = "locked"
    protocol.write_text(
        yaml.safe_dump(protocol_value, sort_keys=False),
        encoding="utf-8",
    )

    issues, _ = validate_development_protocol(
        protocol,
        project_root=tmp_path,
        require_locked=False,
    )

    assert "V3 evaluation policy is not locked" in issues
    assert "V3 evaluation policy is not approved by both authors" in issues
    assert "locked evaluation policy cannot retain unresolved decisions" in issues
    assert "evaluation-policy status does not match the bound policy" in issues
    assert "locked evaluation-policy readiness receipt is not pass" in issues
    assert "locked evaluation-policy receipt is not eligible" in issues
    assert "locked evaluation-policy schedule cannot be provisional" in issues
    assert (
        "locked evaluation policy requires a bound licensed-market package receipt"
        in issues
    )
