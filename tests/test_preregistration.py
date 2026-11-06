from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.campaigns.preregistration import (
    preregistration_gate_validator,
    validate_preregistration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = PROJECT_ROOT / "config" / "research_preregistration.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sidecar(path: Path) -> None:
    path.with_suffix(".sha256").write_text(_sha256(path) + "\n", encoding="utf-8")


def _copy_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    config = project / "config" / PREREGISTRATION.name
    receipt = project / "reports" / "data_quality_market" / "market_data_readiness.json"
    config.parent.mkdir(parents=True)
    receipt.parent.mkdir(parents=True)
    shutil.copy2(PREREGISTRATION, config)
    shutil.copy2(PREREGISTRATION.with_suffix(".sha256"), config.with_suffix(".sha256"))
    source_receipt = PROJECT_ROOT / "reports" / "data_quality_market" / receipt.name
    shutil.copy2(source_receipt, receipt)
    shutil.copy2(source_receipt.with_suffix(".sha256"), receipt.with_suffix(".sha256"))
    return project, config, receipt


def test_repository_preregistration_passes_and_exposes_gate_references() -> None:
    issues, referenced = validate_preregistration(PREREGISTRATION, project_root=PROJECT_ROOT)

    assert issues == []
    assert PREREGISTRATION.with_suffix(".sha256") in referenced
    assert PROJECT_ROOT / "reports" / "data_quality_market" / "market_data_readiness.json" in referenced
    assert PROJECT_ROOT / "reports" / "data_quality_market" / "market_data_readiness.sha256" in referenced

    gate_issues, gate_references = preregistration_gate_validator(
        SimpleNamespace(project_root=PROJECT_ROOT),
        PREREGISTRATION,
    )
    assert gate_issues == []
    assert gate_references == referenced


def test_validator_rejects_resigned_publication_and_test_selection_changes(tmp_path: Path) -> None:
    project, config, _ = _copy_bundle(tmp_path)
    value = yaml.safe_load(config.read_text(encoding="utf-8"))
    value["scope"]["publication_claim_eligible"] = True
    value["analysis_plan"]["selection"]["split"] = "test"
    value["analysis_plan"]["selection"]["test_data_allowed"] = True
    value["analysis_plan"]["post_test_tuning"] = "allowed"
    value["primary_paired_outcomes"][0]["candidate_metric"] = "annualized_return"
    config.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    _write_sidecar(config)

    issues, _ = validate_preregistration(config, project_root=project)
    message = " ".join(issues)

    assert "publication_claim_eligible must be false" in message
    assert "selection must be validation-only" in message
    assert "post-test tuning must be forbidden" in message
    assert "excess_cumulative_return has changed metric or direction" in message
    assert "sidecar does not match" not in message


def test_validator_rejects_rebound_synthetic_receipt(tmp_path: Path) -> None:
    project, config, receipt = _copy_bundle(tmp_path)
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_value["data_origin"] = "generated"
    receipt_value["publication_claim_eligible"] = True
    receipt.write_text(json.dumps(receipt_value, indent=2) + "\n", encoding="utf-8")
    _write_sidecar(receipt)

    config_value = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_value["evidence_binding"]["readiness_receipt"]["sha256"] = _sha256(receipt)
    config.write_text(yaml.safe_dump(config_value, sort_keys=False), encoding="utf-8")
    _write_sidecar(config)

    issues, _ = validate_preregistration(config, project_root=project)
    message = " ".join(issues)

    assert "data_origin must be 'observed_market'" in message
    assert "publication_claim_eligible must be False" in message
    assert "sidecar does not match" not in message


def test_validator_rejects_receipt_drift_and_missing_preregistration_sidecar(tmp_path: Path) -> None:
    project, config, receipt = _copy_bundle(tmp_path)
    receipt.write_text(receipt.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    config.with_suffix(".sha256").unlink()

    issues, _ = validate_preregistration(config, project_root=project)
    message = " ".join(issues)

    assert "preregistration SHA-256 sidecar is missing" in message
    assert "readiness receipt SHA-256 sidecar does not match" in message
    assert "bound readiness receipt SHA-256 does not match" in message
