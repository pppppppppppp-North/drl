from __future__ import annotations

import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from scripts.build_v3_evaluation_policy_readiness import (
    V2_IMMUTABILITY_BINDINGS,
    build,
)
from src.v3.evaluation_policy import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_inputs(root: Path) -> None:
    relative_paths = [
        "config/v3/evaluation_policy_draft.yaml",
        (
            "data/processed/v3_development/"
            "set100_membership_session_mask_2021_2024.csv.gz"
        ),
        (
            "data/processed/v3_development/"
            "set100_membership_session_mask_2021_2024.csv.gz.sha256"
        ),
        (
            "reports/data_quality_v3/"
            "v3_membership_mask_readiness_20260731.json"
        ),
        (
            "reports/data_quality_v3/"
            "v3_membership_mask_readiness_20260731.json.sha256"
        ),
        "reports/data_quality_v3/v3_market_package_gap_20260731.json",
        "reports/data_quality_v3/v3_market_package_gap_20260731.json.sha256",
        "src/v3/evaluation_policy.py",
        "scripts/build_v3_evaluation_policy_readiness.py",
        *[binding["path"] for binding in V2_IMMUTABILITY_BINDINGS.values()],
    ]
    for relative in relative_paths:
        source = PROJECT_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _build(root: Path) -> dict:
    return build(
        project_root=root,
        policy_path=Path("config/v3/evaluation_policy_draft.yaml"),
        membership_path=Path(
            "data/processed/v3_development/"
            "set100_membership_session_mask_2021_2024.csv.gz"
        ),
        membership_receipt_path=Path(
            "reports/data_quality_v3/"
            "v3_membership_mask_readiness_20260731.json"
        ),
        market_receipt_path=Path(
            "reports/data_quality_v3/v3_market_package_gap_20260731.json"
        ),
        schedule_path=Path(
            "reports/data_quality_v3/"
            "v3_evaluation_fold_schedule_20260820.csv"
        ),
        summary_path=Path(
            "reports/data_quality_v3/"
            "v3_evaluation_fold_summary_20260820.csv"
        ),
        candidates_path=Path(
            "reports/data_quality_v3/v3_candidate_ledger_20260820.csv"
        ),
        receipt_path=Path(
            "reports/data_quality_v3/"
            "v3_evaluation_policy_readiness_20260820.json"
        ),
        chart_path=Path(
            "reports/figures/v3_evaluation_fold_schedule_20260820.png"
        ),
        report_path=Path(
            "reports/v3_evaluation_policy_readiness_20260820.md"
        ),
        generated_at_utc="2026-08-20T00:00:00+00:00",
    )


def _assert_sidecar(root: Path, relative: str) -> None:
    artifact = root / relative
    sidecar = Path(f"{artifact}.sha256")
    tokens = sidecar.read_text(encoding="utf-8").split()
    assert tokens == [sha256_file(artifact), artifact.name]


def test_evaluation_policy_builder_is_deterministic_and_fail_closed(
    tmp_path: Path,
) -> None:
    _copy_inputs(tmp_path)
    first = _build(tmp_path)
    output_paths = [
        "config/v3/evaluation_policy_draft.yaml.sha256",
        (
            "reports/data_quality_v3/"
            "v3_evaluation_fold_schedule_20260820.csv"
        ),
        (
            "reports/data_quality_v3/"
            "v3_evaluation_fold_schedule_20260820.csv.sha256"
        ),
        (
            "reports/data_quality_v3/"
            "v3_evaluation_fold_summary_20260820.csv"
        ),
        (
            "reports/data_quality_v3/"
            "v3_evaluation_fold_summary_20260820.csv.sha256"
        ),
        "reports/data_quality_v3/v3_candidate_ledger_20260820.csv",
        "reports/data_quality_v3/v3_candidate_ledger_20260820.csv.sha256",
        (
            "reports/data_quality_v3/"
            "v3_evaluation_policy_readiness_20260820.json"
        ),
        (
            "reports/data_quality_v3/"
            "v3_evaluation_policy_readiness_20260820.json.sha256"
        ),
        "reports/figures/v3_evaluation_fold_schedule_20260820.png",
        "reports/figures/v3_evaluation_fold_schedule_20260820.png.sha256",
        "reports/v3_evaluation_policy_readiness_20260820.md",
        "reports/v3_evaluation_policy_readiness_20260820.md.sha256",
    ]
    first_hashes = {
        relative: sha256_file(tmp_path / relative) for relative in output_paths
    }

    second = _build(tmp_path)
    second_hashes = {
        relative: sha256_file(tmp_path / relative) for relative in output_paths
    }

    assert first == second
    assert first_hashes == second_hashes
    assert first["status"] == "blocked"
    assert first["evaluation_policy_eligible"] is False
    assert first["schedule_structural_eligible"] is True
    assert first["candidate_registry_eligible"] is True
    assert first["training_enabled"] is False
    assert first["test_access_enabled"] is False
    assert first["all_access_enabled"] is False
    assert first["hpc_jobs_planned_maximum"] == 3
    assert first["hpc_jobs_submitted"] == 0
    assert first["metrics"]["fold_count"] == 6
    assert first["metrics"]["validation_transitions"] == 719
    assert first["metrics"]["candidate_configurations"] == 30
    assert first["metrics"]["planned_model_fits"] == 540
    assert first["metrics"]["planned_training_timesteps"] == 63_000_000
    assert first["metrics"]["maximum_hpc_array_jobs"] == 3
    assert first["scientific_boundary"]["no_performance_metrics_generated"] is True
    assert first["scientific_boundary"]["known_period_is_not_final_holdout"] is True

    for relative in (
        "config/v3/evaluation_policy_draft.yaml",
        "reports/data_quality_v3/v3_evaluation_fold_schedule_20260820.csv",
        "reports/data_quality_v3/v3_evaluation_fold_summary_20260820.csv",
        "reports/data_quality_v3/v3_candidate_ledger_20260820.csv",
        (
            "reports/data_quality_v3/"
            "v3_evaluation_policy_readiness_20260820.json"
        ),
        "reports/figures/v3_evaluation_fold_schedule_20260820.png",
        "reports/v3_evaluation_policy_readiness_20260820.md",
    ):
        _assert_sidecar(tmp_path, relative)

    with Image.open(
        tmp_path
        / "reports/figures/v3_evaluation_fold_schedule_20260820.png"
    ) as chart:
        assert chart.size == (1920, 1080)

    report = (
        tmp_path / "reports/v3_evaluation_policy_readiness_20260820.md"
    ).read_text(encoding="utf-8")
    assert "structural readiness, not a performance result" in report
    assert "expanding temporal folds" in report
    assert "540 model fits" in report
    assert "63,000,000 training timesteps" in report
    assert "at most **3 array-job submissions**" in report
    assert "submitted **0 HPC jobs**" in report
    assert "cannot become a new final holdout" in report
    assert "does not report investment performance" in report


def test_evaluation_builder_rejects_tampered_membership_receipt_binding(
    tmp_path: Path,
) -> None:
    _copy_inputs(tmp_path)
    receipt = (
        tmp_path
        / "reports/data_quality_v3/v3_membership_mask_readiness_20260731.json"
    )
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["outputs"]["membership_panel"]["sha256"] = "0" * 64
    receipt.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(f"{receipt}.sha256").write_text(
        f"{sha256_file(receipt)}  {receipt.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="membership receipt does not bind exact membership_panel",
    ):
        _build(tmp_path)


def test_evaluation_builder_rejects_tampered_market_receipt_binding(
    tmp_path: Path,
) -> None:
    _copy_inputs(tmp_path)
    receipt = (
        tmp_path
        / "reports/data_quality_v3/v3_market_package_gap_20260731.json"
    )
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["inputs"]["membership_panel"]["sha256"] = "0" * 64
    receipt.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(f"{receipt}.sha256").write_text(
        f"{sha256_file(receipt)}  {receipt.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="market receipt does not bind exact membership_panel",
    ):
        _build(tmp_path)
