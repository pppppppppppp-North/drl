from __future__ import annotations

import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from scripts.build_v3_investability_policy_readiness import (
    V2_IMMUTABILITY_BINDINGS,
    build,
)
from src.v3.investability_policy import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_inputs(root: Path) -> None:
    relative_paths = [
        "config/v3/investability_policy_draft.yaml",
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
        "src/v3/execution_ledger.py",
        "src/v3/investability_policy.py",
        "src/v3/trading_env.py",
        "src/v3/market_data_readiness.py",
        "src/v3/development_gate.py",
        "scripts/build_v3_investability_policy_readiness.py",
        *[binding["path"] for binding in V2_IMMUTABILITY_BINDINGS.values()],
    ]
    for relative in relative_paths:
        source = PROJECT_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _build(root: Path):
    return build(
        project_root=root,
        policy_path=Path("config/v3/investability_policy_draft.yaml"),
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
        ledger_path=Path(
            "reports/data_quality_v3/"
            "v3_membership_exit_path_inventory_20260731.csv"
        ),
        receipt_path=Path(
            "reports/data_quality_v3/"
            "v3_investability_policy_readiness_20260731.json"
        ),
        chart_path=Path(
            "reports/figures/v3_membership_exit_transitions_20260731.png"
        ),
        report_path=Path(
            "reports/v3_investability_policy_readiness_20260731.md"
        ),
        generated_at_utc="2026-07-31T00:00:00+00:00",
    )


def test_investability_builder_is_deterministic_and_fail_closed(tmp_path) -> None:
    _copy_inputs(tmp_path)
    first = _build(tmp_path)
    output_paths = [
        "config/v3/investability_policy_draft.yaml.sha256",
        (
            "reports/data_quality_v3/"
            "v3_membership_exit_path_inventory_20260731.csv"
        ),
        (
            "reports/data_quality_v3/"
            "v3_membership_exit_path_inventory_20260731.csv.sha256"
        ),
        (
            "reports/data_quality_v3/"
            "v3_investability_policy_readiness_20260731.json"
        ),
        (
            "reports/data_quality_v3/"
            "v3_investability_policy_readiness_20260731.json.sha256"
        ),
        "reports/figures/v3_membership_exit_transitions_20260731.png",
        "reports/figures/v3_membership_exit_transitions_20260731.png.sha256",
        "reports/v3_investability_policy_readiness_20260731.md",
        "reports/v3_investability_policy_readiness_20260731.md.sha256",
    ]
    first_hashes = {
        relative: sha256_file(tmp_path / relative) for relative in output_paths
    }
    second = _build(tmp_path)
    second_hashes = {
        relative: sha256_file(tmp_path / relative) for relative in output_paths
    }

    assert first_hashes == second_hashes
    assert first == second
    assert first["status"] == "blocked"
    assert first["metrics"]["exit_transitions"] == 62
    assert first["metrics"]["exit_tickers"] == 54
    assert first["metrics"]["exit_batches"] == 9
    assert first["metrics"]["exit_paths_covered"] == 0
    assert first["metrics"]["unresolved_author_decisions"] == 4
    assert first["training_enabled"] is False
    assert first["hpc_jobs_submitted"] == 0

    chart = Image.open(
        tmp_path
        / "reports/figures/v3_membership_exit_transitions_20260731.png"
    )
    assert chart.size == (1920, 1080)
    report = (
        tmp_path / "reports/v3_investability_policy_readiness_20260731.md"
    ).read_text(encoding="utf-8")
    assert "62 membership exits" in report
    assert "0/62 exit paths" in report
    assert "11/37 = 29.7%" in report
    assert "No HPC job was submitted" in report

    market_receipt = (
        tmp_path
        / "reports/data_quality_v3/v3_market_package_gap_20260731.json"
    )
    market_receipt_value = json.loads(
        market_receipt.read_text(encoding="utf-8")
    )
    market_receipt_value["inputs"]["membership_panel"]["sha256"] = "0" * 64
    market_receipt.write_text(
        json.dumps(market_receipt_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(f"{market_receipt}.sha256").write_text(
        f"{sha256_file(market_receipt)}  {market_receipt.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="market-package receipt does not bind the exact membership_panel",
    ):
        _build(tmp_path)
