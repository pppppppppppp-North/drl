from __future__ import annotations

import json
from pathlib import Path

from src.campaigns.control import load_definition
from src.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = PROJECT_ROOT / "campaigns/large_scale_thai_drl_v2"
ORIGINAL_SPEC = PROJECT_ROOT / "config/research_campaign.yaml"
CUTOVER_SPEC = PROJECT_ROOT / "config/research_campaign_storage_safe.yaml"
CUTOVER_MANIFEST = (
    CAMPAIGN_DIR
    / "deployment/bistka_screening_storage_safe_cutover_v2_manifest.json"
)


def _definition(path: Path):
    definition, _ = load_definition(
        path,
        project_root=PROJECT_ROOT,
        campaign_dir=CAMPAIGN_DIR,
    )
    return definition


def test_cutover_keeps_original_contract_as_rollback() -> None:
    original = _definition(ORIGINAL_SPEC)
    cutover = _definition(CUTOVER_SPEC)

    original_smoke = original.stage_map["screening_smoke"]
    original_screening = original.stage_map["screening"]
    assert original_smoke.job_script == "slurm/screening_smoke.sbatch"
    assert original_smoke.run_glob == "results/screening_smoke_*"
    assert original_screening.job_script == "slurm/large_scale_screening.sbatch"

    smoke = cutover.stage_map["screening_smoke"]
    screening = cutover.stage_map["screening"]
    expected_manifest = (
        "{campaign_dir}/deployment/"
        "bistka_screening_storage_safe_cutover_v2_manifest.json"
    )
    expected_preflight = (
        "reports/bistka_screening_storage_safe_cutover_v2_preflight.json"
    )
    assert smoke.job_script == "slurm/screening_storage_safe_smoke.sbatch"
    assert smoke.run_glob == "results/screening_storage_safe_smoke_*"
    assert smoke.run_dir_template == "results/screening_storage_safe_smoke_{job_id}"
    assert smoke.deployment_manifest == expected_manifest
    assert smoke.preflight_receipt == expected_preflight
    assert screening.job_script == "slurm/large_scale_screening_storage_safe.sbatch"
    assert screening.deployment_manifest == expected_manifest
    assert screening.preflight_receipt == expected_preflight


def test_cutover_smoke_requires_separate_storage_safe_receipt() -> None:
    smoke = _definition(CUTOVER_SPEC).stage_map["screening_smoke"]
    requirements = {requirement.name: requirement for requirement in smoke.requirements}

    receipt = requirements["storage-safe smoke validation receipt"]
    assert receipt.patterns == ("storage_safe_smoke_validation.json",)
    assert receipt.validator == "sha256_sidecar"
    assert requirements[
        "storage-safe smoke validation SHA-256 sidecar"
    ].patterns == ("storage_safe_smoke_validation.json.sha256",)


def test_cutover_actions_use_only_the_versioned_spec() -> None:
    definition = _definition(CUTOVER_SPEC)
    actions = [stage.action for stage in definition.stages if stage.action]
    original_spec_actions = [
        action
        for action in actions
        if "--spec config/research_campaign.yaml" in action
    ]

    assert any("screening_storage_safe_cutover.yaml" in action for action in actions)
    assert len(original_spec_actions) == 1
    assert "materialize_calibrated_screening" in original_spec_actions[0]
    assert all(
        "--spec config/research_campaign_storage_safe.yaml" in action
        for action in actions
        if "--spec" in action and action not in original_spec_actions
    )


def test_cutover_release_binds_its_contract_and_operators() -> None:
    manifest = json.loads(CUTOVER_MANIFEST.read_text(encoding="utf-8"))
    paths = {record["path"] for record in manifest["files"]}
    expected = {
        "config/bistka_profiles_v2/screening_storage_safe_cutover.yaml",
        "config/research_campaign_storage_safe.yaml",
        "scripts/apply_bistka_storage_safe_cutover.sh",
        "scripts/create_campaign_cutover_snapshot.py",
        "scripts/fetch_bistka_screening_storage_safe_smoke_results.sh",
        "scripts/validate_screening_storage_safe_smoke.py",
        "slurm/large_scale_screening_storage_safe.sbatch",
        "slurm/screening_storage_safe_smoke.sbatch",
        "src/experiments/run_screening_storage_safe.py",
        "src/experiments/run_screening_storage_safe_smoke.py",
        "src/experiments/screening_storage.py",
    }
    assert expected <= paths
    assert len(manifest["release_id"]) == 64
    assert set(manifest["release_id"]) <= set("0123456789abcdef")


def test_cutover_profile_is_based_on_the_tested_storage_safe_candidate() -> None:
    profile = load_config(
        PROJECT_ROOT / "config/bistka_profiles_v2/screening_storage_safe_cutover.yaml"
    )["campaign_release_profile"]

    assert profile["base"]["manifest"].endswith(
        "/bistka_screening_storage_safe_v2_manifest.json"
    )
    assert profile["runtime_configs"] == []
