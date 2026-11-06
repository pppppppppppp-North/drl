from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import yaml

from src.campaigns.bistka_deployment import (
    build_deployment,
    run_preflight,
    sha256_file,
    validate_deployment_bundle,
    validate_deployment_manifest,
    validate_preflight_receipt,
)


def _fixture_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "reports/data_quality_market").mkdir(parents=True)
    (project / "src/example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "reports/data_quality_market/market_data_readiness.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "data_origin": "observed_market",
                "evidence_class": "restricted_private_research_pilot",
                "execution_eligible": True,
            }
        ),
        encoding="utf-8",
    )
    config = project / "deployment.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "deployment": {
                    "campaign_id": "test_campaign",
                    "target_project_root": str(project),
                    "required_hostname_prefix": "bistka-",
                    "data_readiness_path": "reports/data_quality_market/market_data_readiness.json",
                    "required_packages": [],
                    "inputs": ["src/*.py", "reports/data_quality_market/*.json"],
                }
            }
        ),
        encoding="utf-8",
    )
    return project, config


def test_content_addressed_bundle_and_preflight_fail_closed(tmp_path: Path) -> None:
    project, config = _fixture_project(tmp_path)
    output_dir = project / "campaigns/test/deployment"
    manifest = output_dir / "manifest.json"
    bundle = output_dir / "release.tar.gz"
    first = build_deployment(
        project_root=project,
        config_path=config,
        output_manifest=manifest,
        output_bundle=bundle,
    )
    first_bundle = bundle.read_bytes()
    second = build_deployment(
        project_root=project,
        config_path=config,
        output_manifest=manifest,
        output_bundle=bundle,
    )
    assert first["release_id"] == second["release_id"]
    assert first_bundle == bundle.read_bytes()
    assert manifest.with_suffix(".sha256").read_text(encoding="utf-8").strip() == sha256_file(manifest)
    with tarfile.open(bundle, "r:gz") as archive:
        names = set(archive.getnames())
    assert ".bistka/deployment_manifest.json" in names
    assert "campaigns/test/deployment/manifest.json" in names
    bundle_issues, _ = validate_deployment_bundle(
        bundle,
        manifest_path=manifest,
        project_root=project,
    )
    assert bundle_issues == []

    receipt_path = project / "reports/preflight.json"
    receipt = run_preflight(
        project_root=project,
        manifest_path=manifest,
        output_path=receipt_path,
        allow_non_bistka=True,
    )
    assert receipt["status"] == "pass"

    # A local-only receipt must not satisfy the actual BistKA gate.
    issues, _ = validate_preflight_receipt(receipt_path, expected_manifest_path=manifest)
    assert "hostname" in " ".join(issues).lower()

    (project / "src/example.py").write_text("VALUE = 2\n", encoding="utf-8")
    failed = run_preflight(
        project_root=project,
        manifest_path=manifest,
        output_path=receipt_path,
        allow_non_bistka=True,
    )
    assert failed["status"] == "fail"
    assert "hash mismatch" in " ".join(failed["issues"])


def test_deployment_manifest_rejects_shadowable_preferred_feature_path(tmp_path: Path) -> None:
    project, deployment_config = _fixture_project(tmp_path)
    runtime_config = project / "config" / "screening.yaml"
    fallback = project / "data" / "features.csv"
    runtime_config.parent.mkdir(parents=True)
    fallback.parent.mkdir(parents=True)
    runtime_config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "features": "data/features.parquet",
                    "fallback_features_csv": "data/features.csv",
                }
            }
        ),
        encoding="utf-8",
    )
    fallback.write_text("date,ticker,close\n2024-01-02,A.BK,100.0\n", encoding="utf-8")

    deployment = yaml.safe_load(deployment_config.read_text(encoding="utf-8"))
    deployment["deployment"]["inputs"].extend(["config/*.yaml", "data/*.csv"])
    deployment["deployment"]["runtime_configs"] = ["config/screening.yaml"]
    deployment_config.write_text(yaml.safe_dump(deployment), encoding="utf-8")
    output_dir = project / "campaigns" / "test" / "deployment"
    manifest_path = output_dir / "manifest.json"
    with pytest.raises(
        FileNotFoundError,
        match="config/screening.yaml.*data/features.parquet",
    ):
        build_deployment(
            project_root=project,
            config_path=deployment_config,
            output_manifest=manifest_path,
            output_bundle=output_dir / "release.tar.gz",
        )


def test_deployment_and_preflight_bind_the_preferred_runtime_feature(tmp_path: Path) -> None:
    project, deployment_config = _fixture_project(tmp_path)
    runtime_config = project / "config" / "screening.yaml"
    features = project / "data" / "features.csv"
    runtime_config.parent.mkdir(parents=True)
    features.parent.mkdir(parents=True)
    runtime_config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "features": "data/features.csv",
                    "fallback_features_csv": "data/features.csv",
                }
            }
        ),
        encoding="utf-8",
    )
    features.write_text("date,ticker,close\n2024-01-02,A.BK,100.0\n", encoding="utf-8")
    deployment = yaml.safe_load(deployment_config.read_text(encoding="utf-8"))
    deployment["deployment"]["inputs"].extend(["config/*.yaml", "data/*.csv"])
    deployment["deployment"]["runtime_configs"] = ["config/screening.yaml"]
    deployment_config.write_text(yaml.safe_dump(deployment), encoding="utf-8")
    output_dir = project / "campaigns" / "test" / "deployment"
    manifest_path = output_dir / "manifest.json"
    bundle_path = output_dir / "release.tar.gz"

    built = build_deployment(
        project_root=project,
        config_path=deployment_config,
        output_manifest=manifest_path,
        output_bundle=bundle_path,
    )

    binding = built["runtime_feature_bindings"][0]
    assert binding["config_path"] == "config/screening.yaml"
    assert binding["config_sha256"] == sha256_file(runtime_config)
    assert binding["feature_path"] == "data/features.csv"
    assert binding["feature_sha256"] == sha256_file(features)
    assert binding["feature_bytes"] == features.stat().st_size
    manifest_issues, _ = validate_deployment_manifest(
        manifest_path,
        project_root=project,
    )
    assert manifest_issues == []
    receipt = run_preflight(
        project_root=project,
        manifest_path=manifest_path,
        output_path=project / "reports/runtime_preflight.json",
        allow_non_bistka=True,
    )
    assert receipt["status"] == "pass"
    assert receipt["runtime_feature_bindings"] == built["runtime_feature_bindings"]
