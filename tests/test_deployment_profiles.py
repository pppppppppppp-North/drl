from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.campaigns.bistka_deployment import build_deployment, sha256_file
from src.campaigns.deployment_profiles import build_cumulative_deployment_config


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_readiness(project: Path, feature: Path | None = None) -> Path:
    path = project / "reports" / "readiness.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    if feature is not None:
        artifacts["features"] = {
            "path": str(feature.relative_to(project)),
            "bytes": feature.stat().st_size,
            "sha256": sha256_file(feature),
        }
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "data_origin": "observed_market",
                "execution_eligible": True,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return path


def _base_deployment_config(
    project: Path,
    *,
    inputs: list[str],
    runtime_configs: list[str] | None = None,
) -> Path:
    path = project / "config" / "base_deployment.yaml"
    _write_yaml(
        path,
        {
            "deployment": {
                "campaign_id": "test_campaign_v2",
                "target_project_root": str(project),
                "python_executable": "/opt/test/python",
                "required_hostname_prefix": "bistka-",
                "data_readiness_path": "reports/readiness.json",
                "required_packages": [],
                "runtime_configs": runtime_configs or [],
                "inputs": inputs,
            }
        },
    )
    return path


def _runtime_config(project: Path, relative: str, feature_relative: str) -> Path:
    path = project / relative
    _write_yaml(
        path,
        {
            "paths": {
                "features": feature_relative,
                "fallback_features_csv": feature_relative,
            },
            "features": {"lookback": 2, "columns": ["return_1d"]},
        },
    )
    return path


def _profile(path: Path, value: dict) -> Path:
    _write_yaml(path, {"campaign_release_profile": value})
    return path


def test_cumulative_profile_inherits_exact_manifest_and_builds_deterministically(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    campaign = project / "campaigns" / "test_campaign_v2"
    static = project / "src" / "static.py"
    base_feature = project / "data" / "base.csv"
    derived_feature = project / "data" / "derived.csv"
    for path, payload in (
        (static, "VALUE = 1\n"),
        (base_feature, "date,ticker,return_1d\n2024-01-01,A,0.1\n"),
        (derived_feature, "date,ticker,return_1d\n2024-01-02,A,0.2\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    readiness = _write_readiness(project, derived_feature)
    base_runtime = _runtime_config(project, "config/base_runtime.yaml", "data/base.csv")
    base_config = _base_deployment_config(
        project,
        inputs=[
            str(static.relative_to(project)),
            str(readiness.relative_to(project)),
            str(base_runtime.relative_to(project)),
            str(base_feature.relative_to(project)),
        ],
        runtime_configs=[str(base_runtime.relative_to(project))],
    )
    base_manifest = campaign / "deployment" / "initial_manifest.json"
    build_deployment(
        project_root=project,
        config_path=base_config,
        output_manifest=base_manifest,
        output_bundle=campaign / "deployment" / "initial.tar.gz",
    )

    derived_runtime = _runtime_config(
        project,
        "campaigns/test_campaign_v2/derived/locked_budget.yaml",
        "data/derived.csv",
    )
    decision = derived_runtime.with_suffix(".decision.json")
    decision.write_text('{"selected_budget": 100000}\n', encoding="utf-8")
    profile = _profile(
        project / "config" / "profiles" / "budget.yaml",
        {
            "schema_version": 1,
            "name": "budget",
            "base": {"manifest": "{campaign_dir}/deployment/initial_manifest.json"},
            "include": {
                "derived_configs": ["{campaign_dir}/derived/locked_budget.yaml"],
                "decisions": ["{campaign_dir}/derived/locked_budget.decision.json"],
            },
            "runtime_configs": ["{campaign_dir}/derived/locked_budget.yaml"],
            "feature_expectations": {
                "{campaign_dir}/derived/locked_budget.yaml": {
                    "path": "data/derived.csv",
                    "sha256": sha256_file(derived_feature),
                    "bytes": derived_feature.stat().st_size,
                }
            },
        },
    )
    generated = campaign / "deployment" / "budget_deployment.yaml"

    first = build_cumulative_deployment_config(
        project_root=project,
        profile_path=profile,
        output_path=generated,
        campaign_dir=campaign,
    )
    first_bytes = generated.read_bytes()
    second = build_cumulative_deployment_config(
        project_root=project,
        profile_path=profile,
        output_path=generated,
        campaign_dir=campaign,
    )

    assert first["deployment_config_sha256"] == second["deployment_config_sha256"]
    assert generated.read_bytes() == first_bytes
    assert first["runtime_configs"] == [
        "campaigns/test_campaign_v2/derived/locked_budget.yaml",
        "config/base_runtime.yaml",
    ]
    deployment = yaml.safe_load(generated.read_text(encoding="utf-8"))["deployment"]
    assert all(not any(character in path for character in "*?[") for path in deployment["inputs"])
    assert str(base_manifest.relative_to(project)) in deployment["inputs"]
    assert str(base_manifest.with_suffix(".sha256").relative_to(project)) in deployment["inputs"]
    assert str(generated.relative_to(project)) in deployment["inputs"]
    assert str(profile.relative_to(project)) in deployment["inputs"]

    final_manifest = campaign / "deployment" / "budget_manifest.json"
    final_bundle = campaign / "deployment" / "budget.tar.gz"
    built = build_deployment(
        project_root=project,
        config_path=generated,
        output_manifest=final_manifest,
        output_bundle=final_bundle,
    )
    first_release = built["release_id"]
    rebuilt = build_deployment(
        project_root=project,
        config_path=generated,
        output_manifest=final_manifest,
        output_bundle=final_bundle,
    )
    assert rebuilt["release_id"] == first_release
    assert {row["config_path"] for row in built["runtime_feature_bindings"]} == {
        "config/base_runtime.yaml",
        "campaigns/test_campaign_v2/derived/locked_budget.yaml",
    }


def test_feature_ablation_resolver_adds_group_configs_features_and_extends(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    campaign = project / "campaigns" / "test_campaign_v2"
    static = project / "src" / "static.py"
    static.parent.mkdir(parents=True)
    static.write_text("VALUE = 1\n", encoding="utf-8")
    _write_readiness(project)
    base = _base_deployment_config(
        project,
        inputs=["src/*.py", "reports/readiness.json"],
    )
    parent_config = project / "config" / "group_parent.yaml"
    _write_yaml(parent_config, {"features": {"lookback": 2, "columns": ["return_1d"]}})
    group_paths = []
    feature_paths = []
    for index in (1, 2):
        feature = project / "data" / f"group_{index}.csv"
        feature.parent.mkdir(parents=True, exist_ok=True)
        feature.write_text(
            f"date,ticker,return_1d\n2024-01-0{index},A,0.{index}\n",
            encoding="utf-8",
        )
        config = project / "config" / f"group_{index}.yaml"
        _write_yaml(
            config,
            {
                "extends": "group_parent.yaml",
                "paths": {
                    "features": str(feature.relative_to(project)),
                    "fallback_features_csv": str(feature.relative_to(project)),
                },
            },
        )
        group_paths.append(config)
        feature_paths.append(feature)
    orchestrator = campaign / "derived" / "locked_ablation.yaml"
    _write_yaml(
        orchestrator,
        {
            "feature_ablation": {
                "groups": [
                    {"name": f"group_{index}", "config": f"config/group_{index}.yaml"}
                    for index in (1, 2)
                ]
            }
        },
    )
    profile = _profile(
        project / "config" / "profiles" / "protocol.yaml",
        {
            "schema_version": 1,
            "name": "protocol",
            "base": {"deployment_config": str(base.relative_to(project))},
            "include": {"derived_configs": ["{campaign_dir}/derived/locked_ablation.yaml"]},
            "resolve": {
                "feature_ablation_configs": ["{campaign_dir}/derived/locked_ablation.yaml"]
            },
        },
    )
    generated = campaign / "deployment" / "protocol_deployment.yaml"

    result = build_cumulative_deployment_config(
        project_root=project,
        profile_path=profile,
        output_path=generated,
        campaign_dir=campaign.relative_to(project),
    )

    assert result["runtime_configs"] == ["config/group_1.yaml", "config/group_2.yaml"]
    deployment = yaml.safe_load(generated.read_text(encoding="utf-8"))["deployment"]
    inputs = set(deployment["inputs"])
    assert str(orchestrator.relative_to(project)) in inputs
    assert str(parent_config.relative_to(project)) in inputs
    assert {str(path.relative_to(project)) for path in group_paths}.issubset(inputs)
    assert {str(path.relative_to(project)) for path in feature_paths}.issubset(inputs)
    assert "src/static.py" in inputs
    assert "src/*.py" not in inputs

    built = build_deployment(
        project_root=project,
        config_path=generated,
        output_manifest=campaign / "deployment" / "protocol_manifest.json",
        output_bundle=campaign / "deployment" / "protocol.tar.gz",
    )
    assert len(built["runtime_feature_bindings"]) == 2


def test_stress_registry_resolver_binds_every_model_and_source_artifact(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    campaign = project / "campaigns" / "test_campaign_v2"
    static = project / "src" / "static.py"
    feature = project / "data" / "features.csv"
    for path, payload in (
        (static, "VALUE = 1\n"),
        (feature, "date,ticker,return_1d\n2024-01-01,A,0.1\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    _write_readiness(project, feature)
    base = _base_deployment_config(
        project,
        inputs=["src/static.py", "reports/readiness.json"],
    )
    artifact_payloads = {
        "results/selection.json": b'{"selected_algorithms":["ppo"]}\n',
        "results/robustness/model.zip": b"model",
        "results/robustness/test_curve.csv": b"date,portfolio_value\n2024-01-01,100\n",
        "results/robustness/experiment_manifest.json": b'{"status":"complete"}\n',
        "results/robustness/resolved_config.json": b'{"features":{}}\n',
        "results/robustness/metrics.csv": b"sharpe\n0.1\n",
    }
    artifacts: dict[str, Path] = {}
    for relative, payload in artifact_payloads.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts[relative] = path
    model = {
        "path": "results/robustness/model.zip",
        "bytes": artifacts["results/robustness/model.zip"].stat().st_size,
        "sha256": sha256_file(artifacts["results/robustness/model.zip"]),
        "source_test_curve_path": "results/robustness/test_curve.csv",
        "source_test_curve_sha256": sha256_file(
            artifacts["results/robustness/test_curve.csv"]
        ),
        "manifest_path": "results/robustness/experiment_manifest.json",
        "manifest_sha256": sha256_file(
            artifacts["results/robustness/experiment_manifest.json"]
        ),
        "resolved_config_path": "results/robustness/resolved_config.json",
        "resolved_config_sha256": sha256_file(
            artifacts["results/robustness/resolved_config.json"]
        ),
        "source_metrics_path": "results/robustness/metrics.csv",
        "source_metrics_sha256": sha256_file(artifacts["results/robustness/metrics.csv"]),
        "feature_data_sha256": sha256_file(feature),
    }
    registry = campaign / "derived" / "model_registry.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(
            {
                "screening_selection_path": "results/selection.json",
                "screening_selection_sha256": sha256_file(artifacts["results/selection.json"]),
                "feature_data_sha256": sha256_file(feature),
                "models": [model],
            }
        ),
        encoding="utf-8",
    )
    registry.with_suffix(".sha256").write_text(sha256_file(registry) + "\n", encoding="utf-8")
    stress_config = campaign / "derived" / "locked_stress.yaml"
    _write_yaml(
        stress_config,
        {
            "paths": {
                "features": "data/features.csv",
                "fallback_features_csv": "data/features.csv",
            },
            "features": {"lookback": 2, "columns": ["return_1d"]},
            "execution_stress": {
                "require_models": True,
                "model_registry": str(registry.relative_to(project)),
                "model_paths": [model],
            },
        },
    )
    profile = _profile(
        project / "config" / "profiles" / "stress.yaml",
        {
            "schema_version": 1,
            "name": "stress",
            "base": {"deployment_config": str(base.relative_to(project))},
            "include": {
                "derived_configs": ["{campaign_dir}/derived/locked_stress.yaml"],
                "registries": [
                    "{campaign_dir}/derived/model_registry.json",
                    "{campaign_dir}/derived/model_registry.sha256",
                ],
            },
            "runtime_configs": ["{campaign_dir}/derived/locked_stress.yaml"],
            "resolve": {"stress_registries": ["{campaign_dir}/derived/model_registry.json"]},
        },
    )
    generated = campaign / "deployment" / "stress_deployment.yaml"

    build_cumulative_deployment_config(
        project_root=project,
        profile_path=profile,
        output_path=generated,
        campaign_dir=campaign.relative_to(project),
    )
    inputs = set(yaml.safe_load(generated.read_text(encoding="utf-8"))["deployment"]["inputs"])
    assert set(artifact_payloads).issubset(inputs)
    assert str(registry.relative_to(project)) in inputs
    assert str(registry.with_suffix(".sha256").relative_to(project)) in inputs

    curve = artifacts["results/robustness/test_curve.csv"]
    original = curve.read_bytes()
    curve.write_bytes(b"drift")
    with pytest.raises(ValueError, match="stress registry SHA-256 mismatch"):
        build_cumulative_deployment_config(
            project_root=project,
            profile_path=profile,
            output_path=generated,
            campaign_dir=campaign.relative_to(project),
        )
    curve.write_bytes(original)

    stress_value = yaml.safe_load(stress_config.read_text(encoding="utf-8"))
    del stress_value["execution_stress"]["model_registry"]
    _write_yaml(stress_config, stress_value)
    with pytest.raises(ValueError, match="not bound by a runtime config"):
        build_cumulative_deployment_config(
            project_root=project,
            profile_path=profile,
            output_path=generated,
            campaign_dir=campaign.relative_to(project),
        )


def test_profile_rejects_broad_missing_unsafe_and_stale_preferred_inputs(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / "src" / "static.py"
    canonical = project / "data" / "canonical.csv"
    stale = project / "data" / "stale.parquet"
    for path, payload in (
        (source, "VALUE = 1\n"),
        (canonical, "date,ticker,return_1d\n2024-01-01,A,0.1\n"),
        (stale, "not-a-current-feature-table\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    _write_readiness(project, canonical)
    base = _base_deployment_config(
        project,
        inputs=["src/static.py", "reports/readiness.json"],
    )
    profile_path = project / "config" / "profile.yaml"
    output = project / "campaigns" / "test" / "deployment.yaml"

    common = {
        "schema_version": 1,
        "name": "reject",
        "base": {"deployment_config": str(base.relative_to(project))},
    }
    _profile(profile_path, {**common, "include": {"files": ["src/*.py"]}})
    with pytest.raises(ValueError, match="broad glob"):
        build_cumulative_deployment_config(
            project_root=project,
            profile_path=profile_path,
            output_path=output,
        )

    _profile(profile_path, {**common, "include": {"files": ["missing.txt"]}})
    with pytest.raises(FileNotFoundError, match="missing or unsafe"):
        build_cumulative_deployment_config(
            project_root=project,
            profile_path=profile_path,
            output_path=output,
        )

    _profile(profile_path, {**common, "include": {"files": ["../outside.txt"]}})
    with pytest.raises(FileNotFoundError, match="missing or unsafe"):
        build_cumulative_deployment_config(
            project_root=project,
            profile_path=profile_path,
            output_path=output,
        )

    runtime = project / "config" / "stale_runtime.yaml"
    _write_yaml(
        runtime,
        {
            "paths": {
                "features": str(stale.relative_to(project)),
                "fallback_features_csv": str(canonical.relative_to(project)),
            }
        },
    )
    _profile(
        profile_path,
        {**common, "runtime_configs": [str(runtime.relative_to(project))]},
    )
    with pytest.raises(ValueError, match="shadowable or stale preferred feature"):
        build_cumulative_deployment_config(
            project_root=project,
            profile_path=profile_path,
            output_path=output,
        )


def test_profile_requires_complete_and_readiness_certified_feature_expectations(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / "src" / "static.py"
    feature = project / "data" / "certified.csv"
    for path, payload in (
        (source, "VALUE = 1\n"),
        (feature, "date,ticker,return_1d\n2024-01-01,A,0.1\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    _write_readiness(project)
    base = _base_deployment_config(
        project,
        inputs=["src/static.py", "reports/readiness.json"],
    )
    runtime = _runtime_config(project, "config/runtime.yaml", "data/certified.csv")
    profile_path = project / "config" / "profile.yaml"
    output = project / "campaigns" / "test" / "deployment.yaml"
    profile = {
        "schema_version": 1,
        "name": "certified",
        "base": {"deployment_config": str(base.relative_to(project))},
        "runtime_configs": [str(runtime.relative_to(project))],
        "require_feature_expectations": True,
    }
    _profile(profile_path, profile)

    with pytest.raises(ValueError, match="lack feature expectations"):
        build_cumulative_deployment_config(
            project_root=project,
            profile_path=profile_path,
            output_path=output,
        )

    profile["feature_expectations"] = {
        "config/runtime.yaml": {
            "path": "data/certified.csv",
            "require_readiness": True,
        }
    }
    _profile(profile_path, profile)
    with pytest.raises(ValueError, match="not certified by the data-readiness receipt"):
        build_cumulative_deployment_config(
            project_root=project,
            profile_path=profile_path,
            output_path=output,
        )

    _write_readiness(project, feature)
    result = build_cumulative_deployment_config(
        project_root=project,
        profile_path=profile_path,
        output_path=output,
    )
    assert result["runtime_feature_bindings"][0]["feature_sha256"] == sha256_file(feature)

    readiness = project / "reports" / "readiness.json"
    receipt = json.loads(readiness.read_text(encoding="utf-8"))
    receipt["status"] = "fail"
    readiness.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="not a passing execution-eligible"):
        build_cumulative_deployment_config(
            project_root=project,
            profile_path=profile_path,
            output_path=output,
        )
