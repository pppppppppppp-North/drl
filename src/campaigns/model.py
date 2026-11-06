from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


SAFE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
SAFE_EXPORT_VALUE = re.compile(r"^[A-Za-z0-9_./:-]+$")


class CampaignConfigurationError(ValueError):
    pass


class StageState(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    VERIFY_PENDING = "VERIFY_PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    STALE = "STALE"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class ArtifactRequirement:
    name: str
    patterns: tuple[str, ...]
    minimum: int = 1
    validator: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ArtifactRequirement":
        patterns = value.get("patterns", value.get("pattern", []))
        if isinstance(patterns, str):
            patterns = [patterns]
        return cls(
            name=str(value.get("name", "artifact")),
            patterns=tuple(str(item) for item in patterns),
            minimum=int(value.get("minimum", 1)),
            validator=str(value["validator"]) if value.get("validator") else None,
            options=dict(value.get("options", {})),
        )


@dataclass(frozen=True)
class CampaignStage:
    id: str
    title: str
    kind: str
    depends_on: tuple[str, ...] = ()
    run_glob: str | None = None
    run_source: str | None = None
    requirements: tuple[ArtifactRequirement, ...] = ()
    manifest_pattern: str | None = None
    expected_manifests: int = 0
    require_protocol_hash: bool = False
    job_script: str | None = None
    config_path: str | None = None
    runtime_config_paths: tuple[str, ...] = ()
    deployment_manifest: str | None = None
    preflight_receipt: str | None = None
    run_dir_template: str | None = None
    run_dir_env: str = "RUN_DIR"
    action: str | None = None
    fingerprint_paths: tuple[str, ...] = ()
    missing_state: StageState = StageState.READY
    planned_runs: int = 0
    planned_timesteps: int = 0
    array_tasks: int = 0

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "CampaignStage":
        runtime_configs_value = value.get("runtime_configs")
        if "runtime_configs" in value:
            if not isinstance(runtime_configs_value, list) or not runtime_configs_value:
                raise CampaignConfigurationError(
                    f"stage {value.get('id', '<unknown>')} runtime_configs must be a non-empty list"
                )
            if any(
                not isinstance(item, str) or not item.strip()
                for item in runtime_configs_value
            ):
                raise CampaignConfigurationError(
                    f"stage {value.get('id', '<unknown>')} runtime_configs contains an invalid path"
                )
            if len(runtime_configs_value) != len(set(runtime_configs_value)):
                raise CampaignConfigurationError(
                    f"stage {value.get('id', '<unknown>')} runtime_configs must be unique"
                )
        else:
            runtime_configs_value = []
        return cls(
            id=str(value["id"]),
            title=str(value.get("title", value["id"])),
            kind=str(value.get("kind", "manual")),
            depends_on=tuple(str(item) for item in value.get("depends_on", [])),
            run_glob=str(value["run_glob"]) if value.get("run_glob") else None,
            run_source=str(value["run_source"]) if value.get("run_source") else None,
            requirements=tuple(
                ArtifactRequirement.from_mapping(item) for item in value.get("requirements", [])
            ),
            manifest_pattern=str(value["manifest_pattern"]) if value.get("manifest_pattern") else None,
            expected_manifests=int(value.get("expected_manifests", 0)),
            require_protocol_hash=bool(value.get("require_protocol_hash", False)),
            job_script=str(value["job_script"]) if value.get("job_script") else None,
            config_path=str(value["config_path"]) if value.get("config_path") else None,
            runtime_config_paths=tuple(str(item) for item in runtime_configs_value),
            deployment_manifest=(
                str(value["deployment_manifest"]) if value.get("deployment_manifest") else None
            ),
            preflight_receipt=(
                str(value["preflight_receipt"]) if value.get("preflight_receipt") else None
            ),
            run_dir_template=str(value["run_dir_template"]) if value.get("run_dir_template") else None,
            run_dir_env=str(value.get("run_dir_env", "RUN_DIR")),
            action=str(value["action"]) if value.get("action") else None,
            fingerprint_paths=tuple(str(item) for item in value.get("fingerprint_paths", [])),
            missing_state=StageState(str(value.get("missing_state", "READY")).upper()),
            planned_runs=int(value.get("planned_runs", 0)),
            planned_timesteps=int(value.get("planned_timesteps", 0)),
            array_tasks=int(value.get("array_tasks", 0)),
        )


@dataclass(frozen=True)
class CampaignLimits:
    account: str
    allowed_partitions: tuple[str, ...]
    max_array_tasks: int
    max_concurrent_tasks: int
    max_cpus_per_task: int
    max_rendered_submissions: int


@dataclass(frozen=True)
class LockInput:
    name: str
    pattern: str
    select: str = "latest"


@dataclass(frozen=True)
class CampaignDefinition:
    campaign_id: str
    title: str
    project_root: Path
    campaign_dir: Path
    spec_path: Path
    limits: CampaignLimits
    stages: tuple[CampaignStage, ...]
    protocol_lock_path: Path
    lock_inputs: tuple[LockInput, ...]
    decisions: dict[str, Any]
    strict_deployment_resolution: bool = False

    @property
    def stage_map(self) -> dict[str, CampaignStage]:
        return {stage.id: stage for stage in self.stages}

    def expand_project_relative(self, raw_path: str) -> str:
        try:
            campaign_relative = self.campaign_dir.relative_to(self.project_root)
        except ValueError as exc:
            raise CampaignConfigurationError("campaign directory must stay inside the project") from exc
        return raw_path.replace("{campaign_dir}", str(campaign_relative))

    @classmethod
    def from_mapping(
        cls,
        config: dict[str, Any],
        *,
        project_root: Path,
        campaign_dir: Path,
        spec_path: Path,
    ) -> "CampaignDefinition":
        campaign = dict(config.get("campaign", {}))
        limits = dict(campaign.get("limits", {}))
        lock = dict(campaign.get("protocol_lock", {}))
        definition = cls(
            campaign_id=str(campaign.get("id", "large_scale_thai_drl")),
            title=str(campaign.get("title", "Large-Scale Thai DRL Research Campaign")),
            project_root=project_root.resolve(),
            campaign_dir=campaign_dir.resolve(),
            spec_path=spec_path.resolve(),
            limits=CampaignLimits(
                account=str(limits.get("account", "25sfcs03")),
                allowed_partitions=tuple(str(item) for item in limits.get("allowed_partitions", [])),
                max_array_tasks=int(limits.get("max_array_tasks", 5)),
                max_concurrent_tasks=int(limits.get("max_concurrent_tasks", 2)),
                max_cpus_per_task=int(limits.get("max_cpus_per_task", 32)),
                max_rendered_submissions=int(limits.get("max_rendered_submissions", 1)),
            ),
            stages=tuple(CampaignStage.from_mapping(item) for item in campaign.get("stages", [])),
            protocol_lock_path=(campaign_dir / str(lock.get("path", "protocol_lock.json"))).resolve(),
            lock_inputs=tuple(
                LockInput(
                    name=str(item["name"]),
                    pattern=str(item["pattern"]),
                    select=str(item.get("select", "latest")),
                )
                for item in lock.get("inputs", [])
            ),
            decisions=dict(campaign.get("decisions", {})),
            strict_deployment_resolution=bool(
                campaign.get("strict_deployment_resolution", False)
            ),
        )
        definition.validate()
        return definition

    def validate(self) -> None:
        errors: list[str] = []
        if not SAFE_ID.fullmatch(self.campaign_id):
            errors.append(f"unsafe campaign id: {self.campaign_id!r}")
        if not self.stages:
            errors.append("campaign stages must not be empty")
        ids = [stage.id for stage in self.stages]
        if len(ids) != len(set(ids)):
            errors.append("campaign stage ids must be unique")
        known = set(ids)
        for stage in self.stages:
            if not SAFE_ID.fullmatch(stage.id):
                errors.append(f"unsafe stage id: {stage.id!r}")
            unknown = sorted(set(stage.depends_on) - known)
            if unknown:
                errors.append(f"stage {stage.id} has unknown dependencies: {unknown}")
            if stage.id in stage.depends_on:
                errors.append(f"stage {stage.id} cannot depend on itself")
            if stage.run_source and stage.run_source not in known:
                errors.append(f"stage {stage.id} has unknown run_source: {stage.run_source}")
            if stage.run_source and stage.run_source == stage.id:
                errors.append(f"stage {stage.id} cannot use itself as run_source")
            if stage.expected_manifests < 0 or stage.planned_runs < 0 or stage.planned_timesteps < 0:
                errors.append(f"stage {stage.id} has negative workload values")
            if stage.array_tasks < 0 or stage.array_tasks > self.limits.max_array_tasks:
                errors.append(
                    f"stage {stage.id} array_tasks={stage.array_tasks} exceeds limit {self.limits.max_array_tasks}"
                )
            if stage.job_script:
                self._validate_relative_file(stage.job_script, f"stage {stage.id} job_script", errors)
            if stage.config_path:
                self._validate_relative_file(stage.config_path, f"stage {stage.id} config_path", errors, optional=True)
            for runtime_config_path in stage.runtime_config_paths:
                self._validate_relative_file(
                    runtime_config_path,
                    f"stage {stage.id} runtime_config",
                    errors,
                    optional=True,
                )
            if stage.deployment_manifest:
                self._validate_relative_file(
                    stage.deployment_manifest,
                    f"stage {stage.id} deployment_manifest",
                    errors,
                    optional=True,
                )
            if stage.preflight_receipt:
                self._validate_relative_file(
                    stage.preflight_receipt,
                    f"stage {stage.id} preflight_receipt",
                    errors,
                    optional=True,
                )
            for fingerprint_path in stage.fingerprint_paths:
                self._validate_relative_file(
                    fingerprint_path,
                    f"stage {stage.id} fingerprint_path",
                    errors,
                )
            if stage.run_dir_template and not SAFE_EXPORT_VALUE.fullmatch(
                stage.run_dir_template.replace("{job_id}", "123")
            ):
                errors.append(f"unsafe run_dir_template for stage {stage.id}: {stage.run_dir_template!r}")
            for requirement in stage.requirements:
                if not requirement.patterns or requirement.minimum <= 0:
                    errors.append(f"stage {stage.id} has an invalid artifact requirement: {requirement.name}")
                for pattern in requirement.patterns:
                    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                        errors.append(f"unsafe artifact pattern for stage {stage.id}: {pattern!r}")
        if self.strict_deployment_resolution:
            for stage in self.stages:
                if stage.kind != "slurm":
                    continue
                if not stage.config_path:
                    errors.append(
                        f"strict SLURM stage {stage.id} must declare config_path"
                    )
                try:
                    self.resolve_deployment_pair(stage)
                except CampaignConfigurationError as exc:
                    errors.append(str(exc))
                try:
                    self.resolve_runtime_config_paths(stage)
                except CampaignConfigurationError as exc:
                    errors.append(str(exc))
        if not errors:
            try:
                self.topological_stages()
            except CampaignConfigurationError as exc:
                errors.append(str(exc))
        if errors:
            raise CampaignConfigurationError("; ".join(errors))

    def resolve_deployment_pair(
        self,
        stage: CampaignStage | str,
    ) -> tuple[str, str]:
        """Resolve the exact deployment/preflight pair for a SLURM stage.

        Strict execution stages declare the pair themselves. Aggregate stages
        may inherit it through one or more ``run_source`` links, but a partial
        declaration, missing source, non-SLURM source, or source cycle is an
        error. This deliberately has no global fallback.
        """

        current = self.stage_map[stage] if isinstance(stage, str) else stage
        seen: list[str] = []
        while True:
            if current.id in seen:
                chain = " -> ".join([*seen, current.id])
                raise CampaignConfigurationError(
                    f"strict deployment run_source cycle: {chain}"
                )
            seen.append(current.id)
            manifest = current.deployment_manifest
            preflight = current.preflight_receipt
            if bool(manifest) != bool(preflight):
                raise CampaignConfigurationError(
                    f"strict SLURM stage {current.id} must declare both "
                    "deployment_manifest and preflight_receipt"
                )
            if manifest and preflight:
                return manifest, preflight
            if not current.run_source:
                raise CampaignConfigurationError(
                    f"strict SLURM execution stage {current.id} must declare both "
                    "deployment_manifest and preflight_receipt"
                )
            source = self.stage_map.get(current.run_source)
            if source is None:
                raise CampaignConfigurationError(
                    f"strict SLURM stage {current.id} cannot inherit deployment from "
                    f"unknown run_source {current.run_source!r}"
                )
            if source.kind != "slurm":
                raise CampaignConfigurationError(
                    f"strict SLURM stage {current.id} cannot inherit deployment from "
                    f"non-SLURM run_source {source.id}"
                )
            current = source

    def resolve_runtime_config_paths(
        self,
        stage: CampaignStage | str,
    ) -> tuple[str, ...]:
        """Resolve explicit runtime configs through a run-source chain.

        A stage's explicit list wins. Otherwise the first explicit list on its
        transitive run source is inherited. If no stage in the chain declares
        one, the requesting stage's primary ``config_path`` is the runtime
        config. This final fallback is stage-local and never global.
        """

        requested = self.stage_map[stage] if isinstance(stage, str) else stage
        current = requested
        seen: list[str] = []
        while True:
            if current.id in seen:
                chain = " -> ".join([*seen, current.id])
                raise CampaignConfigurationError(
                    f"strict runtime-config run_source cycle: {chain}"
                )
            seen.append(current.id)
            if current.runtime_config_paths:
                return current.runtime_config_paths
            if not current.run_source:
                break
            source = self.stage_map.get(current.run_source)
            if source is None:
                raise CampaignConfigurationError(
                    f"strict SLURM stage {current.id} cannot inherit runtime configs from "
                    f"unknown run_source {current.run_source!r}"
                )
            if source.kind != "slurm":
                raise CampaignConfigurationError(
                    f"strict SLURM stage {current.id} cannot inherit runtime configs from "
                    f"non-SLURM run_source {source.id}"
                )
            current = source
        if not requested.config_path:
            raise CampaignConfigurationError(
                f"strict SLURM stage {requested.id} has no effective runtime config"
            )
        return (requested.config_path,)

    def _validate_relative_file(
        self,
        raw_path: str,
        label: str,
        errors: list[str],
        *,
        optional: bool = False,
    ) -> None:
        raw_path = self.expand_project_relative(raw_path)
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"{label} must stay inside the project: {raw_path!r}")
            return
        resolved = (self.project_root / path).resolve()
        if not resolved.is_relative_to(self.project_root):
            errors.append(f"{label} escapes the project: {raw_path!r}")
        if not optional and not resolved.is_file():
            errors.append(f"{label} does not exist: {raw_path!r}")

    def topological_stages(self) -> list[CampaignStage]:
        stages = self.stage_map
        indegree = {stage.id: 0 for stage in self.stages}
        children = {stage.id: [] for stage in self.stages}
        for stage in self.stages:
            for parent in stage.depends_on:
                indegree[stage.id] += 1
                children[parent].append(stage.id)
        ready = [stage.id for stage in self.stages if indegree[stage.id] == 0]
        ordered: list[CampaignStage] = []
        while ready:
            current = ready.pop(0)
            ordered.append(stages[current])
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(ordered) != len(self.stages):
            raise CampaignConfigurationError("campaign stage graph contains a cycle")
        return ordered

    def workload_summary(self) -> dict[str, int]:
        return {
            "planned_model_runs": sum(stage.planned_runs for stage in self.stages),
            "planned_training_timesteps": sum(stage.planned_timesteps for stage in self.stages),
            "array_elements": sum(stage.array_tasks for stage in self.stages),
        }
