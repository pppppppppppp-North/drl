from __future__ import annotations

import re
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.campaigns.model import CampaignConfigurationError, CampaignDefinition, CampaignStage


SBATCH_PATTERN = re.compile(r"^#SBATCH\s+--([A-Za-z0-9-]+)(?:=|\s+)(.+?)\s*$")
ARRAY_PATTERN = re.compile(r"^(\d+)-(\d+)(?::(\d+))?(?:%(\d+))?$")


def parse_sbatch_directives(path: Path) -> dict[str, str]:
    directives: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SBATCH_PATTERN.match(line)
        if match:
            directives[match.group(1)] = match.group(2).strip()
    return directives


def parse_array_shape(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = ARRAY_PATTERN.fullmatch(value)
    if not match:
        raise CampaignConfigurationError(f"unsupported SLURM array expression: {value!r}")
    start, end = int(match.group(1)), int(match.group(2))
    step = int(match.group(3) or 1)
    if end < start or step <= 0:
        raise CampaignConfigurationError(f"invalid SLURM array expression: {value!r}")
    count = ((end - start) // step) + 1
    concurrency = int(match.group(4) or count)
    return count, concurrency


def validate_slurm_stage(
    definition: CampaignDefinition,
    stage: CampaignStage,
    *,
    directives: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    if not stage.job_script:
        return None
    path = definition.project_root / stage.job_script
    directives = directives if directives is not None else parse_sbatch_directives(path)
    errors: list[str] = []
    required = {"partition", "account", "cpus-per-task", "output", "error"}
    missing = sorted(required - set(directives))
    if missing:
        errors.append(f"missing SBATCH directives {missing}")
    partition = directives.get("partition")
    if definition.limits.allowed_partitions and partition not in definition.limits.allowed_partitions:
        errors.append(f"partition {partition!r} is not allowed")
    if directives.get("account") != definition.limits.account:
        errors.append(
            f"account {directives.get('account')!r} does not match campaign account {definition.limits.account!r}"
        )
    try:
        cpus = int(directives.get("cpus-per-task", "0"))
    except ValueError:
        cpus = 0
    if cpus <= 0 or cpus > definition.limits.max_cpus_per_task:
        errors.append(f"cpus-per-task={cpus} exceeds safe range")
    array = parse_array_shape(directives.get("array"))
    actual_tasks = array[0] if array else 0
    concurrency = array[1] if array else 1
    if stage.array_tasks != actual_tasks:
        errors.append(
            f"declared array_tasks={stage.array_tasks} but {stage.job_script} defines {actual_tasks}"
        )
    if actual_tasks > definition.limits.max_array_tasks:
        errors.append(f"array task count {actual_tasks} exceeds limit {definition.limits.max_array_tasks}")
    if concurrency > definition.limits.max_concurrent_tasks:
        errors.append(
            f"array concurrency {concurrency} exceeds limit {definition.limits.max_concurrent_tasks}"
        )
    if errors:
        raise CampaignConfigurationError(f"SLURM stage {stage.id}: {'; '.join(errors)}")
    return {
        "script": stage.job_script,
        "partition": partition,
        "account": directives.get("account"),
        "cpus_per_task": cpus,
        "memory": directives.get("mem"),
        "walltime": directives.get("time"),
        "array_tasks": actual_tasks,
        "array_concurrency": concurrency,
        "output": directives.get("output"),
        "error": directives.get("error"),
    }


def validate_all_slurm_stages(definition: CampaignDefinition) -> dict[str, dict[str, Any]]:
    stages = [stage for stage in definition.stages if stage.job_script]
    unique_paths = sorted({definition.project_root / str(stage.job_script) for stage in stages})
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(unique_paths)))) as executor:
        parsed_values = list(executor.map(parse_sbatch_directives, unique_paths))
    parsed = dict(zip(unique_paths, parsed_values))
    inventory = {}
    for stage in stages:
        path = definition.project_root / str(stage.job_script)
        resources = validate_slurm_stage(definition, stage, directives=parsed[path])
        if resources:
            inventory[stage.id] = resources
    return inventory


def render_stage_command(
    definition: CampaignDefinition,
    stage: CampaignStage,
    *,
    run_dir: Path | None = None,
    protocol_hash: str | None = None,
    extra_exports: dict[str, str] | None = None,
) -> str:
    if stage.kind in {"manual", "local", "lock"}:
        if not stage.action:
            raise CampaignConfigurationError(f"stage {stage.id} does not define an action")
        try:
            campaign_relative = definition.campaign_dir.relative_to(definition.project_root)
        except ValueError as exc:
            raise CampaignConfigurationError("campaign directory must stay inside the project") from exc
        return stage.action.replace("{campaign_dir}", str(campaign_relative))
    if stage.kind != "slurm" or not stage.job_script:
        raise CampaignConfigurationError(f"stage {stage.id} is not command-renderable")

    exports = {
        "CAMPAIGN_ID": definition.campaign_id,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
    if stage.config_path:
        exports["CONFIG_PATH"] = definition.expand_project_relative(stage.config_path)
    if protocol_hash:
        exports["PROTOCOL_LOCK_HASH"] = protocol_hash
    if run_dir is not None:
        try:
            relative_run_dir = run_dir.resolve().relative_to(definition.project_root)
        except ValueError as exc:
            raise CampaignConfigurationError("run directory must stay inside the project") from exc
        exports[stage.run_dir_env] = str(relative_run_dir)
    exports.update(extra_exports or {})
    for name, value in exports.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or not re.fullmatch(r"[A-Za-z0-9_./:-]+", value):
            raise CampaignConfigurationError(f"unsafe SLURM export {name}={value!r}")
    export_value = ",".join(f"{key}={value}" for key, value in sorted(exports.items()))
    command = " ".join(
        [
            "sbatch",
            "--parsable",
            f"--export={shlex.quote(export_value)}",
            shlex.quote(stage.job_script),
        ]
    )
    variable = stage.id.upper() + "_SUBMISSION"
    job_variable = stage.id.upper() + "_JOB_ID"
    lines = [
        "mkdir -p logs",
        f"{variable}=$({command})",
        f"{job_variable}=\"${{{variable}%%;*}}\"",
    ]
    if stage.run_dir_template:
        template = stage.run_dir_template.replace("{job_id}", f"${{{job_variable}}}")
        lines.append(f"echo \"run_dir={template}\"")
    lines.append(f"echo \"stage={stage.id} job_id=${{{job_variable}}}\"")
    return "\n".join(lines)
