from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.campaigns.gates import StageStatus, atomic_json, protocol_lock_status
from src.campaigns.model import CampaignDefinition, StageState


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_status_markdown(definition: CampaignDefinition, statuses: list[StageStatus]) -> str:
    counts = Counter(status.state for status in statuses)
    workload = definition.workload_summary()
    lock_valid, lock_hash, lock_issues = protocol_lock_status(definition)
    blockers = [status for status in statuses if status.state in {StageState.BLOCKED, StageState.FAILED, StageState.STALE}]
    ready = [status for status in statuses if status.state == StageState.READY]
    status_map = {status.stage_id: status for status in statuses}
    data_verified = status_map.get("data_readiness") is not None and status_map["data_readiness"].state == StageState.VERIFIED
    capacity_verified = status_map.get("capacity") is not None and status_map["capacity"].state == StageState.VERIFIED
    deployment_verified = (
        status_map.get("deployment_bundle") is not None
        and status_map["deployment_bundle"].state == StageState.VERIFIED
    )
    preflight_verified = (
        status_map.get("cluster_preflight") is not None
        and status_map["cluster_preflight"].state == StageState.VERIFIED
    )
    lines = [
        f"# {definition.title}: Live Campaign Status",
        "",
        "## Technical Summary",
        "",
        (
            f"The campaign contains **{len(statuses)} dependency-gated stages**, "
            f"**{workload['planned_model_runs']:,} planned model runs**, "
            f"**{workload['planned_training_timesteps']:,} requested training timesteps**, and "
            f"**{workload['array_elements']} bounded SLURM array elements**. "
            f"{counts[StageState.VERIFIED]} stages are explicitly verified; "
            f"{counts[StageState.READY]} are ready for an operator command."
        ),
        "",
        (
            f"The scientific protocol lock is **{'valid' if lock_valid else 'not yet valid'}**"
            + (f" (`{lock_hash[:12]}…`)." if lock_hash else ".")
            + " Test-reading stages remain pending until validation-only selections are materialized and locked."
        ),
        "",
        "## The Funnel Advances Only Through Verified Gates",
        "",
        "```mermaid",
        "flowchart TD",
    ]
    for stage in definition.topological_stages():
        state = next(item.state.value for item in statuses if item.stage_id == stage.id)
        lines.append(f'  {stage.id}["{stage.title}<br/>{state}"]')
        for dependency in stage.depends_on:
            lines.append(f"  {dependency} --> {stage.id}")
    lines.extend(
        [
            "```",
            "",
            "Each compute array and its aggregate are separate stages. This prevents the controller from pre-submitting the whole DAG or consuming BistKA's small submitted-job allowance before prior artifacts have been verified.",
            "",
            "## Stage Evidence and Readiness",
            "",
            "| Stage | State | Progress | Evidence directory | Gate interpretation |",
            "|---|---|---:|---|---|",
        ]
    )
    for status in statuses:
        run_dir = f"`{status.run_dir}`" if status.run_dir else "—"
        reason = status.reason.replace("|", "\\|")
        lines.append(
            f"| `{status.stage_id}` | **{status.state.value}** | {status.progress} | {run_dir} | {reason} |"
        )
    lines.extend(["", "## Current Blocking Evidence", ""])
    if blockers:
        for status in blockers:
            lines.append(f"- **{status.title}:** {status.reason}")
            for issue in status.gate.issues[:3]:
                lines.append(f"  - {issue}")
    else:
        lines.append("No stage is currently failed, stale, or externally blocked.")
    if lock_issues:
        lines.append(
            f"- **Pending protocol-lock prerequisite:** {'; '.join(lock_issues)} "
            "This is expected before validation selections are frozen."
        )
    lines.extend(["", "## Next Safe Operator Action", ""])
    if ready:
        first = ready[0]
        lines.append(
            f"Render the command for **{first.title}** with "
            f"`python -m src.campaigns.control commands --stage {first.stage_id}`. "
            "The controller only writes or prints a command; it never submits it."
        )
    else:
        lines.append("No compute submission is currently ready. Resolve and explicitly verify the earliest blocked or pending gate first.")
    data_safety = (
        "- The verified observed-market receipt rejects generated-origin data and detects drift in the frozen stock, index, feature, or source-manifest files."
        if data_verified
        else "- Missing real data remains a hard block; the controller never downloads or substitutes synthetic data."
    )
    recommendations = []
    if not data_verified:
        recommendations.append("Supply and verify the real OHLCV, SET-index, processed features, and observed-market readiness receipt.")
    if not capacity_verified:
        recommendations.append("Capture and verify the live BistKA credit, storage, partition, and queue snapshot before rendering calibration work.")
    if data_verified and capacity_verified and not deployment_verified:
        recommendations.append(
            "Build and verify the content-addressed initial BistKA deployment bundle."
        )
    elif deployment_verified and not preflight_verified:
        recommendations.append(
            "Inspect the deployment dry run, then apply the exact release only with explicit operator intent and verify the returned BistKA preflight receipt."
        )
    elif preflight_verified:
        recommendations.append(
            "Run the observed vector-throughput and independent-process determinism calibrations one at a time, then verify their attempt-bound receipts."
        )
    recommendations.extend(
        [
            "Advance one compute or aggregate stage at a time; verify each gate before rendering the next command.",
            "Materialize validation selections, create the immutable protocol lock, and only then expose test-reading stages.",
            "Let the controller register the fixed-seed model grid, run window-aligned stress tests, and verify the final archive before using results in the abstract or conclusion.",
        ]
    )
    lines.extend(
        [
            "",
            "## Safety and Scientific Guarantees",
            "",
            data_safety,
            "- A scheduler exit is not success. `_SUCCESS.json` appears only after exact artifact and manifest verification.",
            "- Selection stages are validation-only. Post-lock test stages require the immutable protocol-lock hash in every experiment manifest.",
            "- Old verified evidence becomes `STALE` when its configuration, code-facing script, data artifact, or protocol fingerprint changes.",
            "- Command rendering is one stage at a time and contains at most one `sbatch` invocation.",
            "",
            "## Limitations and Uncertainty",
            "",
            "This report describes orchestration readiness, not investment performance. A verified data gate establishes integrity for the declared execution scope; it does not grant redistribution rights or make a restricted pilot publication-grade. No missing experiment is treated as evidence, and no model conclusion is available until the robustness, stress, ablation, and archive gates are verified. The verified BistKA snapshot covers credit, storage quota, live modules, visible partition and queue state, plus the five-submitted/two-running/32-CPU/two-GPU group policy and partition walltimes. It must be recaptured if cluster policy changes.",
            "",
            "## Recommended Next Steps",
            "",
            *[f"{index}. {recommendation}" for index, recommendation in enumerate(recommendations, start=1)],
            "",
            "## Further Questions",
            "",
            "- Will live BistKA `MaxSubmit` treat each array element as a submitted job for this account?",
            "- Can trustworthy historical data support more than the current two broad test windows?",
            "- Should a preregistered sensitivity analysis repeat execution stress for additional seeds while preserving seed 42 as the primary registry?",
            "",
            f"Generated: `{_utc_now()}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_campaign_outputs(
    definition: CampaignDefinition,
    statuses: list[StageStatus],
) -> tuple[Path, Path]:
    state_path = definition.campaign_dir / "state.json"
    report_path = definition.campaign_dir / "reports" / "campaign_status.md"
    atomic_json(
        state_path,
        {
            "campaign_id": definition.campaign_id,
            "generated_at": _utc_now(),
            "workload": definition.workload_summary(),
            "stages": [status.to_dict() for status in statuses],
        },
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_status_markdown(definition, statuses), encoding="utf-8")
    return state_path, report_path
