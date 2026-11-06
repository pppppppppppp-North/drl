from __future__ import annotations

"""Build the quantitative figures used by the KVIS current-results report.

The script reads only returned, checksum-bound project evidence.  It does not
contact BistKA, submit a scheduler job, or infer scientific performance from
the five-algorithm smoke run.
"""

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


_MPL_CACHE = Path(tempfile.gettempdir()) / (
    f"final_project_plan_kvis_matplotlib_{getattr(os, 'getuid', lambda: 0)()}"
)
_MPL_CACHE.mkdir(mode=0o700, parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#2563EB"
GOLD = "#D97706"
ORANGE = "#EA580C"
OLIVE = "#6B7D2A"
PINK = "#BE185D"
INK = "#1F2937"
MUTED = "#4B5563"
GRID = "#D1D5DB"
LIGHT = "#E5E7EB"

ALGORITHMS = ("ppo", "a2c", "ddpg", "td3", "sac")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.8,
        }
    )


def _save(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _artifact_bytes(directory: Path) -> int:
    files = [path for path in directory.rglob("*") if path.is_file()]
    if not files:
        raise ValueError(f"no returned artifacts found below {directory}")
    return sum(path.stat().st_size for path in files)


def _load_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = ["train", "validation", "test"]
    if [row.get("split") for row in rows] != expected:
        raise ValueError(f"unexpected chronological split rows in {path}")
    return rows


def _load_frozen_pilot_metrics(path: Path) -> list[dict[str, Any]]:
    """Read the accepted Markdown table without re-running the historical job."""

    text = path.read_text(encoding="utf-8")
    heading = "## Real OHLCV Validation Results"
    start = text.find(heading)
    if start < 0:
        raise ValueError(f"missing {heading!r} in {path}")
    block = text[start:]
    next_heading = block.find("\n## ", len(heading))
    if next_heading >= 0:
        block = block[:next_heading]

    rows: list[dict[str, Any]] = []
    for line in block.splitlines():
        if not line.startswith("|") or "---" in line or "policy" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            continue
        rows.append(
            {
                "policy": cells[0],
                "cumulative_return": float(cells[1]),
                "annualized_return": float(cells[2]),
                "sharpe": float(cells[3]),
                "max_drawdown": float(cells[4]),
            }
        )
    expected_policies = [
        "buy_hold_PTT.BK",
        "buy_hold_ADVANC.BK",
        "equal_weight",
        "a2c",
        "ppo",
        "mean_variance",
    ]
    if [row["policy"] for row in rows] != expected_policies:
        raise ValueError(f"unexpected frozen pilot rows in {path}")
    return rows


def plot_chronological_split(project_root: Path, output_dir: Path) -> None:
    split_path = (
        project_root
        / "results/screening_storage_safe_smoke_3550/algorithm_ppo/split_summary.csv"
    )
    rows = _load_split_rows(split_path)
    counts = [int(row["unique_dates"]) for row in rows]
    if counts != [580, 194, 194] or any(int(row["tickers"]) != 5 for row in rows):
        raise ValueError("unexpected observed-market split dimensions")

    _style()
    figure, axis = plt.subplots(figsize=(10.4, 3.7))
    figure.patch.set_facecolor("white")
    left = 0
    colors = [BLUE, GOLD, LIGHT]
    labels = ["Train", "Validation", "Test"]
    for row, count, color, label in zip(rows, counts, colors, labels):
        axis.barh(0, count, left=left, height=0.52, color=color, edgecolor=INK, linewidth=0.8)
        axis.text(
            left + count / 2,
            0,
            f"{label}\n{count} dates / {int(row['rows']):,} rows",
            ha="center",
            va="center",
            fontsize=10,
            color="white" if label == "Train" else INK,
            weight="bold",
        )
        axis.text(
            left + count / 2,
            -0.41,
            f"{row['start_date']} to {row['end_date']}",
            ha="center",
            va="top",
            fontsize=8.5,
            color=MUTED,
        )
        left += count
    axis.set_xlim(0, sum(counts))
    axis.set_ylim(-0.75, 0.55)
    axis.set_yticks([])
    axis.set_xticks([0, counts[0], counts[0] + counts[1], sum(counts)])
    axis.set_xlabel("Unique observed trading dates in chronological order")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.set_title("Chronological observed-market split", loc="left", fontsize=16, weight="bold", pad=34)
    figure.text(
        0.125,
        0.88,
        "Five fixed Thai tickers; validation supports selection and the later test period is protected for conclusions",
        fontsize=10,
        color=MUTED,
    )
    figure.text(
        0.125,
        0.01,
        "Rows equal dates x five tickers.  The split is chronological; observations are not randomly shuffled.",
        fontsize=9,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.86))
    _save(figure, output_dir / "kvis_chronological_split_3550")


def plot_pilot_sharpe(project_root: Path, output_dir: Path) -> None:
    source_path = project_root / "docs/final_methodology_results.md"
    rows = _load_frozen_pilot_metrics(source_path)
    display = {
        "buy_hold_PTT.BK": "PTT buy-and-hold",
        "buy_hold_ADVANC.BK": "ADVANC buy-and-hold",
        "equal_weight": "Equal weight",
        "a2c": "A2C",
        "ppo": "PPO",
        "mean_variance": "Mean variance",
    }
    labels = [display[row["policy"]] for row in rows]
    values = [float(row["sharpe"]) for row in rows]
    colors = [GOLD, GOLD, LIGHT, BLUE, BLUE, LIGHT]
    hatches = ["", "", "//", "xx", "xx", "//"]

    _style()
    figure, axis = plt.subplots(figsize=(9.4, 5.3))
    figure.patch.set_facecolor("white")
    y_positions = list(range(len(rows)))
    bars = axis.barh(y_positions, values, color=colors, edgecolor=INK, linewidth=0.9, height=0.64)
    for index, (bar, hatch, value) in enumerate(zip(bars, hatches, values)):
        bar.set_hatch(hatch)
        if value >= 0:
            x = value + 0.035
            ha = "left"
            value_color = INK
        else:
            x = value + 0.035
            ha = "left"
            value_color = "white" if index in {3, 4} else INK
        axis.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.6f}",
            va="center",
            ha=ha,
            fontsize=10,
            color=value_color,
            weight="bold" if value < 0 else "normal",
        )
    axis.axvline(0, color=INK, linewidth=1.2)
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(-1.25, 0.55)
    axis.set_xlabel("Reported annualized return-to-volatility ratio")
    axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_title("Real-OHLCV pilot validation risk-adjusted ratio", loc="left", fontsize=16, weight="bold", pad=34)
    figure.text(
        0.125,
        0.91,
        "Frozen May 2026 table; compounded annualized return divided by annualized volatility",
        fontsize=10,
        color=MUTED,
    )
    figure.text(
        0.125,
        0.02,
        "PPO and A2C did not outperform the simple baselines in this pilot; no whole-SET claim is supported.",
        fontsize=9,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.89))
    _save(figure, output_dir / "kvis_pilot_validation_sharpe_2855")


def plot_campaign_progress(project_root: Path, output_dir: Path) -> None:
    state_path = project_root / "campaigns/large_scale_thai_drl_v2/state.json"
    state = _load_json(state_path)
    stages = state.get("stages")
    if not isinstance(stages, list):
        raise ValueError("campaign state has no stage list")
    counts = Counter(row.get("state") for row in stages if isinstance(row, dict))
    active = [counts["VERIFIED"], counts["READY"], counts["PENDING"]]
    if active != [9, 1, 25]:
        raise ValueError(f"unexpected active campaign state: {active}")

    canonical = [11, 1, 25]
    labels = ["Canonical project plan", "Active consolidated graph"]
    rows = [canonical, active]
    colors = [BLUE, GOLD, LIGHT]
    state_labels = ["Verified", "Ready", "Pending"]

    _style()
    figure, axis = plt.subplots(figsize=(9.4, 4.2))
    figure.patch.set_facecolor("white")
    y_positions = [1, 0]
    for y, values in zip(y_positions, rows):
        left = 0
        for value, color, state_label in zip(values, colors, state_labels):
            axis.barh(y, value, left=left, height=0.48, color=color, edgecolor=INK, linewidth=0.8)
            if value >= 3:
                axis.text(
                    left + value / 2,
                    y,
                    f"{value} {state_label.lower()}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if state_label == "Verified" else INK,
                    weight="bold" if state_label != "Pending" else "normal",
                )
            else:
                axis.text(left + value / 2, y, str(value), ha="center", va="center", fontsize=10, weight="bold")
            left += value

    axis.set_yticks(y_positions, labels)
    axis.set_xlim(0, 37)
    axis.set_xlabel("Dependency-gated stages")
    axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_title("Campaign stage status", loc="left", fontsize=16, weight="bold", pad=34)
    figure.text(
        0.125,
        0.90,
        "Evidence through 17 July 2026; the active graph consolidates three lineage gates into one",
        fontsize=10,
        color=MUTED,
    )
    figure.text(
        0.125,
        0.02,
        "Canonical completion is 11/37 = 29.7%; the active 9/35 fraction is structural and is not a loss of work.",
        fontsize=9,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.88))
    _save(figure, output_dir / "kvis_campaign_progress_20260717")


def plot_runtime_projection(project_root: Path, output_dir: Path) -> None:
    summary_path = project_root / "results/screening_storage_safe_smoke_3550/screening_smoke_summary.json"
    summary = _load_json(summary_path)
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("smoke summary has no runs")

    by_algorithm: dict[str, float] = {}
    for row in runs:
        if not isinstance(row, dict):
            continue
        algorithm = row.get("algorithm")
        projection = row.get("runtime_projection")
        if algorithm in ALGORITHMS and isinstance(projection, dict):
            by_algorithm[str(algorithm)] = float(projection["projected_task_hours"])
    if tuple(by_algorithm) != ALGORITHMS:
        raise ValueError(f"unexpected algorithm order in smoke summary: {tuple(by_algorithm)}")
    values = [by_algorithm[name] for name in ALGORITHMS]
    if any(value <= 0 or value >= 72 for value in values):
        raise ValueError("runtime projections must be positive and below the declared 72-hour gate")

    _style()
    figure, axis = plt.subplots(figsize=(9.4, 5.2))
    figure.patch.set_facecolor("white")
    y_positions = list(range(len(ALGORITHMS)))
    colors = [BLUE, GOLD, ORANGE, OLIVE, PINK]
    bars = axis.barh(y_positions, values, color=colors, edgecolor=INK, linewidth=0.8, height=0.62)
    axis.axvline(72, color=INK, linewidth=1.4, linestyle="--")
    axis.text(71.2, 4.58, "72 h scheduler gate", ha="right", va="bottom", fontsize=9, color=MUTED)
    for bar, value in zip(bars, values):
        axis.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.4f} h", va="center", fontsize=10)
    axis.set_yticks(y_positions, [name.upper() for name in ALGORITHMS])
    axis.invert_yaxis()
    axis.set_xlim(0, 75)
    axis.set_xlabel("Projected hours for one full 30-trial algorithm task")
    axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_title("Full-task runtime projection by algorithm", loc="left", fontsize=16, weight="bold", pad=34)
    figure.text(
        0.125,
        0.91,
        "Job 3550 smoke observations scaled by declared work ratio with a 1.5 safety margin",
        fontsize=10,
        color=MUTED,
    )
    figure.text(
        0.125,
        0.02,
        "All five projections pass the runtime gate; this does not compare investment performance.",
        fontsize=9,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.89))
    _save(figure, output_dir / "kvis_runtime_projection_3550")


def plot_observed_storage(project_root: Path, output_dir: Path) -> None:
    baseline_dir = project_root / "results/screening_smoke_3543"
    safe_dir = project_root / "results/screening_storage_safe_smoke_3550"
    baseline_bytes = _artifact_bytes(baseline_dir)
    safe_bytes = _artifact_bytes(safe_dir)
    expected = (275_209_516, 137_777_145)
    if (baseline_bytes, safe_bytes) != expected:
        raise ValueError(
            "returned smoke artifact inventories changed: "
            f"observed {(baseline_bytes, safe_bytes)}, expected {expected}"
        )
    values_mib = [baseline_bytes / 1024**2, safe_bytes / 1024**2]
    reduction = 1 - safe_bytes / baseline_bytes

    _style()
    figure, axis = plt.subplots(figsize=(8.2, 5.1))
    figure.patch.set_facecolor("white")
    bars = axis.bar(
        [0, 1],
        values_mib,
        width=0.56,
        color=[GOLD, BLUE],
        edgecolor=INK,
        linewidth=0.9,
    )
    for bar, byte_count, value in zip(bars, [baseline_bytes, safe_bytes], values_mib):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 7,
            f"{value:.3f} MiB\n({byte_count:,} bytes)",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
    axis.annotate(
        f"{reduction:.2%} reduction",
        xy=(1, values_mib[1]),
        xytext=(0.5, 230),
        ha="center",
        fontsize=12,
        weight="bold",
        color=INK,
        arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 1.2},
    )
    axis.set_xticks([0, 1], ["Checkpoint-writing\njob 3543", "Storage-safe\njob 3550"])
    axis.set_ylim(0, 310)
    axis.set_ylabel("Returned output size (MiB; 1 MiB = $2^{20}$ bytes)")
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("Observed five-algorithm smoke output", loc="left", fontsize=16, weight="bold", pad=34)
    figure.text(
        0.125,
        0.91,
        "Complete returned file inventories; job 3550 retained final models and added checksum-bound retention evidence",
        fontsize=10,
        color=MUTED,
    )
    figure.text(
        0.125,
        0.02,
        "The comparison measures artifact storage, not training quality or algorithm return.",
        fontsize=9,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.89))
    _save(figure, output_dir / "kvis_observed_storage_3543_3550")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KVIS report figures from verified project evidence")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    plot_chronological_split(project_root, output_dir)
    plot_pilot_sharpe(project_root, output_dir)
    plot_campaign_progress(project_root, output_dir)
    plot_runtime_projection(project_root, output_dir)
    plot_observed_storage(project_root, output_dir)


if __name__ == "__main__":
    main()
