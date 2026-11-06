from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#2563EB"
BLUE_LIGHT = "#BFDBFE"
GOLD = "#D97706"
INK = "#1F2937"
GRID = "#D1D5DB"


def _positive_number(row: dict[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and positive")
    return value


def load_summary(path: Path) -> tuple[list[dict[str, float]], int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("summary_by_env_count")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("metadata has no summary_by_env_count rows")

    rows: list[dict[str, float]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("summary rows must be objects")
        row = {
            "num_envs": _positive_number(raw, "num_envs"),
            "repeats": _positive_number(raw, "repeats"),
            "median": _positive_number(raw, "median_samples_per_second"),
            "minimum": _positive_number(raw, "min_samples_per_second"),
            "maximum": _positive_number(raw, "max_samples_per_second"),
            "rss": _positive_number(raw, "max_peak_rss_mib"),
        }
        if not row["minimum"] <= row["median"] <= row["maximum"]:
            raise ValueError("throughput minimum, median, and maximum are inconsistent")
        rows.append(row)

    rows.sort(key=lambda row: row["num_envs"])
    env_counts = [int(row["num_envs"]) for row in rows]
    if len(env_counts) != len(set(env_counts)):
        raise ValueError("environment counts must be unique")
    selected = int(payload["selected_num_envs"])
    if selected not in env_counts:
        raise ValueError("selected_num_envs is absent from the summary")
    selection_fraction = float(payload.get("selection_fraction_of_max", 0.95))
    if not 0 < selection_fraction <= 1:
        raise ValueError("selection_fraction_of_max must be in (0, 1]")
    return rows, selected, selection_fraction


def render(metadata_path: Path, output_path: Path) -> None:
    rows, selected, selection_fraction = load_summary(metadata_path)
    labels = [str(int(row["num_envs"])) for row in rows]
    positions = list(range(len(rows)))
    medians = [row["median"] for row in rows]
    lower = [row["median"] - row["minimum"] for row in rows]
    upper = [row["maximum"] - row["median"] for row in rows]
    memory = [row["rss"] for row in rows]
    selected_index = labels.index(str(selected))
    threshold = max(medians) * selection_fraction
    repeats = sorted({int(row["repeats"]) for row in rows})
    repeat_text = str(repeats[0]) if len(repeats) == 1 else "/".join(map(str, repeats))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    figure, (throughput_axis, memory_axis) = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1]},
    )
    figure.patch.set_facecolor("white")

    throughput_axis.errorbar(
        positions,
        medians,
        yerr=[lower, upper],
        color=BLUE,
        marker="o",
        markersize=7,
        linewidth=2,
        capsize=5,
        ecolor=BLUE_LIGHT,
        elinewidth=5,
        label=f"Median and min–max range ({repeat_text} repeats)",
    )
    throughput_axis.scatter(
        [selected_index],
        [medians[selected_index]],
        s=110,
        facecolor=GOLD,
        edgecolor=INK,
        linewidth=1,
        zorder=5,
        label=f"Selected: {selected} environment{'s' if selected != 1 else ''}",
    )
    throughput_axis.axhline(
        threshold,
        color=INK,
        linewidth=1.3,
        linestyle="--",
        label=f"{selection_fraction:.0%} of maximum median",
    )
    throughput_axis.set_title("Observed vectorized-training calibration", loc="left", fontsize=15, weight="bold")
    throughput_axis.set_ylabel("Samples per second")
    throughput_axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    throughput_axis.spines[["top", "right"]].set_visible(False)
    throughput_axis.legend(frameon=False, loc="best", fontsize=9)

    memory_axis.plot(positions, memory, color=BLUE, marker="o", markersize=6, linewidth=2)
    memory_axis.scatter(
        [selected_index],
        [memory[selected_index]],
        s=90,
        facecolor=GOLD,
        edgecolor=INK,
        linewidth=1,
        zorder=5,
    )
    memory_axis.set_ylabel("Peak RSS (MiB)")
    memory_axis.set_xlabel("Parallel vector environments")
    memory_axis.set_xticks(positions, labels)
    memory_axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    memory_axis.spines[["top", "right"]].set_visible(False)

    job_match = re.search(r"_(\d+)\.json$", metadata_path.name)
    job_label = f"BistKA job {job_match.group(1)}" if job_match else "Accepted BistKA job"
    figure.suptitle(
        f"{job_label}; observed Thai-market training split",
        x=0.125,
        y=0.955,
        ha="left",
        fontsize=10,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an accepted vector calibration metadata file")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.metadata, args.output)


if __name__ == "__main__":
    main()
