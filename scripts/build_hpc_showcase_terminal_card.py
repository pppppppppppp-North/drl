#!/usr/bin/env python3
"""Render a poster-ready terminal card from verified HPC job 3624 facts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "poster_assets" / "hpc_showcase_terminal_job_3624.png"


def main() -> None:
    fig = plt.figure(figsize=(16, 9), facecolor="#07111F")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    card = FancyBboxPatch(
        (0.035, 0.06),
        0.93,
        0.88,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.4,
        edgecolor="#24445F",
        facecolor="#0B1727",
    )
    ax.add_patch(card)

    for index, color in enumerate(("#FF5F57", "#FFBD2E", "#28C840")):
        ax.add_patch(plt.Circle((0.068 + index * 0.024, 0.902), 0.008, color=color))

    mono = "DejaVu Sans Mono"
    ax.text(0.055, 0.845, "THAI STOCK DRL  /  HPC COMPUTING", color="#E6EDF3", fontsize=25,
            fontweight="bold", fontfamily=mono)
    ax.text(0.055, 0.792, "$ sacct -j 3624  &&  inspect five-model output", color="#7DD3FC",
            fontsize=16, fontfamily=mono)
    ax.text(0.055, 0.738, "JOB 3624    COMPLETED    00:01:40    2 CPUs",
            color="#86EFAC", fontsize=17, fontfamily=mono)
    ax.text(0.055, 0.690, "Analysed 5 models x 164 observed test dates", color="#CBD5E1",
            fontsize=15, fontfamily=mono)

    rows = (
        ("PPO ", "░░░░░░░░░░░░░░░░░░░░░░░░░░░░", "-19.08%", "22.52%", "#60A5FA"),
        ("A2C ", "████████████████████████████", " -0.33%", " 0.42%", "#FBBF24"),
        ("DDPG", "████████████████████████░░░░", " -3.06%", "10.32%", "#FB923C"),
        ("TD3 ", "█████████████████████████░░░", " -2.57%", " 9.88%", "#A3E635"),
        ("SAC ", "██████████████████████░░░░░░", " -4.35%", "11.46%", "#F472B6"),
    )
    ax.text(0.055, 0.627, "MODEL   CUMULATIVE CHANGE                         FINAL     MAX DD",
            color="#64748B", fontsize=13.5, fontfamily=mono)
    y = 0.575
    for model, meter, final, drawdown, color in rows:
        ax.text(0.055, y, f"[{model}]", color=color, fontsize=16.5, fontweight="bold", fontfamily=mono)
        ax.text(0.140, y, meter, color=color, fontsize=16.5, fontfamily=mono)
        ax.text(0.700, y, final, color="#E6EDF3", fontsize=16.5, fontfamily=mono)
        ax.text(0.840, y, drawdown, color="#E6EDF3", fontsize=16.5, fontfamily=mono)
        y -= 0.073

    ax.text(0.055, 0.170, "> figure exported: cumulative_profit_3550.png", color="#7DD3FC",
            fontsize=15, fontfamily=mono)
    ax.text(0.055, 0.125, "> STATUS: COMPLETE  [no errors]", color="#86EFAC",
            fontsize=15.5, fontweight="bold", fontfamily=mono)
    ax.text(0.620, 0.085, "SMOKE TEST / NOT FINAL RANKING", color="#94A3B8",
            fontsize=12.5, fontfamily=mono)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
