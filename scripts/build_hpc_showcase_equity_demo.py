#!/usr/bin/env python3
"""Build poster-ready equity visuals from the five-model HPC smoke run.

The output is deliberately labelled as an illustrative pipeline check.  The
underlying job was designed to validate storage and workflow behaviour, not to
support a final scientific ranking or an investment conclusion.
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


ALGORITHMS = ("PPO", "A2C", "DDPG", "TD3", "SAC")
COLORS = {
    "PPO": "#1769AA",
    "A2C": "#D99A00",
    "DDPG": "#E66101",
    "TD3": "#6B8E23",
    "SAC": "#C51B7D",
}
LINESTYLES = {
    "PPO": "-",
    "A2C": "--",
    "DDPG": ":",
    "TD3": "-.",
    "SAC": "-",
}


@dataclass(frozen=True)
class Curve:
    algorithm: str
    dates: list[datetime]
    values: list[float]
    drawdowns: list[float]

    @property
    def cumulative_profit_pct(self) -> list[float]:
        initial = self.values[0]
        return [(value / initial - 1.0) * 100.0 for value in self.values]

    @property
    def final_profit_pct(self) -> float:
        return self.cumulative_profit_pct[-1]

    @property
    def max_drawdown_pct(self) -> float:
        return max(self.drawdowns) * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--capture-seconds", type=int, default=0)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "3550"))
    return parser.parse_args()


def load_curve(input_dir: Path, algorithm: str) -> Curve:
    path = input_dir / f"algorithm_{algorithm.lower()}" / f"{algorithm.lower()}_equity_curve.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing equity curve: {path}")

    dates: list[datetime] = []
    values: list[float] = []
    drawdowns: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dates.append(datetime.strptime(row["date"], "%Y-%m-%d"))
            values.append(float(row["portfolio_value"]))
            drawdowns.append(float(row["drawdown"]))

    if len(dates) < 2 or values[0] <= 0:
        raise ValueError(f"Invalid curve in {path}")
    return Curve(algorithm, dates, values, drawdowns)


def validate_curves(curves: list[Curve]) -> None:
    reference_dates = curves[0].dates
    for curve in curves[1:]:
        if curve.dates != reference_dates:
            raise ValueError(f"Date mismatch for {curve.algorithm}")


def write_summary(curves: list[Curve], output_dir: Path) -> Path:
    output_path = output_dir / "hpc_showcase_model_summary_3550.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "algorithm",
                "observed_test_dates",
                "initial_portfolio_value",
                "final_portfolio_value",
                "cumulative_profit_pct",
                "max_drawdown_pct",
                "evidence_status",
            ]
        )
        for curve in curves:
            writer.writerow(
                [
                    curve.algorithm,
                    len(curve.dates),
                    f"{curve.values[0]:.2f}",
                    f"{curve.values[-1]:.2f}",
                    f"{curve.final_profit_pct:.4f}",
                    f"{curve.max_drawdown_pct:.4f}",
                    "illustrative smoke-test result; not final ranking",
                ]
            )
    return output_path


def build_figure(curves: list[Curve], output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )

    fig, (ax_profit, ax_drawdown) = plt.subplots(
        2,
        1,
        figsize=(12.8, 8.0),
        gridspec_kw={"height_ratios": [2.35, 1.0], "hspace": 0.13},
        sharex=True,
    )
    fig.patch.set_facecolor("white")

    for curve in curves:
        label = f"{curve.algorithm}  ({curve.final_profit_pct:+.2f}% final)"
        ax_profit.plot(
            curve.dates,
            curve.cumulative_profit_pct,
            color=COLORS[curve.algorithm],
            linestyle=LINESTYLES[curve.algorithm],
            linewidth=2.25,
            label=label,
        )
        ax_drawdown.plot(
            curve.dates,
            [-value * 100.0 for value in curve.drawdowns],
            color=COLORS[curve.algorithm],
            linestyle=LINESTYLES[curve.algorithm],
            linewidth=1.55,
            alpha=0.92,
        )

    ax_profit.axhline(0.0, color="#64748B", linewidth=1.0, linestyle="--")
    ax_profit.set_ylabel("Cumulative change (%)")
    ax_profit.grid(True, axis="y", color="#E2E8F0", linewidth=0.8)
    ax_profit.spines[["top", "right"]].set_visible(False)
    ax_profit.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=9.6,
    )

    ax_drawdown.axhline(0.0, color="#64748B", linewidth=0.9)
    ax_drawdown.set_ylabel("Drawdown from peak (%)")
    ax_drawdown.set_xlabel("Observed test date")
    ax_drawdown.grid(True, axis="y", color="#E2E8F0", linewidth=0.8)
    ax_drawdown.spines[["top", "right"]].set_visible(False)
    ax_drawdown.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_drawdown.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    fig.suptitle(
        "Five-model cumulative portfolio change",
        x=0.065,
        y=0.972,
        ha="left",
        fontsize=21,
        color="#0F172A",
    )
    fig.text(
        0.066,
        0.925,
        "Storage-safe smoke job 3550 · 164 observed test dates · five DRL algorithms",
        ha="left",
        fontsize=11.5,
        color="#475569",
    )
    fig.text(
        0.066,
        0.018,
        "Illustrative pipeline check — not a final algorithm ranking or investment advice. "
        "Source: project-generated HPC smoke-run equity curves.",
        ha="left",
        fontsize=9.1,
        color="#64748B",
    )
    fig.subplots_adjust(left=0.075, right=0.975, top=0.845, bottom=0.105)

    png_path = output_dir / "hpc_showcase_smoke_cumulative_profit_3550.png"
    pdf_path = output_dir / "hpc_showcase_smoke_cumulative_profit_3550.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def bar(value: float, low: float, high: float, width: int = 28) -> str:
    span = max(high - low, 1e-9)
    filled = round((value - low) / span * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def print_dashboard(curves: list[Curve], output_paths: tuple[Path, Path], capture_seconds: int) -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    host = os.environ.get("SLURMD_NODENAME", socket.gethostname())
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    final_values = [curve.final_profit_pct for curve in curves]
    low, high = min(final_values), max(final_values)

    print("+--------------------------------------------------------------------+", flush=True)
    print("|  THAI STOCK DRL  ·  HPC LIVE ANALYSIS                              |", flush=True)
    print("+--------------------------------------------------------------------+", flush=True)
    print(f"  Job {job_id}  |  Node {host}  |  CPUs {cpus}", flush=True)
    print(f"  Analysed {len(curves)} models × {len(curves[0].dates)} observed test dates", flush=True)
    print("", flush=True)
    for index, curve in enumerate(curves, start=1):
        progress = bar(curve.final_profit_pct, low, high)
        print(
            f"  [{index}/5] {curve.algorithm:<4} {progress} "
            f"final {curve.final_profit_pct:+7.2f}%  "
            f"max DD {curve.max_drawdown_pct:7.2f}%",
            flush=True,
        )
    print("", flush=True)
    print(f"  Figure ready: {output_paths[0]}", flush=True)
    print("  Evidence label: SMOKE TEST / NOT FINAL RANKING", flush=True)

    if capture_seconds > 0:
        step = 5
        print("", flush=True)
        print("  POSTER CAPTURE WINDOW — monitor with squeue + this log", flush=True)
        for remaining in range(capture_seconds, 0, -step):
            meter = bar(capture_seconds - remaining, 0, capture_seconds, width=32)
            print(f"  LIVE  {meter}  {remaining:>3}s remaining", flush=True)
            time.sleep(min(step, remaining))

    print("", flush=True)
    print("  STATUS: COMPLETE", flush=True)
    print("+--------------------------------------------------------------------+", flush=True)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    input_dir = (
        args.input_dir.resolve()
        if args.input_dir
        else project_root / "results" / "screening_storage_safe_smoke_3550"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "output" / "poster_assets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = [load_curve(input_dir, algorithm) for algorithm in ALGORITHMS]
    validate_curves(curves)
    summary_path = write_summary(curves, output_dir)
    output_paths = build_figure(curves, output_dir)
    print_dashboard(curves, output_paths, max(0, args.capture_seconds))
    print(f"  Summary ready: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
