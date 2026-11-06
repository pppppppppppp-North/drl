from __future__ import annotations

"""Render the checkpoint-free screening storage projection as PNG and SVG.

The source is the machine-readable decision emitted by
``scripts.assess_screening_storage``.  A ``hold`` decision is still valid
plotting evidence when its projection is complete; malformed or incomplete
projections fail closed.
"""

import argparse
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from typing import Any, Mapping, Sequence


# Matplotlib otherwise tries to create a cache below HOME.  Use a predictable,
# process-independent directory on the writable local temporary filesystem and
# configure it before importing matplotlib.  No renderer path needs network
# access.
_MPL_CACHE = Path(tempfile.gettempdir()) / (
    f"final_project_plan_matplotlib_{getattr(os, 'getuid', lambda: 0)()}"
)
if _MPL_CACHE.exists() and (_MPL_CACHE.is_symlink() or not _MPL_CACHE.is_dir()):
    _MPL_CACHE = Path(tempfile.mkdtemp(prefix="final_project_plan_matplotlib_"))
else:
    _MPL_CACHE.mkdir(mode=0o700, parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_ALGORITHMS: tuple[str, ...] = ("ppo", "a2c", "ddpg", "td3", "sac")
EXPECTED_TRIALS_PER_ALGORITHM = 30
GIB = 1024**3

BLUE = "#2563EB"
ORANGE = "#D97706"
INK = "#1F2937"
MUTED = "#4B5563"
GRID = "#D1D5DB"
PANEL = "#F3F4F6"


@dataclass(frozen=True)
class AlgorithmProjection:
    algorithm: str
    retained_bytes: int
    avoided_checkpoint_bytes: int

    @property
    def retained_gib(self) -> float:
        return self.retained_bytes / GIB

    @property
    def avoided_checkpoint_gib(self) -> float:
        return self.avoided_checkpoint_bytes / GIB


@dataclass(frozen=True)
class ScreeningStorageProjection:
    decision: str
    trials_per_algorithm: int
    algorithms: tuple[AlgorithmProjection, ...]
    projected_retained_bytes: int
    smoke_projected_original_bytes: int
    required_bytes: int
    safety_factor: Decimal
    fixed_reserve_bytes: int
    quota_soft_headroom_bytes: int | None
    smoke_evidence_note: str

    @property
    def avoided_checkpoint_bytes(self) -> int:
        return sum(row.avoided_checkpoint_bytes for row in self.algorithms)

    @property
    def original_projection_bytes(self) -> int:
        return self.smoke_projected_original_bytes


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _nonnegative_bytes(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a nonnegative integer number of bytes")
    if not math.isfinite(float(value)) or value < 0 or int(value) != value:
        raise ValueError(f"{name} must be a nonnegative integer number of bytes")
    return int(value)


def _safety_factor(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("projection.safety_factor must be a finite number at least 1")
    try:
        factor = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            "projection.safety_factor must be a finite number at least 1"
        ) from exc
    if not factor.is_finite() or factor < 1:
        raise ValueError("projection.safety_factor must be a finite number at least 1")
    return factor


def _smoke_evidence_note(payload: Mapping[str, Any]) -> str:
    generic = "completed five-algorithm smoke-job artifact inventories"
    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, dict):
        return generic
    raw_smoke = raw_inputs.get("smoke_summary")
    if not isinstance(raw_smoke, dict):
        return generic

    path = raw_smoke.get("path")
    digest = raw_smoke.get("sha256")
    path_text = path if isinstance(path, str) and path.strip() else ""
    digest_text = digest if isinstance(digest, str) and digest.strip() else ""
    job_match = re.search(r"screening_smoke_(\d+)", path_text)

    if job_match:
        evidence = f"smoke job {job_match.group(1)} summary"
    elif path_text:
        evidence = f"smoke summary {Path(path_text).name}"
    else:
        evidence = generic
    if digest_text:
        evidence += f" (SHA-256 {digest_text[:12]}…)"
    return evidence


def _optional_quota_headroom(payload: Mapping[str, Any]) -> int | None:
    quota = payload.get("quota")
    if quota is None:
        return None
    quota_object = _object(quota, "quota")
    group = _object(quota_object.get("group"), "quota.group")
    byte_view = _object(group.get("bytes"), "quota.group.bytes")
    if "soft_headroom" not in byte_view:
        raise ValueError("quota.group.bytes.soft_headroom is missing")
    return _nonnegative_bytes(
        byte_view["soft_headroom"], "quota.group.bytes.soft_headroom"
    )


def parse_projection(payload: Mapping[str, Any]) -> ScreeningStorageProjection:
    """Validate and recompute the projection needed by the figure."""

    decision = payload.get("decision")
    if decision not in {"pass", "hold"}:
        raise ValueError("decision must be either 'pass' or 'hold'")

    projection = _object(payload.get("projection"), "projection")
    trials = projection.get("full_screening_trials_per_algorithm")
    if (
        isinstance(trials, bool)
        or not isinstance(trials, int)
        or trials != EXPECTED_TRIALS_PER_ALGORITHM
    ):
        raise ValueError(
            "projection.full_screening_trials_per_algorithm must equal 30"
        )

    raw_rows = projection.get("per_algorithm")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(EXPECTED_ALGORITHMS):
        raise ValueError("projection.per_algorithm must contain exactly five rows")

    rows_by_algorithm: dict[str, AlgorithmProjection] = {}
    for index, raw_row in enumerate(raw_rows):
        row = _object(raw_row, f"projection.per_algorithm[{index}]")
        algorithm = row.get("algorithm")
        if not isinstance(algorithm, str) or algorithm not in EXPECTED_ALGORITHMS:
            raise ValueError(
                f"projection.per_algorithm[{index}].algorithm is unexpected"
            )
        if algorithm in rows_by_algorithm:
            raise ValueError(f"projection.per_algorithm duplicates {algorithm}")

        retained = _nonnegative_bytes(
            row.get("projected_retained_bytes"),
            f"projection.per_algorithm[{index}].projected_retained_bytes",
        )
        excluded_per_trial = _nonnegative_bytes(
            row.get("checkpoint_bytes_excluded_per_trial"),
            (
                f"projection.per_algorithm[{index}]"
                ".checkpoint_bytes_excluded_per_trial"
            ),
        )
        avoided = excluded_per_trial * trials
        if "projected_checkpoint_bytes_avoided" in row:
            claimed_avoided = _nonnegative_bytes(
                row["projected_checkpoint_bytes_avoided"],
                (
                    f"projection.per_algorithm[{index}]"
                    ".projected_checkpoint_bytes_avoided"
                ),
            )
            if claimed_avoided != avoided:
                raise ValueError(
                    f"projected checkpoint avoidance is inconsistent for {algorithm}"
                )
        rows_by_algorithm[algorithm] = AlgorithmProjection(
            algorithm=algorithm,
            retained_bytes=retained,
            avoided_checkpoint_bytes=avoided,
        )

    if set(rows_by_algorithm) != set(EXPECTED_ALGORITHMS):
        raise ValueError("projection must contain the exact five expected algorithms")
    algorithms = tuple(rows_by_algorithm[name] for name in EXPECTED_ALGORITHMS)

    projected_retained = _nonnegative_bytes(
        projection.get("projected_retained_bytes"),
        "projection.projected_retained_bytes",
    )
    recomputed_retained = sum(row.retained_bytes for row in algorithms)
    if projected_retained != recomputed_retained:
        raise ValueError(
            "projection.projected_retained_bytes does not equal the per-algorithm sum"
        )

    if "projected_checkpoint_bytes_avoided" in projection:
        claimed_avoided = _nonnegative_bytes(
            projection["projected_checkpoint_bytes_avoided"],
            "projection.projected_checkpoint_bytes_avoided",
        )
        recomputed_avoided = sum(row.avoided_checkpoint_bytes for row in algorithms)
        if claimed_avoided != recomputed_avoided:
            raise ValueError(
                "projection.projected_checkpoint_bytes_avoided is inconsistent"
            )
    original_projection = _nonnegative_bytes(
        projection.get(
            "smoke_projected_original_bytes",
            projected_retained
            + sum(row.avoided_checkpoint_bytes for row in algorithms),
        ),
        "projection.smoke_projected_original_bytes",
    )

    factor = _safety_factor(projection.get("safety_factor"))
    reserve = _nonnegative_bytes(
        projection.get("fixed_reserve_bytes"), "projection.fixed_reserve_bytes"
    )
    required = _nonnegative_bytes(
        projection.get("required_bytes"), "projection.required_bytes"
    )
    recomputed_required = int(
        (Decimal(projected_retained) * factor).to_integral_value(
            rounding=ROUND_CEILING
        )
    ) + reserve
    if required != recomputed_required:
        raise ValueError(
            "projection.required_bytes is inconsistent with safety factor and reserve"
        )

    if "safety_adjusted_retained_bytes" in projection:
        safety_adjusted = _nonnegative_bytes(
            projection["safety_adjusted_retained_bytes"],
            "projection.safety_adjusted_retained_bytes",
        )
        if safety_adjusted + reserve != required:
            raise ValueError(
                "projection.safety_adjusted_retained_bytes is inconsistent"
            )

    return ScreeningStorageProjection(
        decision=decision,
        trials_per_algorithm=trials,
        algorithms=algorithms,
        projected_retained_bytes=projected_retained,
        smoke_projected_original_bytes=original_projection,
        required_bytes=required,
        safety_factor=factor,
        fixed_reserve_bytes=reserve,
        quota_soft_headroom_bytes=_optional_quota_headroom(payload),
        smoke_evidence_note=_smoke_evidence_note(payload),
    )


def load_projection(path: str | Path) -> ScreeningStorageProjection:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read capacity decision JSON: {exc}") from exc
    return parse_projection(_object(payload, "capacity decision"))


def format_gib(byte_count: int) -> str:
    """Return the exact plotted value to four decimal places."""

    return f"{byte_count / GIB:.4f} GiB"


def _summary_annotation(projection: ScreeningStorageProjection) -> str:
    items = [
        f"Original smoke projection  {format_gib(projection.original_projection_bytes)}",
        f"Storage-safe retained  {format_gib(projection.projected_retained_bytes)}",
        (
            f"Required  {format_gib(projection.required_bytes)} "
            f"({projection.safety_factor.normalize()}× + "
            f"{format_gib(projection.fixed_reserve_bytes)} reserve)"
        ),
    ]
    if projection.quota_soft_headroom_bytes is not None:
        items.append(
            "Quota soft headroom  "
            f"{format_gib(projection.quota_soft_headroom_bytes)}"
        )
    return "    |    ".join(items)


def render(
    decision_path: str | Path,
    png_path: str | Path,
    svg_path: str | Path,
) -> ScreeningStorageProjection:
    """Render both requested static formats and return the validated data."""

    projection = load_projection(decision_path)
    png_destination = Path(png_path)
    svg_destination = Path(svg_path)
    if png_destination.resolve() == svg_destination.resolve():
        raise ValueError("PNG and SVG destinations must be different files")

    retained = [row.retained_gib for row in projection.algorithms]
    avoided = [row.avoided_checkpoint_gib for row in projection.algorithms]
    labels = [row.algorithm.upper() for row in projection.algorithms]
    positions = list(range(len(labels)))
    bar_height = 0.30
    offset = 0.18

    style = {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "screening-storage-projection-v1",
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(11.4, 7.0))
        figure.patch.set_facecolor("white")
        axis.set_facecolor("white")

        retained_bars = axis.barh(
            [position - offset for position in positions],
            retained,
            height=bar_height,
            color=BLUE,
            edgecolor=INK,
            linewidth=0.7,
            label="Storage-safe retained (incl. evidence budget)",
            zorder=3,
        )
        avoided_bars = axis.barh(
            [position + offset for position in positions],
            avoided,
            height=bar_height,
            color=ORANGE,
            edgecolor=INK,
            linewidth=0.7,
            hatch="///",
            label="Avoided checkpoint storage",
            zorder=3,
        )

        maximum = max([*retained, *avoided], default=0.0)
        label_padding = max(maximum * 0.014, 0.006)
        axis.set_xlim(0, maximum * 1.32 if maximum > 0 else 1.0)
        for bars, rows, attribute in (
            (retained_bars, projection.algorithms, "retained_bytes"),
            (avoided_bars, projection.algorithms, "avoided_checkpoint_bytes"),
        ):
            for bar, row in zip(bars, rows, strict=True):
                byte_count = getattr(row, attribute)
                axis.text(
                    bar.get_width() + label_padding,
                    bar.get_y() + bar.get_height() / 2,
                    format_gib(byte_count),
                    va="center",
                    ha="left",
                    fontsize=9,
                    color=INK,
                )

        axis.set_yticks(positions, labels, fontsize=10, weight="bold")
        axis.invert_yaxis()
        axis.set_xlabel("Projected storage (GiB; 1 GiB = 2³⁰ bytes)", labelpad=10)
        axis.axvline(0, color=INK, linewidth=1.0, zorder=4)
        axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.75, zorder=0)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=9)
        axis.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.015),
            frameon=False,
            ncol=2,
            fontsize=9.5,
            handlelength=2.0,
            columnspacing=2.0,
        )

        figure.text(
            0.105,
            0.955,
            "Full-screening storage by algorithm",
            ha="left",
            va="top",
            fontsize=17,
            weight="bold",
            color=INK,
        )
        figure.text(
            0.105,
            0.915,
            (
                f"{projection.trials_per_algorithm} trials per algorithm; "
                "projected from completed smoke-job evidence"
            ),
            ha="left",
            va="top",
            fontsize=10.5,
            color=MUTED,
        )
        figure.text(
            0.105,
            0.855,
            _summary_annotation(projection),
            ha="left",
            va="center",
            fontsize=9.0,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": PANEL,
                "edgecolor": GRID,
                "linewidth": 0.8,
            },
        )
        figure.text(
            0.105,
            0.035,
            (
                "Evidence: "
                f"{projection.smoke_evidence_note}. Avoided storage is the observed "
                "checkpoint bytes per smoke trial × 30; retained storage includes "
                "bounded receipt and task-summary overhead."
            ),
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=MUTED,
        )
        figure.subplots_adjust(left=0.14, right=0.955, top=0.745, bottom=0.145)

        png_destination.parent.mkdir(parents=True, exist_ok=True)
        svg_destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            png_destination,
            format="png",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "final_project_plan"},
        )
        figure.savefig(
            svg_destination,
            format="svg",
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "final_project_plan", "Date": None},
        )
        plt.close(figure)

    return projection


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot checkpoint-free full-screening storage capacity"
    )
    parser.add_argument("decision", type=Path, help="Capacity decision JSON")
    parser.add_argument("--png", required=True, type=Path, help="PNG output path")
    parser.add_argument("--svg", required=True, type=Path, help="SVG output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    render(args.decision, args.png, args.svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
