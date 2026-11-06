from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.plot_screening_storage_projection import (
    EXPECTED_ALGORITHMS,
    GIB,
    format_gib,
    load_projection,
    render,
)


def _capacity_decision(*, decision: str = "hold") -> dict[str, object]:
    retained = {
        "ppo": 132_154_080,
        "a2c": 89_445_720,
        "ddpg": 1_161_276_480,
        "td3": 1_740_554_280,
        "sac": 1_008_166_650,
    }
    checkpoint_per_trial = {
        "ppo": 4_343_562,
        "a2c": 2_918_334,
        "ddpg": 38_659_416,
        "td3": 57_913_515,
        "sac": 33_605_108,
    }
    projected_retained = sum(retained.values())
    projected_avoided = 30 * sum(checkpoint_per_trial.values())
    reserve = 256 * 1024 * 1024
    safety_adjusted = (projected_retained * 3 + 1) // 2
    return {
        "schema_version": 1,
        "decision": decision,
        "projection": {
            "checkpoint_policy": "exclude_path_component_named_checkpoints",
            "full_screening_trials_per_algorithm": 30,
            "per_algorithm": [
                {
                    "algorithm": algorithm,
                    "checkpoint_bytes_excluded_per_trial": checkpoint_per_trial[
                        algorithm
                    ],
                    "projected_retained_bytes": retained[algorithm],
                }
                for algorithm in EXPECTED_ALGORITHMS
            ],
            "projected_retained_bytes": projected_retained,
            "projected_checkpoint_bytes_avoided": projected_avoided,
            "smoke_projected_original_bytes": projected_retained + projected_avoided,
            "required_bytes": safety_adjusted + reserve,
            "safety_adjusted_retained_bytes": safety_adjusted,
            "safety_factor": 1.5,
            "fixed_reserve_bytes": reserve,
        },
        "quota": {
            "group": {
                "bytes": {
                    "soft_headroom": 6_926_224_384,
                    "hard_headroom": 20_000_000_000,
                }
            }
        },
        "inputs": {
            "smoke_summary": {
                "path": "results/screening_smoke_3543/screening_smoke_summary.json",
                "sha256": "4d724b25b430e3574e1e9d8ca1064ece12499a3c6001a98abf522f5aff6c1883",
            }
        },
    }


def _write_decision(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_projection_recomputes_totals_and_accepts_hold(tmp_path: Path) -> None:
    payload = _capacity_decision(decision="hold")
    source = tmp_path / "capacity.json"
    _write_decision(source, payload)

    projection = load_projection(source)

    expected_retained = sum(
        row["projected_retained_bytes"]
        for row in payload["projection"]["per_algorithm"]  # type: ignore[index]
    )
    expected_avoided = 30 * sum(
        row["checkpoint_bytes_excluded_per_trial"]
        for row in payload["projection"]["per_algorithm"]  # type: ignore[index]
    )
    assert projection.decision == "hold"
    assert [row.algorithm for row in projection.algorithms] == list(
        EXPECTED_ALGORITHMS
    )
    assert projection.projected_retained_bytes == expected_retained
    assert projection.avoided_checkpoint_bytes == expected_avoided
    assert projection.original_projection_bytes == expected_retained + expected_avoided
    assert projection.quota_soft_headroom_bytes == 6_926_224_384
    assert projection.smoke_evidence_note.startswith("smoke job 3543 summary")
    assert format_gib(GIB) == "1.0000 GiB"


@pytest.mark.parametrize("malformation", ["missing_algorithm", "negative_bytes"])
def test_load_projection_rejects_malformed_algorithm_projection(
    tmp_path: Path, malformation: str
) -> None:
    payload = copy.deepcopy(_capacity_decision())
    rows = payload["projection"]["per_algorithm"]  # type: ignore[index]
    if malformation == "missing_algorithm":
        rows[-1]["algorithm"] = "ppo"
    else:
        rows[0]["checkpoint_bytes_excluded_per_trial"] = -1
    source = tmp_path / "capacity.json"
    _write_decision(source, payload)

    with pytest.raises(ValueError):
        load_projection(source)


def test_render_creates_nonempty_png_and_svg(tmp_path: Path) -> None:
    source = tmp_path / "capacity.json"
    png = tmp_path / "screening_storage.png"
    svg = tmp_path / "screening_storage.svg"
    _write_decision(source, _capacity_decision(decision="pass"))

    projection = render(source, png, svg)

    assert projection.decision == "pass"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert png.stat().st_size > 20_000
    svg_text = svg.read_text(encoding="utf-8")
    assert svg.stat().st_size > 10_000
    assert "Full-screening storage by algorithm" in svg_text
    assert "30 trials per algorithm" in svg_text
    assert "Storage-safe retained" in svg_text
    assert "Avoided checkpoint storage" in svg_text
