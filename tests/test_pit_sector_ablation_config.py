"""The sector arm adds only the sector-relative stock inputs to the locked point-in-time design."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from src.features.pit_sector_context import SECTOR_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text())


def test_the_sector_arm_is_the_locked_variant_with_four_more_stock_inputs():
    base, spec = _load("pit_walk_forward_experiment.yaml"), _load("pit_sector_ablation_cashcap.yaml")
    for key in ("splits", "universe", "features", "base"):
        assert spec[key] == base[key], key
    assert spec["data"] == {**base["data"], "sector": True}
    assert spec["seeds"] == [42, 43, 44, 45, 46]
    assert [v["name"] for v in spec["variants"]] == ["pit_cashcap_src_sector"]
    locked = next(v for v in base["variants"] if v["name"] == "pit_plan_weekly_lstm_shared_cashcap")
    arm = copy.deepcopy(spec["variants"][0])
    assert arm.pop("features") == {"stock_columns": base["features"]["stock_columns"] + SECTOR_COLUMNS}
    strip = lambda v: {k: x for k, x in v.items() if k not in {"name", "description"}}  # noqa: E731
    assert strip(arm) == strip(locked)
    # The market inputs, which the actor reads separately, are the locked design's.
    assert "market_feature_columns" not in arm["training"]["policy"]
