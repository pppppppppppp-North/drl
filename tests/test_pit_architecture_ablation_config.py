"""The architecture-ablation arm changes only the temporal encoder of the locked point-in-time design."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text())


def test_the_mlp_arm_is_the_locked_variant_with_another_temporal_encoder():
    base, spec = _load("pit_walk_forward_experiment.yaml"), _load("pit_architecture_ablation_cashcap.yaml")
    for key in ("data", "splits", "universe", "features", "base"):
        assert spec[key] == base[key], key
    assert spec["seeds"] == [42, 43, 44, 45, 46]
    assert [v["name"] for v in spec["variants"]] == ["pit_cashcap_arch_mlp"]
    locked = next(v for v in base["variants"] if v["name"] == "pit_plan_weekly_lstm_shared_cashcap")
    arm = copy.deepcopy(spec["variants"][0])
    assert arm["training"]["policy"].pop("temporal_encoder") == "mlp"
    strip = lambda v: {k: x for k, x in v.items() if k not in {"name", "description"}}  # noqa: E731
    assert strip(arm) == strip(locked)
    assert "temporal_encoder" not in base["base"]["training"]["policy"]
