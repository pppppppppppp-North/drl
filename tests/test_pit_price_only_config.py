"""The price-only arm keeps only the daily return of the locked point-in-time design's inputs."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text())


def _strip(variant: dict) -> dict:
    return {k: x for k, x in variant.items() if k not in {"name", "description"}}


def test_the_price_arm_is_the_technical_only_arm_with_the_daily_return_alone():
    base, spec = _load("pit_walk_forward_experiment.yaml"), _load("pit_price_only_cashcap.yaml")
    for key in ("data", "splits", "universe", "features", "base"):
        assert spec[key] == base[key], key
    assert spec["seeds"] == [42, 43, 44, 45, 46]
    assert [v["name"] for v in spec["variants"]] == ["pit_cashcap_src_price"]
    arm = copy.deepcopy(spec["variants"][0])
    assert arm["features"] == {"stock_columns": ["return_1d"], "market_columns": []}
    assert "return_1d" in base["features"]["stock_columns"]
    # Apart from the stock inputs it is the technical-only arm of the source ablation ...
    technical = next(v for v in _load("pit_source_ablation_cashcap.yaml")["variants"]
                     if v["name"] == "pit_cashcap_src_technical")
    arm["features"].pop("stock_columns")
    assert _strip(arm) == _strip(technical)
    # ... which is the locked variant without the market-level inputs.
    locked = next(v for v in base["variants"] if v["name"] == "pit_plan_weekly_lstm_shared_cashcap")
    arm.pop("features")
    assert arm["training"]["policy"].pop("market_feature_columns") == []
    assert _strip(arm) == _strip(locked)
