"""The leave-one-source-out arms change only the inputs of the all-sources arm of the locked design."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from src.features.pit_sector_context import SECTOR_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text())


def _strip(variant: dict) -> dict:
    return {k: x for k, x in variant.items() if k not in {"name", "description"}}


def test_the_arms_are_the_all_sources_arm_with_one_source_added_or_left_out():
    src, spec = _load("pit_source_ablation_cashcap.yaml"), _load("pit_source_loo_cashcap.yaml")
    for key in ("splits", "universe", "features", "base"):
        assert spec[key] == src[key], key
    assert spec["data"] == {**src["data"], "sector": True}
    assert spec["seeds"] == [42, 43, 44, 45, 46]

    everything = next(v for v in src["variants"] if v["name"] == "pit_cashcap_src_all")
    technical = list(src["features"]["stock_columns"])
    market = list(everything["features"]["market_columns"])
    etf = [c for c in market if c.startswith("set50_etf_")]
    policy = [c for c in market if c.startswith("bot_")]
    macro = [c for c in market if c not in etf + policy]
    assert (len(etf), len(policy), len(macro)) == (5, 5, 8)
    expected = {
        "pit_cashcap_src_all_sector": (technical + SECTOR_COLUMNS, etf + policy + macro),
        "pit_cashcap_src_loo_etf": (technical + SECTOR_COLUMNS, policy + macro),
        "pit_cashcap_src_loo_policy": (technical + SECTOR_COLUMNS, etf + macro),
        "pit_cashcap_src_loo_macro": (technical + SECTOR_COLUMNS, etf + policy),
        "pit_cashcap_src_loo_technical": (["return_1d"] + SECTOR_COLUMNS, etf + policy + macro),
    }
    assert [v["name"] for v in spec["variants"]] == list(expected)
    for variant in spec["variants"]:
        stock, columns = expected[variant["name"]]
        arm = copy.deepcopy(variant)
        assert arm.pop("features") == {"stock_columns": stock, "market_columns": columns}
        # The actor reads the market inputs from its own list, which must match the environment's.
        assert arm["training"]["policy"].pop("market_feature_columns") == columns
        reference = copy.deepcopy(everything)
        reference.pop("features")
        reference["training"]["policy"].pop("market_feature_columns")
        assert _strip(arm) == _strip(reference), variant["name"]
