"""The source-ablation arms change only the market-level inputs of the locked point-in-time design."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text())


def _variant(spec: dict, name: str) -> dict:
    return next(v for v in spec["variants"] if v["name"] == name)


def test_everything_but_the_inputs_matches_the_locked_config():
    base, spec = _load("pit_walk_forward_experiment.yaml"), _load("pit_source_ablation_cashcap.yaml")
    for key in ("splits", "universe", "features", "base"):
        assert spec[key] == base[key], key
    extra = {"policy_rate", "macro", "macro_names"}
    assert {k: v for k, v in spec["data"].items() if k not in extra} == base["data"]
    assert spec["data"]["policy_rate"] == _load("pit_walk_forward_policy_rate.yaml")["data"]["policy_rate"]
    macro = _load("pit_walk_forward_macro.yaml")["data"]
    assert (spec["data"]["macro"], spec["data"]["macro_names"]) == (macro["macro"], macro["macro_names"])
    assert spec["seeds"] == [42, 43, 44, 45, 46]


def test_each_arm_is_the_locked_variant_with_other_market_inputs():
    base, spec = _load("pit_walk_forward_experiment.yaml"), _load("pit_source_ablation_cashcap.yaml")
    locked = _variant(base, "pit_plan_weekly_lstm_shared_cashcap")
    market = base["features"]["market_columns"]
    assert base["base"]["training"]["policy"]["market_feature_columns"] == market
    policy = [c for c in _variant(_load("pit_walk_forward_policy_rate.yaml"), "pit_plan_weekly_lstm_shared_policy")
              ["features"]["market_columns"] if c not in market]
    macro = [c for c in _variant(_load("pit_walk_forward_macro.yaml"), "pit_plan_weekly_lstm_shared_macro")
             ["features"]["market_columns"] if c not in market]
    assert len(policy) == 5 and len(macro) == 8
    expected = {
        "pit_cashcap_src_technical": [],
        "pit_cashcap_src_policy": market + policy,
        "pit_cashcap_src_macro": market + macro,
        "pit_cashcap_src_all": market + policy + macro,
    }
    assert [v["name"] for v in spec["variants"]] == list(expected)
    for name, columns in expected.items():
        arm = copy.deepcopy(_variant(spec, name))
        assert arm.pop("features") == {"market_columns": columns}, name
        assert arm["training"]["policy"].pop("market_feature_columns") == columns, name
        arm.pop("description")
        reference = {k: v for k, v in locked.items() if k != "description"}
        assert {**arm, "name": locked["name"]} == reference, name
