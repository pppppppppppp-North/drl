from __future__ import annotations

import json
from pathlib import Path

from scripts.materialize_storage_safe_calibration_lineage import (
    validate_lineage_receipt,
)
from src.campaigns.control import load_definition
from src.campaigns.gates import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = PROJECT_ROOT / "campaigns/large_scale_thai_drl_v2"
SPEC = PROJECT_ROOT / "config/research_campaign_storage_safe.yaml"
RECEIPT = (
    CAMPAIGN_DIR / "imports/storage_safe_cutover_calibration_lineage.json"
)


def test_lineage_receipt_validates_without_recomputed_hpc_jobs() -> None:
    assert validate_lineage_receipt(
        RECEIPT,
        project_root=PROJECT_ROOT,
        spec_path=SPEC,
        campaign_dir=CAMPAIGN_DIR,
    ) == []
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert value["status"] == "accepted"
    assert value["decision"] == "reuse_without_compute"
    assert value["submitted_hpc_jobs"] == 0
    assert value["accepted_stages"]["vector_benchmark"]["receipts"] == [
        {
            "path": "reports/vectorized_training_benchmark_3540.csv.receipt.json",
            "bytes": 4886,
            "sha256": (
                "e0b089755046ff53e78bc53c5f7816545c32d242956538dc63361992daf94524"
            ),
        }
    ]
    assert value["accepted_stages"]["determinism"]["receipts"] == [
        {
            "path": "reports/determinism_3541.json.receipt.json",
            "bytes": 4483,
            "sha256": (
                "c2009de48486aec4b51271f385afdf659c239dffe3014a2d93d839e15422bd4d"
            ),
        }
    ]


def test_lineage_restores_exact_attempt_bound_predecessor_records() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    restored = {
        record["path"]: record["sha256"]
        for record in value["restored_predecessor_records"]
    }
    assert restored == {
        "campaigns/large_scale_thai_drl_v2/stages/cluster_preflight/gate_report.json": (
            "bb175bb075b05784c08db6ddb24c9e7e681946204fdcab311199d5f8d06ff4d0"
        ),
        "campaigns/large_scale_thai_drl_v2/stages/cluster_preflight/_SUCCESS.json": (
            "0e2f6ced32670ab86a28b70882cb21ae982ad61d5f758fdd5784f4ada4ece052"
        ),
        "campaigns/large_scale_thai_drl_v2/stages/vector_benchmark/gate_report.json": (
            "6a806ec61c814a96da8caa0041968ac25027d0a40ed5e4bac81f31ec22894658"
        ),
        "campaigns/large_scale_thai_drl_v2/stages/vector_benchmark/_SUCCESS.json": (
            "24a543a87a8de9efeaddef77aaeb6a4a8715840881edf48822da6adb9daa9436"
        ),
        "campaigns/large_scale_thai_drl_v2/stages/determinism/gate_report.json": (
            "0de02de36c2a8e4a06e724f1efa06cb2d559d70c2aedb0e9c272a9da723bd215"
        ),
        "campaigns/large_scale_thai_drl_v2/stages/determinism/_SUCCESS.json": (
            "88bc5bfef522bc5b73b56ab148351b5f17ee46ff0c3e73b344ea8c966489dbff"
        ),
    }
    for relative, expected in restored.items():
        assert sha256_file(PROJECT_ROOT / relative) == expected


def test_storage_safe_graph_replaces_only_the_base_preflight_control_stage() -> None:
    definition, _ = load_definition(
        SPEC,
        project_root=PROJECT_ROOT,
        campaign_dir=CAMPAIGN_DIR,
    )
    assert "cluster_preflight" not in definition.stage_map
    assert "vector_benchmark" not in definition.stage_map
    assert "determinism" not in definition.stage_map
    assert definition.stage_map["calibration_lineage"].depends_on == (
        "deployment_bundle",
    )
    assert definition.stage_map["screening_config"].depends_on == (
        "calibration_lineage",
    )
