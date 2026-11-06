from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.validate_bot_macro_exploratory_features import validate_integration


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize(tmp_path: Path, *, future_release: bool = False) -> dict[str, Path]:
    base = tmp_path / "base.csv"
    merged = tmp_path / "merged.csv"
    macro = tmp_path / "macro.csv"
    config = tmp_path / "config.yaml"
    receipt = tmp_path / "receipt.json"
    dates = ["2021-02-26", "2021-03-01"]
    base_frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "A"],
            "close": [10.0, 11.0],
            "return_1d": [0.0, 0.1],
        }
    )
    base_frame.to_csv(base, index=False)
    merged_frame = base_frame.copy()
    merged_frame["bot_lei"] = [100.0, 100.0]
    merged_frame["bot_change"] = [1.0, 1.0]
    merged_frame["bot_macro_release_date"] = (
        ["2021-03-02", "2021-03-02"]
        if future_release
        else ["2021-02-26", "2021-02-26"]
    )
    merged_frame["bot_macro_release_age_days"] = (
        [-4, -1] if future_release else [0, 3]
    )
    merged_frame.to_csv(merged, index=False)
    pd.DataFrame(
        {
            "period": ["2021-01-01", "2021-01-01"],
            "metric": ["leading_economic_index", "index_change"],
            "value": [100.0, 1.0],
            "point_in_time_eligible": [False, False],
        }
    ).to_csv(macro, index=False)
    config.write_text(
        yaml.safe_dump(
            {
                "official_macro_context": {
                    "metrics": {
                        "leading_economic_index": "bot_lei",
                        "index_change": "bot_change",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    receipt_value = {
        "status": "accepted_with_point_in_time_restriction",
        "normalized": {
            "bytes": macro.stat().st_size,
            "sha256": _sha256(macro),
        },
        "scientific_eligibility": {
            "structural_source_ready": True,
            "point_in_time_eligible": False,
            "publication_performance_claim_eligible": False,
        },
    }
    receipt.write_text(json.dumps(receipt_value) + "\n", encoding="utf-8")
    Path(str(receipt) + ".sha256").write_text(
        _sha256(receipt) + "\n",
        encoding="utf-8",
    )
    return {
        "base": base,
        "merged": merged,
        "macro": macro,
        "config": config,
        "receipt": receipt,
    }


def test_validator_accepts_explicitly_restricted_exploratory_merge(
    tmp_path: Path,
) -> None:
    paths = _materialize(tmp_path)

    result = validate_integration(
        base_features_path=paths["base"],
        merged_features_path=paths["merged"],
        macro_path=paths["macro"],
        source_receipt_path=paths["receipt"],
        config_path=paths["config"],
    )

    assert result["status"] == "pass_with_point_in_time_restriction"
    assert result["alignment"]["future_release_rows"] == 0
    assert result["scientific_eligibility"]["point_in_time_eligible"] is False


def test_validator_rejects_future_scheduled_release(tmp_path: Path) -> None:
    paths = _materialize(tmp_path, future_release=True)

    with pytest.raises(ValueError, match="future scheduled release"):
        validate_integration(
            base_features_path=paths["base"],
            merged_features_path=paths["merged"],
            macro_path=paths["macro"],
            source_receipt_path=paths["receipt"],
            config_path=paths["config"],
        )
