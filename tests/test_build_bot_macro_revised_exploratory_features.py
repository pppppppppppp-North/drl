from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.build_bot_macro_revised_exploratory_features import (
    build_revised_exploratory_features,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path / "base.csv"
    macro = tmp_path / "macro.csv"
    config = tmp_path / "config.yaml"
    receipt = tmp_path / "receipt.json"
    pd.DataFrame(
        {
            "date": ["2021-02-25", "2021-02-26", "2021-03-01"],
            "ticker": ["A", "A", "A"],
            "close": [10.0, 10.5, 11.0],
        }
    ).to_csv(base, index=False)
    pd.DataFrame(
        {
            "period": ["2021-01-01"],
            "release_date": ["2021-02-26"],
            "metric": ["leading_economic_index"],
            "value": [100.0],
            "point_in_time_eligible": [False],
        }
    ).to_csv(macro, index=False)
    config.write_text(
        yaml.safe_dump(
            {
                "official_macro_context": {
                    "metrics": {
                        "leading_economic_index": "bot_lei",
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
        "macro": macro,
        "config": config,
        "receipt": receipt,
    }


def test_builder_requires_explicit_revised_history_acknowledgement(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)

    with pytest.raises(ValueError, match="explicit acknowledgement"):
        build_revised_exploratory_features(
            base_features_path=paths["base"],
            macro_path=paths["macro"],
            source_receipt_path=paths["receipt"],
            config_path=paths["config"],
            acknowledge_revised_history=False,
        )


def test_builder_marks_pre_release_rows_zero_and_never_claims_eligibility(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)

    merged, metadata = build_revised_exploratory_features(
        base_features_path=paths["base"],
        macro_path=paths["macro"],
        source_receipt_path=paths["receipt"],
        config_path=paths["config"],
        acknowledge_revised_history=True,
    )

    assert merged.iloc[0]["bot_lei"] == 0.0
    assert merged.iloc[1]["bot_lei"] == 100.0
    assert metadata["point_in_time_eligible"] is False
    assert metadata["publication_performance_claim_eligible"] is False
