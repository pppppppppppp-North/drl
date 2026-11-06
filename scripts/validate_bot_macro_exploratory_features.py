from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_file() or value.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")
    return value


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_integration(
    *,
    base_features_path: str | Path,
    merged_features_path: str | Path,
    macro_path: str | Path,
    source_receipt_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    base_path = _regular(base_features_path, label="base feature table")
    merged_path = _regular(merged_features_path, label="merged feature table")
    macro_file = _regular(macro_path, label="normalized BOT table")
    receipt_path = _regular(source_receipt_path, label="BOT source receipt")
    config_file = _regular(config_path, label="official macro configuration")
    sidecar = _regular(
        Path(str(receipt_path) + ".sha256"),
        label="BOT source receipt sidecar",
    )
    if sidecar.read_text(encoding="utf-8") != f"{_sha256(receipt_path)}\n":
        raise ValueError("BOT source receipt sidecar does not match")

    receipt = _json(receipt_path, label="BOT source receipt")
    eligibility = receipt.get("scientific_eligibility")
    normalized_record = receipt.get("normalized")
    if (
        receipt.get("status") != "accepted_with_point_in_time_restriction"
        or not isinstance(eligibility, dict)
        or eligibility.get("structural_source_ready") is not True
        or eligibility.get("point_in_time_eligible") is not False
        or eligibility.get("publication_performance_claim_eligible") is not False
        or not isinstance(normalized_record, dict)
        or normalized_record.get("sha256") != _sha256(macro_file)
        or normalized_record.get("bytes") != macro_file.stat().st_size
    ):
        raise ValueError("BOT source receipt eligibility or source binding does not match")

    config = load_config(config_file)
    metric_map = dict(config.get("official_macro_context", {}).get("metrics", {}))
    if not metric_map:
        raise ValueError("official macro configuration has no metric map")
    required_feature_columns = list(metric_map.values())

    base = pd.read_csv(base_path)
    merged = pd.read_csv(merged_path)
    macro = pd.read_csv(macro_file)
    keys = ["date", "ticker"]
    if any(column not in base for column in keys) or any(
        column not in merged for column in keys
    ):
        raise ValueError("feature tables are missing date/ticker keys")
    if base.duplicated(keys).any() or merged.duplicated(keys).any():
        raise ValueError("feature tables contain duplicate date/ticker rows")
    base_sorted = base.sort_values(keys).reset_index(drop=True)
    merged_sorted = merged.sort_values(keys).reset_index(drop=True)
    if base_sorted[keys].to_dict("records") != merged_sorted[keys].to_dict("records"):
        raise ValueError("exploratory merge changed the base feature panel keys")
    shared_columns = list(base.columns)
    try:
        pd.testing.assert_frame_equal(
            base_sorted[shared_columns],
            merged_sorted[shared_columns],
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError("exploratory merge changed existing feature values") from exc

    required = {
        *required_feature_columns,
        "bot_macro_release_date",
        "bot_macro_release_age_days",
    }
    missing = sorted(required - set(merged.columns))
    if missing:
        raise ValueError(f"exploratory feature table is missing BOT columns: {missing}")
    feature_values = merged[required_feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        feature_values.isna().any().any()
        or not np.isfinite(feature_values.to_numpy(dtype=float)).all()
    ):
        raise ValueError("BOT exploratory features contain missing or non-finite values")
    empty_features = [
        column
        for column in required_feature_columns
        if not feature_values[column].ne(0.0).any()
    ]
    if empty_features:
        raise ValueError(f"BOT exploratory features are all zero: {empty_features}")

    feature_dates = pd.to_datetime(merged["date"], errors="coerce")
    release_dates = pd.to_datetime(
        merged["bot_macro_release_date"],
        errors="coerce",
    )
    release_age = pd.to_numeric(
        merged["bot_macro_release_age_days"],
        errors="coerce",
    )
    if (
        feature_dates.isna().any()
        or release_dates.isna().any()
        or release_age.isna().any()
    ):
        raise ValueError("exploratory BOT release alignment contains missing values")
    future_rows = int((release_dates > feature_dates).sum())
    negative_age_rows = int((release_age < 0).sum())
    if future_rows or negative_age_rows:
        raise ValueError("exploratory BOT merge uses a future scheduled release")

    point_in_time_values = (
        macro.get("point_in_time_eligible", pd.Series([], dtype=str))
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if point_in_time_values.empty or not point_in_time_values.eq("false").all():
        raise ValueError("normalized BOT table is not consistently marked ineligible")

    return {
        "schema_version": 1,
        "status": "pass_with_point_in_time_restriction",
        "validation_kind": "bot_revised_history_exploratory_feature_integration",
        "base_features": {
            "path": str(base_path),
            "rows": len(base),
            "dates": int(pd.to_datetime(base["date"]).nunique()),
            "tickers": int(base["ticker"].nunique()),
            "sha256": _sha256(base_path),
        },
        "merged_features": {
            "path": str(merged_path),
            "rows": len(merged),
            "dates": int(feature_dates.nunique()),
            "tickers": int(merged["ticker"].nunique()),
            "columns": len(merged.columns),
            "sha256": _sha256(merged_path),
        },
        "macro_source": {
            "path": str(macro_file),
            "rows": len(macro),
            "periods": int(macro["period"].nunique()),
            "metrics": int(macro["metric"].nunique()),
            "sha256": _sha256(macro_file),
            "source_receipt": str(receipt_path),
            "source_receipt_sha256": _sha256(receipt_path),
        },
        "alignment": {
            "configured_features": required_feature_columns,
            "nonzero_rows": {
                column: int(feature_values[column].ne(0.0).sum())
                for column in required_feature_columns
            },
            "distinct_values": {
                column: int(feature_values[column].nunique())
                for column in required_feature_columns
            },
            "scheduled_release_dates": int(release_dates.nunique()),
            "missing_release_rows": int(release_dates.isna().sum()),
            "future_release_rows": future_rows,
            "negative_release_age_rows": negative_age_rows,
        },
        "scientific_eligibility": {
            "point_in_time_eligible": False,
            "publication_performance_claim_eligible": False,
            "model_performance_result_created": False,
            "usage": (
                "Exploratory source-integration and revised-series sensitivity "
                "analysis only; historical vintages are required for causal backtesting."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an explicitly exploratory BOT revised-history feature merge"
        )
    )
    parser.add_argument(
        "--base-features",
        default="data/processed/features_real_ohlcv_market.csv",
    )
    parser.add_argument(
        "--merged-features",
        default=(
            "data/processed/"
            "features_real_ohlcv_official_macro_revised_exploratory.csv"
        ),
    )
    parser.add_argument(
        "--macro",
        default="data/external/bot_historical_macro.csv",
    )
    parser.add_argument(
        "--source-receipt",
        default="reports/data_sources/bot_historical_macro_20260725.json",
    )
    parser.add_argument(
        "--config",
        default="config/real_ohlcv_official_macro.yaml",
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/data_sources/"
            "bot_macro_exploratory_feature_validation_20260725.json"
        ),
    )
    args = parser.parse_args()
    result = validate_integration(
        base_features_path=args.base_features,
        merged_features_path=args.merged_features,
        macro_path=args.macro,
        source_receipt_path=args.source_receipt,
        config_path=args.config,
    )
    output = Path(args.output)
    _atomic_text(output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    _atomic_text(Path(str(output) + ".sha256"), f"{_sha256(output)}\n")
    print(
        f"validated exploratory BOT feature merge rows="
        f"{result['merged_features']['rows']} future_release_rows=0 "
        "point_in_time_eligible=false"
    )


if __name__ == "__main__":
    main()
