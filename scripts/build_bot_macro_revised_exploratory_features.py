from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.ingest_ohlcv import write_csv_atomic
from src.features.official_macro_context import merge_official_macro_context
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


def build_revised_exploratory_features(
    *,
    base_features_path: str | Path,
    macro_path: str | Path,
    source_receipt_path: str | Path,
    config_path: str | Path,
    acknowledge_revised_history: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not acknowledge_revised_history:
        raise ValueError(
            "building revised-history features requires an explicit "
            "acknowledgement that the output is exploratory and not "
            "point-in-time eligible"
        )
    base_path = _regular(base_features_path, label="base feature table")
    macro_file = _regular(macro_path, label="normalized BOT table")
    receipt_path = _regular(source_receipt_path, label="BOT source receipt")
    sidecar = _regular(
        Path(str(receipt_path) + ".sha256"),
        label="BOT source receipt sidecar",
    )
    config_file = _regular(config_path, label="official macro configuration")
    if sidecar.read_text(encoding="utf-8") != f"{_sha256(receipt_path)}\n":
        raise ValueError("BOT source receipt sidecar does not match")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    eligibility = receipt.get("scientific_eligibility", {})
    normalized = receipt.get("normalized", {})
    if (
        receipt.get("status") != "accepted_with_point_in_time_restriction"
        or eligibility.get("point_in_time_eligible") is not False
        or eligibility.get("publication_performance_claim_eligible") is not False
        or normalized.get("sha256") != _sha256(macro_file)
        or normalized.get("bytes") != macro_file.stat().st_size
    ):
        raise ValueError("BOT source receipt restriction or source binding does not match")

    base = pd.read_csv(base_path)
    macro = pd.read_csv(macro_file)
    point_in_time = (
        macro.get("point_in_time_eligible", pd.Series([], dtype=str))
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if point_in_time.empty or not point_in_time.eq("false").all():
        raise ValueError("BOT table is not consistently marked point-in-time ineligible")
    config = load_config(config_file)
    metric_map = dict(config.get("official_macro_context", {}).get("metrics", {}))
    if not metric_map:
        raise ValueError("official macro configuration has no metric map")
    merged = merge_official_macro_context(base, macro, metric_map=metric_map)
    metadata = {
        "schema_version": 1,
        "status": "exploratory_revised_history_only",
        "base_features_sha256": _sha256(base_path),
        "macro_sha256": _sha256(macro_file),
        "source_receipt_sha256": _sha256(receipt_path),
        "rows": len(merged),
        "dates": int(pd.to_datetime(merged["date"]).nunique()),
        "tickers": int(merged["ticker"].nunique()),
        "configured_features": list(metric_map.values()),
        "point_in_time_eligible": False,
        "publication_performance_claim_eligible": False,
    }
    return merged, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an explicitly exploratory feature table from a current "
            "revised BOT history"
        )
    )
    parser.add_argument(
        "--base-features",
        default="data/processed/features_real_ohlcv_market.csv",
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
            "data/processed/"
            "features_real_ohlcv_official_macro_revised_exploratory.csv"
        ),
    )
    parser.add_argument(
        "--metadata-output",
        default=(
            "reports/data_sources/"
            "bot_macro_exploratory_feature_build_20260725.json"
        ),
    )
    parser.add_argument(
        "--acknowledge-revised-history",
        action="store_true",
    )
    args = parser.parse_args()
    merged, metadata = build_revised_exploratory_features(
        base_features_path=args.base_features,
        macro_path=args.macro,
        source_receipt_path=args.source_receipt,
        config_path=args.config,
        acknowledge_revised_history=args.acknowledge_revised_history,
    )
    output = Path(args.output)
    write_csv_atomic(merged, output)
    metadata["output"] = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
    }
    metadata_output = Path(args.metadata_output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(str(metadata_output) + ".sha256").write_text(
        f"{_sha256(metadata_output)}\n",
        encoding="utf-8",
    )
    print(
        f"wrote exploratory revised-history features {output} "
        f"rows={len(merged)} point_in_time_eligible=false"
    )


if __name__ == "__main__":
    main()
