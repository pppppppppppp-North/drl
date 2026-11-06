from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
from calendar import monthrange
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.ingest_ohlcv import write_csv_atomic
from src.utils.config import load_config


BOT_FULL_CSV_URL = (
    "https://app.bot.or.th/BTWS_STAT/statistics/DownloadFile.aspx?"
    "file=EC_EI_002_S2_ENG_ALL.CSV"
)
BOT_REPORT_URL = (
    "https://app.bot.or.th/BTWS_STAT/statistics/"
    "BOTWEBSTAT.aspx?reportID=887&language=ENG"
)
BOT_METADATA_URL = (
    "https://app.bot.or.th/BTWS_STAT/statistics/DownloadFile.aspx?"
    "file=EC_EI_002_S2_ENG.PDF"
)
SOURCE_TABLE = "EC_EI_002_S2"
EXPECTED_METRICS = (
    "authorized_capital_of_newly_registered_companies_million_baht",
    "broad_money_at_2000_prices_million_baht",
    "business_sentiment_index_3_months",
    "construction_areas_permitted_1000_sq_m",
    "export_volume_index_exclude_gold",
    "index_change",
    "leading_economic_index",
    "oil_price_inverse_index_dubai",
    "set_index",
)
POINT_IN_TIME_CAVEAT = (
    "The BOT full-history export is a current revised vintage. Historical values "
    "marked revised are not point-in-time observations, so they must not be "
    "backdated into causal model evaluation or publication-grade performance claims."
)
MONTH_PATTERN = re.compile(r"^[A-Z]{3}\s+\d{4}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value).strip().lower())
    cleaned = re.sub(r"\d+/", "", cleaned)
    return re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")


def _parse_numeric(value: object) -> float | None:
    text = str(value).replace(",", "").strip()
    if not text or text in {"....", "nan", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _period_start(period_label: str) -> pd.Timestamp:
    clean = str(period_label).replace(" p", "").replace(" r", "").strip()
    return pd.to_datetime(
        f"{clean[-4:]}-{clean[:3]}-01",
        format="%Y-%b-%d",
    )


def _last_business_day_following_month(
    period_start: pd.Timestamp,
) -> pd.Timestamp:
    following = period_start + pd.DateOffset(months=1)
    day = monthrange(int(following.year), int(following.month))[1]
    release = pd.Timestamp(
        year=int(following.year),
        month=int(following.month),
        day=day,
    )
    while release.weekday() >= 5:
        release -= pd.Timedelta(days=1)
    return release


def _revision_flag(period_label: str) -> str:
    clean = str(period_label).strip().lower()
    if clean.endswith(" r"):
        return "revised"
    if clean.endswith(" p"):
        return "preliminary"
    return "not_marked"


def _parse_historical_table(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    period_columns = [
        column
        for column in frame.columns
        if MONTH_PATTERN.match(
            str(column).replace(" p", "").replace(" r", "")
        )
    ]
    if not period_columns:
        raise ValueError("BOT historical table has no monthly period columns")
    descriptor_columns = [
        column for column in frame.columns if column not in period_columns
    ]
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        descriptor_values = [
            str(row[column]).strip()
            for column in descriptor_columns
            if str(row[column]).strip()
        ]
        metric_label = descriptor_values[-1] if descriptor_values else ""
        metric = _safe_name(metric_label)
        if not metric:
            continue
        for period_column in period_columns:
            value = _parse_numeric(row[period_column])
            if value is None:
                continue
            period = _period_start(str(period_column))
            rows.append(
                {
                    "period": period.date().isoformat(),
                    "release_date": (
                        _last_business_day_following_month(period)
                        .date()
                        .isoformat()
                    ),
                    "period_label": str(period_column).strip(),
                    "revision_flag": _revision_flag(str(period_column)),
                    "metric": metric,
                    "metric_label": metric_label,
                    "value": value,
                    "source_table": SOURCE_TABLE,
                    "source": "Bank of Thailand",
                }
            )
    if not rows:
        raise ValueError("BOT historical table contains no numeric observations")
    return (
        pd.DataFrame(rows)
        .sort_values(["period", "metric"])
        .reset_index(drop=True)
    )


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    _atomic_bytes(path, payload.encode("utf-8"))


def normalize_bot_historical_export(
    raw_bytes: bytes,
    *,
    retrieved_at_utc: str,
    start: str,
    end: str,
    warmup_months: int = 3,
    expected_metrics: tuple[str, ...] = EXPECTED_METRICS,
    minimum_full_history_periods: int = 300,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not raw_bytes:
        raise ValueError("BOT historical export is empty")
    if warmup_months < 0:
        raise ValueError("warmup months must be nonnegative")
    retrieved_timestamp = pd.Timestamp(retrieved_at_utc)
    if retrieved_timestamp.tzinfo is None:
        raise ValueError("retrieval timestamp must include a timezone")
    retrieved_at_utc = retrieved_timestamp.tz_convert("UTC").isoformat()
    try:
        table = pd.read_csv(io.BytesIO(raw_bytes), skiprows=5)
    except Exception as exc:
        raise ValueError(f"BOT historical export is unreadable: {exc}") from exc
    full = _parse_historical_table(table)
    metrics = tuple(sorted(full["metric"].astype(str).unique()))
    if metrics != tuple(sorted(expected_metrics)):
        missing = sorted(set(expected_metrics) - set(metrics))
        extra = sorted(set(metrics) - set(expected_metrics))
        raise ValueError(
            f"BOT historical metric contract changed; missing={missing} extra={extra}"
        )
    full_periods = int(full["period"].nunique())
    if full_periods < minimum_full_history_periods:
        raise ValueError(
            "BOT historical export has too few monthly periods: "
            f"{full_periods} < {minimum_full_history_periods}"
        )
    if full.duplicated(["period", "metric"]).any():
        raise ValueError("BOT historical export contains duplicate period/metric rows")
    if full["value"].isna().any():
        raise ValueError("BOT historical export contains missing numeric values")

    start_timestamp = pd.Timestamp(start) - pd.DateOffset(months=warmup_months)
    end_timestamp = pd.Timestamp(end)
    periods = pd.to_datetime(full["period"], errors="coerce")
    if periods.isna().any():
        raise ValueError("BOT historical export contains invalid periods")
    normalized = full[periods.between(start_timestamp, end_timestamp)].copy()
    if normalized.empty:
        raise ValueError("BOT historical export does not overlap the requested window")
    expected_window_periods = pd.date_range(
        start=start_timestamp,
        end=end_timestamp,
        freq="MS",
    )
    actual_periods = pd.DatetimeIndex(pd.to_datetime(normalized["period"]).unique())
    missing_periods = expected_window_periods.difference(actual_periods)
    if not missing_periods.empty:
        raise ValueError(
            "BOT historical export is missing requested months: "
            + ", ".join(timestamp.date().isoformat() for timestamp in missing_periods[:10])
        )
    expected_rows = len(expected_window_periods) * len(expected_metrics)
    if len(normalized) != expected_rows:
        raise ValueError(
            f"BOT historical export window has {len(normalized)} rows; "
            f"expected {expected_rows}"
        )

    normalized["source_url"] = BOT_FULL_CSV_URL
    normalized["retrieved_at_utc"] = retrieved_at_utc
    normalized["vintage_kind"] = "latest_revised_full_history_export"
    normalized["point_in_time_eligible"] = False
    normalized["release_date_basis"] = (
        "scheduled_last_business_day_following_month_not_observed_vintage"
    )
    normalized = normalized[
        [
            "period",
            "release_date",
            "period_label",
            "revision_flag",
            "metric",
            "metric_label",
            "value",
            "source_table",
            "source",
            "source_url",
            "retrieved_at_utc",
            "vintage_kind",
            "point_in_time_eligible",
            "release_date_basis",
        ]
    ].sort_values(["period", "metric"]).reset_index(drop=True)

    profile = {
        "full_history": {
            "first_period": str(full["period"].min()),
            "last_period": str(full["period"].max()),
            "periods": full_periods,
            "metrics": len(metrics),
            "rows": len(full),
            "missing_values": int(full["value"].isna().sum()),
        },
        "modeling_window_with_warmup": {
            "first_period": str(normalized["period"].min()),
            "last_period": str(normalized["period"].max()),
            "periods": int(normalized["period"].nunique()),
            "metrics": int(normalized["metric"].nunique()),
            "rows": len(normalized),
            "missing_values": int(normalized["value"].isna().sum()),
            "revision_flags": {
                str(key): int(value)
                for key, value in normalized["revision_flag"].value_counts().items()
            },
        },
        "metrics": list(metrics),
    }
    return normalized, profile


def ingest_bot_historical_export(
    input_path: Path,
    *,
    original_output: Path,
    normalized_output: Path,
    model_output: Path,
    receipt_output: Path,
    manifest_path: Path,
    config_path: Path,
    retrieved_at_utc: str | None = None,
    warmup_months: int = 3,
) -> dict[str, Any]:
    if input_path.is_symlink() or not input_path.is_file():
        raise ValueError("BOT input must be a regular file")
    retrieved_at_utc = retrieved_at_utc or datetime.now(timezone.utc).isoformat()
    retrieved_timestamp = pd.Timestamp(retrieved_at_utc)
    if retrieved_timestamp.tzinfo is None:
        raise ValueError("retrieval timestamp must include a timezone")
    retrieved_at_utc = retrieved_timestamp.tz_convert("UTC").isoformat()
    raw_bytes = input_path.read_bytes()
    config = load_config(config_path)
    normalized, profile = normalize_bot_historical_export(
        raw_bytes,
        retrieved_at_utc=retrieved_at_utc,
        start=str(config["data"]["start"]),
        end=str(config["data"]["end"]),
        warmup_months=warmup_months,
    )

    raw_sha256 = _sha256_bytes(raw_bytes)
    if original_output.exists():
        if original_output.is_symlink() or not original_output.is_file():
            raise ValueError("preserved BOT original path is unsafe")
        if _sha256_file(original_output) != raw_sha256:
            raise ValueError(
                "refusing to replace a different preserved BOT source vintage"
            )
    else:
        _atomic_bytes(original_output, raw_bytes)
    write_csv_atomic(normalized, normalized_output)
    write_csv_atomic(normalized, model_output)

    receipt = {
        "schema_version": 1,
        "status": "accepted_with_point_in_time_restriction",
        "source": {
            "provider": "Bank of Thailand",
            "table": SOURCE_TABLE,
            "title": "Leading Economic Index and Components (Seasonally Adjusted)",
            "download_url": BOT_FULL_CSV_URL,
            "report_url": BOT_REPORT_URL,
            "metadata_url": BOT_METADATA_URL,
            "retrieved_at_utc": retrieved_at_utc,
        },
        "original": {
            "path": str(original_output),
            "bytes": len(raw_bytes),
            "sha256": raw_sha256,
        },
        "normalized": {
            "path": str(normalized_output),
            "bytes": normalized_output.stat().st_size,
            "sha256": _sha256_file(normalized_output),
        },
        "exploratory_model_input_copy": {
            "path": str(model_output),
            "bytes": model_output.stat().st_size,
            "sha256": _sha256_file(model_output),
        },
        "profile": profile,
        "scientific_eligibility": {
            "structural_source_ready": True,
            "point_in_time_eligible": False,
            "publication_performance_claim_eligible": False,
            "reason": POINT_IN_TIME_CAVEAT,
        },
        "usage_boundary": (
            "Suitable for source integration, descriptive context, and revised-series "
            "sensitivity analysis. Not eligible for causal or publication-grade "
            "performance evaluation without historical vintage snapshots."
        ),
    }
    _atomic_json(receipt_output, receipt)
    _atomic_bytes(
        Path(str(receipt_output) + ".sha256"),
        f"{_sha256_file(receipt_output)}\n".encode("utf-8"),
    )
    manifest_row = pd.DataFrame(
        [
            {
                "source": "bot_official_leading_indicator_historical_revised",
                "provider": "Bank of Thailand",
                "source_table": SOURCE_TABLE,
                "title": (
                    "Leading Economic Index and Components "
                    "(Seasonally Adjusted)"
                ),
                "access_method": (
                    "Bank of Thailand BOTWEBSTAT full-history CSV download"
                ),
                "frequency": "monthly",
                "date_range": (
                    f"{normalized['period'].min()} to "
                    f"{normalized['period'].max()}"
                ),
                "rows": len(normalized),
                "periods": int(normalized["period"].nunique()),
                "metrics": int(normalized["metric"].nunique()),
                "missing_values": int(normalized["value"].isna().sum()),
                "normalized_path": str(normalized_output),
                "normalized_sha256": _sha256_file(normalized_output),
                "original_path": str(original_output),
                "original_sha256": raw_sha256,
                "source_url": BOT_FULL_CSV_URL,
                "retrieved_at_utc": retrieved_at_utc,
                "point_in_time_eligible": False,
                "redistribution_note": (
                    "Check BOT terms and third-party component-source rights "
                    "before redistribution."
                ),
            }
        ]
    )
    write_csv_atomic(manifest_row, manifest_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize the official BOT EC_EI_002_S2 full-history CSV while "
            "preserving its revised-vintage claim boundary"
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--config",
        default="config/real_ohlcv_official_macro.yaml",
    )
    parser.add_argument(
        "--original-output",
        default=None,
        help=(
            "Preserved original path. Defaults to a retrieval-date-stamped path "
            "under data/external/original."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/external/bot_historical_macro.csv",
    )
    parser.add_argument(
        "--model-output",
        default=(
            "data/external/"
            "bot_historical_macro_revised_exploratory_input.csv"
        ),
        help=(
            "Explicitly revised-history copy for the guarded exploratory builder. "
            "This intentionally does not overwrite the legacy point-in-time model "
            "input path under data/raw."
        ),
    )
    parser.add_argument(
        "--receipt",
        default="reports/data_sources/bot_historical_macro.json",
    )
    parser.add_argument(
        "--manifest",
        default="data/external/bot_historical_source_manifest.csv",
    )
    parser.add_argument("--warmup-months", type=int, default=3)
    parser.add_argument("--retrieved-at-utc", default=None)
    args = parser.parse_args()

    retrieved_at = args.retrieved_at_utc or datetime.now(timezone.utc).isoformat()
    retrieval_date = pd.Timestamp(retrieved_at).date().strftime("%Y%m%d")
    original_output = Path(
        args.original_output
        or f"data/external/original/EC_EI_002_S2_ENG_ALL_{retrieval_date}.CSV"
    )
    receipt = ingest_bot_historical_export(
        Path(args.input),
        original_output=original_output,
        normalized_output=Path(args.output),
        model_output=Path(args.model_output),
        receipt_output=Path(args.receipt),
        manifest_path=Path(args.manifest),
        config_path=Path(args.config),
        retrieved_at_utc=retrieved_at,
        warmup_months=args.warmup_months,
    )
    window = receipt["profile"]["modeling_window_with_warmup"]
    print(
        "accepted BOT historical export "
        f"rows={window['rows']} periods={window['periods']} "
        f"metrics={window['metrics']} point_in_time_eligible=false"
    )


if __name__ == "__main__":
    main()
