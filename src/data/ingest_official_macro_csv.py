from __future__ import annotations

import argparse
from calendar import monthrange
from pathlib import Path

import pandas as pd

from src.data.ingest_bot_macro import _safe_name
from src.data.ingest_ohlcv import write_manifest_row
from src.utils.config import ensure_dirs, load_config


def _last_business_day_following_month(period: pd.Timestamp) -> pd.Timestamp:
    following = period + pd.DateOffset(months=1)
    day = monthrange(int(following.year), int(following.month))[1]
    release = pd.Timestamp(year=int(following.year), month=int(following.month), day=day)
    while release.weekday() >= 5:
        release -= pd.Timedelta(days=1)
    return release


def assign_release_dates(periods: pd.Series, rule: str) -> pd.Series:
    parsed = pd.to_datetime(periods, errors="coerce")
    if parsed.isna().any():
        bad = periods[parsed.isna()].astype(str).head(5).tolist()
        raise ValueError(f"could not parse period values: {bad}")

    if rule == "same_day":
        return parsed
    if rule == "last_business_day_following_month":
        return parsed.map(_last_business_day_following_month)
    raise ValueError(f"unsupported release date rule: {rule}")


def normalize_long_macro_csv(
    raw: pd.DataFrame,
    *,
    period_column: str,
    release_date_column: str | None,
    metric_column: str,
    value_column: str,
    release_rule: str,
    source_table: str,
    source_name: str,
) -> pd.DataFrame:
    required = {period_column, metric_column, value_column}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"official macro CSV is missing columns: {sorted(missing)}")

    df = raw.copy()
    df["period"] = pd.to_datetime(df[period_column], errors="coerce")
    if df["period"].isna().any():
        bad = df.loc[df["period"].isna(), period_column].astype(str).head(5).tolist()
        raise ValueError(f"could not parse period values: {bad}")

    if release_date_column and release_date_column in df.columns:
        df["release_date"] = pd.to_datetime(df[release_date_column], errors="coerce")
        if df["release_date"].isna().any():
            bad = df.loc[df["release_date"].isna(), release_date_column].astype(str).head(5).tolist()
            raise ValueError(f"could not parse release date values: {bad}")
    else:
        df["release_date"] = assign_release_dates(df["period"], release_rule)

    df["metric_label"] = df[metric_column].astype(str).str.strip()
    df["metric"] = df["metric_label"].map(_safe_name)
    df["value"] = pd.to_numeric(df[value_column].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        raise ValueError("official macro CSV contains no numeric macro observations")

    df["source_table"] = source_table
    df["source"] = source_name
    normalized = df[["period", "release_date", "metric", "metric_label", "value", "source_table", "source"]].copy()
    normalized["period"] = normalized["period"].dt.date.astype(str)
    normalized["release_date"] = normalized["release_date"].dt.date.astype(str)
    return normalized.sort_values(["period", "metric"]).reset_index(drop=True)


def normalize_wide_macro_csv(
    raw: pd.DataFrame,
    *,
    period_column: str,
    release_date_column: str | None,
    release_rule: str,
    source_table: str,
    source_name: str,
) -> pd.DataFrame:
    if period_column not in raw.columns:
        raise ValueError(f"official macro CSV is missing period column: {period_column}")
    id_columns = [period_column]
    if release_date_column and release_date_column in raw.columns:
        id_columns.append(release_date_column)
    value_columns = [column for column in raw.columns if column not in id_columns]
    if not value_columns:
        raise ValueError("wide official macro CSV has no metric columns")

    long = raw.melt(id_vars=id_columns, value_vars=value_columns, var_name="metric", value_name="value")
    return normalize_long_macro_csv(
        long,
        period_column=period_column,
        release_date_column=release_date_column if release_date_column in long.columns else None,
        metric_column="metric",
        value_column="value",
        release_rule=release_rule,
        source_table=source_table,
        source_name=source_name,
    )


def normalize_macro_csv(
    raw: pd.DataFrame,
    *,
    csv_format: str,
    period_column: str,
    release_date_column: str | None,
    metric_column: str,
    value_column: str,
    release_rule: str,
    source_table: str,
    source_name: str,
) -> pd.DataFrame:
    if csv_format == "auto":
        csv_format = "long" if {metric_column, value_column}.issubset(raw.columns) else "wide"
    if csv_format == "long":
        return normalize_long_macro_csv(
            raw,
            period_column=period_column,
            release_date_column=release_date_column,
            metric_column=metric_column,
            value_column=value_column,
            release_rule=release_rule,
            source_table=source_table,
            source_name=source_name,
        )
    if csv_format == "wide":
        return normalize_wide_macro_csv(
            raw,
            period_column=period_column,
            release_date_column=release_date_column,
            release_rule=release_rule,
            source_table=source_table,
            source_name=source_name,
        )
    raise ValueError(f"unsupported official macro CSV format: {csv_format}")


def write_official_macro_csv_manifest(
    manifest_path: Path,
    macro: pd.DataFrame,
    raw_file_path: Path,
    *,
    source_name: str,
) -> None:
    manifest_prices = macro.rename(columns={"period": "date"}).copy()
    manifest_prices["ticker"] = manifest_prices["source_table"] + ":" + manifest_prices["metric"]
    for column in ["open", "high", "low", "close", "volume"]:
        manifest_prices[column] = manifest_prices["value"] if column != "volume" else 0.0
    write_manifest_row(
        manifest_path,
        source_name=source_name,
        access_method="User-provided official macro CSV import",
        prices=manifest_prices,
        raw_file_path=raw_file_path,
        license_note="User-provided official macro CSV; verify source terms, redistribution rights, and release-date assumptions before publication.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_official_macro.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default="data/data_manifest.csv")
    parser.add_argument("--format", choices=["auto", "long", "wide"], default="auto")
    parser.add_argument("--period-column", default="period")
    parser.add_argument("--release-date-column", default="release_date")
    parser.add_argument("--metric-column", default="metric")
    parser.add_argument("--value-column", default="value")
    parser.add_argument("--source-table", default="official_macro_csv")
    parser.add_argument("--source-name", default="user_official_macro_csv")
    parser.add_argument("--manifest-source-name", default="official_macro_csv_import")
    args = parser.parse_args()

    config = load_config(args.config)
    macro_config = config.get("official_macro_context", {})
    output = Path(args.output or macro_config.get("raw_path", "data/raw/bot_official_macro.csv"))
    release_rule = str(macro_config.get("release_rule", "last_business_day_following_month"))
    raw = pd.read_csv(args.input)
    macro = normalize_macro_csv(
        raw,
        csv_format=args.format,
        period_column=args.period_column,
        release_date_column=args.release_date_column,
        metric_column=args.metric_column,
        value_column=args.value_column,
        release_rule=release_rule,
        source_table=args.source_table,
        source_name=args.source_name,
    )
    ensure_dirs(output.parent)
    macro.to_csv(output, index=False)
    write_official_macro_csv_manifest(
        Path(args.manifest),
        macro,
        output,
        source_name=args.manifest_source_name,
    )
    print(
        f"wrote {output} rows={len(macro)} "
        f"periods={macro['period'].nunique()} metrics={macro['metric'].nunique()}"
    )


if __name__ == "__main__":
    main()
