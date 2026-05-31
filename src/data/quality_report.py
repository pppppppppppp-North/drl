from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.envs.trading_env import load_feature_table
from src.utils.config import ensure_dirs, load_config


def coverage_by_ticker(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    all_dates = pd.Index(sorted(df["date"].dropna().unique()))
    expected_dates = len(all_dates)
    rows = []
    for ticker, part in df.groupby("ticker", sort=True):
        unique_dates = part["date"].nunique()
        rows.append(
            {
                "ticker": ticker,
                "start_date": part["date"].min().date().isoformat(),
                "end_date": part["date"].max().date().isoformat(),
                "rows": int(len(part)),
                "unique_dates": int(unique_dates),
                "expected_dates": int(expected_dates),
                "missing_panel_dates": int(expected_dates - unique_dates),
                "coverage_ratio": float(unique_dates / expected_dates) if expected_dates else 0.0,
            }
        )
    return pd.DataFrame(rows)


def missingness_by_column(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    row_count = len(features)
    for column in features.columns:
        missing = int(features[column].isna().sum())
        rows.append(
            {
                "column": column,
                "missing_values": missing,
                "missing_ratio": float(missing / row_count) if row_count else 0.0,
                "dtype": str(features[column].dtype),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_ratio", "column"], ascending=[False, True]).reset_index(drop=True)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(
    output_path: Path,
    *,
    config_path: str,
    features_path: str,
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    missingness: pd.DataFrame,
) -> None:
    dates = pd.to_datetime(features["date"])
    lines = [
        "# Real OHLCV Data Quality Report",
        "",
        f"Config: `{config_path}`",
        f"Feature table: `{features_path}`",
        "",
        "## Summary",
        "",
        f"- Rows: {len(features)}",
        f"- Tickers: {features['ticker'].nunique()}",
        f"- Date range: {dates.min().date().isoformat()} to {dates.max().date().isoformat()}",
        f"- Unique trading dates: {dates.nunique()}",
        f"- Columns: {len(features.columns)}",
        "",
        "## Ticker Coverage",
        "",
        dataframe_to_markdown(coverage),
        "",
        "## Column Missingness",
        "",
        dataframe_to_markdown(missingness),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv.yaml")
    parser.add_argument("--output-dir", default="reports/data_quality")
    args = parser.parse_args()

    config = load_config(args.config)
    features = load_feature_table(config["paths"]["features"], config["paths"].get("fallback_features_csv"))
    output_dir = Path(args.output_dir)
    ensure_dirs(output_dir)

    coverage = coverage_by_ticker(features)
    missingness = missingness_by_column(features)
    coverage.to_csv(output_dir / "ticker_coverage.csv", index=False)
    missingness.to_csv(output_dir / "column_missingness.csv", index=False)
    write_markdown_report(
        output_dir / "real_ohlcv_data_quality.md",
        config_path=args.config,
        features_path=config["paths"]["features"],
        features=features,
        coverage=coverage,
        missingness=missingness,
    )
    print(coverage.to_string(index=False))
    print(missingness.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
