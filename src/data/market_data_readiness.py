from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.leakage import run_leakage_checks
from src.data.quality_report import coverage_by_ticker, dataframe_to_markdown, missingness_by_column
from src.data.ticker_universe import resolve_tickers_from_config
from src.envs.trading_env import load_feature_table
from src.utils.config import ensure_dirs, load_config


PRICE_COLUMNS = ("date", "ticker", "open", "high", "low", "close", "volume")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    ensure_dirs(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _artifact(path: Path, project_root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(project_root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _check(
    rows: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    *,
    severity: str,
    observed: Any,
    requirement: str,
    details: str,
) -> None:
    rows.append(
        {
            "check": check_id,
            "status": "pass" if passed else "fail",
            "severity": severity,
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def _resolve_feature_path(config: dict[str, Any], project_root: Path) -> Path:
    primary = project_root / str(config["paths"]["features"])
    fallback = project_root / str(
        config["paths"].get("fallback_features_csv", primary.with_suffix(".csv"))
    )
    if primary.is_file():
        return primary
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(primary)


def evaluate_market_data(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    project_root = project_root.resolve()
    raw_path = (project_root / str(config["paths"]["raw_prices"])).resolve()
    index_path = (
        project_root
        / str(config.get("market_indices", {}).get("raw_prices", "data/raw/prices_market_indices.csv"))
    ).resolve()
    feature_path = _resolve_feature_path(config, project_root).resolve()
    manifest_path = (project_root / "data/data_manifest.csv").resolve()
    for path in (raw_path, index_path, feature_path, manifest_path):
        if not path.is_relative_to(project_root) or not path.is_file():
            raise FileNotFoundError(f"required market-data artifact is missing or outside the project: {path}")

    stocks = pd.read_csv(raw_path)
    indices = pd.read_csv(index_path)
    features = load_feature_table(feature_path)
    manifest = pd.read_csv(manifest_path)
    checks: list[dict[str, Any]] = []
    expected_tickers = sorted(resolve_tickers_from_config(config))
    expected_index_symbols = sorted(str(value) for value in config.get("market_indices", {}).get("symbols", {}).values())
    start = pd.Timestamp(str(config["data"]["start"]))
    end_exclusive = pd.Timestamp(str(config["data"]["end"]))
    requested_business_days = len(pd.bdate_range(start, end_exclusive - pd.Timedelta(days=1)))
    minimum_dates = max(1, math.floor(requested_business_days * 0.85))

    stock_required = set(PRICE_COLUMNS)
    missing_stock_columns = sorted(stock_required - set(stocks.columns))
    _check(
        checks,
        "stock_required_columns",
        not missing_stock_columns,
        severity="critical",
        observed=missing_stock_columns,
        requirement=f"columns include {sorted(stock_required)}",
        details="Raw stock OHLCV must contain the canonical daily panel fields.",
    )
    index_required = stock_required | {"index_name"}
    missing_index_columns = sorted(index_required - set(indices.columns))
    _check(
        checks,
        "index_required_columns",
        not missing_index_columns,
        severity="critical",
        observed=missing_index_columns,
        requirement=f"columns include {sorted(index_required)}",
        details="SET-index rows must include the canonical daily OHLCV fields and index name.",
    )

    if not missing_stock_columns:
        stocks["date"] = pd.to_datetime(stocks["date"], errors="coerce")
        for column in ("open", "high", "low", "close", "volume"):
            stocks[column] = pd.to_numeric(stocks[column], errors="coerce")
        actual_tickers = sorted(stocks["ticker"].dropna().astype(str).unique())
        _check(
            checks,
            "stock_ticker_contract",
            actual_tickers == expected_tickers,
            severity="critical",
            observed=actual_tickers,
            requirement=str(expected_tickers),
            details="The observed stock universe must match the preregistered five-ticker pilot exactly.",
        )
        duplicates = int(stocks.duplicated(["date", "ticker"]).sum())
        _check(
            checks,
            "stock_unique_daily_grain",
            duplicates == 0,
            severity="critical",
            observed=duplicates,
            requirement="0 duplicate date/ticker keys",
            details="Duplicate panel keys would overweight observations and corrupt temporal splits.",
        )
        missing_required = int(stocks[list(PRICE_COLUMNS)].isna().sum().sum())
        _check(
            checks,
            "stock_required_values_complete",
            missing_required == 0,
            severity="critical",
            observed=missing_required,
            requirement="0 missing required values",
            details="Every daily stock row needs a complete date, ticker, OHLC, and volume record.",
        )
        numeric = stocks[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        invalid_numeric = int((~np.isfinite(numeric)).sum())
        nonpositive_prices = int((stocks[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
        negative_volume = int((stocks["volume"] < 0).sum())
        _check(
            checks,
            "stock_numeric_domain",
            invalid_numeric == 0 and nonpositive_prices == 0 and negative_volume == 0,
            severity="critical",
            observed={
                "nonfinite_values": invalid_numeric,
                "nonpositive_price_rows": nonpositive_prices,
                "negative_volume_rows": negative_volume,
            },
            requirement="finite positive prices and nonnegative volume",
            details="Invalid prices or volume make environment transitions and returns undefined.",
        )
        tolerance = 1e-8
        envelope_violations = int(
            (
                (stocks["high"] + tolerance < stocks[["open", "close"]].max(axis=1))
                | (stocks["low"] - tolerance > stocks[["open", "close"]].min(axis=1))
                | (stocks["low"] > stocks["high"] + tolerance)
            ).sum()
        )
        _check(
            checks,
            "stock_ohlc_envelope",
            envelope_violations == 0,
            severity="high",
            observed=envelope_violations,
            requirement="0 OHLC envelope violations after deterministic normalization",
            details="Adjusted provider rows must still satisfy low <= open/close <= high.",
        )
        repair_flags = (
            stocks["ohlc_repaired"].fillna(False).astype(str).str.lower().isin({"1", "true", "yes"})
            if "ohlc_repaired" in stocks.columns
            else pd.Series(False, index=stocks.index)
        )
        repaired_by_ticker = {
            str(ticker): int(count)
            for ticker, count in stocks.loc[repair_flags].groupby("ticker").size().items()
        }
        repaired_ohlc_rows = int(repair_flags.sum())
        checks.append(
            {
                "check": "stock_ohlc_repair_disclosure",
                "status": "warning" if repaired_ohlc_rows else "pass",
                "severity": "medium",
                "observed": {
                    "repaired_rows": repaired_ohlc_rows,
                    "repaired_by_ticker": repaired_by_ticker,
                },
                "requirement": "provider OHLC repairs are counted and disclosed",
                "details": (
                    "Adjusted-price OHLC envelopes were normalized deterministically before feature generation."
                    if repaired_ohlc_rows
                    else "No provider OHLC row required deterministic envelope normalization."
                ),
            }
        )
        unique_dates = int(stocks["date"].nunique())
        date_min = stocks["date"].min()
        date_max = stocks["date"].max()
        date_bounds_ok = (
            pd.notna(date_min)
            and pd.notna(date_max)
            and date_min <= start + pd.Timedelta(days=10)
            and date_max >= end_exclusive - pd.Timedelta(days=10)
            and unique_dates >= minimum_dates
        )
        _check(
            checks,
            "stock_temporal_coverage",
            bool(date_bounds_ok),
            severity="critical",
            observed={
                "start": date_min.date().isoformat() if pd.notna(date_min) else None,
                "end": date_max.date().isoformat() if pd.notna(date_max) else None,
                "unique_dates": unique_dates,
            },
            requirement=f"near {start.date()}..{(end_exclusive - pd.Timedelta(days=1)).date()} and >= {minimum_dates} dates",
            details="The requested four-year period must be substantially present before walk-forward splitting.",
        )
    else:
        stocks["date"] = pd.NaT
        repaired_ohlc_rows = 0

    coverage = coverage_by_ticker(stocks) if {"date", "ticker"}.issubset(stocks.columns) else pd.DataFrame()
    minimum_coverage = float(coverage["coverage_ratio"].min()) if not coverage.empty else 0.0
    _check(
        checks,
        "stock_balanced_panel",
        bool(not coverage.empty and minimum_coverage >= 0.99),
        severity="high",
        observed=minimum_coverage,
        requirement=">= 0.99 coverage for every ticker against the union of trading dates",
        details="Materially unbalanced ticker histories can create changing-universe bias.",
    )

    if not missing_index_columns:
        indices["date"] = pd.to_datetime(indices["date"], errors="coerce")
        for column in ("open", "high", "low", "close", "volume"):
            indices[column] = pd.to_numeric(indices[column], errors="coerce")
        actual_indices = sorted(indices["ticker"].dropna().astype(str).unique())
        index_duplicates = int(indices.duplicated(["date", "index_name"]).sum())
        index_missing = int(indices[list(index_required)].isna().sum().sum())
        index_dates = int(indices["date"].nunique())
        _check(
            checks,
            "index_contract",
            actual_indices == expected_index_symbols and index_duplicates == 0 and index_missing == 0 and index_dates >= minimum_dates,
            severity="critical",
            observed={
                "symbols": actual_indices,
                "duplicates": index_duplicates,
                "missing_required": index_missing,
                "unique_dates": index_dates,
            },
            requirement=f"symbols={expected_index_symbols}, unique daily grain, complete values, >= {minimum_dates} dates",
            details="The SET benchmark must be a complete, uniquely keyed daily series.",
        )
        stock_dates = set(pd.to_datetime(stocks["date"]).dropna().unique())
        index_date_set = set(indices["date"].dropna().unique())
        overlap = len(stock_dates & index_date_set) / len(stock_dates) if stock_dates else 0.0
        missing_index_dates = sorted(pd.Timestamp(value).date().isoformat() for value in stock_dates - index_date_set)
        if overlap >= 0.99:
            checks.append(
                {
                    "check": "index_stock_date_overlap",
                    "status": "pass" if overlap == 1.0 else "warning",
                    "severity": "medium" if overlap < 1.0 else "high",
                    "observed": {"coverage_ratio": overlap, "missing_stock_dates": missing_index_dates},
                    "requirement": ">= 0.99 of stock dates have a SET-index observation",
                    "details": (
                        "Missing SET dates are forward-filled at the index level with an explicit availability flag."
                        if overlap < 1.0
                        else "Every stock date has a same-date SET observation."
                    ),
                }
            )
        else:
            _check(
                checks,
                "index_stock_date_overlap",
                False,
                severity="high",
                observed={"coverage_ratio": overlap, "missing_stock_dates": missing_index_dates},
                requirement=">= 0.99 of stock dates have a SET-index observation",
                details="SET-index coverage is too sparse for the declared market-context features.",
            )

    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    configured_features = [str(value) for value in config.get("features", {}).get("columns", [])]
    missing_features = sorted(set(configured_features) - set(features.columns))
    _check(
        checks,
        "configured_feature_contract",
        not missing_features,
        severity="critical",
        observed=missing_features,
        requirement=f"all {len(configured_features)} configured model features are present",
        details="Training must not silently omit preregistered observation columns.",
    )
    feature_duplicates = int(features.duplicated(["date", "ticker"]).sum())
    feature_missing = int(features[["date", "ticker", *configured_features]].isna().sum().sum()) if not missing_features else -1
    feature_numeric = features[configured_features].apply(pd.to_numeric, errors="coerce") if not missing_features else pd.DataFrame()
    feature_nonfinite = int((~np.isfinite(feature_numeric.to_numpy(dtype=float))).sum()) if not feature_numeric.empty else -1
    _check(
        checks,
        "feature_values_valid",
        feature_duplicates == 0 and feature_missing == 0 and feature_nonfinite == 0,
        severity="critical",
        observed={
            "duplicate_keys": feature_duplicates,
            "missing_required": feature_missing,
            "nonfinite_values": feature_nonfinite,
        },
        requirement="unique date/ticker keys with finite, non-missing configured features",
        details="The model observation table must be numerically valid at its declared grain.",
    )
    stock_keys = set(zip(pd.to_datetime(stocks["date"]), stocks["ticker"].astype(str)))
    feature_keys = set(zip(features["date"], features["ticker"].astype(str)))
    _check(
        checks,
        "feature_stock_key_integrity",
        feature_keys == stock_keys,
        severity="critical",
        observed={
            "stock_keys": len(stock_keys),
            "feature_keys": len(feature_keys),
            "missing_feature_keys": len(stock_keys - feature_keys),
            "orphan_feature_keys": len(feature_keys - stock_keys),
        },
        requirement="feature keys exactly equal raw stock keys",
        details="Feature generation must neither drop observations nor introduce orphan dates.",
    )
    leakage = run_leakage_checks(features)
    leakage_failures = leakage[leakage["status"].ne("pass")].to_dict("records")
    _check(
        checks,
        "feature_leakage_checks",
        not leakage_failures,
        severity="critical",
        observed=leakage_failures,
        requirement="all configured leakage checks pass",
        details="Chronology, duplicate-key, and release-date checks must fail closed.",
    )
    scaffold_columns = [column for column in ("macro_rate_change", "sentiment_score") if column in features]
    constant_scaffolds = [column for column in scaffold_columns if features[column].nunique(dropna=False) <= 1]
    checks.append(
        {
            "check": "context_scaffold_scope",
            "status": "warning" if constant_scaffolds else "pass",
            "severity": "medium",
            "observed": constant_scaffolds,
            "requirement": "constant scaffold fields are explicitly excluded from contextual claims",
            "details": (
                "These columns are pipeline placeholders, not observed historical macro or sentiment evidence."
                if constant_scaffolds
                else "Configured contextual fields vary in the modeling period."
            ),
        }
    )

    required_manifest_sources = {"yahoo_thai_ohlcv", "yahoo_thai_market_indices"}
    manifest_sources = set(manifest.get("source", pd.Series(dtype=str)).dropna().astype(str))
    _check(
        checks,
        "source_manifest_records",
        required_manifest_sources.issubset(manifest_sources),
        severity="critical",
        observed=sorted(manifest_sources & required_manifest_sources),
        requirement=str(sorted(required_manifest_sources)),
        details="Both observed feeds need auditable source and access records.",
    )

    checks_frame = pd.DataFrame(checks)
    failed = checks_frame[checks_frame["status"].eq("fail")]
    warnings = checks_frame[checks_frame["status"].eq("warning")]
    receipt = {
        "schema_version": 1,
        "status": "pass" if failed.empty else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_origin": "observed_market",
        "evidence_class": "restricted_private_research_pilot",
        "execution_eligible": bool(failed.empty),
        "publication_claim_eligible": False,
        "claim_scope": (
            "Five-ticker Yahoo Finance Thai-market pilot with SET-index context; not a survivorship-free "
            "SET50/SET100 or licensed-market study."
        ),
        "usage_boundary": (
            "Private educational/research execution only. Do not redistribute raw Yahoo data or use this "
            "bundle for publication-grade claims without provider permission or a licensed SET replacement."
        ),
        "config_source": str(config.get("data", {}).get("source", "")),
        "requested_period": {
            "start": str(config["data"]["start"]),
            "end_exclusive": str(config["data"]["end"]),
        },
        "summary": {
            "stock_rows": int(len(stocks)),
            "stock_tickers": int(stocks["ticker"].nunique()),
            "stock_dates": int(pd.to_datetime(stocks["date"]).nunique()),
            "index_rows": int(len(indices)),
            "index_dates": int(pd.to_datetime(indices["date"]).nunique()),
            "feature_rows": int(len(features)),
            "feature_columns": int(len(features.columns)),
            "repaired_ohlc_rows": repaired_ohlc_rows,
            "failed_checks": int(len(failed)),
            "warnings": int(len(warnings)),
        },
        "artifacts": {
            "stocks": _artifact(raw_path, project_root),
            "market_index": _artifact(index_path, project_root),
            "features": _artifact(feature_path, project_root),
            "source_manifest": _artifact(manifest_path, project_root),
        },
        "sources": [
            {
                "name": "Yahoo Finance daily market data",
                "url": "https://finance.yahoo.com/",
                "access_method": "yfinance download, daily interval, auto_adjust=True, repair=True",
            },
            {
                "name": "Stock Exchange of Thailand historical data service",
                "url": "https://www.set.or.th/en/services/connectivity-and-data/data/historical",
                "access_method": "preferred licensed/request source for final publication-grade replacement",
            },
        ],
        "checks": checks,
    }
    return receipt, checks_frame, coverage, missingness_by_column(features)


def write_readiness_outputs(
    output_dir: Path,
    *,
    config_path: str,
    receipt: dict[str, Any],
    checks: pd.DataFrame,
    coverage: pd.DataFrame,
    missingness: pd.DataFrame,
) -> Path:
    ensure_dirs(output_dir)
    receipt_path = output_dir / "market_data_readiness.json"
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, default=str) + "\n")
    _atomic_text(receipt_path.with_suffix(".sha256"), _sha256(receipt_path) + "\n")
    checks.to_csv(output_dir / "market_data_quality_checks.csv", index=False)
    coverage.to_csv(output_dir / "ticker_coverage.csv", index=False)
    missingness.to_csv(output_dir / "column_missingness.csv", index=False)
    summary = receipt["summary"]
    failed = checks[checks["status"].eq("fail")]
    warnings = checks[checks["status"].eq("warning")]
    lines = [
        "# Real Thai Market Data Readiness Report",
        "",
        "## Technical Summary",
        "",
        (
            f"The observed five-ticker pilot data bundle **{receipt['status'].upper()}ED** its strict readiness gate: "
            f"{len(checks) - len(failed) - len(warnings)} checks passed, {len(warnings)} warnings remain, "
            f"and {len(failed)} checks failed."
        ),
        "",
        f"- Config: `{config_path}`",
        f"- Scope: {receipt['claim_scope']}",
        f"- Stock panel: {summary['stock_rows']:,} rows, {summary['stock_tickers']} tickers, {summary['stock_dates']} dates",
        f"- Deterministically normalized OHLC rows: {summary['repaired_ohlc_rows']:,}",
        f"- SET index: {summary['index_rows']:,} rows, {summary['index_dates']} dates",
        f"- Feature panel: {summary['feature_rows']:,} rows and {summary['feature_columns']} columns",
        "",
        "## Quality Checks",
        "",
        dataframe_to_markdown(checks),
        "",
        "## Ticker Coverage",
        "",
        dataframe_to_markdown(coverage),
        "",
        "## Feature Missingness",
        "",
        dataframe_to_markdown(missingness),
        "",
        "## Scientific Boundary",
        "",
        (
            "This bundle may run the preregistered five-ticker private research pilot, but it is not eligible "
            "for redistribution or publication-grade claims. It is not a survivorship-free SET50/SET100 panel, "
            "and its Yahoo access path must be replaced by licensed SET/SETSMART data or used with explicit "
            "provider permission before publication. Constant macro and sentiment scaffolds are not observed "
            "contextual evidence and must not support contextual conclusions."
        ),
        "",
    ]
    _atomic_text(output_dir / "real_market_data_quality.md", "\n".join(lines))
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed readiness checks for the real Thai market pilot")
    parser.add_argument("--config", default="config/real_ohlcv_market.yaml")
    parser.add_argument("--output-dir", default="reports/data_quality_market")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_config(project_root / args.config)
    if str(config.get("data", {}).get("source", "")).lower() == "synthetic":
        raise ValueError("generated data cannot satisfy the observed-market readiness gate")
    receipt, checks, coverage, missingness = evaluate_market_data(config, project_root=project_root)
    receipt_path = write_readiness_outputs(
        project_root / args.output_dir,
        config_path=args.config,
        receipt=receipt,
        checks=checks,
        coverage=coverage,
        missingness=missingness,
    )
    print(checks[["check", "status", "severity", "observed"]].to_string(index=False))
    print(f"wrote {receipt_path} status={receipt['status']}")
    if receipt["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
