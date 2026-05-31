from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LeakageCheckConfig:
    date_column: str = "date"
    ticker_column: str = "ticker"
    release_date_columns: tuple[str, ...] = ()


def _status(ok: bool) -> str:
    return "pass" if ok else "fail"


def _check_row(name: str, ok: bool, details: str) -> dict[str, str]:
    return {"check": name, "status": _status(ok), "details": details}


def infer_release_date_columns(features: pd.DataFrame) -> tuple[str, ...]:
    suffixes = ("release_date", "available_date", "asof_date")
    columns = [
        column
        for column in features.columns
        if column != "date" and any(column == suffix or column.endswith(f"_{suffix}") for suffix in suffixes)
    ]
    return tuple(columns)


def run_leakage_checks(features: pd.DataFrame, config: LeakageCheckConfig | None = None) -> pd.DataFrame:
    config = config or LeakageCheckConfig()
    rows: list[dict[str, str]] = []

    required = {config.date_column, config.ticker_column}
    missing = sorted(required - set(features.columns))
    rows.append(_check_row("required_columns", not missing, f"missing={missing}" if missing else "present"))
    if missing:
        return pd.DataFrame(rows)

    checked = features.copy()
    checked[config.date_column] = pd.to_datetime(checked[config.date_column], errors="coerce")
    invalid_dates = int(checked[config.date_column].isna().sum())
    rows.append(_check_row("valid_dates", invalid_dates == 0, f"invalid_dates={invalid_dates}"))

    duplicate_count = int(checked.duplicated([config.date_column, config.ticker_column]).sum())
    rows.append(_check_row("duplicate_date_ticker", duplicate_count == 0, f"duplicates={duplicate_count}"))

    unsorted_tickers = []
    for ticker, group in checked.groupby(config.ticker_column, sort=False):
        if not group[config.date_column].is_monotonic_increasing:
            unsorted_tickers.append(str(ticker))
    rows.append(
        _check_row(
            "ticker_date_monotonic",
            not unsorted_tickers,
            f"unsorted_tickers={unsorted_tickers[:10]}" if unsorted_tickers else "all ticker series monotonic",
        )
    )

    release_columns = config.release_date_columns or infer_release_date_columns(checked)
    missing_release_columns = sorted(set(release_columns) - set(checked.columns))
    rows.append(
        _check_row(
            "release_columns_present",
            not missing_release_columns,
            f"missing={missing_release_columns}" if missing_release_columns else f"columns={list(release_columns)}",
        )
    )

    for column in release_columns:
        if column not in checked.columns:
            continue
        release_dates = pd.to_datetime(checked[column], errors="coerce")
        invalid_release_dates = int(release_dates.isna().sum())
        future_rows = int((release_dates > checked[config.date_column]).sum())
        rows.append(
            _check_row(
                f"{column}_valid",
                invalid_release_dates == 0,
                f"invalid_release_dates={invalid_release_dates}",
            )
        )
        rows.append(
            _check_row(
                f"{column}_not_future",
                future_rows == 0,
                f"future_rows={future_rows}",
            )
        )

    return pd.DataFrame(rows)


def assert_no_leakage(features: pd.DataFrame, config: LeakageCheckConfig | None = None) -> pd.DataFrame:
    results = run_leakage_checks(features, config)
    failures = results[results["status"] != "pass"]
    if not failures.empty:
        detail = "; ".join(f"{row.check}: {row.details}" for row in failures.itertuples(index=False))
        raise ValueError(f"leakage checks failed: {detail}")
    return results


def check_split_boundaries(splits: dict[str, pd.DataFrame], date_column: str = "date") -> pd.DataFrame:
    rows = []
    seen_dates: set[pd.Timestamp] = set()
    previous_end: pd.Timestamp | None = None

    for split_name in ("train", "validation", "test"):
        split = splits.get(split_name)
        if split is None or split.empty:
            rows.append(_check_row(f"{split_name}_non_empty", False, "split missing or empty"))
            continue

        dates = set(pd.to_datetime(split[date_column]))
        overlap = seen_dates.intersection(dates)
        rows.append(_check_row(f"{split_name}_date_overlap", not overlap, f"overlap_dates={len(overlap)}"))

        start = min(dates)
        end = max(dates)
        boundary_ok = previous_end is None or previous_end < start
        rows.append(
            _check_row(
                f"{split_name}_after_previous",
                boundary_ok,
                f"start={start.date().isoformat()}, end={end.date().isoformat()}",
            )
        )
        seen_dates.update(dates)
        previous_end = end

    return pd.DataFrame(rows)
