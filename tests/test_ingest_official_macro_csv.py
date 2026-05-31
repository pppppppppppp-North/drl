from __future__ import annotations

import pandas as pd
import pytest

from src.data.ingest_official_macro_csv import normalize_macro_csv


def test_normalize_wide_macro_csv_assigns_release_dates() -> None:
    raw = pd.DataFrame(
        {
            "period": ["2023-01-01", "2023-02-01"],
            "Leading Economic Index": ["100.5", "101.25"],
            "Index Change": ["0.5", "0.75"],
        }
    )

    normalized = normalize_macro_csv(
        raw,
        csv_format="wide",
        period_column="period",
        release_date_column="release_date",
        metric_column="metric",
        value_column="value",
        release_rule="last_business_day_following_month",
        source_table="historical_bot_export",
        source_name="Bank of Thailand historical export",
    )

    assert len(normalized) == 4
    assert set(normalized["metric"]) == {"leading_economic_index", "index_change"}
    assert set(normalized["release_date"]) == {"2023-02-28", "2023-03-31"}
    assert set(normalized["source_table"]) == {"historical_bot_export"}


def test_normalize_long_macro_csv_preserves_release_dates() -> None:
    raw = pd.DataFrame(
        {
            "period": ["2023-01-01", "2023-01-01"],
            "release_date": ["2023-02-15", "2023-02-15"],
            "metric": ["Leading Economic Index", "Index Change"],
            "value": ["100.0", "1.5"],
        }
    )

    normalized = normalize_macro_csv(
        raw,
        csv_format="auto",
        period_column="period",
        release_date_column="release_date",
        metric_column="metric",
        value_column="value",
        release_rule="last_business_day_following_month",
        source_table="manual_bot_csv",
        source_name="manual official macro csv",
    )

    assert normalized["release_date"].tolist() == ["2023-02-15", "2023-02-15"]
    assert normalized["value"].tolist() == [1.5, 100.0]


def test_normalize_macro_csv_rejects_missing_period_column() -> None:
    raw = pd.DataFrame({"Leading Economic Index": [100.0]})

    with pytest.raises(ValueError, match="missing period column"):
        normalize_macro_csv(
            raw,
            csv_format="wide",
            period_column="period",
            release_date_column="release_date",
            metric_column="metric",
            value_column="value",
            release_rule="last_business_day_following_month",
            source_table="historical_bot_export",
            source_name="Bank of Thailand historical export",
        )
