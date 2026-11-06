from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingest_bot_historical_export import (
    EXPECTED_METRICS,
    normalize_bot_historical_export,
)


METRIC_LABELS = (
    "Authorized Capital of Newly Registered Companies (Million Baht)",
    "Broad Money at 2000 prices (Million Baht)",
    "Business Sentiment Index (3 months)",
    "Construction Areas Permitted (1000 sq. m)",
    "Export Volume index (exclude Gold)",
    "Index Change 1/",
    "Leading Economic Index",
    "Oil Price Inverse Index (Dubai)",
    "SET index",
)


def _export_bytes(*, missing_last_metric: bool = False) -> bytes:
    periods = ("OCT 2020 r", "NOV 2020 r", "DEC 2020 r", "JAN 2021 p")
    rows = []
    labels = METRIC_LABELS[:-1] if missing_last_metric else METRIC_LABELS
    for index, label in enumerate(labels, start=1):
        rows.append([index, label, *[float(index * 10 + offset) for offset in range(4)]])
    table = pd.DataFrame(rows, columns=["", "", *periods])
    payload = io.StringIO()
    payload.write("Bank of Thailand\n")
    payload.write("EC_EI_002_S2 : Leading Economic Index and Components\n")
    payload.write("\n")
    payload.write("Last Updated :\n")
    payload.write("Retrieved date :\n")
    table.to_csv(payload, index=False)
    return payload.getvalue().encode("utf-8")


def test_normalizer_accepts_complete_revised_export_with_warmup() -> None:
    normalized, profile = normalize_bot_historical_export(
        _export_bytes(),
        retrieved_at_utc="2026-07-25T12:00:00+00:00",
        start="2021-01-01",
        end="2021-01-31",
        warmup_months=3,
        minimum_full_history_periods=4,
    )

    assert len(normalized) == 36
    assert normalized["period"].nunique() == 4
    assert tuple(sorted(normalized["metric"].unique())) == tuple(
        sorted(EXPECTED_METRICS)
    )
    assert normalized["point_in_time_eligible"].eq(False).all()
    assert set(normalized["revision_flag"]) == {"revised", "preliminary"}
    assert normalized["release_date"].min() == "2020-11-30"
    assert profile["modeling_window_with_warmup"]["missing_values"] == 0


def test_normalizer_rejects_changed_metric_contract() -> None:
    with pytest.raises(ValueError, match="metric contract changed"):
        normalize_bot_historical_export(
            _export_bytes(missing_last_metric=True),
            retrieved_at_utc="2026-07-25T12:00:00+00:00",
            start="2021-01-01",
            end="2021-01-31",
            warmup_months=3,
            minimum_full_history_periods=4,
        )


def test_normalizer_rejects_missing_requested_month() -> None:
    raw = _export_bytes().replace(b"NOV 2020 r", b"SEP 2020 r")

    with pytest.raises(ValueError, match="missing requested months"):
        normalize_bot_historical_export(
            raw,
            retrieved_at_utc="2026-07-25T12:00:00+00:00",
            start="2021-01-01",
            end="2021-01-31",
            warmup_months=3,
            minimum_full_history_periods=4,
        )
