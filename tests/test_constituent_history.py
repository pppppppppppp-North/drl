from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.constituent_history import (
    acquisition_universe,
    load_constituent_history,
    membership_as_of,
    validate_constituent_history,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = (
    PROJECT_ROOT
    / "data/reference/set50_set100_membership_2021_2024.csv"
)


def test_history_validates_complete_contiguous_intervals() -> None:
    history = load_constituent_history(HISTORY_PATH)

    assert len(history) == 1000
    assert history["record_id"].nunique() == 10
    assert history["effective_from"].min() == pd.Timestamp("2021-01-01")
    assert history["effective_to_exclusive"].max() == pd.Timestamp(
        "2025-01-01"
    )


def test_as_of_membership_changes_at_revision_publication_boundary() -> None:
    history = load_constituent_history(HISTORY_PATH)

    before = membership_as_of(history, "2021-02-10", index_name="SET100")
    after = membership_as_of(history, "2021-02-11", index_name="SET100")

    assert before["record_id"].unique().tolist() == [
        "set_2021_h1_original"
    ]
    assert after["record_id"].unique().tolist() == [
        "set_2021_h1_revised"
    ]
    assert (before["published_at"] <= pd.Timestamp("2021-02-10")).all()
    assert (after["published_at"] <= pd.Timestamp("2021-02-11")).all()


def test_2024_period_boundary_is_exclusive() -> None:
    history = load_constituent_history(HISTORY_PATH)

    january_first = membership_as_of(
        history,
        "2024-01-01",
        index_name="SET50",
    )
    january_second = membership_as_of(
        history,
        "2024-01-02",
        index_name="SET50",
    )

    assert january_first["record_id"].unique().tolist() == ["set_2023_h2"]
    assert january_second["record_id"].unique().tolist() == ["set_2024_h1"]
    assert len(january_first) == 50
    assert len(january_second) == 50


def test_union_is_acquisition_only_superset() -> None:
    history = load_constituent_history(HISTORY_PATH)
    union = acquisition_universe(history)
    active = membership_as_of(history, "2024-07-01", index_name="SET100")

    assert len(union) == 138
    assert len(active) == 100
    assert set(active["ticker"]).issubset(union)


def test_validator_rejects_interval_gap() -> None:
    history = pd.read_csv(HISTORY_PATH)
    history.loc[
        history["record_id"].eq("set_2021_h1_revised"),
        "effective_from",
    ] = "2021-02-12"

    with pytest.raises(ValueError, match="interval gap"):
        validate_constituent_history(history)


def test_validator_rejects_wrong_set50_count() -> None:
    history = pd.read_csv(HISTORY_PATH)
    row = history.index[
        history["record_id"].eq("set_2024_h2") & history["in_set50"]
    ][0]
    history.loc[row, "in_set50"] = False

    with pytest.raises(ValueError, match="does not contain 50 SET50"):
        validate_constituent_history(history)
