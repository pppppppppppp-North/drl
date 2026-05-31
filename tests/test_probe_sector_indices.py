from __future__ import annotations

import pandas as pd

from src.data.probe_sector_indices import _date_range, default_candidates


def test_default_candidates_include_set_sector_patterns() -> None:
    candidates = default_candidates()

    assert candidates["energ"][0] == "^SETENERG.BK"
    assert "ENERG.BK" in candidates["energ"]
    assert candidates["bank"][0] == "^SETBANK.BK"


def test_date_range_handles_empty_frame() -> None:
    assert _date_range(pd.DataFrame()) == (None, None)


def test_date_range_formats_bounds() -> None:
    frame = pd.DataFrame({"date": ["2024-01-03", "2024-01-02"]})

    assert _date_range(frame) == ("2024-01-02", "2024-01-03")
