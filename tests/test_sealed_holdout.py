"""The sealed-holdout loader: a seal written by the download script verifies, and any change is caught.

Every holdout here is synthetic, written by the same function that sealed the real one.
"""
from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from scripts.download_extended_history import _write_holdout
from src.data.sealed_holdout import SealError, check_holdout_lock, open_sealed, verify_seal

APPROVED = {"stage": "holdout_2025_2026_point_in_time", "status": "APPROVED", "approved_by": ["A", "B"],
            "approved_on": "2026-12-01"}


def _seal(tmp_path):
    dates = pd.bdate_range("2025-01-02", periods=5)
    stocks = pd.DataFrame({"date": list(dates) * 2, "ticker": ["AAA.BK"] * 5 + ["BBB.BK"] * 5,
                           "close": [float(i) for i in range(10)]})
    index = pd.DataFrame({"date": dates, "close": [100.0, 101.0, 99.0, 102.0, 103.0]})
    folder = tmp_path / "sealed"
    _write_holdout({"stocks": stocks, "index": index}, folder, {"source": "synthetic"})
    return folder


def test_a_fresh_seal_verifies_and_opens_with_an_approved_lock(tmp_path):
    folder = _seal(tmp_path)
    manifest = verify_seal(folder)
    assert manifest["status"] == "SEALED" and set(manifest["files"]) == {"stocks", "index"}
    frames = open_sealed(folder, APPROVED)
    assert len(frames["stocks"]) == 10 and frames["index"]["date"].max() == pd.Timestamp("2025-01-08")


def test_the_folder_can_move(tmp_path):
    folder = _seal(tmp_path)
    moved = tmp_path / "elsewhere"
    folder.rename(moved)
    assert verify_seal(moved)["status"] == "SEALED"


def test_a_changed_data_file_breaks_the_seal(tmp_path):
    folder = _seal(tmp_path)
    with gzip.open(folder / "index.csv.gz", "at") as handle:
        handle.write("2025-01-09,104.0\n")
    with pytest.raises(SealError, match="index.csv.gz .index. changed"):
        verify_seal(folder)


def test_a_changed_manifest_breaks_the_seal(tmp_path):
    folder = _seal(tmp_path)
    manifest = json.loads((folder / "HOLDOUT_MANIFEST.json").read_text())
    manifest["status"] = "OPEN"
    (folder / "HOLDOUT_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(SealError, match="does not match its recorded hash"):
        verify_seal(folder)


def test_a_missing_file_breaks_the_seal(tmp_path):
    folder = _seal(tmp_path)
    (folder / "stocks.csv.gz").unlink()
    with pytest.raises(SealError, match="stocks.csv.gz .stocks. is missing"):
        verify_seal(folder)


@pytest.mark.parametrize("change, message", [
    ({"status": "DRAFT"}, "not 'APPROVED'"),
    ({"approved_by": []}, "lists no one"),
    ({"approved_on": None}, "approved_on is empty"),
    ({"stage": "test_2023_2024_point_in_time"}, "not a holdout stage"),
])
def test_opening_needs_an_approved_holdout_lock(tmp_path, change, message):
    folder = _seal(tmp_path)
    lock = {**APPROVED, **change}
    assert any(message in p for p in check_holdout_lock(lock))
    with pytest.raises(SealError, match="may not be opened"):
        open_sealed(folder, lock)
