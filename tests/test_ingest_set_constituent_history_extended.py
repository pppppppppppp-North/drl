from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("xlrd")

from src.data import ingest_set_constituent_history as july
from src.data import ingest_set_constituent_history_extended as ext


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / ext.DEFAULT_MANIFEST
ORIGINALS = PROJECT_ROOT / "data/external/original/set_constituent_history"
JULY_PARSEABLE_PDFS = (
    "SET50_100_H1_2019.pdf",
    "SET50_100_H2_2019.pdf",
    "SET50_100_H2_2019_revised.pdf",
    "SET50_100_H1_2020-27082646.pdf",
    "SET50_100_H1_2020_revised-27082541.pdf",
    "SET50_100_H2_2020-27082452.pdf",
    "SET50_100_H2_2020_revised-27082322.pdf",
)


@pytest.fixture(scope="module")
def built():
    return ext.build_extended_archive(PROJECT_ROOT, MANIFEST)


def _pdf(name: str) -> ext.ConstituentList:
    return ext.parse_constituent_source(ORIGINALS / name, source_format="pdf_two_column")


# ------------------------------------------------------------------------------------ archive


def test_builds_continuous_checksum_bound_2012_2024_archive(built) -> None:
    membership, universe, profile, symbol_changes, receipt = built

    assert len(membership) == 3400
    assert membership["record_id"].nunique() == 34
    assert membership["effective_from"].min() == "2012-01-01"
    assert membership["effective_to_exclusive"].max() == "2025-01-01"
    assert len(profile) == 24
    assert profile["set50_count"].eq(50).all() and profile["set100_count"].eq(100).all()
    assert len(universe) == membership["ticker"].nunique() == 227
    assert len(symbol_changes) == len(ext.CORPORATE_ACTIONS)
    assert all(receipt["quality_gates"].values())
    checks = receipt["printed_change_checks"]
    assert checks["checked"] == checks["matched"] == 93
    assert checks["records_without_printed_changes"] == ["set_2012_h1", "set_2012_h2"]
    assert receipt["superseded_before_effective"][0]["record_id"] == "set_2016_h1_original"
    assert receipt["inferred_ranks"] == {"set_2015_h1_revised": ["SET100 70 SAMART"]}


def test_combined_file_passes_the_july_validator(built) -> None:
    history = ext.load_membership_2012_2024(PROJECT_ROOT / ext.DEFAULT_MEMBERSHIP_OUTPUT)

    assert history["record_id"].nunique() == 34
    intervals = history[["record_id", "effective_from", "published_at"]].drop_duplicates()
    assert (intervals["published_at"] <= intervals["effective_from"]).all()


def test_july_rows_are_appended_byte_for_byte(built) -> None:
    payload = (PROJECT_ROOT / ext.BASE_ARCHIVE_PATH).read_bytes()

    assert july.sha256_bytes(payload) == ext.BASE_ARCHIVE_SHA256
    assert july._csv_bytes(built[0]).endswith(payload.split(b"\n", 1)[1])


def test_committed_outputs_are_reproduced_and_bound_by_the_receipt(built) -> None:
    membership, universe, profile, symbol_changes, _ = built
    for frame, relative in (
        (membership, ext.DEFAULT_MEMBERSHIP_OUTPUT),
        (universe, ext.DEFAULT_UNIVERSE_OUTPUT),
        (profile, ext.DEFAULT_PROFILE_OUTPUT),
        (symbol_changes, ext.DEFAULT_SYMBOL_CHANGES_OUTPUT),
    ):
        assert july._csv_bytes(frame) == (PROJECT_ROOT / relative).read_bytes(), relative

    receipt_path = PROJECT_ROOT / (ext.DEFAULT_MEMBERSHIP_OUTPUT + ".receipt.json")
    receipt = json.loads(receipt_path.read_text())
    for output in receipt["outputs"].values():
        assert july.sha256_file(PROJECT_ROOT / output["path"]) == output["sha256"]
    sidecar = receipt_path.with_name(receipt_path.name + ".sha256").read_text().split()[0]
    assert sidecar == july.sha256_file(receipt_path)


def test_as_of_membership_follows_revisions_and_the_2016_merger() -> None:
    history = ext.load_membership_2012_2024(PROJECT_ROOT / ext.DEFAULT_MEMBERSHIP_OUTPUT)

    before = set(ext.membership_as_of_2012_2024(history, "2015-04-28", index_name="SET50")["symbol"])
    after = set(ext.membership_as_of_2012_2024(history, "2015-04-29", index_name="SET50")["symbol"])
    assert after - before == {"THCOM"} and before - after == {"HEMRAJ"}

    # The merger list was published on 2015-12-30, before the period began, so the original
    # 2016 H1 list (BECL and BMCL, without SGP) never applies.
    merged = set(ext.membership_as_of_2012_2024(history, "2016-01-04")["symbol"])
    assert {"BEM", "SGP"} <= merged and not {"BECL", "BMCL"} & merged
    assert "set_2016_h1_original" not in set(history["record_id"])

    seam = ext.membership_as_of_2012_2024(history, "2020-12-30")
    assert seam["record_id"].unique().tolist() == ["set_2020_h2_revised"]
    assert ext.membership_as_of_2012_2024(history, "2021-01-04")["record_id"].iloc[0] == "set_2021_h1_original"


# ------------------------------------------------------------------------------------ parsers


@pytest.mark.parametrize("name", JULY_PARSEABLE_PDFS)
def test_pdf_parser_agrees_with_the_july_parser(name: str) -> None:
    old50, old100, _ = july.parse_set_constituent_pdf(ORIGINALS / name)
    new = _pdf(name)

    for old, records in ((old50, new.set50), (old100, new.set100)):
        assert {rank: (r["symbol"], r["sector"]) for rank, r in old.items()} == {
            rank: (r["symbol"], r["sector"]) for rank, r in records.items()
        }


def test_wrapped_sector_labels_are_read_and_reserves_kept_apart() -> None:
    parsed = _pdf("SET50_100_H1_2018.pdf")

    assert parsed.set50[1] == {
        "symbol": "ADVANC",
        "company_name": "ADVANCED INFO SERVICE PCL",
        "sector": "Information & Communication Technology",
    }
    # The reserve list restarts at rank 1 below rank 50; it must not fill any main rank.
    assert parsed.reserves["SET50"] == ["BCPG", "CK", "ESSO", "SPALI", "TPIPL"]
    assert not set(parsed.reserves["SET50"]) & {r["symbol"] for r in parsed.set50.values()}


def test_truncated_sector_cells_are_matched_by_unique_prefix() -> None:
    parsed = _pdf("SET50_100_H1_2015.pdf")

    assert "ADVANC" in parsed.prefix_matched_sectors
    assert parsed.set50[1]["sector"] == "Information & Communication Technology"


def test_rows_drawn_twice_are_read_once() -> None:
    parsed = _pdf("SET50_100_H1_2015.pdf")

    assert parsed.set50[20] == {
        "symbol": "EGCO",
        "company_name": "ELECTRICITY GENERATING PUBLIC COMPANY LIMITED",
        "sector": "Energy & Utilities",
    }


def test_row_without_a_rank_number_is_placed_between_its_neighbours() -> None:
    parsed = _pdf("SET50_100_H1_2015-Update-HEMRAJ-Out.pdf")

    assert parsed.inferred_ranks == ["SET100 70 SAMART"]
    assert [parsed.set100[rank]["symbol"] for rank in (69, 70, 71)] == ["ROBINS", "SAMART", "SAWAD"]


def test_disclaimer_text_does_not_leak_into_company_names() -> None:
    parsed = _pdf("SET50_100_H1_2019.pdf")
    names = {r["symbol"]: r["company_name"] for r in parsed.set100.values()}

    assert names["GUNKUL"] == "GUNKUL ENGINEERING PCL"
    assert names["TU"] == "THAI UNION GROUP PCL"


def test_2014_h1_workbook_keeps_set50_reserves_out_of_set50() -> None:
    parsed = ext.parse_constituent_source(
        ORIGINALS / "SET50_100_H1_2014-TH_v2.xls", source_format="xls_th_set100_marker"
    )
    set50 = {r["symbol"] for r in parsed.set50.values()}

    assert len(set50) == 50
    assert set(parsed.reserves["SET50"]) == {"BLAND", "HEMRAJ", "KKP", "STEC", "WHA"}
    assert not set50 & set(parsed.reserves["SET50"])
    assert parsed.published_on == dt.date(2013, 12, 16)


def test_thai_sector_names_match_the_official_summary_sheets() -> None:
    xlrd = pytest.importorskip("xlrd")
    from_summaries: dict[str, str] = {}
    for name in (
        "SET50_100_H1_2012.xls",
        "SET50_100_H2_2012-TH.xls",
        "SET50_100_H1_2013-TH-v1.xls",
        "SET50_100_H2_2013-TH-v3.xls",
    ):
        book = xlrd.open_workbook(str(ORIGINALS / name))
        for sheet_name in ("Summary - SET50", "Summary - SET100"):
            sheet = book.sheet_by_name(sheet_name)
            for index in range(sheet.nrows):
                cells = [
                    (column, str(value).strip())
                    for column, value in enumerate(sheet.row_values(index))
                    if str(value).strip()
                ]
                for (left, english), (right, thai) in zip(cells, cells[1:]):
                    if right == left + 1 and english.isascii() and any("฀" <= c <= "๿" for c in thai):
                        matched = ext.match_sector(english, allow_prefix=False)
                        if matched is not None:  # industry groups are not sectors
                            from_summaries[thai] = matched[0]

    added_in_2014 = {"บริการรับเหมาก่อสร้าง"}
    assert {k: v for k, v in ext.THAI_SECTOR_TO_ENGLISH.items() if k not in added_in_2014} == from_summaries


def test_construction_services_mapping_agrees_with_later_english_labels(built) -> None:
    membership = built[0]
    thai = membership[(membership["record_id"] == "set_2014_h2") & (membership["sector"] == "Construction Services")]
    english = membership[membership["record_id"] == "set_2015_h1_original"].set_index("symbol")["sector"]
    common = sorted(set(thai["symbol"]) & set(english.index))

    assert len(common) >= 3
    assert all(english[symbol] == "Construction Services" for symbol in common)


def test_dates_in_english_and_thai_headers() -> None:
    assert ext._period_from_text("for calculating the index during Jan 1 - Jun 30, 2020") == (
        dt.date(2020, 1, 1),
        dt.date(2020, 6, 30),
    )
    assert ext._period_from_text("during July 1, 2016 - December 31, 2016") == (
        dt.date(2016, 7, 1),
        dt.date(2016, 12, 31),
    )
    assert ext._period_from_text("สำหรับการคำนวณค่าดัชนีระหว่าง 1 ม.ค. 2555 ถึง 30 มิ.ย. 2555") == (
        dt.date(2012, 1, 1),
        dt.date(2012, 6, 30),
    )
    assert ext._published_from_text("(ประกาศ ณ วันที่ 16 มิถุนายน 2557)") == dt.date(2014, 6, 16)
    assert ext._published_from_text("(Published on Dec 13, 2017)") == dt.date(2017, 12, 13)


def test_sector_prefix_matching_is_word_aligned_and_unique() -> None:
    assert ext.match_sector("Information & Communication", allow_prefix=True) == (
        "Information & Communication Technology",
        True,
    )
    assert ext.match_sector("Information & Communication", allow_prefix=False) is None
    assert ext.match_sector("Construction", allow_prefix=True) is None
    assert ext.match_sector("Informat", allow_prefix=True) is None
    assert ext.match_sector("Food and Beverage", allow_prefix=False) == ("Food & Beverage", False)


# ------------------------------------------------------------------------------------ checks


def _list(set50: set[str], set100: set[str], **kwargs) -> ext.ConstituentList:
    def records(symbols: set[str]) -> dict[int, dict[str, str]]:
        return {
            rank: {"symbol": symbol, "company_name": symbol, "sector": "Banking"}
            for rank, symbol in enumerate(sorted(symbols), start=1)
        }

    return ext.ConstituentList(
        set50=records(set50),
        set100=records(set100),
        period_start=dt.date(2020, 1, 1),
        period_end=dt.date(2020, 6, 30),
        published_on=None,
        unit_count=2,
        **kwargs,
    )


BASELINE = {"SET50": {"A", "B"}, "SET100": {"A", "B", "C"}}


def test_printed_notes_must_explain_a_scheduled_change() -> None:
    notes = {"inclusion": {"D"}, "exclusion": {"B"}}
    good = _list({"A", "D"}, {"A", "C", "D"}, printed_changes={"SET50": notes, "SET100": notes})
    assert all(check["match"] for check in ext._check_printed_changes("r", good, BASELINE, revision=False))

    wrong = _list({"A", "D"}, {"A", "C", "D"}, printed_changes={"SET50": {"inclusion": {"E"}}})
    with pytest.raises(ValueError, match="SET50 inclusion"):
        ext._check_printed_changes("r", wrong, BASELINE, revision=False)


def test_a_revision_must_name_what_it_removes_and_take_known_replacements() -> None:
    predecessor = _list({"A", "B"}, {"A", "B", "C"}, reserves={"SET50": ["D"], "SET100": ["D"]})
    named = _list({"A", "D"}, {"A", "C", "D"}, replacement_notes={"included": set(), "excluded": {"B"}})
    assert ext._check_printed_changes("r", named, BASELINE, revision=True, predecessor=predecessor)

    unnamed = _list({"A", "D"}, {"A", "C", "D"})
    with pytest.raises(ValueError, match="named_exclusions"):
        ext._check_printed_changes("r", unnamed, BASELINE, revision=True, predecessor=predecessor)

    stranger = _list({"A", "E"}, {"A", "C", "E"}, replacement_notes={"included": set(), "excluded": {"B"}})
    with pytest.raises(ValueError, match="inclusions_named_starred_or_reserve"):
        ext._check_printed_changes("r", stranger, BASELINE, revision=True, predecessor=predecessor)


def test_symbol_changes_are_not_counted_as_index_changes(monkeypatch) -> None:
    monkeypatch.setattr(
        ext,
        "CORPORATE_ACTIONS",
        ({"old": "B", "new": "B2", "kind": "symbol_change", "first_record_id": "r", "note": ""},),
    )
    none = {"inclusion": set(), "exclusion": set()}
    renamed = _list({"A", "B2"}, {"A", "B2", "C"}, printed_changes={"SET50": none, "SET100": none})

    assert all(check["match"] for check in ext._check_printed_changes("r", renamed, BASELINE, revision=False))


# ------------------------------------------------------------------------------------ manifest


def _load(frame: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    return ext.load_extended_manifest(path)


def test_manifest_rejects_gaps_early_revisions_and_orphaned_replacements(tmp_path) -> None:
    frame = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    assert len(_load(frame, tmp_path)) == 25

    gap = frame.copy()
    gap.loc[gap["record_id"] == "set_2012_h1", "effective_to_exclusive"] = "2012-06-30"
    with pytest.raises(ValueError, match="gap"):
        _load(gap, tmp_path)

    early = frame.copy()
    early.loc[early["record_id"] == "set_2019_h1_original", "effective_to_exclusive"] = "2019-05-27"
    early.loc[early["record_id"] == "set_2019_h1_revised", "effective_from"] = "2019-05-27"
    with pytest.raises(ValueError, match="before it was published"):
        _load(early, tmp_path)

    orphan = frame[frame["record_id"] != "set_2016_h1_revised"]
    with pytest.raises(ValueError, match="no replacement"):
        _load(orphan, tmp_path)


def test_a_changed_july_archive_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ext, "BASE_ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="byte-identical"):
        ext._load_base_archive(PROJECT_ROOT, ext.BASE_ARCHIVE_PATH)
