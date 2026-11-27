"""Extend the point-in-time SET50/SET100 constituent archive back to 2012 (milestone M3).

The July importer (``src/data/ingest_set_constituent_history.py``) covers 2021-2024. It and its
outputs stay unchanged; this module only adds to them:

* a parser for the 2015-2020 two-column PDFs. In these files sector labels and company names wrap
  onto a second line, ``*`` replacement markers can be glued to the rank or the symbol, and a
  reserve list whose ranks restart at 1 follows the main list in the right-hand column;
* parsers for the 2012-2014 Excel workbooks (English sheets for 2012-2013, Thai sheets for 2014,
  whose Thai sector names are mapped with the SET's own English/Thai summary tables);
* a manifest for 2012-2020 that publication-gates between-review revisions and records an
  original that was replaced before its period began (2016 H1);
* a builder that appends the checksum-pinned 2021-2024 archive unchanged and writes one
  2012-2024 membership file with a receipt.

Every source must match its SHA-256 and size. Every list must have exactly 50/100 unique members
with SET50 inside SET100. Where a source prints its changes (Inclusion/Exclusion notes, "New Entry"
marks, replacement notes), the change from the previous list must match them exactly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pdfplumber

from src.data.constituent_history import validate_constituent_history
from src.data.ingest_set_constituent_history import (
    MEMBERSHIP_COLUMNS,
    REQUIRED_MANIFEST_COLUMNS,
    SECTORS,
    SYMBOL_PATTERN,
    _atomic_bytes,
    _atomic_json,
    _canonical_text,
    _csv_bytes,
    _group_words_by_line,
    _validate_source_file,
    sha256_bytes,
    sha256_file,
)

EXTENDED_START = "2012-01-01"
BASE_ARCHIVE_START = "2021-01-01"
BASE_ARCHIVE_END = "2025-01-01"
BASE_ARCHIVE_PATH = "data/reference/set50_set100_membership_2021_2024.csv"
# SHA-256 of the July 2021-2024 archive, as bound by reports/data_sources/set_constituent_archive_20260731.json.
BASE_ARCHIVE_SHA256 = "96ea9924830e94ce702f8cd2062dd66670e262a5627774fab5e3b27a4a02c074"

DEFAULT_MANIFEST = "data/external/set_constituent_source_manifest_2012_2020.csv"
DEFAULT_MEMBERSHIP_OUTPUT = "data/reference/set50_set100_membership_2012_2024.csv"
DEFAULT_UNIVERSE_OUTPUT = "data/reference/set50_or_set100_universe_2012_2024.csv"
DEFAULT_PROFILE_OUTPUT = "data/reference/set_constituent_interval_profile_2012_2020.csv"
DEFAULT_SYMBOL_CHANGES_OUTPUT = "data/reference/set_symbol_changes_2012_2020.csv"

EXTRA_MANIFEST_COLUMNS = ("source_format", "published_at_source")
MANIFEST_COLUMNS = REQUIRED_MANIFEST_COLUMNS + EXTRA_MANIFEST_COLUMNS
SOURCE_FORMATS = (
    "xls_en_sheets",  # 2012-2013: SET50-EN / SET100-EN sheets
    "xls_th_set100_marker",  # 2014 H1: one SET100-TH sheet with a SET50/SET100 column
    "xls_th_sheets",  # 2014 H2: SET50-TH / SET100-TH sheets
    "pdf_two_column",  # 2015-2020: two-page, two-column PDF
)
REVISION_STATUSES = ("original", "revised", "superseded_before_effective")
PUBLISHED_AT_SOURCES = ("source_header", "not_stated_period_start")

# Sectors of the older SET classification that no SET50/SET100 member of 2021-2024 carried.
LEGACY_SECTORS = ("Mining", "Paper & Printing Materials")
ALL_SECTORS = SECTORS + LEGACY_SECTORS

# Thai sector names and their English names, transcribed from the "Summary - SET50/SET100" sheets of
# the official 2012-2013 workbooks (tests re-derive this table from the archived files).
THAI_SECTOR_TO_ENGLISH = {
    "ธุรกิจการเกษตร": "Agribusiness",
    "อาหารและเครื่องดื่ม": "Food & Beverage",
    "แฟชั่น": "Fashion",
    "ของใช้ในครัวเรือนและสำนักงาน": "Home & Office Products",
    "ของใช้ส่วนตัวและเวชภัณฑ์": "Personal Products & Pharmaceuticals",
    "ธนาคาร": "Banking",
    "เงินทุนและหลักทรัพย์": "Finance & Securities",
    "ประกันภัยและประกันชีวิต": "Insurance",
    "ยานยนต์": "Automotive",
    "วัสดุอุตสาหกรรมและเครื่องจักร": "Industrial Materials & Machinery",
    "บรรจุภัณฑ์": "Packaging",
    "กระดาษและวัสดุการพิมพ์": "Paper & Printing Materials",
    "ปิโตรเคมีและเคมีภัณฑ์": "Petrochemicals & Chemicals",
    "เหล็ก": "Steel",
    "วัสดุก่อสร้าง": "Construction Materials",
    "พัฒนาอสังหาริมทรัพย์": "Property Development",
    "พลังงานและสาธารณูปโภค": "Energy & Utilities",
    "เหมืองแร่": "Mining",
    "พาณิชย์": "Commerce",
    "การแพทย์": "Health Care Services",
    "สื่อและสิ่งพิมพ์": "Media & Publishing",
    "บริการเฉพาะกิจ": "Professional Services",
    "การท่องเที่ยวและสันทนาการ": "Tourism & Leisure",
    "ขนส่งและโลจิสติกส์": "Transportation & Logistics",
    "ชิ้นส่วนอิเล็กทรอนิกส์": "Electronic Components",
    "เทคโนโลยีสารสนเทศและการสื่อสาร": "Information & Communication Technology",
    # Not in the 2012-2013 summaries: the sector was added in 2014. Tests check it against the
    # English labels of the same companies in the 2015 H1 PDF.
    "บริการรับเหมาก่อสร้าง": "Construction Services",
}
# Industry groups (one level above sectors). The 2014 workbooks put them in one of two columns.
THAI_INDUSTRY_GROUPS = {
    "เกษตรและอุตสาหกรรมอาหาร",
    "สินค้าอุปโภคบริโภค",
    "ธุรกิจการเงิน",
    "วัตถุดิบและสินค้าอุตสาหกรรม",
    "สินค้าอุตสาหกรรม",
    "อสังหาริมทรัพย์และก่อสร้าง",
    "ทรัพยากร",
    "บริการ",
    "เทคโนโลยี",
}

ENGLISH_MONTHS = {
    name: number
    for number, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
THAI_MONTHS = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4, "พฤษภาคม": 5, "มิถุนายน": 6,
    "กรกฎาคม": 7, "สิงหาคม": 8, "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
    "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4, "พ.ค.": 5, "มิ.ย.": 6,
    "ก.ค.": 7, "ส.ค.": 8, "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
}
_EN_DATE = r"([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s*(\d{4})"
_TH_DATE = (
    r"(\d{1,2})\s*("
    + "|".join(re.escape(name) for name in sorted(THAI_MONTHS, key=len, reverse=True))
    + r")\s*(\d{4})"
)
PERIOD_EN = re.compile(
    r"during\s+([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:,?\s*(\d{4}))?\s*[-–]\s*" + _EN_DATE,
    re.IGNORECASE,
)
PUBLISHED_EN = re.compile(r"Published\s+on\s+" + _EN_DATE, re.IGNORECASE)
PERIOD_TH = re.compile(r"ระหว่าง\s*" + _TH_DATE + r"\s*(?:-|–|ถึง)\s*" + _TH_DATE)
PUBLISHED_TH = re.compile(r"ประกาศ\s*ณ\s*วันที่\s*" + _TH_DATE)

# Symbol changes and mergers that explain list differences the printed notes do not mention.
# "first_record_id" is the first list showing the new symbol. Every entry is checked against the
# lists (old symbol in the earlier list, new symbol in the later one) and the receipt records both
# company names as evidence.
CORPORATE_ACTIONS = (
    {"old": "KH", "new": "BCH", "kind": "symbol_change", "first_record_id": "set_2013_h1",
     "note": "Bangkok Chain Hospital changed its symbol"},
    {"old": "GSTEEL", "new": "GSTEL", "kind": "symbol_change", "first_record_id": "set_2013_h2",
     "note": "G Steel changed its symbol"},
    {"old": "KK", "new": "KKP", "kind": "symbol_change", "first_record_id": "set_2014_h1",
     "note": "Kiatnakin Bank changed its symbol; the 2014 H1 notes already use KKP"},
    {"old": "BGH", "new": "BDMS", "kind": "symbol_change", "first_record_id": "set_2015_h2",
     "note": "Bangkok Dusit Medical Services changed its symbol"},
    {"old": "TUF", "new": "TU", "kind": "symbol_change", "first_record_id": "set_2016_h1_original",
     "note": "Thai Union Frozen Products became Thai Union Group"},
    {"old": "BECL", "new": "BEM", "kind": "merger", "first_record_id": "set_2016_h1_revised",
     "note": "Bangkok Expressway and Bangkok Metro amalgamated into Bangkok Expressway and Metro"},
    {"old": "BMCL", "new": "BEM", "kind": "merger", "first_record_id": "set_2016_h1_revised",
     "note": "Bangkok Expressway and Bangkok Metro amalgamated into Bangkok Expressway and Metro"},
    {"old": "MTLS", "new": "MTC", "kind": "symbol_change", "first_record_id": "set_2018_h2",
     "note": "Muangthai Leasing became Muangthai Capital"},
)

RANK_WORD = re.compile(r"(\d{1,3})(\*?)(\*?[A-Z][A-Z0-9&.\-]*\*?)?")
REPLACEMENT_FREE_FLOAT = re.compile(r"due to the fact that\s+([A-Z0-9&.\-]+)\s+has", re.IGNORECASE)
REPLACEMENT_RESTRUCTURE = re.compile(r"\b([A-Z0-9&.\-]{2,12})\s+replaces?\s+([A-Z0-9&.\-]{1,12})\b")
INDEX_NAMES = ("SET50", "SET100")


@dataclass
class ConstituentList:
    """One parsed source file: the two ranked lists plus what the file says about itself."""

    set50: dict[int, dict[str, str]]
    set100: dict[int, dict[str, str]]
    period_start: dt.date
    period_end: dt.date
    published_on: dt.date | None
    unit_count: int
    reserves: dict[str, list[str]] = field(default_factory=dict)
    # Printed change notes, e.g. {"SET50": {"inclusion": {...}, "exclusion": {...}}}.
    printed_changes: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    # Symbols flagged "*" or "New Entry" in each list.
    marked: dict[str, set[str]] = field(default_factory=dict)
    # Replacement notes of between-review revisions: {"included": {...}, "excluded": {...}}.
    replacement_notes: dict[str, set[str]] = field(default_factory=dict)
    # Symbols whose sector cell is cut short in the PDF and was matched by a unique prefix.
    prefix_matched_sectors: list[str] = field(default_factory=list)
    # Rows whose rank number is missing from the PDF's text layer, placed between their neighbours.
    inferred_ranks: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------- text helpers


def _english_date(month: str, day: str, year: str) -> dt.date:
    key = month[:3].lower()
    if key not in ENGLISH_MONTHS:
        raise ValueError(f"unrecognized month name: {month!r}")
    return dt.date(int(year), ENGLISH_MONTHS[key], int(day))


def _thai_date(day: str, month: str, buddhist_year: str) -> dt.date:
    return dt.date(int(buddhist_year) - 543, THAI_MONTHS[month], int(day))


def _period_from_text(text: str) -> tuple[dt.date, dt.date]:
    match = PERIOD_EN.search(text)
    if match:
        month, day, year, *end_parts = match.groups()
        end = _english_date(*end_parts)
        if year is None:  # "Jan 1 - Jun 30, 2020" prints the year once
            year = str(end.year if ENGLISH_MONTHS[month[:3].lower()] <= end.month else end.year - 1)
        return _english_date(month, day, year), end
    match = PERIOD_TH.search(text)
    if match:
        groups = match.groups()
        return _thai_date(*groups[:3]), _thai_date(*groups[3:])
    raise ValueError("could not find the index calculation period in the source header")


def _published_from_text(text: str) -> dt.date | None:
    match = PUBLISHED_EN.search(text)
    if match:
        return _english_date(*match.groups())
    match = PUBLISHED_TH.search(text)
    if match:
        return _thai_date(*match.groups())
    return None


def _sector_tokens(text: str) -> list[str]:
    return _canonical_text(text).casefold().split()


def match_sector(text: str, *, allow_prefix: bool) -> tuple[str, bool] | None:
    """Return (sector, matched_by_prefix) for a sector cell, or None.

    A prefix match is accepted only when the cell is at least two words long, stops at a word
    boundary, and is the start of exactly one known sector (e.g. "Information & Communication").
    """
    tokens = _sector_tokens(text)
    if not tokens:
        return None
    exact = [sector for sector in ALL_SECTORS if _sector_tokens(sector) == tokens]
    if len(exact) == 1:
        return exact[0], False
    if not allow_prefix or len(tokens) < 2:
        return None
    prefix = [
        sector
        for sector in ALL_SECTORS
        if len(_sector_tokens(sector)) > len(tokens)
        and _sector_tokens(sector)[: len(tokens)] == tokens
    ]
    if len(prefix) == 1:
        return prefix[0], True
    return None


def canonical_sector(value: object) -> str:
    matched = match_sector(str(value), allow_prefix=False)
    if matched is None:
        raise ValueError(f"unrecognized SET sector label: {value!r}")
    return matched[0]


def _split_symbol_list(text: str) -> set[str]:
    """Parse a printed list such as "BCP, BEAUTY, CENTEL, TPIPP, SAWAD & WHA"."""
    symbols: set[str] = set()
    for part in re.split(r"\s*,\s*|\s+&\s+|\s+and\s+", text.strip()):
        part = part.strip().strip(".")
        if part in {"", "-"} or part.lower() in {"none", "n/a"}:
            continue
        token = part.split()[0].strip("*").upper()
        if not SYMBOL_PATTERN.fullmatch(token):
            raise ValueError(f"unparseable symbol {token!r} in printed change list {text!r}")
        symbols.add(token)
    return symbols


def _replacement_notes(text: str) -> dict[str, set[str]]:
    included: set[str] = set()
    excluded = {match.upper() for match in REPLACEMENT_FREE_FLOAT.findall(text)}
    for new_symbol, old_symbol in REPLACEMENT_RESTRUCTURE.findall(text):
        included.add(new_symbol.upper())
        excluded.add(old_symbol.upper())
    if not (included or excluded):
        return {}
    return {"included": included, "excluded": excluded}


# ---------------------------------------------------------------------------- PDF (2015-2020)


def _is_marker(word: dict[str, Any]) -> bool:
    """Footnote markers such as "*" or "#" printed as separate words ("&" is not one)."""
    return re.fullmatch(r"[*#]+", str(word["text"])) is not None


def _x0(word: dict[str, Any]) -> float:
    return float(word["x0"])


def _line_of(lines: list[list[dict[str, Any]]], word: dict[str, Any]) -> int:
    for index, line in enumerate(lines):
        if any(item is word for item in line):
            return index
    raise KeyError("word not found in its band")


def _merge_fragments(
    words: list[dict[str, Any]],
    *,
    max_gap: float,
    line_tolerance: float,
) -> list[dict[str, Any]]:
    """Rejoin pieces of one word that the extractor split.

    After de-duplicating a highlighted row, a word can keep characters from both drawn copies,
    which the extractor then treats as two lines ("Utilit" + "ies"). A real space is about a
    quarter of the font size, so pieces closer than ``max_gap`` on the same line are one word.
    """
    merged: list[dict[str, Any]] = []
    for line in _group_words_by_line(words, tolerance=line_tolerance):
        current: dict[str, Any] | None = None
        for word in sorted(line, key=_x0):
            if (
                current is not None
                and 0.0 <= _x0(word) - float(current["x1"]) < max_gap
                and abs(float(word.get("size", 0.0)) - float(current.get("size", 0.0))) < 0.01
            ):
                current = {
                    **current,
                    "text": str(current["text"]) + str(word["text"]),
                    "x1": max(float(current["x1"]), float(word["x1"])),
                    "top": min(float(current["top"]), float(word["top"])),
                    "bottom": max(float(current["bottom"]), float(word["bottom"])),
                }
                merged[-1] = current
                continue
            current = dict(word)
            merged.append(current)
    return merged


def _sector_start(cells: list[dict[str, Any]]) -> float | None:
    """x position where the rest of the rank line reads as the start of a sector label."""
    for position, start in enumerate(cells):
        tokens = _sector_tokens(" ".join(str(word["text"]) for word in cells[position:]))
        if tokens and any(_sector_tokens(sector)[: len(tokens)] == tokens for sector in ALL_SECTORS):
            return _x0(start)
    return None


def _parse_pdf_row(
    first_word: dict[str, Any],
    match: re.Match[str] | None,
    band: list[dict[str, Any]],
    *,
    pitch: float,
    scale: float,
) -> tuple[int | None, dict[str, str], bool, bool]:
    """Parse one table row; return (rank, record, starred, sector_matched_by_prefix).

    ``first_word`` is the rank number, or the symbol when the PDF's text layer has no rank for
    the row (``match`` is then None and the rank is inferred from the neighbouring rows).
    """
    lines = _group_words_by_line(band, tolerance=2.6 * scale)
    first_index = _line_of(lines, first_word)
    markers = [word for word in lines[first_index] if _is_marker(word)]
    first_line = sorted((word for word in lines[first_index] if not _is_marker(word)), key=_x0)
    rank = int(match.group(1)) if match else None
    label = f"rank {rank}" if rank is not None else "unnumbered row"
    starred = bool(match and match.group(2))

    if match is None:
        symbol_word, symbol_text = first_word, str(first_word["text"])
    elif match.group(3):
        symbol_word, symbol_text = first_word, match.group(3)
    else:
        after = [
            word
            for word in first_line
            if word is not first_word and _x0(word) > float(first_word["x1"]) - 0.1 * scale
        ]
        if not after:
            raise ValueError(f"{label}: no symbol on the rank line")
        symbol_word = after[0]
        symbol_text = str(symbol_word["text"])
        starred = starred or any(
            "*" in str(marker["text"]) and _x0(marker) < _x0(symbol_word) for marker in markers
        )
    starred = starred or "*" in symbol_text
    symbol = symbol_text.strip("*#").upper()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"{label}: invalid symbol {symbol_text!r}")

    cells = [word for word in first_line if _x0(word) > float(symbol_word["x1"]) + 0.5 * scale]
    if not cells:
        raise ValueError(f"{label} {symbol}: nothing after the symbol")
    sector_start = _sector_start(cells)
    aligned_starts = (
        [_x0(cells[0]), sector_start] if sector_start is not None else [_x0(word) for word in cells]
    )
    row_size = statistics.median(float(word.get("size", 0.0)) for word in first_line)

    # Keep the rank line and the wrapped lines under it. A wrapped line starts where the company
    # or the sector cell starts and has the row's font size; anything else (a heading, a footer,
    # a disclaimer overlay, the next row) ends the row.
    row_lines = [first_line]
    previous_top = float(first_line[0]["top"])
    gap_limit = max(10.0 * scale, 0.75 * pitch)
    for line in lines[first_index + 1 :]:
        content = sorted((word for word in line if not _is_marker(word)), key=_x0)
        if not content:
            continue
        line_top = min(float(word["top"]) for word in content)
        size = statistics.median(float(word.get("size", 0.0)) for word in content)
        if (
            line_top - previous_top > gap_limit
            or not any(abs(_x0(content[0]) - start) <= 1.5 * scale for start in aligned_starts)
            or abs(size - row_size) > 0.08 * row_size
        ):
            break
        row_lines.append(content)
        previous_top = line_top
    ordered = [word for line in row_lines for word in line]

    sector: tuple[str, bool] | None = None
    sector_x = 0.0
    for allow_prefix in (False, True):
        for start in cells:
            cell = [str(word["text"]) for word in ordered if _x0(word) >= _x0(start) - 1.0 * scale]
            sector = match_sector(" ".join(cell), allow_prefix=allow_prefix)
            if sector is not None:
                sector_x = _x0(start)
                break
        if sector is not None:
            break
    if sector is None:
        raise ValueError(f"{label} {symbol}: no recognized sector label")

    company = _canonical_text(
        " ".join(
            str(word["text"]).strip("*#")
            for word in ordered
            if word is not first_word
            and word is not symbol_word
            and float(symbol_word["x1"]) + 0.5 * scale < _x0(word) < sector_x - 1.0 * scale
        )
    )
    if not company:
        raise ValueError(f"{label} {symbol}: blank company name")
    return rank, {"symbol": symbol, "company_name": company, "sector": sector[0]}, starred, sector[1]


def _pdf_change_notes(
    words: list[dict[str, Any]],
    boundary: float,
    scale: float,
) -> dict[str, dict[str, set[str]]]:
    """Read "SET50 Inclusion: ..." style notes, keeping each note inside its own column."""
    notes: dict[str, dict[str, set[str]]] = {}
    lines = _group_words_by_line(words, tolerance=2.6 * scale)

    def tokens_in(line: list[dict[str, Any]], low: float, high: float) -> tuple[list[str], bool]:
        tokens: list[str] = []
        for word in line:
            if not low <= _x0(word) < high:
                continue
            if str(word["text"]).upper() in INDEX_NAMES:  # the next note on the same line
                return tokens, True
            tokens.append(str(word["text"]))
        return tokens, False

    for line_index, line in enumerate(lines):
        for position in range(len(line) - 1):
            index_word, label_word = line[position], line[position + 1]
            index_name = str(index_word["text"]).upper()
            label = str(label_word["text"]).lower().rstrip(":")
            if index_name not in INDEX_NAMES or label not in {"inclusion", "exclusion"}:
                continue
            left_column = _x0(index_word) < boundary
            low, high = (0.0, boundary) if left_column else (boundary, float("inf"))
            tokens, stopped = tokens_in(line[position + 2 :], low, high)
            follow = line_index + 1
            while not stopped and tokens and tokens[-1].endswith(",") and follow < len(lines):
                more, stopped = tokens_in(lines[follow], low, high)
                tokens += more
                follow += 1
            notes.setdefault(index_name, {})[label] = _split_symbol_list(" ".join(tokens))
    return notes


def _pdf_page_layout(page: Any, *, index_name: str) -> dict[str, Any]:
    """Words of one constituent page plus the positions of its two side-by-side tables."""
    # Some pages are drawn at four times the usual size; scale every distance by the font size.
    sizes = [float(char["size"]) for char in page.chars if float(char["size"]) > 0]
    if not sizes:
        raise ValueError(f"{index_name}: page has no text")
    median_size = statistics.median(sizes)
    scale = max(1.0, median_size / 6.0)
    # Highlighted rows are drawn twice at a small offset ("fake bold"); drop the second copy.
    page = page.dedupe_chars(tolerance=0.2 * median_size, extra_attrs=("size",))
    text = page.extract_text() or ""
    if not re.search(rf"(?m)^\s*{index_name}\s+INDEX\s+CONSTITUENTS", text):
        raise ValueError(f"page does not carry the {index_name} INDEX CONSTITUENTS heading")
    words = _merge_fragments(
        page.extract_words(
            x_tolerance=0.5 * scale,
            y_tolerance=2 * scale,
            keep_blank_chars=False,
            extra_attrs=["size"],
        ),
        max_gap=0.12 * median_size,
        line_tolerance=2.6 * scale,
    )
    number_headers = [word for word in words if str(word["text"]).lower().rstrip(".") == "no"]
    header: list[dict[str, Any]] = []
    for top in sorted({round(float(word["top"]), 1) for word in number_headers}):
        same = [word for word in number_headers if abs(float(word["top"]) - top) < 1.0 * scale]
        if len(same) == 2:
            header = sorted(same, key=_x0)
            break
    if not header:
        raise ValueError(f"{index_name}: could not locate the two-column table header")
    header_top = float(header[0]["top"])
    symbol_headers = sorted(
        (
            word
            for word in words
            if str(word["text"]).lower() == "symbol"
            and abs(float(word["top"]) - header_top) < 1.5 * scale
        ),
        key=_x0,
    )
    if len(symbol_headers) != 2:
        raise ValueError(f"{index_name}: expected two Symbol column headers")
    # Notes under the right-hand table start a little left of its "No." header.
    boundary = _x0(header[1]) - 4.0 * scale
    return {
        "words": words,
        "text": text,
        "scale": scale,
        "header_top": header_top,
        "boundary": boundary,
        "columns": (
            (0.0, boundary, header[0], symbol_headers[0]),
            (boundary, float(page.width) + 1.0, header[1], symbol_headers[1]),
        ),
    }


def printed_changes_from_two_column_pdf(path: Path) -> dict[str, dict[str, set[str]]]:
    """Only the Inclusion/Exclusion notes of a two-page constituent PDF (used at the 2021 seam)."""
    found: dict[str, dict[str, set[str]]] = {}
    with pdfplumber.open(path) as pdf:
        for page, index_name in zip(pdf.pages, INDEX_NAMES):
            layout = _pdf_page_layout(page, index_name=index_name)
            notes = _pdf_change_notes(layout["words"], layout["boundary"], layout["scale"])
            if notes.get(index_name):
                found[index_name] = notes[index_name]
    return found


def _parse_pdf_page(
    page: Any,
    *,
    index_name: str,
    expected_count: int,
) -> dict[str, Any]:
    layout = _pdf_page_layout(page, index_name=index_name)
    words, text, scale = layout["words"], layout["text"], layout["scale"]
    header_top, boundary, columns = layout["header_top"], layout["boundary"], layout["columns"]

    sequence: list[tuple[int | None, dict[str, str], bool, bool]] = []
    failures: list[str] = []
    for x_low, x_high, number_header, symbol_header in columns:
        column_words = [
            word
            for word in words
            if x_low <= _x0(word) < x_high and float(word["top"]) > header_top + 3.0 * scale
        ]
        contents = [
            sorted((word for word in line if not _is_marker(word)), key=_x0)
            for line in _group_words_by_line(column_words, tolerance=2.6 * scale)
        ]
        contents = [content for content in contents if content]
        numbered = []
        for content in contents:
            match = RANK_WORD.fullmatch(str(content[0]["text"]))
            if match and _x0(number_header) - 6.0 * scale <= _x0(content[0]) < _x0(symbol_header) + 3.0 * scale:
                numbered.append((content[0], match, content))
        starts: list[tuple[dict[str, Any], re.Match[str] | None]] = [
            (first, match) for first, match, _ in numbered
        ]
        # A row whose rank number is missing from the text layer starts in the symbol column.
        symbol_xs = [
            _x0(content[1]) for _, match, content in numbered if not match.group(3) and len(content) > 1
        ]
        if symbol_xs:
            symbol_x = statistics.median(symbol_xs)
            numbered_ids = {id(first) for first, _ in starts}
            for content in contents:
                first = content[0]
                candidate = str(first["text"]).strip("*#")
                if (
                    id(first) not in numbered_ids
                    and abs(_x0(first) - symbol_x) <= 1.5 * scale
                    and SYMBOL_PATTERN.fullmatch(candidate)
                    and candidate.isupper()
                    and len(content) > 1
                ):
                    starts.append((first, None))
        starts.sort(key=lambda item: float(item[0]["top"]))
        gaps = [
            float(after[0]["top"]) - float(before[0]["top"])
            for before, after in zip(starts, starts[1:])
            if before[1] is not None and after[1] is not None
        ]
        pitch = statistics.median(gaps) if gaps else 20.0 * scale
        for position, (first, match) in enumerate(starts):
            top = float(first["top"])
            bottom = (
                float(starts[position + 1][0]["top"]) - 2.5 * scale
                if position + 1 < len(starts)
                else top + 4.0 * pitch
            )
            band = [word for word in column_words if top - 2.5 * scale <= float(word["top"]) < bottom]
            try:
                sequence.append(_parse_pdf_row(first, match, band, pitch=pitch, scale=scale))
            except ValueError as exc:
                if match is not None:  # an unnumbered line that is not a table row is ignored
                    failures.append(str(exc))

    main: dict[int, dict[str, str]] = {}
    reserves: list[str] = []
    marked: set[str] = set()
    prefix_matched: list[str] = []
    inferred: list[str] = []
    expected = 1
    for position, (rank, record, starred, prefix_sector) in enumerate(sequence):
        in_main = expected <= expected_count
        target = expected if in_main else len(reserves) + 1
        if rank is None:
            following = next((item[0] for item in sequence[position + 1 :] if item[0] is not None), None)
            last_main = in_main and target == expected_count
            if not (following == target + 1 or (last_main and following in {None, 1}) or (not in_main and following is None)):
                raise ValueError(f"{index_name}: cannot place unnumbered row {record['symbol']}")
            rank = target
            inferred.append(f"{index_name} {'' if in_main else 'reserve '}{rank} {record['symbol']}")
        if rank != target:
            where = "reserve list" if not in_main else "main list"
            raise ValueError(
                f"{index_name}: {where} expected rank {target}, found {rank} ({record['symbol']}); "
                f"unparsed rows: {failures}"
            )
        if in_main:
            main[rank] = record
            expected += 1
            if starred:
                marked.add(record["symbol"])
            if prefix_sector:
                prefix_matched.append(record["symbol"])
        else:
            reserves.append(record["symbol"])
    if expected <= expected_count:
        raise ValueError(
            f"{index_name}: parsed {expected - 1} of {expected_count} ranks; unparsed rows: {failures}"
        )
    symbols = [main[rank]["symbol"] for rank in sorted(main)]
    return {
        "records": main,
        "reserves": reserves,
        "marked": marked,
        "prefix_matched": prefix_matched,
        "inferred": inferred,
        "alphabetical": symbols == sorted(symbols),
        "notes": _pdf_change_notes(words, boundary, scale),
        "text": text,
    }


def parse_two_column_pdf(
    path: Path,
    *,
    expected_set50_count: int = 50,
    expected_set100_count: int = 100,
) -> ConstituentList:
    try:
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages) != 2:
                raise ValueError(f"two-column SET document must have two pages, found {len(pdf.pages)}")
            page50 = _parse_pdf_page(pdf.pages[0], index_name="SET50", expected_count=expected_set50_count)
            page100 = _parse_pdf_page(pdf.pages[1], index_name="SET100", expected_count=expected_set100_count)
    except ValueError:
        raise
    except Exception as exc:  # pdfplumber/pdfminer errors
        raise ValueError(f"could not read SET constituent PDF {path}: {exc}") from exc
    period_start, period_end = _period_from_text(page50["text"])
    if _period_from_text(page100["text"]) != (period_start, period_end):
        raise ValueError("the SET50 and SET100 pages state different calculation periods")
    published = _published_from_text(page50["text"])
    if published != _published_from_text(page100["text"]):
        raise ValueError("the SET50 and SET100 pages state different publication dates")
    for name, page in (("SET50", page50), ("SET100", page100)):
        if not page["alphabetical"]:
            raise ValueError(f"{name} list is not in the alphabetical order the PDF states")
    printed = {
        "SET50": page50["notes"].get("SET50", {}),
        "SET100": page100["notes"].get("SET100", {}),
    }
    return ConstituentList(
        set50=page50["records"],
        set100=page100["records"],
        period_start=period_start,
        period_end=period_end,
        published_on=published,
        unit_count=2,
        reserves={"SET50": page50["reserves"], "SET100": page100["reserves"]},
        printed_changes={name: value for name, value in printed.items() if value},
        marked={"SET50": page50["marked"], "SET100": page100["marked"]},
        replacement_notes=_replacement_notes(page50["text"] + "\n" + page100["text"]),
        prefix_matched_sectors=sorted(set(page50["prefix_matched"] + page100["prefix_matched"])),
        inferred_ranks=page50["inferred"] + page100["inferred"],
    )


# ---------------------------------------------------------------------------- Excel (2012-2014)


def _open_workbook(path: Path) -> Any:
    try:
        import xlrd  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading the 2012-2014 SET workbooks needs xlrd>=2.0 "
            "(pip install -r requirements-data-extensions.txt)"
        ) from exc
    try:
        return xlrd.open_workbook(str(path))
    except Exception as exc:
        raise ValueError(f"could not read SET constituent workbook {path}: {exc}") from exc


def _sheet_rows(book: Any, name: str) -> list[list[Any]]:
    if name not in book.sheet_names():
        raise ValueError(f"workbook has no sheet named {name!r}")
    sheet = book.sheet_by_name(name)
    return [sheet.row_values(index) for index in range(sheet.nrows)]


def _cell_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _row_text(row: list[Any]) -> str:
    return " | ".join(_cell_text(cell) for cell in row if _cell_text(cell))


def _is_rank(value: Any) -> bool:
    return isinstance(value, float) and value == int(value) and value >= 1


def _split_main_and_reserve(
    rows: list[tuple[int, dict[str, str], bool]],
    *,
    index_name: str,
    expected_count: int,
    reserve_start: int | None,
    row_numbers: list[int],
) -> tuple[dict[int, dict[str, str]], list[str], set[str]]:
    main: dict[int, dict[str, str]] = {}
    reserves: list[str] = []
    marked: set[str] = set()
    for (rank, record, flagged), row_number in zip(rows, row_numbers):
        in_reserve = reserve_start is not None and row_number > reserve_start
        if in_reserve:
            if rank != len(reserves) + 1:
                raise ValueError(f"{index_name}: reserve list out of order at rank {rank}")
            reserves.append(record["symbol"])
            continue
        if rank != len(main) + 1:
            raise ValueError(f"{index_name}: expected rank {len(main) + 1}, found {rank}")
        main[rank] = record
        if flagged:
            marked.add(record["symbol"])
    if len(main) != expected_count:
        raise ValueError(f"{index_name}: found {len(main)} members, expected {expected_count}")
    return main, reserves, marked


def _parse_en_sheet(
    rows: list[list[Any]],
    *,
    index_name: str,
    expected_count: int,
) -> tuple[dict[int, dict[str, str]], list[str], set[str], set[str]]:
    header = next(
        (index for index, row in enumerate(rows) if _cell_text(row[0]).lower().rstrip(".") == "no"),
        None,
    )
    if header is None:
        raise ValueError(f"{index_name}-EN: no header row")
    labels = [_cell_text(cell).lower() for cell in rows[header]]
    if labels[1] != "sector" or not labels[2].startswith("securities"):
        raise ValueError(f"{index_name}-EN: unexpected header {rows[header]!r}")
    reserve_start = next(
        (
            index
            for index, row in enumerate(rows)
            if _cell_text(row[0]).lower().startswith("replacement list")
        ),
        None,
    )
    parsed: list[tuple[int, dict[str, str], bool]] = []
    row_numbers: list[int] = []
    for index, row in enumerate(rows[header + 1 :], start=header + 1):
        if not _is_rank(row[0]):
            continue
        symbol_text = _cell_text(row[3])
        note = _cell_text(row[4]) if len(row) > 4 else ""
        symbol = symbol_text.strip("* ").upper()
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise ValueError(f"{index_name}-EN row {index}: invalid symbol {symbol_text!r}")
        record = {
            "symbol": symbol,
            "company_name": _canonical_text(row[2]),
            "sector": canonical_sector(row[1]),
        }
        flagged = "*" in symbol_text or "new entry" in note.lower()
        parsed.append((int(row[0]), record, flagged))
        row_numbers.append(index)
    main, reserves, marked = _split_main_and_reserve(
        parsed,
        index_name=index_name,
        expected_count=expected_count,
        reserve_start=reserve_start,
        row_numbers=row_numbers,
    )
    exclusions: set[str] = set()
    for row in rows[:header]:
        for cell in row:
            text = _cell_text(cell)
            if text.lower().startswith("exclusion:"):
                exclusions |= _split_symbol_list(text.split(":", 1)[1])
    return main, reserves, marked, exclusions


def _thai_sector_column(rows: list[list[Any]], data_rows: list[int]) -> int:
    """The 2014 workbooks label columns 3/4 the same way but swap their content; decide by content."""
    candidates = []
    for column in (3, 4):
        values = {_cell_text(rows[index][column]) for index in data_rows}
        if values and values <= set(THAI_SECTOR_TO_ENGLISH):
            candidates.append(column)
        elif not values <= THAI_INDUSTRY_GROUPS:
            unknown = sorted(values - set(THAI_SECTOR_TO_ENGLISH) - THAI_INDUSTRY_GROUPS)
            raise ValueError(f"unrecognized Thai sector or industry labels: {unknown}")
    if len(candidates) != 1:
        raise ValueError("could not tell the sector column from the industry-group column")
    return candidates[0]


def _parse_th_sheet(
    rows: list[list[Any]],
    *,
    index_name: str,
    expected_count: int,
    membership_column: int | None = None,
) -> tuple[list[tuple[int, dict[str, str], bool, str, str]], int | None]:
    header = next(
        (index for index, row in enumerate(rows) if _cell_text(row[0]) == "ลำดับ"),
        None,
    )
    if header is None or _cell_text(rows[header][1]) != "ชื่อย่อ":
        raise ValueError(f"{index_name}-TH: no recognizable header row")
    note_column = len(rows[header]) - 1
    if _cell_text(rows[header][note_column]) != "หมายเหตุ":
        raise ValueError(f"{index_name}-TH: last column is not the note column")
    reserve_start = next(
        (
            index
            for index, row in enumerate(rows)
            if _cell_text(row[0]).startswith("รายชื่อหลักทรัพย์สำรอง")
        ),
        None,
    )
    data_rows = [
        index
        for index, row in enumerate(rows[header + 1 :], start=header + 1)
        if _is_rank(row[0]) and (reserve_start is None or index < reserve_start)
    ]
    sector_column = _thai_sector_column(rows, data_rows)
    parsed = []
    for index, row in enumerate(rows[header + 1 :], start=header + 1):
        if not _is_rank(row[0]):
            continue
        symbol_text = _cell_text(row[1])
        symbol = symbol_text.strip("* ").upper()
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise ValueError(f"{index_name}-TH row {index}: invalid symbol {symbol_text!r}")
        thai_sector = _cell_text(row[sector_column])
        if thai_sector not in THAI_SECTOR_TO_ENGLISH:
            raise ValueError(f"{index_name}-TH row {index}: unknown sector {thai_sector!r}")
        record = {
            "symbol": symbol,
            "company_name": _canonical_text(row[2]),
            "sector": THAI_SECTOR_TO_ENGLISH[thai_sector],
        }
        flagged = "*" in symbol_text
        membership = _cell_text(row[membership_column]) if membership_column is not None else ""
        note = _cell_text(row[note_column])
        in_reserve = reserve_start is not None and index > reserve_start
        parsed.append((int(row[0]), record, flagged, membership, note if not in_reserve else "", index))
    return parsed, reserve_start


def _thai_exclusions(rows: list[list[Any]]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for row in rows:
        for cell in row:
            match = re.match(r"(SET50|SET100)\s+Exclusion\s*:\s*(.*)", _cell_text(cell), re.IGNORECASE)
            if match:
                found[match.group(1).upper()] = _split_symbol_list(match.group(2))
    return found


def parse_set_constituent_xls(
    path: Path,
    *,
    source_format: str,
    expected_set50_count: int = 50,
    expected_set100_count: int = 100,
) -> ConstituentList:
    book = _open_workbook(path)
    if source_format == "xls_en_sheets":
        rows50 = _sheet_rows(book, "SET50-EN")
        rows100 = _sheet_rows(book, "SET100-EN")
        set50, reserves50, marked50, excluded50 = _parse_en_sheet(
            rows50, index_name="SET50", expected_count=expected_set50_count
        )
        set100, reserves100, marked100, excluded100 = _parse_en_sheet(
            rows100, index_name="SET100", expected_count=expected_set100_count
        )
        header_text = _row_text(rows50[1]) + "\n" + _row_text(rows100[1])
        period = _period_from_text(_row_text(rows50[1]))
        if _period_from_text(_row_text(rows100[1])) != period:
            raise ValueError("the SET50 and SET100 sheets state different calculation periods")
        printed: dict[str, dict[str, set[str]]] = {}
        # 2012-2013 workbooks print exclusions in the header and flag inclusions as "New Entry".
        if "exclusion" in header_text.lower():
            printed = {
                "SET50": {"exclusion": excluded50, "inclusion": marked50},
                "SET100": {"exclusion": excluded100, "inclusion": marked100},
            }
        return ConstituentList(
            set50=set50,
            set100=set100,
            period_start=period[0],
            period_end=period[1],
            published_on=_published_from_text(header_text),
            unit_count=2,
            reserves={"SET50": reserves50, "SET100": reserves100},
            printed_changes=printed,
            marked={"SET50": marked50, "SET100": marked100},
        )

    if source_format == "xls_th_set100_marker":
        rows = _sheet_rows(book, "SET100-TH")
        header = next((index for index, row in enumerate(rows) if _cell_text(row[0]) == "ลำดับ"), None)
        if header is None:
            raise ValueError("SET100-TH: no header row")
        membership_column = next(
            (
                column
                for column, cell in enumerate(rows[header])
                if _cell_text(cell).replace(" ", "") == "SET50/SET100"
            ),
            None,
        )
        if membership_column is None:
            raise ValueError("SET100-TH: no SET50/SET100 membership column")
        parsed, reserve_start = _parse_th_sheet(
            rows,
            index_name="SET100",
            expected_count=expected_set100_count,
            membership_column=membership_column,
        )
        set100, reserves100, _ = _split_main_and_reserve(
            [(rank, record, flagged) for rank, record, flagged, _, _, _ in parsed],
            index_name="SET100",
            expected_count=expected_set100_count,
            reserve_start=reserve_start,
            row_numbers=[row for *_, row in parsed],
        )
        main_rows = [item for item in parsed if reserve_start is None or item[5] < reserve_start]
        set50_records = [
            record
            for _, record, _, membership, _, _ in main_rows
            if membership.replace(" ", "").upper().startswith("SET50&")
        ]
        reserves50 = [
            record["symbol"] for _, record, _, membership, _, _ in main_rows if "สำรอง" in membership
        ]
        if len(set50_records) != expected_set50_count:
            raise ValueError(f"SET50: found {len(set50_records)} members, expected {expected_set50_count}")
        set50 = {rank: record for rank, record in enumerate(set50_records, start=1)}
        set50_symbols = {record["symbol"] for record in set50_records}
        # "SET50 New Entry" marks a SET100 member that joins SET50; "New Entry" marks a new SET100
        # member, which is also new to SET50 when the row says "SET50 & SET100".
        marked50 = {
            record["symbol"]
            for _, record, _, _, note, _ in main_rows
            if "new entry" in note.lower() and record["symbol"] in set50_symbols
        }
        marked100 = {
            record["symbol"]
            for _, record, _, _, note, _ in main_rows
            if note.lower() == "new entry"
        }
        exclusions = _thai_exclusions(rows)
        header_text = _row_text(rows[0]) + "\n" + _row_text(rows[1])
        period = _period_from_text(header_text)
        return ConstituentList(
            set50=set50,
            set100=set100,
            period_start=period[0],
            period_end=period[1],
            published_on=_published_from_text(header_text),
            unit_count=1,
            reserves={"SET50": reserves50, "SET100": reserves100},
            printed_changes={
                "SET50": {"exclusion": exclusions.get("SET50", set()), "inclusion": marked50},
                "SET100": {"exclusion": exclusions.get("SET100", set()), "inclusion": marked100},
            },
            marked={"SET50": marked50, "SET100": marked100},
        )

    if source_format == "xls_th_sheets":
        lists: dict[str, Any] = {}
        texts = []
        for index_name, expected in (("SET50", expected_set50_count), ("SET100", expected_set100_count)):
            rows = _sheet_rows(book, f"{index_name}-TH")
            parsed, reserve_start = _parse_th_sheet(rows, index_name=index_name, expected_count=expected)
            main, reserves, _ = _split_main_and_reserve(
                [(rank, record, flagged) for rank, record, flagged, _, _, _ in parsed],
                index_name=index_name,
                expected_count=expected,
                reserve_start=reserve_start,
                row_numbers=[row for *_, row in parsed],
            )
            marked = {
                record["symbol"]
                for _, record, _, _, note, row in parsed
                if "new entry" in note.lower() and (reserve_start is None or row < reserve_start)
            }
            exclusions = _thai_exclusions(rows).get(index_name, set())
            lists[index_name] = (main, reserves, marked, exclusions)
            texts.append(_row_text(rows[0]) + "\n" + _row_text(rows[1]))
        period = _period_from_text(texts[0])
        if _period_from_text(texts[1]) != period:
            raise ValueError("the SET50 and SET100 sheets state different calculation periods")
        published = _published_from_text(texts[0])
        if _published_from_text(texts[1]) != published:
            raise ValueError("the SET50 and SET100 sheets state different publication dates")
        return ConstituentList(
            set50=lists["SET50"][0],
            set100=lists["SET100"][0],
            period_start=period[0],
            period_end=period[1],
            published_on=published,
            unit_count=2,
            reserves={name: value[1] for name, value in lists.items()},
            printed_changes={
                name: {"exclusion": value[3], "inclusion": value[2]} for name, value in lists.items()
            },
            marked={name: value[2] for name, value in lists.items()},
        )

    raise ValueError(f"unsupported workbook source format: {source_format!r}")


def parse_constituent_source(
    path: Path,
    *,
    source_format: str,
    expected_set50_count: int = 50,
    expected_set100_count: int = 100,
) -> ConstituentList:
    if source_format == "pdf_two_column":
        parsed = parse_two_column_pdf(
            path,
            expected_set50_count=expected_set50_count,
            expected_set100_count=expected_set100_count,
        )
    else:
        parsed = parse_set_constituent_xls(
            path,
            source_format=source_format,
            expected_set50_count=expected_set50_count,
            expected_set100_count=expected_set100_count,
        )
    validate_constituent_list(parsed, expected_set50_count, expected_set100_count)
    return parsed


def validate_constituent_list(
    parsed: ConstituentList,
    expected_set50_count: int,
    expected_set100_count: int,
) -> None:
    for index_name, records, expected in (
        ("SET50", parsed.set50, expected_set50_count),
        ("SET100", parsed.set100, expected_set100_count),
    ):
        if set(records) != set(range(1, expected + 1)):
            raise ValueError(f"{index_name} rank contract failed")
        symbols = [record["symbol"] for record in records.values()]
        if len(set(symbols)) != len(symbols):
            raise ValueError(f"{index_name} contains duplicate symbols")
        for rank, record in records.items():
            if not record["company_name"]:
                raise ValueError(f"{index_name} rank {rank} has a blank company")
            if record["sector"] not in ALL_SECTORS:
                raise ValueError(f"{index_name} rank {rank} has an unknown sector")
        overlap = set(symbols) & set(parsed.reserves.get(index_name, []))
        if overlap:
            raise ValueError(f"{index_name} reserve list repeats members: {sorted(overlap)}")
    by_symbol = {record["symbol"]: record for record in parsed.set100.values()}
    missing = sorted({record["symbol"] for record in parsed.set50.values()} - set(by_symbol))
    if missing:
        raise ValueError("SET50 is not a subset of SET100: " + ", ".join(missing))
    disagree = sorted(
        record["symbol"]
        for record in parsed.set50.values()
        if by_symbol[record["symbol"]]["sector"] != record["sector"]
    )
    if disagree:
        raise ValueError("SET50/SET100 sector labels disagree for: " + ", ".join(disagree))


# ---------------------------------------------------------------------------- manifest


def load_extended_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("extended SET manifest is missing columns: " + ", ".join(missing))
    frame = frame[list(MANIFEST_COLUMNS)].copy()
    if frame.empty:
        raise ValueError("extended SET manifest is empty")
    if frame["record_id"].duplicated().any():
        raise ValueError("extended SET manifest has duplicate record IDs")
    for column in ("expected_set50_count", "expected_set100_count", "source_bytes"):
        try:
            frame[column] = frame[column].astype(int)
        except ValueError as exc:
            raise ValueError(f"manifest column {column} must be integral") from exc
    for column in ("effective_from", "effective_to_exclusive", "published_at"):
        parsed = pd.to_datetime(frame[column], errors="coerce", format="%Y-%m-%d")
        if parsed.isna().any():
            raise ValueError(f"manifest column {column} contains invalid dates")
        frame[column] = parsed.dt.strftime("%Y-%m-%d")
    retrieval = pd.to_datetime(frame["retrieved_at_utc"], errors="coerce", utc=True)
    if retrieval.isna().any():
        raise ValueError("manifest has invalid retrieval timestamps")
    frame["retrieved_at_utc"] = retrieval.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame = frame.sort_values(
        ["effective_from", "effective_to_exclusive", "record_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    validate_extended_manifest(frame)
    return frame


def validate_extended_manifest(
    frame: pd.DataFrame,
    *,
    start: str = EXTENDED_START,
    end: str = BASE_ARCHIVE_START,
) -> None:
    by_id = {row.record_id: row for row in frame.itertuples(index=False)}
    for row in frame.itertuples(index=False):
        if row.revision_status not in REVISION_STATUSES:
            raise ValueError(f"unsupported revision status for {row.record_id}: {row.revision_status}")
        if row.source_format not in SOURCE_FORMATS:
            raise ValueError(f"unsupported source format for {row.record_id}: {row.source_format}")
        if row.published_at_source not in PUBLISHED_AT_SOURCES:
            raise ValueError(f"unsupported published_at_source for {row.record_id}")
        if row.published_at > row.effective_from:
            raise ValueError(f"manifest would apply {row.record_id} before it was published")
        if row.revision_status == "original":
            if row.supersedes_record_id or row.effective_date_basis != "scheduled_period_start":
                raise ValueError(f"original record {row.record_id} must start at its scheduled period")
        elif row.revision_status == "revised":
            predecessor = by_id.get(row.supersedes_record_id)
            if predecessor is None:
                raise ValueError(f"revised record {row.record_id} supersedes an unknown record")
            if predecessor.revision_status == "superseded_before_effective":
                if (
                    row.effective_date_basis != "scheduled_period_start"
                    or row.effective_from != predecessor.effective_from
                ):
                    raise ValueError(
                        f"{row.record_id} replaces a list before its period began, so it must "
                        "start at that scheduled period start"
                    )
            elif (
                row.effective_date_basis != "publication_date_conservative"
                or row.effective_from != row.published_at
            ):
                raise ValueError(f"revised record {row.record_id} is not publication-gated")
        if row.revision_status == "superseded_before_effective":
            if row.effective_from != row.effective_to_exclusive:
                raise ValueError(f"{row.record_id} never applied, so its interval must be empty")
            replacements = [
                other
                for other in frame.itertuples(index=False)
                if other.supersedes_record_id == row.record_id and other.revision_status == "revised"
            ]
            if len(replacements) != 1 or replacements[0].published_at > row.effective_from:
                raise ValueError(f"{row.record_id} has no replacement published before its period began")
        elif row.effective_from >= row.effective_to_exclusive:
            raise ValueError(f"manifest contains a non-positive interval for {row.record_id}")

    applied = frame[frame["revision_status"] != "superseded_before_effective"]
    starts = list(applied["effective_from"])
    ends = list(applied["effective_to_exclusive"])
    if not starts or starts[0] != start:
        raise ValueError(f"extended manifest must start at {start}")
    if ends[-1] != end:
        raise ValueError(f"extended manifest must end at {end} exclusive")
    for previous_end, current_start in zip(ends, starts[1:]):
        if previous_end != current_start:
            relationship = "gap" if previous_end < current_start else "overlap"
            raise ValueError(f"manifest interval {relationship}: {previous_end} -> {current_start}")


# ---------------------------------------------------------------------------- archive


def _symbols(records: dict[int, dict[str, str]]) -> set[str]:
    return {record["symbol"] for record in records.values()}


def _check_printed_changes(
    record_id: str,
    parsed: ConstituentList,
    baseline: dict[str, set[str]] | None,
    *,
    revision: bool,
    predecessor: ConstituentList | None = None,
) -> list[dict[str, Any]]:
    """Compare the change from the baseline list with what the source prints about it.

    The baseline is the previous period's final list for a scheduled review and the replaced list
    for a revision. Symbol changes and mergers in ``CORPORATE_ACTIONS`` are applied to it first.
    Fresh Inclusion/Exclusion notes or "New Entry" marks must match exactly. A revision without
    fresh notes (or one that reprints its predecessor's notes) must name every removed symbol in a
    replacement note, and every added symbol must be named, starred, or on the replaced list's
    reserve list.
    """
    checks: list[dict[str, Any]] = []
    if baseline is None:
        return checks
    actions = [action for action in CORPORATE_ACTIONS if action["first_record_id"] == record_id]
    mapping = {action["old"]: action["new"] for action in actions}
    current = {"SET50": _symbols(parsed.set50), "SET100": _symbols(parsed.set100)}
    for action in actions:
        if not any(action["old"] in baseline[name] for name in INDEX_NAMES):
            raise ValueError(f"{record_id}: {action['old']} is not in the earlier list")
        if any(action["old"] in current[name] for name in INDEX_NAMES):
            raise ValueError(f"{record_id}: {action['old']} is still listed after {action['kind']}")
        if not any(action["new"] in current[name] for name in INDEX_NAMES):
            raise ValueError(f"{record_id}: {action['new']} is not in the later list")
    for index_name in INDEX_NAMES:
        mapped = {mapping.get(symbol, symbol) for symbol in baseline[index_name]}
        added = current[index_name] - mapped
        removed = mapped - current[index_name]
        printed = parsed.printed_changes.get(index_name, {})
        stale = bool(
            revision
            and printed
            and predecessor is not None
            and printed == predecessor.printed_changes.get(index_name, {})
        )
        expectations: list[tuple[str, set[str], set[str]]] = []
        if printed and not stale:
            if "inclusion" in printed:
                expectations.append(("inclusion", added, printed["inclusion"]))
            if "exclusion" in printed:
                expectations.append(("exclusion", removed, printed["exclusion"]))
        elif revision:
            notes = parsed.replacement_notes or {"included": set(), "excluded": set()}
            named_out = {mapping.get(symbol, symbol) for symbol in notes["excluded"]} & mapped
            expectations.append(("named_exclusions", removed, named_out))
            reserves = set(predecessor.reserves.get(index_name, [])) if predecessor else set()
            explained = parsed.marked.get(index_name, set()) | notes["included"] | {
                mapping.get(symbol, symbol) for symbol in reserves
            }
            expectations.append(("inclusions_named_starred_or_reserve", added, added & explained))
        for kind, computed, stated in expectations:
            check = {
                "record_id": record_id,
                "index": index_name,
                "kind": kind,
                "computed": sorted(computed),
                "printed": sorted(stated),
                "match": computed == stated,
                "stale_notes_ignored": stale,
            }
            checks.append(check)
            if not check["match"]:
                raise ValueError(
                    f"{record_id} {index_name} {kind}: source prints {sorted(stated)} but the "
                    f"lists differ by {sorted(computed)}"
                )
    return checks


def _membership_rows(row: Any, parsed: ConstituentList, source_sha256: str) -> list[dict[str, str]]:
    set50 = _symbols(parsed.set50)
    rows = []
    for rank in sorted(parsed.set100):
        record = parsed.set100[rank]
        symbol = record["symbol"]
        rows.append(
            {
                "record_id": row.record_id,
                "period_label": row.period_label,
                "effective_from": row.effective_from,
                "effective_to_exclusive": row.effective_to_exclusive,
                "published_at": row.published_at,
                "effective_date_basis": row.effective_date_basis,
                "revision_status": row.revision_status,
                "symbol": symbol,
                "ticker": f"{symbol}.BK",
                "company_name": record["company_name"],
                "sector": record["sector"],
                "in_set50": str(symbol in set50),
                "in_set100": "True",
                "source_url": row.source_url,
                "source_file_path": row.source_file_path,
                "source_file_sha256": source_sha256,
                "retrieved_at_utc": row.retrieved_at_utc,
            }
        )
    return rows


def _load_base_archive(project_root: Path, relative_path: str) -> tuple[pd.DataFrame, bytes]:
    path = project_root / relative_path
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    if digest != BASE_ARCHIVE_SHA256:
        raise ValueError(
            f"the 2021-2024 archive {relative_path} changed (sha256 {digest}); "
            "it must stay byte-identical to the July receipt"
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != MEMBERSHIP_COLUMNS:
        raise ValueError("the 2021-2024 archive has unexpected columns")
    return frame, payload


def build_extended_archive(
    project_root: Path,
    manifest_path: Path,
    *,
    base_archive_path: str = BASE_ARCHIVE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    project_root = project_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = load_extended_manifest(manifest_path)
    manifest_bytes = manifest_path.read_bytes()

    rows: list[dict[str, str]] = []
    profiles: list[dict[str, Any]] = []
    raw_files: list[dict[str, Any]] = []
    change_checks: list[dict[str, Any]] = []
    superseded: list[dict[str, Any]] = []
    prefix_matches: dict[str, list[str]] = {}
    inferred_ranks: dict[str, list[str]] = {}
    previous: dict[str, set[str]] | None = None
    parsed_by_id: dict[str, dict[str, set[str]]] = {}
    parsed_lists: dict[str, ConstituentList] = {}
    company_names: dict[str, str] = {}

    for row in manifest.itertuples(index=False):
        source_path, source_sha256, source_bytes = _validate_source_file(project_root, row)
        parsed = parse_constituent_source(
            source_path,
            source_format=row.source_format,
            expected_set50_count=int(row.expected_set50_count),
            expected_set100_count=int(row.expected_set100_count),
        )
        effective_from = dt.date.fromisoformat(row.effective_from)
        effective_to = dt.date.fromisoformat(row.effective_to_exclusive)
        if not (parsed.period_start <= effective_from and effective_to <= parsed.period_end + dt.timedelta(days=1)):
            raise ValueError(
                f"{row.record_id}: interval {row.effective_from}..{row.effective_to_exclusive} lies "
                f"outside the printed period {parsed.period_start}..{parsed.period_end}"
            )
        if row.revision_status == "original" and parsed.period_start != effective_from:
            raise ValueError(f"{row.record_id}: original does not start at its printed period")
        if row.published_at_source == "source_header":
            if parsed.published_on is None or parsed.published_on.isoformat() != row.published_at:
                raise ValueError(
                    f"{row.record_id}: manifest publication date {row.published_at} does not match "
                    f"the source header ({parsed.published_on})"
                )
        elif parsed.published_on is not None:
            raise ValueError(f"{row.record_id}: source states a publication date; use source_header")
        elif row.published_at != row.effective_from:
            raise ValueError(f"{row.record_id}: an unstated publication date must equal the period start")

        current = {"SET50": _symbols(parsed.set50), "SET100": _symbols(parsed.set100)}
        parsed_by_id[row.record_id] = current
        parsed_lists[row.record_id] = parsed
        for record in parsed.set100.values():
            company_names.setdefault(record["symbol"], record["company_name"])
        raw_files.append(
            {
                "record_id": row.record_id,
                "period_label": row.period_label,
                "source_format": row.source_format,
                "source_url": row.source_url,
                "source_file_path": row.source_file_path,
                "sha256": source_sha256,
                "bytes": source_bytes,
                "units": parsed.unit_count,
                "retrieved_at_utc": row.retrieved_at_utc,
            }
        )
        if parsed.prefix_matched_sectors:
            prefix_matches[row.record_id] = parsed.prefix_matched_sectors
        if parsed.inferred_ranks:
            inferred_ranks[row.record_id] = parsed.inferred_ranks

        revision = row.revision_status == "revised"
        if row.revision_status == "superseded_before_effective":
            # Checked against the previous applied list, then set aside: it never took effect.
            change_checks += _check_printed_changes(row.record_id, parsed, previous, revision=False)
            superseded.append(
                {
                    "record_id": row.record_id,
                    "published_at": row.published_at,
                    "nominal_period_start": parsed.period_start.isoformat(),
                    "superseded_by": next(
                        other.record_id
                        for other in manifest.itertuples(index=False)
                        if other.supersedes_record_id == row.record_id
                    ),
                }
            )
            continue
        baseline = previous
        predecessor = None
        if revision:
            # A revision is checked against the list it replaces, which for 2016 H1 is an original
            # that never took effect.
            baseline = parsed_by_id[row.supersedes_record_id]
            predecessor = parsed_lists[row.supersedes_record_id]
        change_checks += _check_printed_changes(
            row.record_id, parsed, baseline, revision=revision, predecessor=predecessor
        )

        rows += _membership_rows(row, parsed, source_sha256)
        added50 = 0 if previous is None else len(current["SET50"] - previous["SET50"])
        removed50 = 0 if previous is None else len(previous["SET50"] - current["SET50"])
        added100 = 0 if previous is None else len(current["SET100"] - previous["SET100"])
        removed100 = 0 if previous is None else len(previous["SET100"] - current["SET100"])
        profiles.append(
            {
                "record_id": row.record_id,
                "period_label": row.period_label,
                "effective_from": row.effective_from,
                "effective_to_exclusive": row.effective_to_exclusive,
                "published_at": row.published_at,
                "revision_status": row.revision_status,
                "source_format": row.source_format,
                "set50_count": len(current["SET50"]),
                "set100_count": len(current["SET100"]),
                "set50_added": added50,
                "set50_removed": removed50,
                "set100_added": added100,
                "set100_removed": removed100,
                "sector_count": len({record["sector"] for record in parsed.set100.values()}),
            }
        )
        previous = current

    extended = pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS)
    extended = extended.sort_values(["effective_from", "symbol"], kind="mergesort").reset_index(drop=True)
    if extended.duplicated(["record_id", "symbol"]).any():
        raise ValueError("extended archive contains duplicate interval/symbol rows")
    if not extended["ticker"].str.fullmatch(r"[A-Z0-9&.-]+\.BK").all():
        raise ValueError("extended archive contains malformed .BK ticker mappings")

    base, base_payload = _load_base_archive(project_root, base_archive_path)
    if base["effective_from"].min() != BASE_ARCHIVE_START or base["effective_to_exclusive"].max() != BASE_ARCHIVE_END:
        raise ValueError("the 2021-2024 archive does not cover 2021-01-01..2025-01-01")
    if extended["effective_to_exclusive"].max() != BASE_ARCHIVE_START:
        raise ValueError("the extended archive does not end where the 2021-2024 archive starts")
    combined = pd.concat([extended, base], ignore_index=True)
    combined_bytes = _csv_bytes(combined)
    base_body = base_payload.split(b"\n", 1)[1]
    if not combined_bytes.endswith(base_body):
        raise ValueError("the 2021-2024 rows changed while combining the archives")

    # The 2020 -> 2021 seam: the change from the last 2020 list to the July archive's first list
    # must match the Inclusion/Exclusion notes printed in that first list's own PDF.
    first_id = base["record_id"].iloc[0]
    first_base = base[base["record_id"] == first_id]
    seam = {
        "SET50": set(first_base.loc[first_base["in_set50"] == "True", "symbol"]),
        "SET100": set(first_base["symbol"]),
    }
    seam_source = project_root / first_base["source_file_path"].iloc[0]
    if sha256_file(seam_source) != first_base["source_file_sha256"].iloc[0]:
        raise ValueError("the 2021 H1 source PDF does not match the July archive's checksum")
    seam_printed = printed_changes_from_two_column_pdf(seam_source)
    seam_checks = []
    for index_name in INDEX_NAMES:
        for kind, computed in (
            ("inclusion", seam[index_name] - previous[index_name]),
            ("exclusion", previous[index_name] - seam[index_name]),
        ):
            stated = seam_printed.get(index_name, {}).get(kind)
            if stated is None:
                raise ValueError(f"the 2021 H1 PDF prints no {index_name} {kind} note")
            seam_checks.append(
                {
                    "record_id": first_id,
                    "index": index_name,
                    "kind": kind,
                    "computed": sorted(computed),
                    "printed": sorted(stated),
                    "match": computed == stated,
                }
            )
            if computed != stated:
                raise ValueError(
                    f"2020 -> 2021 seam {index_name} {kind}: the 2021 H1 PDF prints {sorted(stated)} "
                    f"but the lists differ by {sorted(computed)}"
                )
    change_checks += seam_checks

    symbol_changes = pd.DataFrame(
        [
            {
                "old_symbol": action["old"],
                "new_symbol": action["new"],
                "kind": action["kind"],
                "first_record_with_new_symbol": action["first_record_id"],
                "old_company_name": company_names.get(action["old"], ""),
                "new_company_name": company_names.get(action["new"], ""),
                "note": action["note"],
            }
            for action in CORPORATE_ACTIONS
        ]
    )

    universe = _acquisition_universe(combined)
    profile = pd.DataFrame(profiles)
    turnover50 = (profile["set50_added"]).iloc[1:]
    receipt = {
        "schema_version": 1,
        "milestone": "M3 point-in-time universe, 2012-2024",
        "source": "Stock Exchange of Thailand official SET50/SET100 constituent archive",
        "source_archive_page": (
            "https://www.set.or.th/en/market/information/securities-list/"
            "constituents-list-set50-set100"
        ),
        "source_manifest": {
            "path": str(manifest_path.relative_to(project_root)),
            "sha256": sha256_bytes(manifest_bytes),
            "bytes": len(manifest_bytes),
            "rows": len(manifest),
        },
        "base_archive_2021_2024": {
            "path": base_archive_path,
            "sha256": BASE_ARCHIVE_SHA256,
            "rows": len(base),
            "unchanged": True,
        },
        "coverage": {
            "effective_from": combined["effective_from"].min(),
            "effective_to_exclusive": combined["effective_to_exclusive"].max(),
            "interval_count": int(combined["record_id"].nunique()),
            "extended_interval_count": int(extended["record_id"].nunique()),
            "normalized_rows": len(combined),
            "unique_tickers": int(combined["ticker"].nunique()),
            "sector_count": int(combined["sector"].nunique()),
            "raw_file_count": len(raw_files),
            "raw_bytes": sum(item["bytes"] for item in raw_files),
        },
        "quality_gates": {
            "source_hashes_match_manifest": True,
            "source_bytes_match_manifest": True,
            "continuous_intervals_2012_2025": True,
            "exactly_50_set50_per_interval": bool(profile["set50_count"].eq(50).all()),
            "exactly_100_set100_per_interval": bool(profile["set100_count"].eq(100).all()),
            "set50_subset_of_set100": True,
            "no_duplicate_interval_symbols": True,
            "revisions_not_applied_before_publication": True,
            "printed_publication_dates_match_manifest": True,
            "printed_changes_match_list_differences": all(check["match"] for check in change_checks),
            "seam_2020_2021_matches_printed_notes": all(check["match"] for check in seam_checks),
            "pdf_lists_in_printed_alphabetical_order": True,
            "base_archive_byte_identical": True,
        },
        "printed_change_checks": {
            "checked": len(change_checks),
            "matched": sum(check["match"] for check in change_checks),
            "records_without_printed_changes": sorted(
                set(manifest["record_id"]) - {check["record_id"] for check in change_checks}
            ),
        },
        "turnover_summary": {
            "median_set50_additions_per_list": float(turnover50.median()) if len(turnover50) else 0.0,
            "max_set50_additions_per_list": int(turnover50.max()) if len(turnover50) else 0,
            "seam_2020_to_2021": {check["index"] + " " + check["kind"]: check["computed"] for check in seam_checks},
        },
        "superseded_before_effective": superseded,
        "corporate_actions": symbol_changes.to_dict(orient="records"),
        "prefix_matched_sectors": prefix_matches,
        "inferred_ranks": inferred_ranks,
        "eligibility": {
            "point_in_time_membership_eligible": True,
            "model_input_eligible": False,
            "static_union_model_eligible": False,
            "publication_grade_performance_claims_eligible": False,
        },
        "limitations": [
            (
                "Between-review revisions state no separate exchange-effective date, so they apply "
                "from their publication date. The 2016 H1 merger list was published before the period "
                "began and replaces the original list for the whole period."
            ),
            (
                "The 2012-2013 workbooks state no publication date. published_at is set to the period "
                "start, the latest date by which the list was public, because the index used it from "
                "that day. The SET archive offers no between-review update files for 2012-2014, so any "
                "replacement inside those periods is not reflected."
            ),
            (
                "Company names for 2014 are in Thai because the 2014 workbooks have Thai sheets only. "
                "Sectors are mapped to English with the SET's own summary tables."
            ),
            (
                "Tickers are the symbols printed in each list. Renamed securities (for example "
                "KK to KKP, TUF to TU, PS to PSH) appear under each historical symbol and are not linked."
            ),
            (
                "The union file is for price acquisition only. Using it as a static model universe "
                "would reintroduce survivorship and future-membership bias."
            ),
        ],
        "raw_files": raw_files,
        "change_checks": change_checks,
    }
    return combined, universe, profile, symbol_changes, receipt


def _acquisition_universe(membership: pd.DataFrame) -> pd.DataFrame:
    ordered = membership.sort_values(["ticker", "effective_from"], kind="mergesort")
    grouped = ordered.groupby("ticker", sort=True)
    latest = grouped.tail(1).set_index("ticker")
    return pd.DataFrame(
        {
            "ticker": sorted(membership["ticker"].unique()),
        }
    ).assign(
        symbol=lambda frame: frame["ticker"].map(latest["symbol"]),
        usage="acquisition_only_not_static_model_membership",
        first_effective_from=lambda frame: frame["ticker"].map(grouped["effective_from"].min()),
        last_effective_to_exclusive=lambda frame: frame["ticker"].map(grouped["effective_to_exclusive"].max()),
        sector=lambda frame: frame["ticker"].map(latest["sector"]),
        sector_history_consistent=lambda frame: frame["ticker"].map(grouped["sector"].nunique().eq(1)),
        set100_intervals=lambda frame: frame["ticker"].map(grouped["record_id"].nunique()),
        set50_intervals=lambda frame: frame["ticker"].map(
            ordered[ordered["in_set50"] == "True"].groupby("ticker")["record_id"].nunique()
        ).fillna(0).astype(int),
    )


def write_extended_archive(
    *,
    project_root: Path,
    membership: pd.DataFrame,
    universe: pd.DataFrame,
    profile: pd.DataFrame,
    symbol_changes: pd.DataFrame,
    receipt: dict[str, Any],
    membership_path: Path,
    universe_path: Path,
    profile_path: Path,
    symbol_changes_path: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    outputs = {
        "membership_history": (membership_path, _csv_bytes(membership), len(membership)),
        "acquisition_universe": (universe_path, _csv_bytes(universe), len(universe)),
        "interval_profile": (profile_path, _csv_bytes(profile), len(profile)),
        "symbol_changes": (symbol_changes_path, _csv_bytes(symbol_changes), len(symbol_changes)),
    }
    receipt["outputs"] = {
        name: {
            "path": str(path.resolve().relative_to(project_root)),
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "rows": rows,
        }
        for name, (path, payload, rows) in outputs.items()
    }
    for path, payload, _ in outputs.values():
        _atomic_bytes(path, payload)
    receipt_path = membership_path.with_name(membership_path.name + ".receipt.json")
    _atomic_json(receipt_path, receipt)
    _atomic_bytes(
        receipt_path.with_name(receipt_path.name + ".sha256"),
        f"{sha256_file(receipt_path)}  {receipt_path.name}\n".encode("utf-8"),
    )
    return receipt


def load_membership_2012_2024(path: Path) -> pd.DataFrame:
    """Load the 2012-2024 archive and check it with the July validator's rules."""
    return validate_constituent_history(
        pd.read_csv(path),
        expected_start=EXTENDED_START,
        expected_end_exclusive=BASE_ARCHIVE_END,
    )


def membership_as_of_2012_2024(
    history: pd.DataFrame,
    as_of: str | pd.Timestamp,
    *,
    index_name: str = "SET100",
) -> pd.DataFrame:
    """Members on a date, from a frame returned by ``load_membership_2012_2024``.

    Same rule as ``constituent_history.membership_as_of``: a list applies from its effective date
    and never before its publication date.
    """
    if index_name not in INDEX_NAMES:
        raise ValueError(f"index_name must be one of {INDEX_NAMES}")
    timestamp = pd.Timestamp(as_of)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    active = history[
        (history["effective_from"] <= timestamp)
        & (timestamp < history["effective_to_exclusive"])
        & (history["published_at"] <= timestamp)
    ]
    active = active[active["in_set50" if index_name == "SET50" else "in_set100"]]
    expected = 50 if index_name == "SET50" else 100
    if len(active) != expected:
        raise ValueError(
            f"{index_name} membership at {timestamp.date()} has {len(active)} rows, expected {expected}"
        )
    return active.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extend the checksum-bound, point-in-time SET50/SET100 archive back to 2012 from "
            "preserved official SET workbooks and PDFs, appending the unchanged 2021-2024 archive."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--membership-output", default=DEFAULT_MEMBERSHIP_OUTPUT)
    parser.add_argument("--universe-output", default=DEFAULT_UNIVERSE_OUTPUT)
    parser.add_argument("--profile-output", default=DEFAULT_PROFILE_OUTPUT)
    parser.add_argument("--symbol-changes-output", default=DEFAULT_SYMBOL_CHANGES_OUTPUT)
    args = parser.parse_args(None if argv is None else list(argv))
    project_root = Path(args.project_root).resolve()
    membership, universe, profile, symbol_changes, receipt = build_extended_archive(
        project_root, project_root / args.manifest
    )
    receipt = write_extended_archive(
        project_root=project_root,
        membership=membership,
        universe=universe,
        profile=profile,
        symbol_changes=symbol_changes,
        receipt=receipt,
        membership_path=project_root / args.membership_output,
        universe_path=project_root / args.universe_output,
        profile_path=project_root / args.profile_output,
        symbol_changes_path=project_root / args.symbol_changes_output,
    )
    summary = {key: receipt[key] for key in ("coverage", "quality_gates", "printed_change_checks", "outputs")}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
