"""Normalize the Bank of Thailand policy-rate decision history (milestone M4: official, point-in-time macro).

Source: "Historical data of policy interest rate and MPC voting result", Bank of Thailand,
https://www.bot.or.th/en/our-roles/monetary-policy/mpc-publication/policy-interest-rate.html
(downloadable .xlsx). Each row is an announced decision with its date, so the series is point-in-time
by construction: nothing is revised after the announcement, unlike the BOT LEI history.

The workbook is read with the standard library (zipfile + XML), so no spreadsheet dependency is
needed. Output columns: announcement_date, meeting_type, decision, change_bp, policy_rate,
votes_hold, votes_raise_25, votes_raise_50, votes_cut_25, votes_cut_50, votes_absent,
dissent_share, source_name. A JSON receipt binds the output to the source file's SHA-256.

    python -m src.data.ingest_bot_policy_rate \
        --input data/external/original/bot_policy_rate_mpc_table_asof_20261028.xlsx \
        --output data/external/bot_policy_rate_history.csv \
        --source-url https://www.bot.or.th/... --retrieved-at 2026-11-06T06:48:00Z
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_EXCEL_EPOCH = dt.date(1899, 12, 30)
VOTE_COLUMNS = {"I": "votes_hold", "J": "votes_raise_25", "K": "votes_raise_50", "L": "votes_cut_25", "M": "votes_cut_50", "N": "votes_absent"}
# The policy rate switched from the 14-day to the 1-day repurchase rate on 17 January 2007 (workbook
# footnote 2/), so the level may jump at that boundary without a matching decision.
POLICY_RATE_SWITCH = pd.Timestamp("2007-01-17")


def read_xlsx_rows(path: str | Path, sheet: str = "xl/worksheets/sheet1.xml") -> list[dict[str, str | None]]:
    """Rows of the first worksheet as {column letter: text}, with shared strings resolved."""

    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", _NS):
                shared.append("".join(node.text or "" for node in item.iter(f"{{{_NS['m']}}}t")))
        sheet_root = ET.fromstring(archive.read(sheet))
    rows = []
    for row in sheet_root.findall(".//m:sheetData/m:row", _NS):
        cells: dict[str, str | None] = {}
        for cell in row.findall("m:c", _NS):
            column = re.sub(r"\d+", "", cell.get("r", ""))
            value = cell.find("m:v", _NS)
            if value is None:
                inline = cell.find(".//m:t", _NS)
                cells[column] = inline.text if inline is not None else None
            elif cell.get("t") == "s":
                cells[column] = shared[int(value.text)]
            else:
                cells[column] = value.text
        rows.append(cells)
    return rows


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip().rstrip(".")


def parse_decision(text: str | None) -> int:
    """Signed change in basis points from 'Hold', 'Raise 25 bps', 'Cut 50 bps', ..."""

    cleaned = _clean(text).lower()
    if cleaned == "hold":
        return 0
    match = re.fullmatch(r"(raise|cut) (\d+) bps", cleaned)
    if not match:
        raise ValueError(f"unrecognized decision text: {text!r}")
    size = int(match.group(2))
    return size if match.group(1) == "raise" else -size


def parse_votes(text: str | None) -> float:
    """Vote counts are single digits; a trailing 'N/' is a footnote marker (e.g. '65/' -> 6)."""

    cleaned = _clean(text)
    if not cleaned:
        return float("nan")
    match = re.fullmatch(r"(\d)(\d+/)?", cleaned)
    if not match:
        raise ValueError(f"unrecognized vote count: {text!r}")
    return float(match.group(1))


def parse_announcement_date(english: str | None, thai: str | None = None) -> dt.date:
    """Announcement date from the English date cell (Excel serial or text), falling back to the
    Thai-date column's serial. Two-day meetings ('28-29 May 2013') announce on the last day."""

    for text in (english, thai):
        if text and re.fullmatch(r"\d+(\.0+)?", text.strip()):
            return _EXCEL_EPOCH + dt.timedelta(days=int(float(text)))
    cleaned = _clean(english)
    two_day = re.fullmatch(r"\d{1,2}\s*-\s*(\d{1,2}) ([A-Za-z]+) (\d{4})", cleaned)
    if two_day:
        cleaned = f"{two_day.group(1)} {two_day.group(2)} {two_day.group(3)}"
    for fmt in ("%d %B %Y", "%d %b %Y", "%d-%b-%Y", "%d-%B-%Y"):
        try:
            return dt.datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized announcement date: {english!r}")


def normalize_policy_rate_history(rows: list[dict[str, str | None]], *, source_name: str) -> pd.DataFrame:
    records = []
    for cells in rows:
        decision_text = _clean(cells.get("G"))
        rate_text = _clean(cells.get("H"))
        if not decision_text or decision_text == "Decision" or not re.fullmatch(r"\d+(\.\d+)?", rate_text):
            continue  # header, blank rows, footnotes
        announced = parse_announcement_date(cells.get("C"), cells.get("B"))
        record = {
            "announcement_date": pd.Timestamp(announced),
            "meeting_type": re.sub(r"\d+/$", "", _clean(cells.get("E"))),
            "decision": _clean(cells.get("G")),
            "change_bp": parse_decision(cells.get("G")),
            "policy_rate": float(_clean(cells.get("H"))),
        }
        for column, name in VOTE_COLUMNS.items():
            record[name] = parse_votes(cells.get(column))
        records.append(record)
    if not records:
        raise ValueError("no decision rows found")
    frame = pd.DataFrame(records).sort_values("announcement_date").reset_index(drop=True)
    if frame["announcement_date"].duplicated().any():
        raise ValueError("duplicate announcement dates")

    voted = frame[["votes_hold", "votes_raise_25", "votes_raise_50", "votes_cut_25", "votes_cut_50"]]
    total = voted.sum(axis=1, min_count=1)
    frame["dissent_share"] = (1.0 - voted.max(axis=1) / total).where(total > 0)

    # Fail closed: each level change must equal its announced decision (except the 2007 regime switch).
    implied = (frame["policy_rate"].diff() * 100).round(6)
    check = frame.iloc[1:].copy()
    check["implied_bp"] = implied.iloc[1:]
    bad = check[(check["implied_bp"] != check["change_bp"]) & (check["announcement_date"] != POLICY_RATE_SWITCH)]
    if not bad.empty:
        raise ValueError(
            "policy-rate levels disagree with announced decisions on "
            f"{bad['announcement_date'].dt.date.astype(str).tolist()[:5]}"
        )
    frame["source_name"] = source_name
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--retrieved-at", required=True, help="UTC retrieval timestamp of the source file")
    parser.add_argument("--source-name", default="bot_policy_rate_mpc_table")
    args = parser.parse_args()

    source, output = Path(args.input), Path(args.output)
    frame = normalize_policy_rate_history(read_xlsx_rows(source), source_name=args.source_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.assign(announcement_date=frame["announcement_date"].dt.date).to_csv(output, index=False)
    receipt = {
        "source_name": args.source_name,
        "source_url": args.source_url,
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "source_bytes": source.stat().st_size,
        "retrieved_at_utc": args.retrieved_at,
        "output_path": str(output),
        "output_sha256": _sha256(output),
        "rows": int(len(frame)),
        "first_announcement": str(frame["announcement_date"].min().date()),
        "last_announcement": str(frame["announcement_date"].max().date()),
        "latest_policy_rate": float(frame["policy_rate"].iloc[-1]),
        "point_in_time": True,
        "note": "Announced decisions; level from the announcement date. 14-day repo rate until 2007-01-16, 1-day repo after.",
    }
    output.with_suffix(output.suffix + ".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
