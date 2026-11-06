from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pdfplumber


REQUIRED_MANIFEST_COLUMNS = (
    "record_id",
    "period_label",
    "effective_from",
    "effective_to_exclusive",
    "published_at",
    "revision_status",
    "supersedes_record_id",
    "effective_date_basis",
    "source_url",
    "source_file_path",
    "source_file_sha256",
    "source_bytes",
    "retrieved_at_utc",
    "expected_set50_count",
    "expected_set100_count",
)

MEMBERSHIP_COLUMNS = (
    "record_id",
    "period_label",
    "effective_from",
    "effective_to_exclusive",
    "published_at",
    "effective_date_basis",
    "revision_status",
    "symbol",
    "ticker",
    "company_name",
    "sector",
    "in_set50",
    "in_set100",
    "source_url",
    "source_file_path",
    "source_file_sha256",
    "retrieved_at_utc",
)

SECTORS = (
    "Information & Communication Technology",
    "Personal Products & Pharmaceuticals",
    "Industrial Materials & Machinery",
    "Petrochemicals & Chemicals",
    "Transportation & Logistics",
    "Health Care Services",
    "Construction Materials",
    "Finance & Securities",
    "Food & Beverage",
    "Media & Publishing",
    "Energy & Utilities",
    "Tourism & Leisure",
    "Electronic Components",
    "Construction Services",
    "Home & Office Products",
    "Property Development",
    "Professional Services",
    "Agribusiness",
    "Automotive",
    "Banking",
    "Insurance",
    "Commerce",
    "Packaging",
    "Fashion",
    "Steel",
)

SYMBOL_PATTERN = re.compile(r"[A-Za-z0-9&.-]{1,12}")
INTEGER_PATTERN = re.compile(r"\d{1,3}")
MODERN_SET50_HEADING = re.compile(
    r"SET50\s*/\s*SET50FF\s+Index\s+Constituents",
    re.IGNORECASE,
)
MODERN_SET100_HEADING = re.compile(
    r"SET100\s*/\s*SET100FF\s+Index\s+Constituents",
    re.IGNORECASE,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    _atomic_bytes(path, payload.encode("utf-8"))


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _canonical_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace(" and ", " & ")
    text = re.sub(r"\s*&\s*", " & ", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_sector(value: object) -> str:
    text = _canonical_text(value)
    matches = [sector for sector in SECTORS if text.casefold() == sector.casefold()]
    if len(matches) != 1:
        raise ValueError(f"unrecognized SET sector label: {value!r}")
    return matches[0]


def _group_words_by_line(
    words: Iterable[dict[str, Any]],
    *,
    tolerance: float = 2.6,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        for group in groups:
            if abs(float(group[0]["top"]) - float(word["top"])) < tolerance:
                group.append(word)
                break
        else:
            groups.append([word])
    return [sorted(group, key=lambda item: item["x0"]) for group in groups]


def _extract_sector(
    words: list[dict[str, Any]],
    *,
    x_min: float,
    x_max: float,
) -> tuple[str, float]:
    right_side = [
        word
        for word in words
        if float(word["x0"]) >= x_min + (x_max - x_min) * 0.45
    ]
    for line in _group_words_by_line(right_side):
        for index, word in enumerate(line):
            tail = _canonical_text(
                " ".join(str(item["text"]) for item in line[index:])
            )
            for sector in sorted(SECTORS, key=len, reverse=True):
                if tail.casefold() == sector.casefold():
                    return sector, float(word["x0"])
    raise ValueError("could not identify a recognized sector in PDF row")


def _header_top(words: list[dict[str, Any]]) -> float:
    number_headers = [
        word
        for word in words
        if str(word["text"]).lower().rstrip(".") == "no"
    ]
    candidate_tops = sorted({round(float(word["top"]), 1) for word in number_headers})
    for top in candidate_tops:
        if sum(abs(float(word["top"]) - top) < 1.0 for word in number_headers) >= 2:
            return top
    raise ValueError("could not locate the two-column SET table header")


def _parse_legacy_page(
    page: Any,
    *,
    expected_count: int,
) -> dict[int, dict[str, str]]:
    words = page.extract_words(
        x_tolerance=0.5,
        y_tolerance=2,
        keep_blank_chars=False,
    )
    header_top = _header_top(words)
    split = float(page.width) / 2.0
    records: dict[int, dict[str, str]] = {}

    for x_min, x_max in ((0.0, split), (split, float(page.width))):
        number_words = [
            word
            for word in words
            if x_min <= float(word["x0"]) < x_max
            and float(word["top"]) > header_top + 3.0
            and INTEGER_PATTERN.fullmatch(str(word["text"]))
            and 1 <= int(word["text"]) <= expected_count
        ]
        first_anchor_by_rank: dict[int, dict[str, Any]] = {}
        for word in sorted(number_words, key=lambda item: item["top"]):
            first_anchor_by_rank.setdefault(int(word["text"]), word)
        anchors = sorted(
            first_anchor_by_rank.values(),
            key=lambda item: item["top"],
        )

        for index, anchor in enumerate(anchors):
            rank = int(anchor["text"])
            previous = anchors[index - 1] if index else None
            following = anchors[index + 1] if index + 1 < len(anchors) else None
            band_top = (
                (float(previous["top"]) + float(anchor["top"])) / 2.0
                if previous is not None
                else header_top + 3.0
            )
            if (
                following is not None
                and int(following["text"]) == rank + 1
            ):
                band_bottom = (
                    float(anchor["top"]) + float(following["top"])
                ) / 2.0
            else:
                prior_top = (
                    float(previous["top"])
                    if previous is not None
                    else header_top
                )
                band_bottom = (
                    float(anchor["top"])
                    + (float(anchor["top"]) - prior_top) * 0.75
                )
            band = [
                word
                for word in words
                if x_min <= float(word["x0"]) < x_max
                and band_top <= float(word["top"]) < band_bottom
            ]
            symbol_candidates = [
                word
                for word in band
                if float(word["x0"]) > float(anchor["x1"])
                and abs(float(word["top"]) - float(anchor["top"])) < 2.0
            ]
            if not symbol_candidates:
                continue
            symbol_word = min(
                symbol_candidates,
                key=lambda item: item["x0"],
            )
            symbol = str(symbol_word["text"]).upper()
            if not SYMBOL_PATTERN.fullmatch(symbol):
                continue
            try:
                sector, sector_x = _extract_sector(
                    band,
                    x_min=x_min,
                    x_max=x_max,
                )
            except ValueError:
                continue
            company_words = [
                word
                for word in band
                if word is not anchor
                and word is not symbol_word
                and float(word["x0"]) > float(symbol_word["x1"]) + 0.5
                and float(word["x0"]) < sector_x - 1.0
            ]
            company = _canonical_text(
                " ".join(
                    str(word["text"])
                    for word in sorted(
                        company_words,
                        key=lambda item: (item["top"], item["x0"]),
                    )
                )
            )
            if not company:
                continue
            records.setdefault(
                rank,
                {
                    "symbol": symbol,
                    "company_name": company,
                    "sector": sector,
                },
            )
    return records


def _parse_modern_document(
    pdf: Any,
) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    parsed: dict[str, dict[int, dict[str, str]]] = {
        "SET50": {},
        "SET100": {},
    }
    mode: str | None = None
    for page in pdf.pages:
        text = page.extract_text() or ""
        if MODERN_SET50_HEADING.search(text):
            mode = "SET50"
        if MODERN_SET100_HEADING.search(text):
            mode = "SET100"
        if mode is None:
            continue
        expected_count = 50 if mode == "SET50" else 100
        words = page.extract_words(
            x_tolerance=0.5,
            y_tolerance=2,
            keep_blank_chars=False,
        )
        rank_words = [
            word
            for word in words
            if float(word["x0"]) < float(page.width) * 0.12
            and INTEGER_PATTERN.fullmatch(str(word["text"]))
            and 1 <= int(word["text"]) <= expected_count
        ]
        for anchor in sorted(rank_words, key=lambda item: item["top"]):
            rank = int(anchor["text"])
            row = [
                word
                for word in words
                if abs(float(word["top"]) - float(anchor["top"])) < 2.5
            ]
            symbol_candidates = [
                word
                for word in row
                if float(word["x0"]) > float(anchor["x1"])
            ]
            if not symbol_candidates:
                continue
            symbol_word = min(
                symbol_candidates,
                key=lambda item: item["x0"],
            )
            symbol = str(symbol_word["text"]).upper()
            if not SYMBOL_PATTERN.fullmatch(symbol):
                continue
            try:
                sector, sector_x = _extract_sector(
                    row,
                    x_min=0.0,
                    x_max=float(page.width),
                )
            except ValueError:
                continue
            company_words = [
                word
                for word in row
                if float(word["x0"]) > float(symbol_word["x1"]) + 0.5
                and float(word["x0"]) < sector_x - 1.0
            ]
            company = _canonical_text(
                " ".join(
                    str(word["text"])
                    for word in sorted(
                        company_words,
                        key=lambda item: item["x0"],
                    )
                )
            )
            if company:
                parsed[mode].setdefault(
                    rank,
                    {
                        "symbol": symbol,
                        "company_name": company,
                        "sector": sector,
                    },
                )
    return parsed["SET50"], parsed["SET100"]


def _validate_ranked_records(
    records: dict[int, dict[str, str]],
    *,
    index_name: str,
    expected_count: int,
) -> None:
    expected_ranks = set(range(1, expected_count + 1))
    actual_ranks = set(records)
    if actual_ranks != expected_ranks:
        missing = sorted(expected_ranks - actual_ranks)
        extra = sorted(actual_ranks - expected_ranks)
        raise ValueError(
            f"{index_name} rank contract failed; missing={missing} extra={extra}"
        )
    symbols = [record["symbol"] for record in records.values()]
    if len(symbols) != len(set(symbols)):
        raise ValueError(f"{index_name} contains duplicate symbols")
    for rank, record in records.items():
        symbol = record["symbol"]
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise ValueError(f"{index_name} rank {rank} has invalid symbol {symbol!r}")
        if not record["company_name"]:
            raise ValueError(f"{index_name} rank {rank} has a blank company")
        record["sector"] = _canonical_sector(record["sector"])


def parse_set_constituent_pdf(
    path: Path,
    *,
    expected_set50_count: int = 50,
    expected_set100_count: int = 100,
) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]], int]:
    try:
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            if page_count > 2:
                set50, set100 = _parse_modern_document(pdf)
            else:
                if page_count != 2:
                    raise ValueError(
                        f"legacy SET document must have two pages, found {page_count}"
                    )
                set50 = _parse_legacy_page(
                    pdf.pages[0],
                    expected_count=expected_set50_count,
                )
                set100 = _parse_legacy_page(
                    pdf.pages[1],
                    expected_count=expected_set100_count,
                )
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"could not read SET constituent PDF {path}: {exc}") from exc

    _validate_ranked_records(
        set50,
        index_name="SET50",
        expected_count=expected_set50_count,
    )
    _validate_ranked_records(
        set100,
        index_name="SET100",
        expected_count=expected_set100_count,
    )
    set50_symbols = {record["symbol"] for record in set50.values()}
    set100_symbols = {record["symbol"] for record in set100.values()}
    if not set50_symbols.issubset(set100_symbols):
        raise ValueError(
            "SET50 is not a subset of SET100: "
            + ", ".join(sorted(set50_symbols - set100_symbols))
        )
    set100_by_symbol = {
        record["symbol"]: record
        for record in set100.values()
    }
    mismatched_sectors = sorted(
        symbol
        for symbol in set50_symbols
        if set50_by_symbol_sector(set50, symbol)
        != set100_by_symbol[symbol]["sector"]
    )
    if mismatched_sectors:
        raise ValueError(
            "SET50/SET100 sector labels disagree for: "
            + ", ".join(mismatched_sectors)
        )
    return set50, set100, page_count


def set50_by_symbol_sector(
    records: dict[int, dict[str, str]],
    symbol: str,
) -> str:
    for record in records.values():
        if record["symbol"] == symbol:
            return record["sector"]
    raise KeyError(symbol)


def load_source_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_columns = [
        column
        for column in REQUIRED_MANIFEST_COLUMNS
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "SET constituent source manifest is missing columns: "
            + ", ".join(missing_columns)
        )
    frame = frame[list(REQUIRED_MANIFEST_COLUMNS)].copy()
    if frame.empty:
        raise ValueError("SET constituent source manifest is empty")
    if frame["record_id"].duplicated().any():
        raise ValueError("SET constituent source manifest has duplicate record IDs")
    for column in (
        "expected_set50_count",
        "expected_set100_count",
        "source_bytes",
    ):
        try:
            frame[column] = frame[column].astype(int)
        except ValueError as exc:
            raise ValueError(f"manifest column {column} must be integral") from exc
    for column in (
        "effective_from",
        "effective_to_exclusive",
        "published_at",
    ):
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"manifest column {column} contains invalid dates")
        frame[column] = parsed.dt.date.astype(str)
    retrieval = pd.to_datetime(frame["retrieved_at_utc"], errors="coerce", utc=True)
    if retrieval.isna().any():
        raise ValueError("manifest has invalid retrieval timestamps")
    frame["retrieved_at_utc"] = retrieval.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame = frame.sort_values(
        ["effective_from", "record_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    validate_manifest_intervals(frame)
    return frame


def validate_manifest_intervals(frame: pd.DataFrame) -> None:
    starts = pd.to_datetime(frame["effective_from"])
    ends = pd.to_datetime(frame["effective_to_exclusive"])
    published = pd.to_datetime(frame["published_at"])
    if (starts >= ends).any():
        raise ValueError("manifest contains a non-positive membership interval")
    if (published > starts).any():
        raise ValueError(
            "manifest would apply a constituent record before it was published"
        )
    for index in range(1, len(frame)):
        previous_end = ends.iloc[index - 1]
        current_start = starts.iloc[index]
        if previous_end != current_start:
            relationship = "gap" if previous_end < current_start else "overlap"
            raise ValueError(
                f"manifest interval {relationship}: "
                f"{previous_end.date()} -> {current_start.date()}"
            )
    record_ids = set(frame["record_id"])
    for row in frame.itertuples(index=False):
        if row.revision_status not in {"original", "revised"}:
            raise ValueError(
                f"unsupported revision status for {row.record_id}: "
                f"{row.revision_status}"
            )
        if row.revision_status == "revised":
            if not row.supersedes_record_id:
                raise ValueError(
                    f"revised record {row.record_id} has no predecessor"
                )
            if row.supersedes_record_id not in record_ids:
                raise ValueError(
                    f"revised record {row.record_id} supersedes an unknown record"
                )
            if row.effective_date_basis != "publication_date_conservative":
                raise ValueError(
                    f"revised record {row.record_id} is not publication-gated"
                )
    if starts.iloc[0] != pd.Timestamp("2021-01-01"):
        raise ValueError("manifest must start at 2021-01-01")
    if ends.iloc[-1] != pd.Timestamp("2025-01-01"):
        raise ValueError("manifest must end at 2025-01-01 exclusive")


def _resolve_source_file(project_root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError("source_file_path must be project-relative")
    candidate = project_root / relative_path
    current = project_root
    for part in Path(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"source path contains a symlink: {relative_path}")
    resolved_root = project_root.resolve()
    resolved = candidate.resolve(strict=True)
    if resolved_root not in resolved.parents:
        raise ValueError(f"source path escapes the project root: {relative_path}")
    if not resolved.is_file():
        raise ValueError(f"source path is not a regular file: {relative_path}")
    return resolved


def _validate_source_file(
    project_root: Path,
    row: Any,
) -> tuple[Path, str, int]:
    source_path = _resolve_source_file(project_root, row.source_file_path)
    size = source_path.stat().st_size
    digest = sha256_file(source_path)
    if size != int(row.source_bytes):
        raise ValueError(
            f"source byte mismatch for {row.record_id}: "
            f"{size} != {row.source_bytes}"
        )
    if digest != row.source_file_sha256:
        raise ValueError(
            f"source hash mismatch for {row.record_id}: "
            f"{digest} != {row.source_file_sha256}"
        )
    return source_path, digest, size


def build_constituent_archive(
    project_root: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    project_root = project_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = load_source_manifest(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    membership_rows: list[dict[str, Any]] = []
    source_profiles: list[dict[str, Any]] = []
    interval_profiles: list[dict[str, Any]] = []
    previous_set50: set[str] | None = None
    previous_set100: set[str] | None = None

    for source_row in manifest.itertuples(index=False):
        source_path, source_sha256, source_bytes = _validate_source_file(
            project_root,
            source_row,
        )
        set50, set100, page_count = parse_set_constituent_pdf(
            source_path,
            expected_set50_count=int(source_row.expected_set50_count),
            expected_set100_count=int(source_row.expected_set100_count),
        )
        set50_symbols = {record["symbol"] for record in set50.values()}
        set100_by_symbol = {
            record["symbol"]: record
            for record in set100.values()
        }
        set100_symbols = set(set100_by_symbol)
        for rank in sorted(set100):
            record = set100[rank]
            symbol = record["symbol"]
            membership_rows.append(
                {
                    "record_id": source_row.record_id,
                    "period_label": source_row.period_label,
                    "effective_from": source_row.effective_from,
                    "effective_to_exclusive": source_row.effective_to_exclusive,
                    "published_at": source_row.published_at,
                    "effective_date_basis": source_row.effective_date_basis,
                    "revision_status": source_row.revision_status,
                    "symbol": symbol,
                    "ticker": f"{symbol}.BK",
                    "company_name": record["company_name"],
                    "sector": record["sector"],
                    "in_set50": symbol in set50_symbols,
                    "in_set100": True,
                    "source_url": source_row.source_url,
                    "source_file_path": source_row.source_file_path,
                    "source_file_sha256": source_sha256,
                    "retrieved_at_utc": source_row.retrieved_at_utc,
                }
            )

        set50_added = (
            0
            if previous_set50 is None
            else len(set50_symbols - previous_set50)
        )
        set50_removed = (
            0
            if previous_set50 is None
            else len(previous_set50 - set50_symbols)
        )
        set100_added = (
            0
            if previous_set100 is None
            else len(set100_symbols - previous_set100)
        )
        set100_removed = (
            0
            if previous_set100 is None
            else len(previous_set100 - set100_symbols)
        )
        interval_profiles.append(
            {
                "record_id": source_row.record_id,
                "period_label": source_row.period_label,
                "effective_from": source_row.effective_from,
                "effective_to_exclusive": source_row.effective_to_exclusive,
                "published_at": source_row.published_at,
                "revision_status": source_row.revision_status,
                "set50_count": len(set50_symbols),
                "set100_count": len(set100_symbols),
                "set50_added": set50_added,
                "set50_removed": set50_removed,
                "set50_turnover": set50_added + set50_removed,
                "set100_added": set100_added,
                "set100_removed": set100_removed,
                "set100_turnover": set100_added + set100_removed,
                "sector_count": len(
                    {record["sector"] for record in set100_by_symbol.values()}
                ),
                "set50_subset_of_set100": True,
            }
        )
        source_profiles.append(
            {
                "record_id": source_row.record_id,
                "period_label": source_row.period_label,
                "source_url": source_row.source_url,
                "source_file_path": source_row.source_file_path,
                "sha256": source_sha256,
                "bytes": source_bytes,
                "page_count": page_count,
                "retrieved_at_utc": source_row.retrieved_at_utc,
            }
        )
        previous_set50 = set50_symbols
        previous_set100 = set100_symbols

    membership = pd.DataFrame(membership_rows, columns=MEMBERSHIP_COLUMNS)
    membership = membership.sort_values(
        ["effective_from", "symbol"],
        kind="mergesort",
    ).reset_index(drop=True)
    if membership.duplicated(["record_id", "symbol"]).any():
        raise ValueError("normalized archive contains duplicate interval/symbol rows")
    if membership[list(MEMBERSHIP_COLUMNS)].isna().any().any():
        raise ValueError("normalized archive contains missing values")
    if not membership["ticker"].str.fullmatch(r"[A-Z0-9&.-]+\.BK").all():
        raise ValueError("normalized archive contains malformed .BK ticker mappings")

    first_seen = membership.groupby("ticker")["effective_from"].min()
    last_seen = membership.groupby("ticker")["effective_to_exclusive"].max()
    latest_rows = (
        membership.sort_values(
            ["ticker", "effective_from"],
            kind="mergesort",
        )
        .groupby("ticker", as_index=False)
        .tail(1)
        .set_index("ticker")
    )
    sector_versions = membership.groupby("ticker")["sector"].nunique()
    union_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    for ticker in sorted(membership["ticker"].unique()):
        latest = latest_rows.loc[ticker]
        consistent = int(sector_versions.loc[ticker]) == 1
        union_rows.append(
            {
                "ticker": ticker,
                "symbol": latest["symbol"],
                "usage": "acquisition_only_not_static_model_membership",
                "first_effective_from": first_seen.loc[ticker],
                "last_effective_to_exclusive": last_seen.loc[ticker],
                "sector": latest["sector"],
                "sector_history_consistent": consistent,
            }
        )
        sector_rows.append(
            {
                "ticker": ticker,
                "sector": latest["sector"],
                "source": "Stock Exchange of Thailand constituent archive",
                "coverage_start": first_seen.loc[ticker],
                "coverage_end_exclusive": last_seen.loc[ticker],
                "usage": "latest_reference_only_use_membership_archive_for_as_of_joins",
                "sector_history_consistent": consistent,
            }
        )
    acquisition_universe = pd.DataFrame(union_rows)
    sector_mapping = pd.DataFrame(sector_rows)
    readiness = pd.DataFrame(interval_profiles)
    changed_sector_tickers = sorted(
        ticker
        for ticker, count in sector_versions.items()
        if int(count) > 1
    )
    receipt = {
        "schema_version": 1,
        "source": "Stock Exchange of Thailand official constituent archive",
        "source_archive_page": (
            "https://www.set.or.th/en/market/information/securities-list/"
            "constituents-list-set50-set100"
        ),
        "retrieved_at_utc": sorted(manifest["retrieved_at_utc"].unique())[-1],
        "source_manifest": {
            "path": str(manifest_path.relative_to(project_root)),
            "sha256": sha256_bytes(manifest_bytes),
            "bytes": len(manifest_bytes),
            "rows": len(manifest),
        },
        "coverage": {
            "effective_from": membership["effective_from"].min(),
            "effective_to_exclusive": membership[
                "effective_to_exclusive"
            ].max(),
            "interval_count": int(membership["record_id"].nunique()),
            "normalized_rows": len(membership),
            "unique_tickers": int(membership["ticker"].nunique()),
            "sector_count": int(membership["sector"].nunique()),
            "raw_file_count": len(source_profiles),
            "raw_bytes": sum(profile["bytes"] for profile in source_profiles),
        },
        "quality_gates": {
            "source_hashes_match_manifest": True,
            "source_bytes_match_manifest": True,
            "continuous_intervals": True,
            "exactly_50_set50_per_interval": True,
            "exactly_100_set100_per_interval": True,
            "set50_subset_of_set100": True,
            "no_duplicate_interval_symbols": True,
            "no_blank_companies_or_sectors": True,
            "valid_bk_ticker_mapping": True,
            "revisions_not_applied_before_publication": True,
        },
        "eligibility": {
            "structural_source_ready": True,
            "point_in_time_membership_eligible": True,
            "point_in_time_sector_history_eligible": True,
            "model_input_eligible": False,
            "static_union_model_eligible": False,
            "publication_grade_performance_claims_eligible": False,
        },
        "sector_history": {
            "consistent_ticker_count": int(
                (sector_versions == 1).sum()
            ),
            "changed_ticker_count": len(changed_sector_tickers),
            "changed_tickers": changed_sector_tickers,
            "static_mapping_rule": (
                "latest official sector label; use dated membership archive "
                "for all as-of joins"
            ),
        },
        "limitations": [
            (
                "The two between-review revisions do not state a separate "
                "exchange-effective date in the archived PDF. They are applied "
                "from publication date, preventing future-information leakage "
                "but not claiming an exact corporate-action transition date."
            ),
            (
                "The union file is for price acquisition only. It must never be "
                "used as a static model universe because that would reintroduce "
                "survivorship and future-membership bias."
            ),
            (
                "No licensed historical OHLCV is included. The current Yahoo "
                "pilot remains private-research-only and cannot support "
                "publication-grade Thai-market performance claims."
            ),
            (
                "The active five-ticker V2 preregistration and deployed release "
                "remain unchanged. A broad-universe experiment requires a new "
                "preregistration and date-aware investability mask."
            ),
        ],
        "raw_files": source_profiles,
        "intervals": interval_profiles,
    }
    return (
        membership,
        acquisition_universe,
        sector_mapping,
        readiness,
        receipt,
    )


def _output_profile(
    project_root: Path,
    path: Path,
    payload: bytes,
    *,
    rows: int,
) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(project_root)),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "rows": rows,
    }


def write_constituent_archive(
    *,
    project_root: Path,
    membership: pd.DataFrame,
    acquisition_universe: pd.DataFrame,
    sector_mapping: pd.DataFrame,
    readiness: pd.DataFrame,
    receipt: dict[str, Any],
    membership_path: Path,
    universe_path: Path,
    sector_path: Path,
    readiness_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    outputs = {
        "membership_history": (
            membership_path,
            _csv_bytes(membership),
            len(membership),
        ),
        "acquisition_universe": (
            universe_path,
            _csv_bytes(acquisition_universe),
            len(acquisition_universe),
        ),
        "sector_mapping": (
            sector_path,
            _csv_bytes(sector_mapping),
            len(sector_mapping),
        ),
        "interval_readiness": (
            readiness_path,
            _csv_bytes(readiness),
            len(readiness),
        ),
    }
    receipt["outputs"] = {
        name: _output_profile(
            project_root,
            path,
            payload,
            rows=rows,
        )
        for name, (path, payload, rows) in outputs.items()
    }
    for path, payload, _ in outputs.values():
        _atomic_bytes(path, payload)
    _atomic_json(receipt_path, receipt)
    receipt_digest = sha256_file(receipt_path)
    sidecar = receipt_path.with_name(receipt_path.name + ".sha256")
    _atomic_bytes(
        sidecar,
        f"{receipt_digest}  {receipt_path.name}\n".encode("utf-8"),
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a checksum-bound, point-in-time SET50/SET100 constituent "
            "archive from preserved official PDFs."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--manifest",
        default="data/external/set_constituent_source_manifest.csv",
    )
    parser.add_argument(
        "--membership-output",
        default="data/reference/set50_set100_membership_2021_2024.csv",
    )
    parser.add_argument(
        "--universe-output",
        default="data/reference/set50_or_set100_universe.csv",
    )
    parser.add_argument(
        "--sector-output",
        default="data/reference/sector_mapping.csv",
    )
    parser.add_argument(
        "--readiness-output",
        default="reports/set_constituent_archive_readiness.csv",
    )
    parser.add_argument(
        "--receipt",
        default="reports/data_sources/set_constituent_archive_20260731.json",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest_path = (project_root / args.manifest).resolve()
    (
        membership,
        acquisition_universe,
        sector_mapping,
        readiness,
        receipt,
    ) = build_constituent_archive(project_root, manifest_path)
    receipt = write_constituent_archive(
        project_root=project_root,
        membership=membership,
        acquisition_universe=acquisition_universe,
        sector_mapping=sector_mapping,
        readiness=readiness,
        receipt=receipt,
        membership_path=(project_root / args.membership_output).resolve(),
        universe_path=(project_root / args.universe_output).resolve(),
        sector_path=(project_root / args.sector_output).resolve(),
        readiness_path=(project_root / args.readiness_output).resolve(),
        receipt_path=(project_root / args.receipt).resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
