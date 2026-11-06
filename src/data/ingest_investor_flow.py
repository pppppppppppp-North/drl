"""Normalize SET daily trading value by investor type.

The Stock Exchange of Thailand publishes daily buy/sell value for four investor groups: foreign
investors, local institutions, proprietary (securities-company) trading, and local individuals.
This importer accepts a provider or SET export in either long form::

    date, investor_type, buy_value, sell_value

or wide form (``foreign_buy, foreign_sell, local_institution_buy, ...``) and writes a validated long
table with ``net_value`` and a checksum sidecar. English and Thai group names are recognized.

Example::

    python -m src.data.ingest_investor_flow --input data/external/original/set_investor_type.csv \
        --output data/external/set_investor_flow.csv --source-name set_investor_type_export
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

INVESTOR_TYPES = ("foreign", "local_institution", "proprietary", "local_individual")
ALIASES = {
    "foreign": "foreign",
    "foreign investors": "foreign",
    "foreign investor": "foreign",
    "นักลงทุนต่างประเทศ": "foreign",
    "local institution": "local_institution",
    "local institutions": "local_institution",
    "local_institution": "local_institution",
    "local institutional investors": "local_institution",
    "สถาบันในประเทศ": "local_institution",
    "นักลงทุนสถาบันในประเทศ": "local_institution",
    "proprietary": "proprietary",
    "proprietary trading": "proprietary",
    "securities company": "proprietary",
    "บัญชีบริษัทหลักทรัพย์": "proprietary",
    "local individual": "local_individual",
    "local individuals": "local_individual",
    "local_individual": "local_individual",
    "local investors": "local_individual",
    "individual": "local_individual",
    "retail": "local_individual",
    "นักลงทุนทั่วไปในประเทศ": "local_individual",
}
OUTPUT_COLUMNS = ["date", "investor_type", "buy_value", "sell_value", "net_value", "source_name"]


def _canonical_type(value: object) -> str:
    key = str(value).strip().lower().replace("-", " ")
    key = " ".join(key.split())
    if key in ALIASES:
        return ALIASES[key]
    raise ValueError(f"unrecognized investor type {value!r}; expected one of {INVESTOR_TYPES}")


def _wide_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for investor in INVESTOR_TYPES:
        buy, sell = f"{investor}_buy", f"{investor}_sell"
        if buy in frame.columns and sell in frame.columns:
            rows.append(
                pd.DataFrame(
                    {"date": frame["date"], "investor_type": investor, "buy_value": frame[buy], "sell_value": frame[sell]}
                )
            )
    if not rows:
        raise ValueError("wide investor-flow input needs <type>_buy and <type>_sell columns")
    return pd.concat(rows, ignore_index=True)


def normalize_investor_flow(frame: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    if "date" not in data.columns:
        raise ValueError("investor-flow input needs a date column")
    if "investor_type" not in data.columns:
        data = _wide_to_long(data)
    missing = {"buy_value", "sell_value"} - set(data.columns)
    if missing:
        raise ValueError(f"investor-flow input is missing columns: {sorted(missing)}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        raise ValueError("investor-flow input contains unparseable dates")
    data["investor_type"] = data["investor_type"].map(_canonical_type)
    for column in ("buy_value", "sell_value"):
        data[column] = pd.to_numeric(
            data[column].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
        if data[column].isna().any():
            raise ValueError(f"investor-flow input has non-numeric {column}")
        if (data[column] < 0).any():
            raise ValueError(f"investor-flow input has negative {column}")
    if data.duplicated(["date", "investor_type"]).any():
        raise ValueError("investor-flow input has duplicate date/investor_type rows")
    counts = data.groupby("date")["investor_type"].nunique()
    incomplete = counts[counts != len(INVESTOR_TYPES)]
    if not incomplete.empty:
        raise ValueError(
            f"every date needs all four investor groups; incomplete dates: {[d.date().isoformat() for d in incomplete.index[:5]]}"
        )
    data["net_value"] = data["buy_value"] - data["sell_value"]
    imbalance = data.groupby("date")["net_value"].sum().abs()
    market_value = data.groupby("date")["buy_value"].sum()
    bad = imbalance[(imbalance > 0.01 * market_value) & (market_value > 0)]
    if not bad.empty:
        raise ValueError(
            "net investor flows must sum to about zero each day (every buy has a seller); "
            f"check units on {[d.date().isoformat() for d in bad.index[:5]]}"
        )
    data["source_name"] = source_name
    return data.sort_values(["date", "investor_type"]).reset_index(drop=True)[OUTPUT_COLUMNS]


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
    parser.add_argument("--source-name", required=True)
    args = parser.parse_args()

    input_path, output_path = Path(args.input), Path(args.output)
    raw = pd.read_csv(input_path)
    normalized = normalize_investor_flow(raw, source_name=args.source_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = normalized.assign(date=normalized["date"].dt.date)
    out.to_csv(output_path, index=False)
    receipt = {
        "source_name": args.source_name,
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "rows": int(len(out)),
        "dates": int(out["date"].nunique()),
        "start": str(out["date"].min()),
        "end": str(out["date"].max()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    output_path.with_suffix(output_path.suffix + ".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
