"""Build the member-to-price-series map for the point-in-time SET50/SET100 lists (milestones M3, M8).

For every list and member of ``data/reference/set50_set100_membership_2012_2024.csv``, records which
price series to use. It replaces the one-off mapping step of 2026-11-07 and reproduces its output byte
for byte:

- ``own_symbol``: the member's own series spans the list's dates within ``--tolerance-days`` (10) at each
  end;
- ``current_symbol``: otherwise, the series of its current symbol from ``symbol_map.csv`` (a rename, or a
  holding company that replaced it through a share swap) spans them;
- ``partial_own_symbol`` / ``partial_current_symbol``: no series spans the list, but one overlaps it. The
  session mask treats these lists as uncovered;
- ``none``: no usable series.

A member's own series that begins after the member's last list is another security that reused the symbol
(KK.BK on Yahoo is not the Kiatnakin Bank of 2012–2013). It is never used, and the receipt lists it.

Coverage comes from the per-ticker coverage files written by the price downloads (first and last date of
each series); ``price_file`` is the price file of the chosen series, relative to ``data/raw/extended``.

    python -m scripts.build_member_price_map            # rebuild the map and its receipt
    python -m scripts.build_member_price_map --check    # compare with the existing map; write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd

EXTENDED = Path("data/raw/extended")
ADDITIONS = EXTENDED / "set100_2012_2024_additions"
DEFAULT_COVERAGE = (
    f"{EXTENDED / 'coverage_set100_union_2012_2024.csv'}=prices_set100_union_2012_2024.csv.gz",
    f"{ADDITIONS / 'coverage_set100_2012_2024_additions.csv'}=set100_2012_2024_additions/prices_set100_2012_2024_additions.csv.gz",
)
COLUMNS = ["record_id", "effective_from", "effective_to_exclusive", "ticker", "in_set50", "price_ticker", "coverage",
           "price_file"]
RULE = ("a member-list is covered when one price series spans its dates within {days} days; current_symbol uses "
        "symbol_map.csv; series that begin after the ticker's last membership are another security and are never used")


def load_coverage(pairs: list[str], root: Path) -> pd.DataFrame:
    """``coverage.csv=price_file`` pairs -> one row per ticker with its first and last date and price file."""

    frames = []
    for pair in pairs:
        path, _, price_file = pair.partition("=")
        if not price_file:
            raise ValueError(f"coverage argument {pair!r} must be <coverage csv>=<price file>")
        frame = pd.read_csv(root / path, usecols=["ticker", "first_date", "last_date"])
        frames.append(frame.assign(price_file=price_file))
    coverage = pd.concat(frames, ignore_index=True)
    duplicated = coverage["ticker"][coverage["ticker"].duplicated()]
    if len(duplicated):
        raise ValueError(f"tickers in more than one coverage file: {sorted(duplicated)}")
    coverage["first_date"] = pd.to_datetime(coverage["first_date"])
    coverage["last_date"] = pd.to_datetime(coverage["last_date"])
    return coverage.set_index("ticker")


def build_map(membership: pd.DataFrame, coverage: pd.DataFrame, symbol_map: pd.Series,
              tolerance_days: int = 10) -> tuple[pd.DataFrame, list[str]]:
    tolerance = pd.Timedelta(days=tolerance_days)
    last_list_end = pd.to_datetime(membership.groupby("ticker")["effective_to_exclusive"].max())
    reused: set[str] = set()
    rows = []
    for row in membership.itertuples(index=False):
        start, end = pd.Timestamp(row.effective_from), pd.Timestamp(row.effective_to_exclusive)
        candidates = []
        if row.ticker in coverage.index:
            if coverage.at[row.ticker, "first_date"] > last_list_end[row.ticker]:
                reused.add(row.ticker)
            else:
                candidates.append(("own_symbol", row.ticker))
        current = symbol_map.get(row.ticker)
        if current is not None and current in coverage.index:
            candidates.append(("current_symbol", current))

        def spans(ticker: str) -> bool:
            return (coverage.at[ticker, "first_date"] <= start + tolerance
                    and coverage.at[ticker, "last_date"] >= end - tolerance)

        def overlaps(ticker: str) -> bool:
            return coverage.at[ticker, "first_date"] < end and coverage.at[ticker, "last_date"] >= start

        choice = next(((kind, t) for kind, t in candidates if spans(t)), None)
        if choice is None:
            choice = next(((f"partial_{kind}", t) for kind, t in candidates if overlaps(t)), ("none", ""))
        kind, ticker = choice
        rows.append({
            "record_id": row.record_id,
            "effective_from": row.effective_from,
            "effective_to_exclusive": row.effective_to_exclusive,
            "ticker": row.ticker,
            "in_set50": bool(row.in_set50),
            "price_ticker": ticker,
            "coverage": kind,
            "price_file": coverage.at[ticker, "price_file"] if ticker else "",
        })
    return pd.DataFrame(rows, columns=COLUMNS), sorted(reused)


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--membership", default="data/reference/set50_set100_membership_2012_2024.csv")
    parser.add_argument("--coverage", nargs="+", default=list(DEFAULT_COVERAGE),
                        help="<coverage csv>=<price file relative to data/raw/extended>, one per price download")
    parser.add_argument("--symbol-map", default=str(ADDITIONS / "symbol_map.csv"))
    parser.add_argument("--tolerance-days", type=int, default=10)
    parser.add_argument("--output", default=str(ADDITIONS / "member_price_map_2012_2024.csv"))
    parser.add_argument("--check", action="store_true", help="compare with the existing output and write nothing")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root)
    membership = pd.read_csv(root / args.membership, usecols=["record_id", "effective_from", "effective_to_exclusive",
                                                              "ticker", "in_set50"])
    coverage = load_coverage(args.coverage, root)
    symbols = pd.read_csv(root / args.symbol_map).set_index("member_ticker")["price_ticker"]
    frame, reused = build_map(membership, coverage, symbols, args.tolerance_days)
    data = to_csv_bytes(frame)
    digest = hashlib.sha256(data).hexdigest()
    output = root / args.output
    counts = frame["coverage"].value_counts().to_dict()
    print(f"{len(frame)} member-lists: {counts}; reused symbols excluded: {reused}; sha256 {digest}")

    if args.check:
        if not output.exists():
            raise SystemExit(f"--check: {output} does not exist")
        existing = output.read_bytes()
        if existing != data:
            old = pd.read_csv(output, dtype=str, keep_default_na=False)
            new = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
            differ = (old.values != new.values).any(axis=1) if old.shape == new.shape else None
            detail = f"{int(differ.sum())} rows differ" if differ is not None else f"shapes {old.shape} vs {new.shape}"
            raise SystemExit(f"--check: the rebuilt map differs from {output} ({detail})")
        print(f"--check: identical to {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, output)
    receipt = {
        "generated_from": [args.membership, *(pair.partition("=")[0] for pair in args.coverage), args.symbol_map],
        "rule": RULE.format(days=args.tolerance_days),
        "reused_symbols_excluded": reused,
        "rows": len(frame),
        "sha256": digest,
        "generated_by": "scripts/build_member_price_map.py",
    }
    receipt_path = output.with_name(output.name + ".receipt.json")
    tmp = receipt_path.with_name(receipt_path.name + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(tmp, receipt_path)
    print(f"wrote {output} and {receipt_path.name}")


if __name__ == "__main__":
    main()
