"""Download long-horizon daily history and seal a post-development holdout (milestone M2).

Development rows (``date <= --dev-end``) are written to ``--dev-output`` for model building.
Rows on or after ``--holdout-start`` are written to a separate gzip file under ``--holdout-dir`` with
a SHA-256 manifest marked ``SEALED``. The script prints only row counts and date ranges for the
holdout, never prices or returns, so nobody looks at holdout performance before the protocol is
locked. Prices use Yahoo Finance ``auto_adjust=True`` (split- and dividend-adjusted), matching
``src.data.ingest_ohlcv``; dividends and splits are kept as audit columns.

Example::

    python -m scripts.download_extended_history --tickers PTT.BK AOT.BK CPALL.BK KBANK.BK ADVANC.BK \
        --index TDEX.BK --index-name set50_etf --start 2012-01-01 --dev-end 2024-12-31 --holdout-start 2025-01-01 \
        --dev-output data/raw/extended/prices_2012_2024.csv \
        --index-dev-output data/raw/extended/index_set50_etf_2012_2024.csv \
        --holdout-dir data/holdout_sealed/2025_onward
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume", "dividends", "stock_splits"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [str(next(part for part in column if part)) for column in frame.columns]
    return frame


def download_one(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        symbol, start=start, end=end, auto_adjust=True, actions=True, progress=False, threads=False
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    frame = _flatten(raw).reset_index()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    if "datetime" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"datetime": "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame["ticker"] = symbol
    for column in ("dividends", "stock_splits"):
        if column not in frame.columns:
            frame[column] = 0.0
    frame = frame.dropna(subset=["close"])
    frame = frame[frame["close"] > 0]
    return frame[COLUMNS].sort_values("date").reset_index(drop=True)


def coverage(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "rows", "first_date", "last_date", "zero_volume_days"])
    grouped = frame.groupby("ticker")
    return pd.DataFrame(
        {
            "rows": grouped.size(),
            "first_date": grouped["date"].min().dt.date,
            "last_date": grouped["date"].max().dt.date,
            "zero_volume_days": grouped["volume"].apply(lambda s: int((s <= 0).sum())),
        }
    ).reset_index()


def _write_dev(frame: pd.DataFrame, path: Path, meta: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.assign(date=frame["date"].dt.date)
    out.to_csv(path, index=False)
    receipt = {
        **meta,
        "path": str(path),
        "rows": int(len(out)),
        "tickers": sorted(out["ticker"].unique().tolist()) if len(out) else [],
        "start": str(out["date"].min()) if len(out) else None,
        "end": str(out["date"].max()) if len(out) else None,
        "sha256": _sha256(path),
    }
    path.with_suffix(path.suffix + ".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _write_holdout(frames: dict[str, pd.DataFrame], holdout_dir: Path, meta: dict) -> dict:
    holdout_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, frame in frames.items():
        path = holdout_dir / f"{name}.csv.gz"
        if path.exists():
            raise FileExistsError(f"{path} already exists; a sealed holdout is never overwritten")
        with gzip.open(path, "wt", newline="") as handle:
            frame.assign(date=frame["date"].dt.date).to_csv(handle, index=False)
        files[name] = {
            "path": str(path),
            "rows": int(len(frame)),
            "start": str(frame["date"].min().date()) if len(frame) else None,
            "end": str(frame["date"].max().date()) if len(frame) else None,
            "sha256": _sha256(path),
        }
    manifest = {
        **meta,
        "status": "SEALED",
        "rule": (
            "Do not load, plot, or evaluate on these files until the protocol (reward, features, algorithm, "
            "hyperparameters, and evaluation metrics) is locked and approved; then evaluate exactly once."
        ),
        "files": files,
    }
    manifest_path = holdout_dir / "HOLDOUT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (holdout_dir / "HOLDOUT_MANIFEST.json.sha256").write_text(f"{_sha256(manifest_path)}  HOLDOUT_MANIFEST.json\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="*", default=[])
    parser.add_argument("--tickers-csv", default=None, help="CSV with a ticker column (for example the 138-ticker union)")
    parser.add_argument(
        "--index",
        default="TDEX.BK",
        help="market series; Yahoo returns only one row for ^SET.BK/^SET50.BK as of 2026-11, so the default is "
        "the investable ThaiDEX SET50 ETF (TDEX.BK)",
    )
    parser.add_argument("--index-name", default="set50_etf", help="index_name label used by market-context features")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--dev-end", default="2024-12-31")
    parser.add_argument("--holdout-start", default="2025-01-01")
    parser.add_argument("--end", default=None, help="exclusive download end date; default today")
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--index-dev-output", required=True)
    parser.add_argument("--holdout-dir", required=True)
    parser.add_argument("--coverage-output", default=None)
    parser.add_argument("--label", default="exploratory_yahoo", help="evidence tier label written to receipts")
    args = parser.parse_args()

    tickers = list(args.tickers)
    if args.tickers_csv:
        tickers += pd.read_csv(args.tickers_csv)["ticker"].astype(str).tolist()
    tickers = sorted(dict.fromkeys(t.strip() for t in tickers if t.strip()))
    if not tickers:
        raise SystemExit("no tickers given")
    if pd.Timestamp(args.holdout_start) <= pd.Timestamp(args.dev_end):
        raise SystemExit("--holdout-start must be after --dev-end")
    end = args.end or (date.today().isoformat())

    stock_frames = [download_one(t, args.start, end) for t in tickers]
    stocks = pd.concat([f for f in stock_frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in stock_frames
    ) else pd.DataFrame(columns=COLUMNS)
    index = download_one(args.index, args.start, end)
    if not index.empty:
        index = index.rename(columns={"ticker": "symbol"}).assign(index_name=args.index_name)

    meta = {
        "source": "Yahoo Finance via yfinance (auto_adjust=True, actions=True)",
        "evidence_tier": args.label,
        "requested_start": args.start,
        "download_end_exclusive": end,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    dev_end, holdout_start = pd.Timestamp(args.dev_end), pd.Timestamp(args.holdout_start)
    dev_receipt = _write_dev(stocks[stocks["date"] <= dev_end], Path(args.dev_output), meta)
    index_dev = index[index["date"] <= dev_end] if not index.empty else index
    if not index_dev.empty:
        index_dev = index_dev.rename(columns={"symbol": "ticker"})
    index_receipt = _write_dev(index_dev, Path(args.index_dev_output), meta) if not index_dev.empty else None
    holdout_frames = {"stocks": stocks[stocks["date"] >= holdout_start]}
    if not index.empty:
        holdout_frames["index"] = index[index["date"] >= holdout_start].rename(columns={"symbol": "ticker"})
    holdout = _write_holdout(holdout_frames, Path(args.holdout_dir), meta)

    cov = coverage(stocks[stocks["date"] <= dev_end])
    missing = sorted(set(tickers) - set(cov["ticker"]))
    if args.coverage_output:
        Path(args.coverage_output).parent.mkdir(parents=True, exist_ok=True)
        cov.to_csv(args.coverage_output, index=False)
    print(json.dumps({"development": dev_receipt, "index": index_receipt, "missing_tickers": missing}, indent=2, default=str))
    print(json.dumps({name: {k: v for k, v in f.items() if k != "sha256"} for name, f in holdout["files"].items()}, indent=2))


if __name__ == "__main__":
    main()
