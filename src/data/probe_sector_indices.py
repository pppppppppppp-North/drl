from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.ingest_ohlcv import _normalize_provider_frame, _yahoo_provider, write_manifest_row
from src.utils.config import ensure_dirs, load_config


SECTOR_CODES = {
    "agri": "AGRI",
    "food": "FOOD",
    "fashion": "FASHION",
    "home": "HOME",
    "person": "PERSON",
    "bank": "BANK",
    "fin": "FIN",
    "insur": "INSUR",
    "auto": "AUTO",
    "imm": "IMM",
    "paper": "PAPER",
    "petro": "PETRO",
    "pkg": "PKG",
    "steel": "STEEL",
    "conmat": "CONMAT",
    "prop": "PROP",
    "pfreit": "PFREIT",
    "cons": "CONS",
    "energ": "ENERG",
    "mine": "MINE",
    "comm": "COMM",
    "helth": "HELTH",
    "media": "MEDIA",
    "prof": "PROF",
    "tourism": "TOURISM",
    "trans": "TRANS",
    "etron": "ETRON",
    "ict": "ICT",
}


@dataclass(frozen=True)
class ProbeResult:
    sector: str
    symbol: str
    status: str
    rows: int
    start_date: str | None
    end_date: str | None
    error: str


def default_candidates() -> dict[str, list[str]]:
    candidates = {}
    for sector, code in SECTOR_CODES.items():
        candidates[sector] = [
            f"^SET{code}.BK",
            f"^SET{code}",
            f"^{code}.BK",
            f"^{code}",
            f"{code}.BK",
        ]
    return candidates


def _date_range(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if frame.empty or "date" not in frame.columns:
        return None, None
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.notna().sum() == 0:
        return None, None
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def probe_symbol(sector: str, symbol: str, start: str, end: str, min_rows: int) -> tuple[ProbeResult, pd.DataFrame | None]:
    try:
        frame = _normalize_provider_frame(_yahoo_provider(symbol, start, end), symbol)
        start_date, end_date = _date_range(frame)
        status = "usable" if len(frame) >= min_rows else "too_few_rows"
        result = ProbeResult(sector, symbol, status, len(frame), start_date, end_date, "")
        if status == "usable":
            prices = frame.copy()
            prices["sector"] = sector
            prices["source_symbol"] = symbol
            return result, prices
        return result, None
    except Exception as exc:  # noqa: BLE001 - source probes should record provider failures.
        return ProbeResult(sector, symbol, "failed", 0, None, None, str(exc)), None


def probe_sector_indices(
    candidates: dict[str, list[str]],
    *,
    start: str,
    end: str,
    min_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[ProbeResult] = []
    usable_frames: list[pd.DataFrame] = []
    for sector, symbols in candidates.items():
        for symbol in symbols:
            result, prices = probe_symbol(sector, symbol, start, end, min_rows)
            results.append(result)
            if prices is not None:
                usable_frames.append(prices)
                break
    result_frame = pd.DataFrame([result.__dict__ for result in results])
    prices_frame = (
        pd.concat(usable_frames, ignore_index=True).sort_values(["date", "sector"]).reset_index(drop=True)
        if usable_frames
        else pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "sector", "source_symbol"])
    )
    return result_frame, prices_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_sector.yaml")
    parser.add_argument("--probe-output", default="data/raw/sector_index_yahoo_probe.csv")
    parser.add_argument("--prices-output", default="data/raw/prices_sector_indices_yahoo.csv")
    parser.add_argument("--manifest", default="data/data_manifest.csv")
    parser.add_argument("--min-rows", type=int, default=200)
    args = parser.parse_args()

    config = load_config(args.config)
    result_frame, prices_frame = probe_sector_indices(
        default_candidates(),
        start=str(config["data"]["start"]),
        end=str(config["data"]["end"]),
        min_rows=args.min_rows,
    )

    probe_output = Path(args.probe_output)
    prices_output = Path(args.prices_output)
    ensure_dirs(probe_output.parent)
    result_frame.to_csv(probe_output, index=False)
    print(f"wrote {probe_output} rows={len(result_frame)} usable={(result_frame['status'] == 'usable').sum()}")

    if not prices_frame.empty:
        ensure_dirs(prices_output.parent)
        prices_frame.to_csv(prices_output, index=False)
        manifest_prices = prices_frame.rename(columns={"sector": "source_sector"}).copy()
        manifest_prices["ticker"] = manifest_prices["source_sector"] + ":" + manifest_prices["ticker"]
        write_manifest_row(
            Path(args.manifest),
            source_name="yahoo_thai_sector_indices_probe",
            access_method="Yahoo Finance via yfinance sector-index candidate probe",
            prices=manifest_prices,
            raw_file_path=prices_output,
            license_note="Yahoo Finance sector-index candidate data access through yfinance; verify usage rights before publication.",
        )
        print(
            f"wrote {prices_output} rows={len(prices_frame)} "
            f"sectors={';'.join(sorted(prices_frame['sector'].dropna().unique()))}"
        )
    else:
        print("no usable sector-index price table was produced")


if __name__ == "__main__":
    main()
