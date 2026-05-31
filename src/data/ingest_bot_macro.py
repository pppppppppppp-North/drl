from __future__ import annotations

import argparse
import re
from calendar import monthrange
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from src.data.ingest_ohlcv import write_manifest_row
from src.utils.config import ensure_dirs, load_config


BOT_REPORT_URL = "https://app.bot.or.th/BTWS_STAT/statistics/BOTWEBSTAT.aspx?language=ENG&reportID={report_id}"
MONTH_PATTERN = re.compile(r"^[A-Z]{3}\s+\d{4}")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value).strip().lower())
    cleaned = re.sub(r"\d+/", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return cleaned


def _parse_numeric(value: object) -> float | None:
    text = str(value).replace(",", "").strip()
    if not text or text in {"....", "nan", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _period_start(period_label: str) -> pd.Timestamp:
    clean = str(period_label).replace(" p", "").replace(" r", "").strip()
    month_text, year_text = clean[:3], clean[-4:]
    return pd.to_datetime(f"{year_text}-{month_text}-01", format="%Y-%b-%d")


def _last_business_day_following_month(period_start: pd.Timestamp) -> pd.Timestamp:
    following = period_start + pd.DateOffset(months=1)
    day = monthrange(int(following.year), int(following.month))[1]
    release = pd.Timestamp(year=int(following.year), month=int(following.month), day=day)
    while release.weekday() >= 5:
        release -= pd.Timedelta(days=1)
    return release


def _select_bot_table(tables: list[pd.DataFrame], required_text: str) -> pd.DataFrame:
    for table in tables:
        text = " ".join(table.astype(str).fillna("").to_numpy().ravel())
        if required_text.lower() in text.lower():
            return table
    raise ValueError(f"could not find BOT table containing {required_text!r}")


def _html_tables(url: str) -> list[pd.DataFrame]:
    from bs4 import BeautifulSoup

    with urlopen(url, timeout=30) as response:
        html = response.read()
    soup = BeautifulSoup(html, "html.parser")
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if len(rows) < 2:
            continue
        width = max(len(row) for row in rows)
        normalized = [row + [""] * (width - len(row)) for row in rows]
        seen: dict[str, int] = {}
        header = []
        for index, value in enumerate(normalized[0]):
            name = value or f"column_{index}"
            count = seen.get(name, 0)
            seen[name] = count + 1
            header.append(name if count == 0 else f"{name}_{count}")
        body = normalized[1:]
        tables.append(pd.DataFrame(body, columns=header))
    return tables


def parse_bot_leading_indicator_table(table: pd.DataFrame) -> pd.DataFrame:
    df = table.copy()
    df.columns = [str(column).strip() for column in df.columns]
    period_columns = [column for column in df.columns if MONTH_PATTERN.match(str(column).replace(" p", "").replace(" r", ""))]
    if not period_columns:
        raise ValueError("BOT table has no monthly period columns")

    descriptor_columns = [column for column in df.columns if column not in period_columns]
    metric_column = descriptor_columns[-1]
    rows = []
    for _, row in df.iterrows():
        descriptor_values = [str(row[column]).strip() for column in descriptor_columns if str(row[column]).strip()]
        metric_raw = descriptor_values[-1] if descriptor_values else str(row[metric_column]).strip()
        if not metric_raw or metric_raw.lower() == "nan":
            continue
        metric_name = _safe_name(metric_raw)
        if not metric_name:
            continue
        for period_column in period_columns:
            value = _parse_numeric(row[period_column])
            if value is None:
                continue
            period_start = _period_start(str(period_column))
            rows.append(
                {
                    "period": period_start.date().isoformat(),
                    "release_date": _last_business_day_following_month(period_start).date().isoformat(),
                    "metric": metric_name,
                    "metric_label": metric_raw,
                    "value": value,
                    "source_table": "EC_EI_002_S2",
                    "source": "Bank of Thailand",
                }
            )
    if not rows:
        raise ValueError("BOT table did not contain any numeric macro observations")
    return pd.DataFrame(rows).sort_values(["period", "metric"]).reset_index(drop=True)


def download_bot_leading_indicator(report_id: int = 887) -> pd.DataFrame:
    url = BOT_REPORT_URL.format(report_id=report_id)
    tables = _html_tables(url)
    table = _select_bot_table(tables, "Leading Economic Index")
    return parse_bot_leading_indicator_table(table)


def write_bot_macro_manifest_row(manifest_path: Path, macro: pd.DataFrame, raw_file_path: Path) -> None:
    manifest_prices = macro.rename(columns={"period": "date"}).copy()
    manifest_prices["ticker"] = manifest_prices["source_table"] + ":" + manifest_prices["metric"]
    for column in ["open", "high", "low", "close", "volume"]:
        manifest_prices[column] = manifest_prices["value"] if column != "volume" else 0.0
    write_manifest_row(
        manifest_path,
        source_name="bot_official_leading_indicator",
        access_method="Bank of Thailand BOTWEBSTAT public statistics table",
        prices=manifest_prices,
        raw_file_path=raw_file_path,
        license_note="Bank of Thailand public statistics; verify terms of use before publication.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real_ohlcv_official_macro.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default="data/data_manifest.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    macro_config = config.get("official_macro_context", {})
    output = Path(args.output or macro_config.get("raw_path", "data/raw/bot_official_macro.csv"))
    ensure_dirs(output.parent)
    macro = download_bot_leading_indicator(int(macro_config.get("report_id", 887)))
    start = pd.Timestamp(config["data"]["start"])
    end = pd.Timestamp(config["data"]["end"])
    in_window = macro[pd.to_datetime(macro["period"]).between(start - pd.DateOffset(months=3), end)].copy()
    if not in_window.empty:
        macro = in_window
    macro.to_csv(output, index=False)
    write_bot_macro_manifest_row(Path(args.manifest), macro, output)
    print(
        f"wrote {output} rows={len(macro)} "
        f"periods={macro['period'].nunique()} metrics={macro['metric'].nunique()}"
    )


if __name__ == "__main__":
    main()
