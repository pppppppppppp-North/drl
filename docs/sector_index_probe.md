# SET Sector Index Source Probe

Last updated: 2026-05-31

This note records the reproducible Yahoo Finance candidate-symbol probe for SET sector indices. It does not replace the existing pilot sector mapping. It is evidence for the remaining source gap: most sector-index candidates are not available through the public Yahoo Finance/yfinance path used by the starter project.

## Command

```bash
python -m src.data.probe_sector_indices --config config/real_ohlcv_sector.yaml
```

## HPC Output

- Probe table: `data/raw/sector_index_yahoo_probe.csv`
- Usable price table: `data/raw/prices_sector_indices_yahoo.csv`
- Manifest source row: `yahoo_thai_sector_indices_probe`

## Result

The probe tested SET sector-code symbol patterns such as `^SETENERG.BK`, `^SETENERG`, `^ENERG.BK`, `^ENERG`, and `ENERG.BK` for each SET sector code.

| status | count |
| --- | ---: |
| failed | 138 |
| usable | 1 |

The only usable candidate was:

| sector | symbol | rows | start_date | end_date |
| --- | --- | ---: | --- | --- |
| bank | `^BANK` | 1004 | 2021-01-04 | 2024-12-30 |

## Interpretation

Because the public Yahoo path produced only the banking sector index, it is not sufficient for full sector-relative modeling across the five starter tickers. The project should continue using `data/reference/sector_mapping_thai_pilot.csv` as the current pilot fallback until a broader licensed SET/SETSMART, SET data export, or other complete sector-index source is available.
