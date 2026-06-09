# Data Sources

This file tracks source provenance for every dataset used by the project.

## Synthetic Pilot

- Purpose: local and HPC smoke tests before licensed/public Thai market data is connected.
- Generator: `python -m src.data.synthetic --config config/debug.yaml`
- Raw path: `data/raw/prices_pilot.csv`
- Processed path: `data/processed/features_pilot.parquet` or CSV fallback.
- Limitation: synthetic data must not be used for final conclusions.

## Planned Real Sources

- SET or SETSMART: daily prices, indices, fundamentals, corporate actions.
- Yahoo Finance via `yfinance`: public daily OHLCV for `.BK` tickers, useful as a reproducible starter source before licensed SET/SETSMART data is available.
- Yahoo Finance via `yfinance`: public daily macro proxy series for FX, commodities, and yields; useful for pipeline validation but not a substitute for official release-date macroeconomic data.
- Yahoo Finance via `yfinance`: annual and quarterly statement fundamentals; useful for pipeline validation but not a substitute for licensed SET/SETSMART fundamentals.
- Yahoo Finance via `yfinance`: latest-news endpoint for source probing only; it does not provide the 2021-2024 historical news coverage needed for a valid sentiment experiment.
- Bank of Thailand statistics: macroeconomic variables aligned by release date.
- SET disclosures and financial news sources: Thai text for sentiment features, subject to access and usage rights.

Every real source added later should record access date, license/access note, schema, date coverage, symbol coverage, and missing-value rate in `data/data_manifest.csv`.

## Remaining Source Readiness

The implementation is wired for broader real-data runs, but the full research version is blocked until these source files are provided or exported from licensed/public systems:

| source gap | template | minimum file expected | required fields | command after file is available |
| --- | --- | --- | --- | --- |
| Broader SET50/SET100 OHLCV universe | `data/reference/set50_or_set100_universe_template.csv` | `data/reference/set50_or_set100_universe.csv` | `ticker` values such as `PTT.BK`, or bare `symbol` values with `.BK` suffix configured | Update `data.ticker_universe.path` in a real-data config, then run `python -m src.data.ingest_ohlcv --config <config>` |
| Complete sector membership or sector index history | `data/reference/sector_mapping_template.csv` | licensed/exported sector mapping or sector-index price file | ticker-to-sector labels for all modeled tickers, or daily sector index OHLCV/history | Replace `data/reference/sector_mapping_thai_pilot.csv` usage, then rebuild sector features with `python -m src.features.sector_context --config config/real_ohlcv_sector.yaml` |
| Historical official macro releases | `data/reference/official_macro_long_template.csv` | historical BOT/export CSV | observation date, value, indicator name/code, and release date or enough metadata to derive release date | `python -m src.data.ingest_official_macro_csv --config config/real_ohlcv_official_macro.yaml --input <historical_macro.csv> --output data/raw/bot_official_macro.csv` |
| Historical news/sentiment | `data/reference/historical_news_sentiment_template.csv` | licensed news/disclosure or sentiment-score CSV | ticker or symbol, publication timestamp/date, headline/text or precomputed score | `python -m src.data.ingest_sentiment_csv --config config/real_ohlcv_sentiment.yaml --input <historical_news.csv> --output data/raw/news_historical.csv --daily-output data/processed/sentiment_daily.parquet` |
| Licensed fundamentals | `data/reference/fundamentals_template.csv` | SET/SETSMART or equivalent statements | ticker, period end, report date/effective date, and statement metrics | Normalize to the fundamentals raw schema, then rebuild with `python -m src.features.fundamental_context --config config/real_ohlcv_fundamentals.yaml` |

Until those files exist, the current results should be described as a reproducible five-ticker pilot with implemented import scaffolding, not as a complete SET50/SET100 study.

Check source readiness before claiming full real-data coverage:

```bash
python -m src.data.source_readiness --output reports/source_readiness.csv
```

The command exits nonzero while any required external source file is missing, empty, or missing required columns.

## Real OHLCV Starter Ingestion

- Config: `config/real_ohlcv.yaml`
- CSV-universe config example: `config/real_ohlcv_universe_csv.yaml`
- Command: `python -m src.data.ingest_ohlcv --config config/real_ohlcv.yaml`
- Raw path: `data/raw/prices_real_ohlcv.csv`
- Processed command: `python -m src.features.build_features --config config/real_ohlcv.yaml`
- Processed path: `data/processed/features_real_ohlcv.parquet` or CSV fallback.
- Starter universe CSV: `data/reference/thai_starter_universe.csv`
- Ticker-universe helper: `docs/ticker_universe_csv.md`
- Limitation: Yahoo Finance access and adjustment methodology must be checked before final publication; licensed SET/SETSMART remains preferred for final results.

## Macro Proxy Ingestion

- Config: `config/real_ohlcv_macro.yaml`
- Raw command: `python -m src.data.ingest_macro --config config/real_ohlcv_macro.yaml`
- Raw path: `data/raw/prices_macro_yahoo.csv`
- Feature command: `python -m src.features.macro_context --config config/real_ohlcv_macro.yaml`
- Processed path: `data/processed/features_real_ohlcv_macro.parquet` or CSV fallback.
- Series: `USDTHB=X`, `BZ=F`, `CL=F`, `GC=F`, and `^TNX`.
- Limitation: these are daily market proxies. Official Bank of Thailand or other macro releases still need release-date alignment before final macroeconomic claims.

## SET Sector Index Probe

- Probe command: `python -m src.data.probe_sector_indices --config config/real_ohlcv_sector.yaml`
- Probe path: `data/raw/sector_index_yahoo_probe.csv`
- Usable raw path, when candidates are found: `data/raw/prices_sector_indices_yahoo.csv`
- Latest result: 139 Yahoo Finance/yfinance sector-code candidates were tested on HPC; 138 failed and one was usable.
- Usable candidate: banking sector symbol `^BANK`, with 1,004 rows dated 2021-01-04 to 2024-12-30.
- Limitation: this is not broad enough for full sector-relative modeling, so the current sector-context pipeline still uses the five-ticker pilot mapping in `data/reference/sector_mapping_thai_pilot.csv`. A licensed SET/SETSMART, SET export, or other complete sector-index source is still required.

## Official Macro Source Probe And Merge Scaffold

- Config: `config/real_ohlcv_official_macro.yaml`
- Raw command: `python -m src.data.ingest_bot_macro --config config/real_ohlcv_official_macro.yaml`
- Raw path: `data/raw/bot_official_macro.csv`
- Historical CSV import command: `python -m src.data.ingest_official_macro_csv --config config/real_ohlcv_official_macro.yaml --input <historical_macro.csv> --output data/raw/bot_official_macro.csv`
- Feature command: `python -m src.features.official_macro_context --config config/real_ohlcv_official_macro.yaml`
- Processed path: `data/processed/features_real_ohlcv_official_macro.parquet` or CSV fallback.
- Source: Bank of Thailand BOTWEBSTAT public statistics table `EC_EI_002_S2`, Leading Economic Indicator.
- Merge rule: each monthly observation is assigned an explicit release date using the table's last-business-day-of-following-month release pattern, then merged backward-only by release date to each stock trading date.
- CSV import helper: `docs/official_macro_csv_import.md`
- Current limitation: the web table probe returned 54 rows across nine indicators dated 2025-11-01 to 2026-04-01. This does not overlap the 2021-2024 modeling table, so merged historical BOT official macro columns are all zero. The CSV importer can normalize a downloaded historical export, but the historical source file is still required before official macro results can be interpreted.

## Fundamentals Ingestion

- Config: `config/real_ohlcv_fundamentals.yaml`
- Raw command: `python -m src.data.ingest_fundamentals --config config/real_ohlcv_fundamentals.yaml`
- Raw path: `data/raw/fundamentals_yahoo_quarterly.csv`
- Feature command: `python -m src.features.fundamental_context --config config/real_ohlcv_fundamentals.yaml`
- Processed path: `data/processed/features_real_ohlcv_fundamentals.parquet` or CSV fallback.
- Merge rule: annual and quarterly statement rows are converted to ratios and growth fields, then merged into stock features after a 60-day reporting lag.
- Limitation: Yahoo statement availability is uneven, and annual rows are used as fallback coverage. Licensed SET/SETSMART fundamentals remain preferred for final results.

## Sentiment Source Probe And Merge Scaffold

- Config: `config/real_ohlcv_sentiment.yaml`
- Latest-news probe command: `python -m src.data.ingest_news_yahoo --config config/real_ohlcv_sentiment.yaml`
- Raw path: `data/raw/news_yahoo_latest.csv`
- Historical CSV import command: `python -m src.data.ingest_sentiment_csv --config config/real_ohlcv_sentiment.yaml --input <historical_news.csv> --output data/raw/news_historical.csv --daily-output data/processed/sentiment_daily.parquet`
- Daily sentiment command: `python -m src.sentiment.extract_embeddings --input data/raw/news_yahoo_latest.csv --output data/processed/sentiment_daily.parquet`
- Feature command: `python -m src.features.sentiment_context --config config/real_ohlcv_sentiment.yaml`
- Processed path: `data/processed/features_real_ohlcv_sentiment.parquet` or CSV fallback.
- Merge rule: daily ticker sentiment is merged backward-only to stock dates with a seven-day maximum age, so future news cannot be used.
- CSV import helper: `docs/sentiment_csv_import.md`
- Current limitation: the Yahoo latest-news probe returned 11 rows across four tickers dated 2025-06-24 to 2026-03-31. This does not overlap the 2021-2024 modeling table, so the merged historical sentiment columns are all zero. The CSV importer can normalize a licensed historical news/disclosure or sentiment-score export, but the historical source file is still required before sentiment results can be interpreted.
