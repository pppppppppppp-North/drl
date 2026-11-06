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
- Yahoo Finance via `yfinance`: technically accessible daily OHLCV for `.BK` tickers, restricted to a private educational pilot. The [`yfinance` legal disclaimer](https://ranaroussi.github.io/yfinance/) states that Yahoo data is intended for personal use, and [Yahoo's terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) restrict automated collection and redistribution. Do not treat this path as publication-grade or redistribute its raw files.
- Yahoo Finance via `yfinance`: public daily macro proxy series for FX, commodities, and yields; useful for pipeline validation but not a substitute for official release-date macroeconomic data.
- Yahoo Finance via `yfinance`: annual and quarterly statement fundamentals; useful for pipeline validation but not a substitute for licensed SET/SETSMART fundamentals.
- Yahoo Finance via `yfinance`: latest-news endpoint for source probing only; it does not provide the 2021-2024 historical news coverage needed for a valid sentiment experiment.
- Bank of Thailand statistics: macroeconomic variables aligned by release date.
- SET disclosures and financial news sources: Thai text for sentiment features, subject to access and usage rights.

Before a deployment is locked, every real source should record its access date, license/access note, schema, date coverage, symbol coverage, and missing-value rate in `data/data_manifest.csv`. After a deployment is locked, new source evidence must first use a separate external manifest, such as `data/external/bot_historical_source_manifest.csv`; promote it into `data/data_manifest.csv` only through a newly built and verified release.

The preferred publication-grade replacement is the [Stock Exchange of Thailand historical-data service](https://www.set.or.th/en/services/connectivity-and-data/data/historical) or a licensed [SETSMART](https://www.setsmart.com/ssm/package) export. The current local Yahoo snapshot may be used only for private pipeline and modeling rehearsal; generated data remains engineering-only and cannot satisfy the real campaign gate.

## Official SET Point-in-Time Constituent And Sector Archive

- Archive page: [SET50 and SET100 constituent archive](https://www.set.or.th/en/market/information/securities-list/constituents-list-set50-set100)
- Preserved official PDFs: `data/external/original/set_constituent_history/`
- Source manifest: `data/external/set_constituent_source_manifest.csv`
- Rebuild command: `python -m src.data.ingest_set_constituent_history --project-root .`
- Dated membership output: `data/reference/set50_set100_membership_2021_2024.csv`
- Acquisition-only union: `data/reference/set50_or_set100_universe.csv`
- Latest-label compatibility mapping: `data/reference/sector_mapping.csv`
- Receipt: `reports/data_sources/set_constituent_archive_20260731.json`
- Readiness report: `reports/set_constituent_archive_readiness_20260731.md`
- Detailed method: `docs/set_constituent_history_intake.md`

Current validated coverage is ten continuous effective intervals from 2021-01-01 through 2025-01-01 exclusive: 1,000 normalized rows, exactly 50 SET50 and 100 SET100 members per interval, 138 unique acquisition tickers, and 23 canonical sector labels. The archive includes the published 2021 H1 and 2023 H1 revisions.

The union CSV exists only to acquire price history for every security that appeared during the study window. It must not be used as a static model universe. Historical selection and sector joins must use `src.data.constituent_history.membership_as_of`.

`MINT.BK` and `NEX.BK` have real sector changes in the archive, so a single static sector label is scientifically unsafe for earlier dates. The compatibility sector file contains the latest label and explicitly directs as-of work back to the dated archive.

This source closes the former broader-universe and sector-history structural gaps, raising the current source-readiness count from 1/5 to 3/5. It does not supply licensed OHLCV and does not alter the locked five-ticker V2 experiment.

The isolated V3 development builder now materializes this archive on the observed SET session grid at `data/processed/v3_development/set100_membership_session_mask_2021_2024.csv.gz`. It contains 133,170 rows across 965 sessions and 138 fixed acquisition slots, with exactly 100 members per session. Same-day revisions are delayed until the next observed session. The output is explicitly labeled membership-only, not final investability; its receipt is `reports/data_quality_v3/v3_membership_mask_readiness_20260731.json`.

The separate V3 market-package gate at `src/v3/market_data_readiness.py` now prevents the five-ticker pilot receipt from certifying the broader study. The current gap receipt measures 5/138 acquisition tickers, 4,825/96,500 active member/session bar keys, zero explicit status keys, and no authoritative calendar/action/symbol/license package. See `reports/v3_market_package_gap_20260731.md`.

## Remaining Source Readiness

The implementation now has official point-in-time membership and sector history, but a publication-grade broad-universe run remains unavailable until the remaining market and multi-source data are provided or exported from licensed/public systems:

| source gap | current status | minimum file expected | required fields | command after file is available |
| --- | --- | --- | --- | --- |
| Licensed market package for the 138-ticker acquisition union | Missing; V3 gap receipt is FAIL | Five separately bound bars, trading-status, SET-session, corporate-action, and symbol-history CSVs | adjusted OHLCV, availability/source IDs, explicit daily status, authoritative sessions, actions, stable instrument intervals | Fill `config/v3/market_package_manifest_template.yaml`, then run `python -m scripts.build_v3_market_package_gap --candidate-manifest <path> --require-pass` |
| Point-in-time SET50/SET100 membership and sectors | **Complete structurally** | `data/reference/set50_set100_membership_2021_2024.csv` | effective interval, publication date, symbol, company, sector, SET50/SET100 flags, source hash | Use `membership_as_of`; do not use the 138-ticker union or latest sector map as historical membership |
| Point-in-time official macro vintages | Structurally complete revised history exists; historical as-published vintages remain missing | Historical BOT vintage snapshots or another release-as-known source | observation date, value as first released, indicator name/code, actual release date, and vintage identifier | Normalize into a separate candidate input, verify `point_in_time_eligible=true`, then promote it only through a newly locked campaign release |
| Historical news/sentiment | Missing | Licensed news/disclosure or sentiment-score CSV | ticker or symbol, publication timestamp/date, headline/text or precomputed score | `python -m src.data.ingest_sentiment_csv --config config/real_ohlcv_sentiment.yaml --input <historical_news.csv> --output data/raw/news_historical.csv --daily-output data/processed/sentiment_daily.parquet` |
| Licensed fundamentals | Missing | SET/SETSMART or equivalent statements | ticker, period end, report date/effective date, and statement metrics | Normalize to the fundamentals raw schema, then rebuild with `python -m src.features.fundamental_context --config config/real_ohlcv_fundamentals.yaml` |

Until licensed broad-universe OHLCV and the remaining multi-source files exist, the model results should still be described as a reproducible five-ticker pilot. The project may separately state that official point-in-time SET50/SET100 membership and sector history for 2021–2024 has been implemented and validated.

Check source readiness before claiming full real-data coverage:

```bash
python -m src.data.source_readiness --output reports/source_readiness.csv
```

The command currently reports 3/5 structurally ready and exits nonzero while historical news/sentiment and licensed fundamentals remain missing. The broader-universe and sector rows additionally require an exact checksum-bound receipt; a header-only or drifted CSV no longer passes.

## Real OHLCV Starter Ingestion

- Config: `config/real_ohlcv.yaml`
- CSV-universe config example: `config/real_ohlcv_universe_csv.yaml`
- Command: `python -m src.data.ingest_ohlcv --config config/real_ohlcv.yaml`
- Raw path: `data/raw/prices_real_ohlcv.csv`
- Processed command: `python -m src.features.build_features --config config/real_ohlcv.yaml`
- Processed path: `data/processed/features_real_ohlcv.parquet` or CSV fallback.
- Starter universe CSV: `data/reference/thai_starter_universe.csv`
- Ticker-universe helper: `docs/ticker_universe_csv.md`
- Strict readiness command: `python -m src.data.market_data_readiness --config config/real_ohlcv_market.yaml --output-dir reports/data_quality_market`
- Readiness receipt: `reports/data_quality_market/market_data_readiness.json`
- Quality report: `reports/data_quality_market/real_market_data_quality.md`
- Current frozen snapshot: 4,840 stock rows across five tickers and 968 dates; 965 SET-index dates; 4,840 processed market-context rows.
- Limitation: the snapshot is private-research-only, cannot be redistributed, and is not eligible for publication-grade claims. Licensed SET/SETSMART data or explicit provider permission is required before publication.

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

## Official Macro Historical Source And Merge

- Config: `config/real_ohlcv_official_macro.yaml`
- Latest-table probe command: `python -m src.data.ingest_bot_macro --config config/real_ohlcv_official_macro.yaml`
- Legacy point-in-time model-input path: `data/raw/bot_official_macro.csv`; the revised full-history importer intentionally does not overwrite this path.
- Official full-history CSV: [Bank of Thailand `EC_EI_002_S2_ENG_ALL.CSV`](https://app.bot.or.th/BTWS_STAT/statistics/DownloadFile.aspx?file=EC_EI_002_S2_ENG_ALL.CSV)
- Historical transposed-export command: `python -m src.data.ingest_bot_historical_export --input <downloaded EC_EI_002_S2_ENG_ALL.CSV> --receipt reports/data_sources/bot_historical_macro_20260725.json`
- Preserved source vintage: `data/external/original/EC_EI_002_S2_ENG_ALL_20260725.CSV`
- Normalized structural-intake path: `data/external/bot_historical_macro.csv`
- Explicitly revised exploratory-input copy: `data/external/bot_historical_macro_revised_exploratory_input.csv`
- Normalization receipt: `reports/data_sources/bot_historical_macro_20260725.json`
- Guarded exploratory feature command: `python -m scripts.build_bot_macro_revised_exploratory_features --acknowledge-revised-history`
- Exploratory validation command: `python -m scripts.validate_bot_macro_exploratory_features`
- Exploratory processed path: `data/processed/features_real_ohlcv_official_macro_revised_exploratory.csv`
- Build receipt: `reports/data_sources/bot_macro_exploratory_feature_build_20260725.json`
- Source: Bank of Thailand BOTWEBSTAT public statistics table `EC_EI_002_S2`, Leading Economic Indicator.
- Merge rule: each monthly observation is assigned an inferred scheduled date using the last-business-day-of-following-month rule, then merged backward-only to each stock trading date. This schedule is not evidence of the value actually published in that historical vintage.
- CSV import helper: `docs/official_macro_csv_import.md`
- Current normalized coverage: 459 rows across nine indicators and 51 monthly periods from October 2020 through December 2024, including a three-month warm-up before the modeling window.
- Exploratory merge: `data/processed/features_real_ohlcv_official_macro_revised_exploratory.csv` contains 4,840 stock rows with five populated BOT feature columns and no joins to a future scheduled release date.
- Exploratory validation receipt: `reports/data_sources/bot_macro_exploratory_feature_validation_20260725.json`.
- Scientific limitation: every retained 2020-2024 observation is marked revised (`r`). This is a latest revised vintage, not the value actually known on each historical release date. It is therefore not point-in-time eligible and must not support causal backtests or publication-grade performance claims. Only the standalone guarded builder accepts it, and only after `--acknowledge-revised-history`; the legacy model-input path remains untouched.

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
