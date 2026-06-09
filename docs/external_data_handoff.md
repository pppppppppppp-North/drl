# External Data Handoff Guide

Last updated: 2026-06-09

This guide turns the remaining project blockers into a concrete data handoff workflow. The current implementation is a reproducible five-ticker pilot. A full SET50/SET100 study requires external files that are not currently available in the repository.

For a provider-facing request packet and email template, see `docs/external_data_provider_request.md`.

## Required Files

Place the real exported or licensed files at these default paths before claiming full real-data coverage:

| source | default path | template | minimum required columns |
| --- | --- | --- | --- |
| Broader ticker universe | `data/reference/set50_or_set100_universe.csv` | `data/reference/set50_or_set100_universe_template.csv` | `ticker` |
| Full ticker sector mapping | `data/reference/sector_mapping.csv` | `data/reference/sector_mapping_template.csv` | `ticker`, `sector` |
| Historical official macro | `data/external/bot_historical_macro.csv` | `data/reference/official_macro_long_template.csv` | `period`, `metric`, `value` |
| Historical news or sentiment | `data/external/historical_news.csv` | `data/reference/historical_news_sentiment_template.csv` | `date`, `ticker` |
| Licensed fundamentals | `data/external/fundamentals.csv` | `data/reference/fundamentals_template.csv` | `ticker`, `metric`, `period_end`, `value` |

The templates are schema examples only. They are not research data and should not be used for final results.

## Readiness Check

Run the source-readiness gate after adding or replacing external files:

```bash
python -m src.data.source_readiness --output reports/source_readiness.csv
```

The command exits with status `1` while any required file is missing, empty, or missing required columns. It exits with status `0` only when all required source files are present and structurally ready.

## Recommended Handoff Steps

1. Export or license the broader SET50/SET100 ticker universe.
2. Save the ticker file as `data/reference/set50_or_set100_universe.csv`.
3. Export sector membership or a full sector-index history source.
4. Save static sector membership as `data/reference/sector_mapping.csv`, or document a dated sector-index price source separately.
5. Export historical official macro data with release-date metadata when available.
6. Save the official macro file as `data/external/bot_historical_macro.csv`.
7. Export historical Thai news, disclosures, or precomputed sentiment scores.
8. Save the sentiment/news file as `data/external/historical_news.csv`.
9. Export licensed fundamentals with period-end and report/effective-date metadata when available.
10. Save fundamentals as `data/external/fundamentals.csv`.
11. Run the readiness check.
12. Update `data/data_manifest.csv` with access method, license note, date range, symbols, columns, and missing-value rate.

## Rebuild Commands

After the readiness check passes, rerun the affected ingestion and feature steps.
For the consolidated full external-data config and stage order, see `docs/full_external_rebuild.md`.

Broader OHLCV universe:

```bash
python -m src.data.ingest_ohlcv --config config/real_ohlcv_universe_csv.yaml
python -m src.features.build_features --config config/real_ohlcv_universe_csv.yaml
```

Official macro:

```bash
python -m src.data.ingest_official_macro_csv \
  --config config/real_ohlcv_official_macro.yaml \
  --input data/external/bot_historical_macro.csv \
  --output data/raw/bot_official_macro.csv

python -m src.features.official_macro_context --config config/real_ohlcv_official_macro.yaml
```

Historical news or sentiment:

```bash
python -m src.data.ingest_sentiment_csv \
  --config config/real_ohlcv_sentiment.yaml \
  --input data/external/historical_news.csv \
  --output data/raw/news_historical.csv \
  --daily-output data/processed/sentiment_daily.parquet

python -m src.features.sentiment_context --config config/real_ohlcv_sentiment.yaml
```

Fundamentals:

```bash
python -m src.features.fundamental_context \
  --config config/real_ohlcv_fundamentals.yaml \
  --fundamentals data/external/fundamentals.csv
```

Sector context:

```bash
python -m src.features.sector_context --config config/real_ohlcv_sector.yaml
```

If `config/real_ohlcv_sector.yaml` still points to `data/reference/sector_mapping_thai_pilot.csv`, update it or pass a config that points to `data/reference/sector_mapping.csv` before using the result for full-universe claims.

## Validation After Rebuild

Run these checks before interpreting broader results:

```bash
python -m src.data.source_readiness --output reports/source_readiness.csv
python -m pytest -q
```

Then regenerate data-quality reports, leakage checks, ablations, walk-forward summaries, and baseline comparisons for the broader universe.

## Claim Boundary

Until this handoff is complete, report the project as:

> A reproducible five-ticker Thai equity DRL pilot with implemented scaffolding for broader SET50/SET100 data, sector context, official macro data, fundamentals, and historical sentiment.

Do not describe current results as a complete SET50/SET100 performance study until the required external files are present, validated, and used in the rebuilt experiments.
