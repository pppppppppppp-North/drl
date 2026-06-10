# Full External-Data Rebuild

Last updated: 2026-06-09

This note describes how to move from the verified five-ticker pilot to the full external-data feature table. It assumes the required files in `docs/external_data_handoff.md` are already present and that `python -m src.data.source_readiness --output reports/source_readiness.csv` exits with status `0`.

## Config

Use:

```text
config/real_ohlcv_full_external.yaml
```

The config points to the default handoff paths:

- `data/reference/set50_or_set100_universe.csv`
- `data/reference/sector_mapping.csv`
- `data/external/fundamentals.csv`
- `data/external/bot_historical_macro.csv`
- `data/external/historical_news.csv`

It intentionally omits sector one-hot columns from `features.columns`. A broad external sector map can contain sectors that differ from the five-ticker pilot, while the numeric sector-relative features have stable names.

## Rebuild Order

Run the steps in this order after the readiness gate passes.

```bash
python -m src.data.ingest_ohlcv --config config/real_ohlcv_full_external.yaml
python -m src.features.build_features \
  --config config/real_ohlcv_full_external.yaml \
  --output data/processed/features_real_ohlcv_full_external_base.parquet

python -m src.features.market_context \
  --config config/real_ohlcv_full_external.yaml \
  --features data/processed/features_real_ohlcv_full_external_base.parquet \
  --output data/processed/features_real_ohlcv_full_external_market.parquet

python -m src.features.sector_context \
  --config config/real_ohlcv_full_external.yaml \
  --input data/processed/features_real_ohlcv_full_external_market.parquet \
  --output data/processed/features_real_ohlcv_full_external_sector.parquet

python -m src.features.macro_context \
  --config config/real_ohlcv_full_external.yaml \
  --features data/processed/features_real_ohlcv_full_external_sector.parquet \
  --output data/processed/features_real_ohlcv_full_external_macro.parquet

python -m src.features.fundamental_context \
  --config config/real_ohlcv_full_external.yaml \
  --features data/processed/features_real_ohlcv_full_external_macro.parquet \
  --output data/processed/features_real_ohlcv_full_external_fundamentals.parquet \
  --fundamentals data/external/fundamentals.csv

python -m src.data.ingest_official_macro_csv \
  --config config/real_ohlcv_full_external.yaml \
  --input data/external/bot_historical_macro.csv \
  --output data/raw/bot_historical_macro.csv

python -m src.features.official_macro_context \
  --config config/real_ohlcv_full_external.yaml \
  --features data/processed/features_real_ohlcv_full_external_fundamentals.parquet \
  --macro data/raw/bot_historical_macro.csv \
  --output data/processed/features_real_ohlcv_full_external_official_macro.parquet

python -m src.data.ingest_sentiment_csv \
  --config config/real_ohlcv_full_external.yaml \
  --input data/external/historical_news.csv \
  --output data/raw/news_historical.csv \
  --daily-output data/processed/sentiment_daily_external.parquet

python -m src.features.sentiment_context \
  --config config/real_ohlcv_full_external.yaml \
  --features data/processed/features_real_ohlcv_full_external_official_macro.parquet \
  --sentiment data/processed/sentiment_daily_external.parquet \
  --output data/processed/features_real_ohlcv_full_external.parquet
```

## Path Boundary

The final all-real feature table should be:

```text
data/processed/features_real_ohlcv_full_external.parquet
```

The current merge modules each read `paths.base_features` and write `paths.features` unless explicit CLI paths are passed. The commands above pass explicit stage inputs and outputs so each stage preserves the previous table. Do not interpret final all-real results until `data/processed/features_real_ohlcv_full_external.parquet` contains all required feature groups and passes the validation below.

## Validation

After rebuilding:

```bash
python -m src.data.source_readiness --output reports/source_readiness.csv
python -m src.data.intake_validation --output reports/external_intake_validation.csv
python -m pytest -q
python -m src.data.quality_report \
  --config config/real_ohlcv_full_external.yaml \
  --output-dir reports/data_quality_full_external
```

Then run leakage checks through the existing test suite or a small Python harness using `src.data.leakage.assert_no_leakage`, and rerun baselines, PPO/A2C comparison, ablations, walk-forward validation, and regime tests against `config/real_ohlcv_full_external.yaml`.
