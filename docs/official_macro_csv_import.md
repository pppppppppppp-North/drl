# Official Macro CSV Import

Last updated: 2026-05-31

The Bank of Thailand web-table probe validates the release-date merge path, but its public table output does not overlap the 2021-2024 modeling window. This importer handles the next practical path: downloading or receiving a historical official macro CSV and normalizing it into the schema already consumed by `src.features.official_macro_context`.

## Expected Output Schema

The importer writes:

```text
period,release_date,metric,metric_label,value,source_table,source
```

That schema can be passed directly to:

```bash
python -m src.features.official_macro_context --config config/real_ohlcv_official_macro.yaml --macro data/raw/bot_official_macro.csv
```

## Long CSV Input

Use this when the source already has one observation per row.

Required columns by default:

- `period`
- `release_date`, optional
- `metric`
- `value`

Example:

```bash
python -m src.data.ingest_official_macro_csv \
  --config config/real_ohlcv_official_macro.yaml \
  --input data/external/bot_historical_macro.csv \
  --output data/raw/bot_official_macro.csv \
  --format long \
  --source-table historical_bot_export \
  --source-name "Bank of Thailand historical export"
```

## Wide CSV Input

Use this when the source has one date row and one column per metric.

Required columns by default:

- `period`
- optional `release_date`
- one or more metric columns

Example:

```bash
python -m src.data.ingest_official_macro_csv \
  --config config/real_ohlcv_official_macro.yaml \
  --input data/external/bot_historical_macro_wide.csv \
  --output data/raw/bot_official_macro.csv \
  --format wide \
  --source-table historical_bot_export \
  --source-name "Bank of Thailand historical export"
```

If `release_date` is absent, the importer uses the `official_macro_context.release_rule` from the config. The current rule is `last_business_day_following_month`.

## Limitation

This importer does not create historical official macro data. It only normalizes a legitimate historical export and records provenance in `data/data_manifest.csv`. Source licensing, redistribution rights, and exact release-date rules still need to be verified before publication.
