# Full External-Data Rebuild

Last updated: 2026-07-31

This note records the original staged rebuild order from the verified five-ticker pilot to a full external-data feature table. The isolated V3 membership, sector-context, and market-package validation code is implemented locally, but broad-universe rebuilding remains blocked until a licensed package passes and the V3 policy/protocol is approved, checksum-locked, and released separately.

## Mandatory V3 Stop Gate

The 2021–2024 official archive, fixed-slot membership panel, dated sector context, past-only development eligibility, and mask-aware environment mechanics are implemented under `src/v3/`. They are not wired into the legacy config, and the V3 protocol intentionally remains an unsigned, training-disabled draft. Before using the commands below:

1. acquire the five-file licensed market package for the 138-ticker acquisition union;
2. pass `scripts.build_v3_market_package_gap --candidate-manifest <path> --require-pass`;
3. rebuild the membership panel on the authoritative licensed session calendar;
4. construct the final investability mask without using next-session price availability to decide today's mask;
5. materialize the implemented `src.v3.sector_context` features from explicit observed-return/status rows;
6. author-approve and checksum-lock entry, exit, suspension, ticker-change, stale-valuation, and delisting rules;
7. bind the passing market receipt and locked policy into the V3 protocol;
8. extend constituent history beyond 2024 and embargo a genuinely untouched V3 final holdout;
9. build and verify a new release rather than changing locked V2 paths.

`data/reference/set50_or_set100_universe.csv` is an acquisition list only. `data/reference/sector_mapping.csv` is a latest-label compatibility file only.

The structural V3 artifact can be rebuilt safely without enabling training:

```bash
python -m scripts.build_v3_membership_readiness --project-root .
python -m scripts.build_v3_market_package_gap --project-root .
python -m src.v3.development_gate --project-root . --split structural
```

The resulting panel is `data/processed/v3_development/set100_membership_session_mask_2021_2024.csv.gz`. It is explicitly labeled membership-only, not final investability.

## Config

Use:

```text
config/real_ohlcv_full_external.yaml
```

The legacy config points to these default handoff paths:

- `data/reference/set50_or_set100_universe.csv`
- `data/reference/sector_mapping.csv`
- `data/external/fundamentals.csv`
- `data/external/bot_historical_macro.csv`
- `data/external/historical_news.csv`

It intentionally omits sector one-hot columns from `features.columns`. The isolated V3 module implements active-member-only dated sector engineering, but the legacy configuration does not call it and does not carry the required explicit observed-return/status evidence. The V3 stop gate above remains mandatory.

## Rebuild Order

Run the steps in this order only after the readiness gate and the V3 stop gate both pass.

```bash
python -m src.data.ingest_ohlcv --config config/real_ohlcv_full_external.yaml
python -m src.features.build_features \
  --config config/real_ohlcv_full_external.yaml \
  --output data/processed/features_real_ohlcv_full_external_base.parquet

python -m src.features.market_context \
  --config config/real_ohlcv_full_external.yaml \
  --features data/processed/features_real_ohlcv_full_external_base.parquet \
  --output data/processed/features_real_ohlcv_full_external_market.parquet

# Do not call src.features.sector_context for V3: it uses static labels.
# Build explicit observed-return/status rows from the accepted package, then
# call src.v3.sector_context.build_active_member_sector_context and preserve
# its availability flags in:
# data/processed/features_real_ohlcv_full_external_sector.parquet

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
python -m src.data.intake_validation \
  --output reports/external_intake_validation.csv \
  --markdown-output reports/external_intake_validation.md
python -m pytest -q
python -m src.data.quality_report \
  --config config/real_ohlcv_full_external.yaml \
  --output-dir reports/data_quality_full_external
```

Then run leakage checks through the existing test suite or a small Python harness using `src.data.leakage.assert_no_leakage`. The V3 test suite already rejects future constituent/sector records, same-day revision use, future price-based eligibility, incomplete masks, missing valuation prices, and `test`/`all` access. Only after the remaining licensed-data and protocol gates pass should baselines, algorithm comparisons, ablations, walk-forward validation, and regime tests be rerun under a new V3 configuration.
