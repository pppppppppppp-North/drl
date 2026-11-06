# External Data Handoff Guide

Last updated: 2026-07-31

This guide turns the remaining project blockers into a concrete data handoff workflow. The current model implementation remains a reproducible five-ticker pilot. Official point-in-time SET50/SET100 membership and sector history is now present, but a broad study still requires licensed OHLCV and the remaining multi-source files.

For a provider-facing request packet and email template, see `docs/external_data_provider_request.md`. After files are received, use `docs/external_data_intake_checklist.md`.

## Required Files

Place the real exported or licensed files at these default paths before claiming full real-data coverage:

| source | default path | template | minimum required columns |
| --- | --- | --- | --- |
| Licensed adjusted OHLCV for the 138-ticker union | Separate candidate file under `data/external/v3_candidate/` | `config/v3/market_package_manifest_template.yaml` | `date`, `ticker`, OHLC, adjusted close, volume, availability, source ID |
| Daily trading status | `data/external/v3_candidate/trading_status.csv` | Same V3 manifest | date, ticker, explicit state, availability, source ID |
| Authoritative SET sessions | `data/external/v3_candidate/set_sessions.csv` | Same V3 manifest | date, calendar availability, source ID |
| Corporate actions | `data/external/v3_candidate/corporate_actions.csv` | Same V3 manifest | action ID/type, announcement, effective date, factor/cash/successor, source ID |
| Stable-instrument symbol history | `data/external/v3_candidate/symbol_history.csv` | Same V3 manifest | instrument ID, ticker, half-open effective interval, source ID |
| Point-in-time membership and sectors | **Already complete:** `data/reference/set50_set100_membership_2021_2024.csv` | `docs/set_constituent_history_intake.md` | effective interval, publication date, symbol, sector, SET50/SET100 flags |
| Historical official macro | `data/external/bot_historical_macro.csv` | `data/reference/official_macro_long_template.csv` | `period`, `metric`, `value` |
| Historical news or sentiment | `data/external/historical_news.csv` | `data/reference/historical_news_sentiment_template.csv` | `date`, `ticker` |
| Licensed fundamentals | `data/external/fundamentals.csv` | `data/reference/fundamentals_template.csv` | `ticker`, `metric`, `period_end`, `value` |

The templates are schema examples only. They are not research data and should not be used for final results.

For the five-file V3 market package, the preferred official route is a licensed SETSMART Enterprise extract using `CALENDAR`, `DAILY_SECURITY_TRADING`, `ADJUST_FACTOR`, `RIGHTS_BENEFIT`, `SECURITY`, `SIGN_POSTING`, `SIGN_LIFTING`, and effective-dated symbol-change events. The prepared advisor/institution request is `docs/setsmart_academic_data_request_draft.md`.

## Readiness Check

Run the source-readiness gate after adding or replacing external files:

```bash
python -m src.data.source_readiness --output reports/source_readiness.csv
```

The command currently reports 3/5 ready and exits with status `1` while historical news/sentiment or licensed fundamentals is missing. The broader-universe and sector rows pass only when their checksum-bound archive receipt also matches.

## Recommended Handoff Steps

1. Use the receipt-bound 138-ticker union only as the provider acquisition list.
2. Export or license adjusted OHLCV, corporate actions, and trading-status history for that list into separate candidate paths.
3. Include an authoritative SET session calendar and stable-instrument/ticker history.
4. Fill a copied V3 market manifest from `config/v3/market_package_manifest_template.yaml`, bind every file by path/hash/bytes/rows, and checksum the completed manifest.
5. Preserve the existing dated constituent archive; do not overwrite it with a static current list.
6. Export point-in-time official macro vintages with actual release-date metadata when available.
7. Export historical Thai news, disclosures, or precomputed sentiment scores.
8. Save the sentiment/news file as `data/external/historical_news.csv`.
9. Export licensed fundamentals with period-end and report/effective-date metadata.
10. Save fundamentals as `data/external/fundamentals.csv`.
11. Fill `data/external/external_data_manifest.csv` from `data/reference/external_data_manifest_template.csv` for the non-market sources.
12. Run the readiness and intake checks.
13. Update `data/data_manifest.csv` only through a newly built and verified release.

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

V3 market-package and sector-context intake:

```bash
python -m scripts.build_v3_market_package_gap \
  --project-root . \
  --candidate-manifest data/external/v3_candidate/market_package.yaml \
  --require-pass
python -m scripts.build_v3_investability_policy_readiness --project-root .
```

The isolated V3 panel performs the as-of membership and sector join and preserves `sector_as_of` on active rows. `src/v3/sector_context.py` now computes active-member-only dated sector returns with explicit observed-return, contributor-count, peer-count, and availability evidence. Do not point the shared static-label module at `sector_mapping.csv`.

The market package must also cover each active-to-inactive membership transition until a valid liquidation bar or explicit terminal settlement. The current panel contains 62 such transitions. Missing post-exit status is a critical failure even when active-member status coverage is otherwise complete.

Without `--candidate-manifest`, the command assesses the current five-ticker snapshot and intentionally writes a failed gap receipt. A provider package is accepted only when all five bound market files, license declarations, checksum bindings, active-key rules, exit-path rules, and point-in-time timing checks pass.

## Validation After Rebuild

Run these checks before interpreting broader results:

```bash
python -m src.data.source_readiness --output reports/source_readiness.csv
python -m pytest -q
```

Then regenerate data-quality reports, leakage checks, ablations, walk-forward summaries, and baseline comparisons for the broader universe.

## Claim Boundary

Until this handoff is complete, report the project as:

> A reproducible five-ticker Thai equity DRL pilot, with a separately validated point-in-time SET50/SET100 archive and an isolated, training-disabled V3 membership-development capsule.

Do not describe current results as a complete SET50/SET100 performance study until the required external files are present, validated, and used in the rebuilt experiments.
