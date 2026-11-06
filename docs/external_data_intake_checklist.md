# External Data Intake Checklist

Last updated: 2026-07-31

Use this checklist after a provider sends the files requested in `docs/external_data_provider_request.md`. The goal is to make each incoming data package auditable before any full SET50/SET100 result is reported.

## Current Internal Source Boundary

- `[x]` Official point-in-time SET50/SET100 membership and sector history for 2021–2024 is already preserved and receipt-bound.
- `[x]` The generated acquisition union contains 138 tickers; it is not a static model membership mask.
- `[x]` The dated archive, rebuild command, quality gates, and limitations are documented in `docs/set_constituent_history_intake.md`.
- `[!]` Do not overwrite `data/reference/set50_set100_membership_2021_2024.csv`, `data/reference/set50_or_set100_universe.csv`, or `data/reference/sector_mapping.csv` with an unverified provider file.
- `[!]` A replacement membership package must first use separate candidate paths and must pass source-by-source checksum, interval, count, and as-of validation before promotion through a new release.

## 1. Preserve Originals

Save the provider package outside generated outputs before normalizing it:

```text
data/external/original/
```

Keep the original file names, provider email or download note, license terms, and received date. Do not overwrite these originals during cleaning.

## 2. Place Working Copies

Copy or normalize the working files to these project paths:

| source | required working path |
| --- | --- |
| Licensed adjusted OHLCV for the receipt-bound 138-ticker acquisition union | `data/external/v3_candidate/market_bars.csv` |
| Daily trading/suspension status | `data/external/v3_candidate/trading_status.csv` |
| Authoritative SET session calendar | `data/external/v3_candidate/set_sessions.csv` |
| Corporate actions | `data/external/v3_candidate/corporate_actions.csv` |
| Stable-instrument and ticker history | `data/external/v3_candidate/symbol_history.csv` |
| Alternate SET50/SET100 membership or sector package | Separate candidate paths; compare against the official dated archive before promotion |
| Historical official macro | `data/external/bot_historical_macro.csv` |
| Historical news or sentiment | `data/external/historical_news.csv` |
| Fundamentals | `data/external/fundamentals.csv` |

## 3. Fill The Manifest

For the five-file market package, copy:

```text
config/v3/market_package_manifest_template.yaml
```

Recommended candidate path:

```text
data/external/v3_candidate/market_package.yaml
```

Set every license and completeness declaration only when supported by provider documentation. Bind each CSV by project-relative path, SHA-256, byte count, and data-row count, then create `market_package.yaml.sha256`.

For an official SET request, use `docs/setsmart_academic_data_request_draft.md`. The preferred extract maps the five package components to SETSMART `CALENDAR`, `DAILY_SECURITY_TRADING`, `ADJUST_FACTOR`, `RIGHTS_BENEFIT`, `SECURITY`, trading-sign tables, and effective-dated symbol changes.

For the remaining multi-source files, create a provider-data manifest from:

```text
data/reference/external_data_manifest_template.csv
```

Recommended working path:

```text
data/external/external_data_manifest.csv
```

For each source, fill:

- provider,
- access method,
- license or terms,
- whether redistribution is allowed,
- whether publication of derived results is allowed,
- exact date range,
- symbols or coverage,
- columns,
- raw file path,
- received date,
- missing-value rate or `unknown` before profiling.

Append a summary row to `data/data_manifest.csv` only after the file is accepted for use.

## 4. Structural Readiness

Run:

```bash
python -m src.data.source_readiness --output reports/source_readiness.csv
python -m src.data.intake_validation \
  --output reports/external_intake_validation.csv \
  --markdown-output reports/external_intake_validation.md
python -m scripts.build_v3_market_package_gap \
  --project-root . \
  --candidate-manifest data/external/v3_candidate/market_package.yaml \
  --require-pass
python -m scripts.build_v3_investability_policy_readiness --project-root .
```

If provider files are being staged outside the default paths, pass explicit paths:

```bash
python -m src.data.intake_validation \
  --universe data/reference/set50_or_set100_universe.csv \
  --sector-mapping data/reference/sector_mapping.csv \
  --official-macro data/external/bot_historical_macro.csv \
  --historical-news data/external/historical_news.csv \
  --fundamentals data/external/fundamentals.csv \
  --manifest data/external/external_data_manifest.csv \
  --start 2021-01-01 \
  --end 2024-12-31 \
  --output reports/external_intake_validation.csv \
  --markdown-output reports/external_intake_validation.md
```

Acceptance criteria:

- command exits with status `0`,
- every row in `reports/source_readiness.csv` has `ready=True`,
- every row in `reports/external_intake_validation.csv` has `passed=True`,
- `reports/external_intake_validation.md` summarizes the same checks as `PASS`,
- no required file is empty,
- no required column is missing.
- the broader-universe and sector readiness rows are bound to the exact official archive receipt and sidecar;
- licensed OHLCV is joined through date-aware membership, not by treating the 138-ticker acquisition union as active on every date.
- `reports/data_quality_v3/v3_market_package_gap_20260731.json` reports `licensed_market_package_eligible=true`;
- every active member/session key has either a valid trading bar or an explicit suspended, halted, or no-trade status;
- every active trading key has a valid adjusted OHLCV row;
- all 62 currently observed membership-exit transitions have contiguous status through a valid liquidation bar or explicit terminal settlement;
- the authoritative session package exactly matches the membership grid, or the grid is rebuilt before reassessment;
- corporate actions and symbol intervals pass uniqueness, timing, factor, successor, overlap, and lineage checks;
- research use and publication of derived results are explicitly allowed; and
- the V3 market-package manifest sidecar and all five path/hash/byte/row bindings pass.

## 5. Manual Coverage Checks

Before rebuilding features, verify:

- every acquired ticker is declared in `data/reference/set50_or_set100_universe.csv`,
- every modeled date uses the SET50/SET100 membership active on that date,
- sector labels come from the dated archive; `sector_mapping.csv` is latest-label compatibility only,
- ticker format is consistent across all files,
- official macro rows cover the modeling window or have a clear release-date rule,
- historical news or sentiment overlaps 2021-2024,
- fundamentals include period-end dates and either report/effective dates or a documented reporting-lag assumption,
- license terms allow the intended research reporting.
- a missing bar is never being used as an inferred suspension;
- index removal is never being used as evidence that a holding was sold;
- same-session bar/status availability is used only for an after-close later decision, not an unproved same-close fill;
- every historical sector aggregate uses `sector_as_of` and active membership, not `sector_mapping.csv`.

## 6. Rebuild

After the checks pass, follow:

```text
docs/full_external_rebuild.md
```

Do not skip the staged outputs. They make it possible to identify which feature group introduced coverage or leakage problems.

## 7. Validation Before Claims

Before describing results as a full SET50/SET100 study, run:

```bash
python -m src.data.source_readiness --output reports/source_readiness.csv
python -m src.data.intake_validation \
  --output reports/external_intake_validation.csv \
  --markdown-output reports/external_intake_validation.md
python -m scripts.build_v3_market_package_gap \
  --project-root . \
  --candidate-manifest data/external/v3_candidate/market_package.yaml \
  --require-pass
python -m pytest -q
python -m src.data.quality_report \
  --config config/real_ohlcv_full_external.yaml \
  --output-dir reports/data_quality_full_external
```

Then run leakage checks through `src.data.leakage.assert_no_leakage` or the existing test suite, followed by baselines, algorithm comparisons, ablations, walk-forward validation, and regime tests.

## Claim Boundary

If any intake step fails, keep the project wording as:

```text
A reproducible five-ticker Thai equity DRL pilot, with a separately validated
point-in-time SET50/SET100 archive and an isolated, training-disabled V3
membership-development capsule.
```
