# External Data Intake Checklist

Last updated: 2026-06-10

Use this checklist after a provider sends the files requested in `docs/external_data_provider_request.md`. The goal is to make each incoming data package auditable before any full SET50/SET100 result is reported.

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
| SET50/SET100 universe | `data/reference/set50_or_set100_universe.csv` |
| Sector mapping | `data/reference/sector_mapping.csv` |
| Historical official macro | `data/external/bot_historical_macro.csv` |
| Historical news or sentiment | `data/external/historical_news.csv` |
| Fundamentals | `data/external/fundamentals.csv` |

## 3. Fill The Manifest

Create a provider-data manifest from:

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
```

Acceptance criteria:

- command exits with status `0`,
- every row in `reports/source_readiness.csv` has `ready=True`,
- no required file is empty,
- no required column is missing.

## 5. Manual Coverage Checks

Before rebuilding features, verify:

- every ticker in `data/reference/set50_or_set100_universe.csv` appears in `data/reference/sector_mapping.csv`,
- ticker format is consistent across all files,
- official macro rows cover the modeling window or have a clear release-date rule,
- historical news or sentiment overlaps 2021-2024,
- fundamentals include period-end dates and either report/effective dates or a documented reporting-lag assumption,
- license terms allow the intended research reporting.

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
python -m pytest -q
python -m src.data.quality_report \
  --config config/real_ohlcv_full_external.yaml \
  --output-dir reports/data_quality_full_external
```

Then run leakage checks through `src.data.leakage.assert_no_leakage` or the existing test suite, followed by baselines, algorithm comparisons, ablations, walk-forward validation, and regime tests.

## Claim Boundary

If any intake step fails, keep the project wording as:

```text
A reproducible five-ticker Thai equity DRL pilot with implemented scaffolding for broader SET50/SET100 data.
```
