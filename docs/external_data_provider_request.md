# External Data Provider Request

Last updated: 2026-06-09

This request packet is for obtaining the five external files needed to move the project from the reproducible five-ticker pilot to a full SET50/SET100 real-data study.

## Study Window

Required modeling window:

```text
2021-01-01 through 2024-12-31
```

Preferred buffer window, if available:

```text
2020-10-01 through 2025-03-31
```

The buffer helps compute lookback features and verify split boundaries.

## Requested Files

Please provide CSV or UTF-8 text exports with one header row, comma delimiters, ISO dates, and one logical observation per row. Keep ticker symbols consistent across files.

| file | required path in project | minimum columns | preferred extra columns |
| --- | --- | --- | --- |
| SET50/SET100 universe | `data/reference/set50_or_set100_universe.csv` | `ticker` | `symbol`, `company_name`, `exchange`, `index_name`, `start_date`, `end_date` |
| Sector membership | `data/reference/sector_mapping.csv` | `ticker`, `sector` | `industry_group`, `industry`, `source`, `effective_date`, `end_date` |
| Historical official macro | `data/external/bot_historical_macro.csv` | `period`, `metric`, `value` | `release_date`, `source_table`, `unit`, `revision_flag` |
| Historical Thai news or sentiment | `data/external/historical_news.csv` | `date`, `ticker` | `published_at`, `title`, `summary`, `text`, `publisher`, `url`, `sentiment_score`, `language` |
| Fundamentals | `data/external/fundamentals.csv` | `ticker`, `metric`, `period_end`, `value` | `report_date`, `effective_date`, `statement_type`, `currency`, `unit`, `source` |

## Field Notes

Ticker universe:

- Use Yahoo-compatible tickers such as `PTT.BK` when possible.
- If the export uses bare Thai symbols such as `PTT`, note that `.BK` must be appended before ingestion.
- Include the exact SET50 or SET100 membership definition used: point-in-time membership is preferred; static membership is acceptable only if clearly labeled.

Sector membership:

- Point-in-time sector membership is preferred.
- Static sector labels are acceptable for the next run if they cover every ticker in the universe.
- Every ticker in `set50_or_set100_universe.csv` must have one sector row.

Official macro:

- Bank of Thailand series are preferred for official Thai macro variables.
- Release dates are strongly preferred. If absent, the pipeline uses `last_business_day_following_month`.
- Use long format when possible: `period,metric,value,release_date`.

Historical news or sentiment:

- Rows can be raw text/news or precomputed sentiment scores.
- If raw text is supplied, include title, summary, or body text.
- If sentiment is supplied, make the score direction clear, such as positive values meaning positive sentiment.
- `published_at` should include time zone when available; otherwise use local Thailand date.

Fundamentals:

- Quarterly data are preferred over annual-only data.
- Include report or effective dates if available; otherwise the pipeline applies a conservative reporting lag.
- Use stable metric names across tickers and periods.

## Acceptance Check

After placing files at the required paths, run:

```bash
python -m src.data.source_readiness --output reports/source_readiness.csv
```

The file package is structurally accepted only when the command exits with status `0`.

Then run the full rebuild guide:

```text
docs/full_external_rebuild.md
```

## Email Template

Subject:

```text
Request for SET50/SET100 historical data exports for student research
```

Body:

```text
Hello,

I am preparing a student research project on deep reinforcement learning for Thai equity portfolio allocation. Could you provide or advise how to export the following CSV datasets for 2021-01-01 through 2024-12-31, preferably with a buffer from 2020-10-01 through 2025-03-31?

1. SET50 or SET100 ticker universe and membership history.
2. Sector membership for every ticker in the universe.
3. Historical Bank of Thailand official macro series with release dates where available.
4. Historical Thai company news, disclosures, or precomputed sentiment scores by ticker and date.
5. Quarterly or annual fundamentals by ticker, metric, period end, and report/effective date.

The expected schemas and project file names are listed in the attached request document. CSV exports with ISO dates and one header row are preferred.

Please also include license or usage restrictions for research reporting, redistribution, and publication of derived results.

Thank you.
```

## Current Claim Boundary

Until these files are present and the readiness check passes, the project should be described as a reproducible five-ticker Thai equity DRL pilot with implemented scaffolding for broader SET50/SET100 data.
