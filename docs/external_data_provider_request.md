# External Data Provider Request

Last updated: 2026-07-31

This request packet is for obtaining the remaining licensed or explicitly authorized data needed to move the project from the reproducible five-ticker pilot to a broad SET50/SET100 real-data study.

The project already has checksum-bound official SET50/SET100 membership and sector history for 2021–2024. The remaining highest-priority market input is adjusted OHLCV and corporate-action history for the 138-ticker acquisition union. A provider does not need to resend constituent history unless it is a licensed package with more exact effective dates or trading-status fields.

## Preferred Official SET Extract

The closest documented match is a licensed SETSMART Enterprise PSIMS extract. Request the following tables/fields rather than relying on public quote-page scraping:

| V3 component | Preferred SETSMART/SET source |
| --- | --- |
| Authoritative sessions | `CALENDAR.D_TRADE` |
| Raw OHLCV and daily state | `DAILY_SECURITY_TRADING`, including `D_TRADE`, `I_SECURITY`, `Z_OPEN`, `Z_HIGH`, `Z_LOW`, `Z_CLOSE`, `Q_TOTAL_VOLUME`, `N_STATUS`, and `R_ADJUST_FACTOR` |
| Adjustment factors | `ADJUST_FACTOR`, including effective dates and action identifiers |
| Dividends, rights, splits, and capital events | `RIGHTS_BENEFIT` |
| Stable identity | `SECURITY.I_SECURITY`, current symbol, and ISIN |
| Suspension/halt intervals | `SIGN_POSTING`, `SIGN_LIFTING`, posting details, and cause tables |
| Symbol changes | Effective-dated A2/reference-data events with old symbol, new symbol, stable security ID, and effective date |

The public SET pages remain useful for manual spot checks, but they are not substitutes for a complete licensed package or written permission for research/HPC use.

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
| Licensed adjusted OHLCV | `data/external/v3_candidate/market_bars.csv` | `date`, `ticker`, OHLC, `adjusted_close`, `volume`, `bar_available_at`, `source_record_id` | adjustment factor, currency, stable instrument ID |
| Daily trading status | `data/external/v3_candidate/trading_status.csv` | `date`, `ticker`, `trading_status`, `status_available_at`, `source_record_id` | reason code, status start/end timestamp |
| Authoritative SET sessions | `data/external/v3_candidate/set_sessions.csv` | `date`, `session_available_at`, `source_record_id` | open/close timestamps, half-day flag |
| Corporate actions | `data/external/v3_candidate/corporate_actions.csv` | action ID/type, ticker, announcement/effective date, factor/cash/currency/successor, source ID | ex/record/payment dates, stable instrument ID |
| Stable-instrument symbol history | `data/external/v3_candidate/symbol_history.csv` | instrument ID, ticker, effective-from/to, source ID | prior/successor ticker and event ID |
| Optional enhanced SET50/SET100 membership | Separate candidate path; do not overwrite the official archive | `symbol`, `index_name`, `effective_from`, `effective_to` | announcement date, exact corporate-action effective time, trading status, source record ID |
| Optional enhanced sector membership | Separate candidate path; do not overwrite the official archive | `ticker`, `sector`, `effective_from`, `effective_to` | industry group, industry, announcement date, source record ID |
| Historical official macro | `data/external/bot_historical_macro.csv` | `period`, `metric`, `value` | `release_date`, `source_table`, `unit`, `revision_flag` |
| Historical Thai news or sentiment | `data/external/historical_news.csv` | `date`, `ticker` | `published_at`, `title`, `summary`, `text`, `publisher`, `url`, `sentiment_score`, `language` |
| Fundamentals | `data/external/fundamentals.csv` | `ticker`, `metric`, `period_end`, `value` | `report_date`, `effective_date`, `statement_type`, `currency`, `unit`, `source` |

## Field Notes

Ticker universe:

- The acquisition list is `data/reference/set50_or_set100_universe.csv` and contains 138 `.BK` tickers.
- A bare SET symbol is acceptable if the mapping to `.BK` and any renamed/delisted successor is documented.
- Include adjusted-price and corporate-action methodology. Raw closes without dividend/split handling are not sufficient for long-horizon return claims.

Sector membership:

- The internal official archive already provides point-in-time sectors.
- Any provider replacement must include effective intervals and source provenance.
- A static sector label is not acceptable for historical modeling because at least `MINT.BK` and `NEX.BK` change sectors during the study.

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
python -m scripts.build_v3_market_package_gap \
  --project-root . \
  --candidate-manifest data/external/v3_candidate/market_package.yaml \
  --require-pass
```

The market package must use a completed copy of `config/v3/market_package_manifest_template.yaml`. The file package is structurally accepted only when both commands exit with status `0` and the V3 receipt states `licensed_market_package_eligible=true`.

The gate additionally requires every observed membership exit to have continuous daily status until the first valid liquidation bar or an explicit terminal settlement. The current membership panel contains 62 such exits; index removal is not treated as an executed sale.

Then run the full rebuild guide:

```text
docs/full_external_rebuild.md
```

For local intake, manifest, and acceptance steps after the provider sends files, use:

```text
docs/external_data_intake_checklist.md
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

1. `CALENDAR`, `DAILY_SECURITY_TRADING`, and `ADJUST_FACTOR` for the attached 138-ticker acquisition union, retaining `I_SECURITY` and availability/correction timestamps.
2. `SIGN_POSTING`, `SIGN_LIFTING`, and cause/detail records for daily suspension and halt reconstruction.
3. `RIGHTS_BENEFIT`, `SECURITY`, and effective-dated old/new-symbol events for corporate-action and stable-instrument continuity.
4. Historical Bank of Thailand official macro vintages with actual release dates where available.
5. Historical Thai company news, disclosures, or precomputed sentiment scores by ticker and publication date.
6. Quarterly or annual fundamentals by ticker, metric, period end, and report/effective date.
7. Optionally, enhanced point-in-time membership or sector records if they include exact effective dates beyond the official archive already preserved by the project.

The expected schemas and project file names are listed in the attached request document. CSV exports with ISO dates and one header row are preferred.

Please also include license or usage restrictions for research reporting, redistribution, and publication of derived results.

Please confirm whether the two student authors and advisor may store and process the access-controlled raw extract on the KVIS BistKA HPC cluster, retain it for the project period, and publish non-reconstructible aggregate metrics, figures, and model results while prohibiting raw-data redistribution.

Thank you.
```

## Official Academic Request Route

SET's research-support page asks for:

1. an institution or faculty letter endorsed by the advisor;
2. the completed data-use request with the exact fields required;
3. the research proposal; and
4. submission to SET for review, which the page says normally takes about two weeks.

Because the page is written for university faculties, confirm KVIS eligibility rather than assuming it. Direct academic-support questions to `ResearchDepartment@set.or.th`; direct extract, licensing, and pricing questions to `infoproducts@set.or.th`.

A project-specific English letter draft is available at `docs/setsmart_academic_data_request_draft.md`.

## Official References

- [SET academic SETSMART data-support process](https://www.set.or.th/th/education-research/research/database/setsmart/approval-process)
- [SETSMART Enterprise PSIMS data model, version 6.2.1](https://media.set.or.th/set/Documents/2022/Apr/SETSMART_MIS_SY_SSE_PSIMS_DataModel_V6_2_1.pdf)
- [SET Reference Data and Corporate Action specification](https://media.set.or.th/set/Documents/2022/Jul/02_1_Reference_Data_Specification.pdf)
- [SET historical-data request service](https://www.set.or.th/en/services/connectivity-and-data/data/historical)

## Current Claim Boundary

Until licensed OHLCV and the remaining multi-source files are present and the readiness checks pass, the model results should be described as a reproducible five-ticker Thai equity DRL pilot. The validated 2021–2024 point-in-time SET50/SET100 membership and sector archive may be reported as a separate completed data-engineering milestone.
