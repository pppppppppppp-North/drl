# Coverage of Each Input Source, Point-in-Time SET50 (M4)

Date: 2026-11-27
Status: **done for the sources the project has.** Window: the training and validation years, 4 January 2012 to 30 December 2022. The 2023–2024 test years and the sealed holdout are not read.
Code: `scripts/source_coverage_table.py` (tests in `tests/test_source_coverage_table.py`); output in `~/drl_runs/source_coverage_20261127/`.

## Why this table

`src.data.source_readiness` checks that a file exists, has the right columns and, for the archive files, matches its receipt. Run from the project folder it reports **3 of 5** sources ready: the broader universe, the sector map and the official macro history. Historical news and licensed fundamentals have no file. (In the GitHub clone it reports 1 of 5, because the archive's receipt in `reports/` is git-ignored.) It does not say how much of the modelling window a source covers, or how much is carried forward or filled. This table measures that for every source the title's "multi-source data" refers to.

## Coverage, 2012–2022

Stock-level sources are counted per SET50 member-session (50 members × 2,678 sessions). Market-level sources are counted per Thai session.

| Source | Unit | Units | From that session | Carried, stale or filled | Missing | Longest carry (days) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stock prices (Yahoo `.BK`) | member-session | 133,900 | 86.3% | 0.0% | 13.7% | |
| Technical inputs (11) | member-session | 133,900 | 85.6% | 0.7% | 13.7% | |
| Dated SET sector (membership archive) | member-session | 133,900 | 84.6% | 0.0% | 15.4% | |
| SET50 ETF (TDEX), market inputs | session | 2,678 | 100.0% | 0.0% | 0.0% | 0 |
| BOT policy rate (MPC decisions) | session | 2,678 | 100.0% | 0.0% | 0.0% | 69 |
| Macro proxy USD/THB | session | 2,678 | 99.9% | 0.1% | 0.0% | 1 |
| Macro proxy Brent | session | 2,678 | 96.0% | 4.0% | 0.0% | 4 |
| Macro proxy WTI | session | 2,678 | 96.6% | 3.4% | 0.0% | 4 |
| Macro proxy gold | session | 2,678 | 96.6% | 3.4% | 0.0% | 4 |
| Macro proxy US 10-year yield | session | 2,678 | 96.7% | 3.3% | 0.0% | 4 |
| Fundamentals (P/E, P/BV, ROE, D/E, yield) | — | 0 | 0% | | 100% | |
| News sentiment | — | 0 | 0% | | 100% | |
| Investor flow (by investor type) | — | 0 | 0% | | 100% | |
| Release-dated CPI | — | 0 | 0% | | 100% | |

How each column is counted:
- **Stock prices:** "missing" means the member has no Yahoo series. These are mostly delisted or renamed members, the survivorship gap of the [point-in-time report](pit_walk_forward_20261107.md). "Stale" means a zero-volume row; there are 0.02% of them in these years. The stale rows that matter are in 2024, which the [test-year report](final_test_results_20261125.md) discloses.
- **Technical inputs:** "filled" means a series' first 20 sessions, whose 20-session inputs the technical builder sets to 0. A member needs 60 sessions of history to trade, so the agent's 30-session windows never contain these rows.
- **Dated sector:** every member has its own list's sector. A member-session counts as covered when at least one other priced SET100 member in the same sector has a real return that session; that holds for 98.0% of priced member-sessions.
- **Policy rate:** this is a step series, so the rate in force is a real value on every session and is not counted as carried. Its "longest carry" is the longest gap since a decision: 87 decisions fall in the window.
- **Macro proxies:** "carried" is the latest earlier close, used on sessions when the foreign market was shut. Yahoo reports zero volume for USD/THB and the 10-year yield on every row, so volume is not used as a staleness test for them.
- **Missing sources:** each would need a data file the project does not have:
  - fundamentals: the licensed SETSMART package (the request has been ready since 2026-09-01 but not sent);
  - news: a news history for 2012–2022;
  - investor flow: SET's export of daily trading value by investor type;
  - CPI: a series with its release dates.

Stock-level coverage by year (share of SET50 member-sessions):

| Year | Member-sessions | No Yahoo series | Priced | Stale | Filled (warm-up) | With a sector peer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2012 | 12,250 | 21.0% | 79.0% | 0.00% | 6.53% | 75.4% |
| 2013 | 12,250 | 19.0% | 81.0% | 0.01% | 0.00% | 79.0% |
| 2014 | 12,250 | 18.0% | 82.0% | 0.07% | 0.00% | 82.0% |
| 2015 | 12,150 | 18.0% | 82.0% | 0.00% | 0.00% | 82.0% |
| 2016 | 12,200 | 13.7% | 86.3% | 0.07% | 0.16% | 83.3% |
| 2017 | 12,100 | 11.0% | 89.0% | 0.00% | 0.00% | 86.0% |
| 2018 | 12,250 | 11.0% | 89.0% | 0.00% | 0.00% | 88.0% |
| 2019 | 12,200 | 12.8% | 87.2% | 0.00% | 0.16% | 87.2% |
| 2020 | 12,150 | 9.3% | 90.7% | 0.01% | 0.33% | 90.4% |
| 2021 | 12,050 | 8.0% | 92.0% | 0.02% | 0.17% | 89.0% |
| 2022 | 12,050 | 9.0% | 91.0% | 0.00% | 0.00% | 88.0% |

## Reading

- **The sources the agent uses cover the window well.** That holds for prices, the technical inputs, the SET50 ETF, the policy rate, the macro proxies and the sector. The ETF and the policy rate have a value on every session, and the macro proxies are carried over at most four days. The weak point is the stock universe itself: 13.7% of SET50 member-sessions have no Yahoo series. The gap falls from 21% in 2012 to 8–9% in 2021–2022. This is the survivorship gap the point-in-time design already reports, and the licensed SETSMART data would close it.
- **Four of the title's sources have no data at all.** These are fundamentals, news sentiment, investor flow and a release-dated CPI. Their code is ready (see M4 in `IMPROVEMENT_MILESTONES.md`), but every model in this project was trained without them. "Multi-source" in the title therefore means prices, a market index, official and market macro series, and the sector. It does not include fundamentals, news or flows.
- **Readiness against coverage:** the check's "3 of 5" counts the universe and the sector map as two sources. Both come from the one membership archive, and neither says how much of the window has prices. Use this table for coverage.

## Command

```bash
python -m scripts.source_coverage_table --config config/pit_source_ablation_cashcap.yaml \
  --output-dir ~/drl_runs/source_coverage_20261127
```
