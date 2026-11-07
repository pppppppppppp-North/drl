# Point-in-Time SET50/SET100 Universe, 2012–2024 (Milestone M3)

Date: 2026-11-07
Evidence tier: **data infrastructure from official SET sources**. No model was trained or evaluated on this universe yet.
Module: `src/data/ingest_set_constituent_history_extended.py` · Tests: `tests/test_ingest_set_constituent_history_extended.py` (27 pass)

## Summary

The July archive covered 2021–2024. This work extends the point-in-time SET50/SET100 membership back to **2012** from 25 official SET files, so it now spans the same years as the M2 price history:

- **34 lists, 3,400 rows, 227 tickers, 25 sectors**, with exactly 50 SET50 and 100 SET100 members in every list and no gap between lists from 2012-01-01 to 2025-01-01.
- The 2021–2024 rows are copied **byte for byte** from the July archive. Its SHA-256 is pinned, and the build refuses to run if that file changes.
- Every list that prints its own changes was checked against the actual change from the list before it: **93 of 93 checks match**. That includes the 2020→2021 seam, where the 2021 H1 PDF's notes match the change from the new 2020 list to the July archive's first list.
- **89 tickers appear only in 2012–2020 lists.** A second Yahoo download raised local price coverage of 2012–2020 SET100 member-lists from 70% to 87%. 38 tickers, mostly delisted or merged, have no Yahoo prices at all (see "Survivorship gap").

## Sources

All files came from the SET's direct `media.set.or.th` links listed on the [SET50/SET100 constituents page](https://www.set.or.th/en/market/information/securities-list/constituents-list-set50-set100), the same way as the July 2021–2024 files. No login was needed.

| Years | Files | Format | Notes |
| --- | ---: | --- | --- |
| 2012–2013 | 4 | Excel, English and Thai sheets | No publication date printed |
| 2014 | 2 | Excel, Thai sheets only | Sectors mapped with the SET's own English/Thai summary tables |
| 2015–2020 | 19 | Two-page PDF | 12 scheduled lists and 7 between-review revisions |
| 2025–2026 | 6 | PDF | Archived with checksums, **not parsed**: they belong to the sealed holdout years |

## Outputs

| File | Contents |
| --- | --- |
| `data/reference/set50_set100_membership_2012_2024.csv` | One row per list and SET100 member: dates, `in_set50`, sector, company, source URL and hash. Same columns as the July file. |
| `…_2012_2024.csv.receipt.json` (+ `.sha256`) | Hashes of every source and output, quality gates, all change checks, corporate actions, limitations |
| `data/reference/set50_or_set100_universe_2012_2024.csv` | The 227 tickers for **price acquisition only**, with first/last dates and the number of lists each ticker is in |
| `data/reference/set_constituent_interval_profile_2012_2020.csv` | Members and additions/removals per list |
| `data/reference/set_symbol_changes_2012_2020.csv` | Six symbol changes and one merger, with both company names |
| `data/external/set_constituent_source_manifest_2012_2020.csv` | URL, SHA-256, size, retrieval time, dates and revision links for each source |
| `data/external/set_constituent_source_manifest_2025_2026_unparsed.csv` | The same for the six unparsed 2025–2026 files |

## How the lists are dated

- **Scheduled reviews** apply from the period start printed in the file.
- **Revisions** apply from their publication date, the same conservative rule as the July archive, because the files state no separate effective date. That covers six revisions: 2015 H1 (HEMRAJ out), 2016 H2 (PSH replaces PS), 2019 H1 (GLOW out), and 2019 H2, 2020 H1 and 2020 H2.
- **2016 H1** is a special case. The original list (published 2015-12-14) was replaced on 2015-12-30, *before* the period began, by a list that reflects the merger of BECL and BMCL into BEM. The replacement applies to the whole period. The original never applies: it is parsed, checked and kept in the receipt, but it contributes no rows.
- The 2012–2013 workbooks print no publication date, so `published_at` is the period start. That is the latest date by which the list was public, because the index used it from that day.

## Validation

1. **Sources:** every file matches its SHA-256 and byte count in the manifest. The publication dates and periods printed in each file match the manifest.
2. **Lists:** each has exactly 50 and 100 unique members, SET50 is inside SET100, and the two tables agree on sectors. Each PDF list is in the alphabetical order the PDF states.
3. **Printed changes (93 checks):**
   - Where a file prints Inclusion/Exclusion notes or "New Entry" marks, the change from the previous list matches them exactly.
   - A revision is checked against the list it replaces. Every symbol it removes must be named in its note (for example "HEMRAJ has free float lower than 20%"). Every symbol it adds must be named, starred, or on the replaced list's reserve list.
   - Only 2012 H1 (the first list) and 2012 H2 (no notes printed) have nothing to check against.
4. **Seam with the July archive:** the 2021 H1 PDF's own notes match the change from the last 2020 list to the July archive's first list. On SET50, BAM, COM7 and DELTA came in and BPP, IRPC and WHA went out.
5. **Agreement with the July parser:** on all seven 2019–2020 files the July parser can read, members, ranks and sectors are identical. The July parser cannot read the 2015–2018 files or the 2019 H1 revision, because their sector labels wrap onto a second line.
6. **July validator:** the combined file passes `constituent_history.validate_constituent_history`, run with a 2012-01-01 start.

### Corporate actions

Seven events explain differences that no printed note lists as an inclusion or exclusion. Each one is checked against the lists: the old symbol is in the earlier list, and the new symbol replaces it in the later one.

| Old | New | Kind | First list with the new symbol | Evidence |
| --- | --- | --- | --- | --- |
| KH | BCH | symbol change | 2013 H1 | Same company name (Bangkok Chain Hospital) |
| GSTEEL | GSTEL | symbol change | 2013 H2 | Same company name (G Steel) |
| KK | KKP | symbol change | 2014 H1 | Kiatnakin Bank; the 2014 H1 exclusion note already uses KKP |
| BGH | BDMS | symbol change | 2015 H2 | Same company name (Bangkok Dusit Medical Services) |
| TUF | TU | symbol change | 2016 H1 | Thai Union Frozen Products became Thai Union Group |
| BECL, BMCL | BEM | merger | 2016 H1 (replacement list) | Bangkok Expressway and Metro. The freed SET100 slot went to SGP, a reserve on the replaced list. |
| MTLS | MTC | symbol change | 2018 H2 | Muangthai Leasing became Muangthai Capital |

### What the parser handles

The July parser assumes one line per row. The 2015–2020 PDFs break that assumption in several ways, and each is covered by a test:

- **Wrapped cells.** Sector labels ("Information & Communication / Technology") and company names wrap onto a second line. When a cell is cut short, a unique word-aligned prefix match is accepted and recorded (for example ADVANC in 2015).
- **Reserve list.** Below rank 50, a reserve list restarts at rank 1. It is read as a separate list and must not overlap the members. In the July parser, a row that failed to parse could be filled by a reserve's rank instead.
- **Page drawing quirks:**
  - several files draw their pages at four times the usual size, so all distances scale with the font size;
  - highlighted rows are drawn twice, which doubles every letter, and the duplicate copy is removed.
- **Missing rank number.** One row has no rank number in the text layer: SAMART, rank 70 in the 2015 H1 revision. It is placed between ROBINS (69) and SAWAD (71) and recorded in the receipt.
- **Overlaid text.** A disclaimer overlaid on the 2019 H1 table is excluded. The July parser leaked it into two company names (GUNKUL and TU) when reading that file. That file is not in the July archive, so no July output is affected.
- **2014 workbooks.**
  - Columns that are labelled the same but hold swapped content are told apart by their values.
  - The 2014 H1 file marks five SET50 reserves inside its SET100 column; they are kept out of SET50.
  - "Construction Services", a sector added in 2014, is not in the 2012–2013 summary tables. The same companies carry that label in the 2015 English PDF, which a test checks.

## Results

- **34 lists:** 24 new ones for 2012–2020 plus the 10 July lists. The new lists are 17 scheduled reviews and 7 revisions; the original 2016 H1 list is held out.
- **227 tickers.** 89 of them appear only before 2021. Ten tickers change sector over time: construction firms move from Property Development to Construction Services in 2014 H2, RS moves from Media & Publishing to Commerce in 2019, IFEC from Home & Office Products to Energy & Utilities in 2016, and MINT and NEX change in 2023 (already in the July archive).
- **Turnover per scheduled review:**
  - 1–8 SET50 symbols are added (median 3) and 3–16 SET100 symbols (median 9).
  - A symbol change counts as one addition and one removal. The 2016 H1 replacement list counts as that period's review.
  - Each of the six publication-gated revisions changes exactly one name per index.

## Survivorship gap

All prices are Yahoo Finance daily history (`auto_adjust=True`) and stay **exploratory**. A member counts as covered in a list if one price series spans the list's dates, within 10 days.

**Two downloads:**

- **6 November:** the Tier B download in `data/raw/extended/prices_set100_union_2012_2024.csv.gz` covered only the 138 tickers of the 2021–2024 union. That union is chosen with hindsight, which is why the early years were thin: the names that left the index are exactly the ones missing.
- **7 November:** I requested the 101 tickers that had no price file, plus STECON (STEC's successor).
  - 58 returned 2012–2024 data: 169,342 rows in `data/raw/extended/set100_2012_2024_additions/`, which is git-ignored and has a SHA-256 receipt.
  - Their 2025–2026 rows went straight into the sealed holdout `data/holdout_sealed/2025_onward_set100_2012_2024_additions/`. Only row counts were printed.

**Renamed members** use the current symbol's history where Yahoo keeps it: KH→BCH, KK→KKP, BGH→BDMS, TUF→TU, MTLS→MTC and TMB→TTB. BECL and BMCL are not mapped to BEM, because BEM's pre-2016 series can belong to only one of them.

| Year | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SET100, 6 Nov | 50% | 52% | 58% | 59% | 65% | 72% | 76% | 84% | 88% | 93% | 91% | 93% | 96% |
| **SET100, 7 Nov** | **80%** | **82%** | **81%** | **82%** | **87%** | **91%** | **92%** | **91%** | **93%** | 93% | 91% | 93% | 96% |
| SET50, 6 Nov | 63% | 67% | 71% | 71% | 80% | 80% | 83% | 86% | 88% | 91% | 91% | 93% | 95% |
| **SET50, 7 Nov** | **79%** | **81%** | **82%** | **82%** | **87%** | **89%** | **89%** | **88%** | **90%** | **92%** | 91% | 93% | 95% |

Over 2012–2020, coverage rose from 70% to 87% of SET100 member-lists and from 78% to 86% of SET50. 189 of the 227 tickers now have prices.

**What is still missing:**

- **No Yahoo prices at all (38 tickers).** Mostly delisted, merged or taken-private names, for example INTUCH, DTAC, ROBINS, GLOW, ESSO, BIGC, HEMRAJ, PS, BECL, BMCL, BPP, THCOM, TICON, MAKRO, SVI, and GULF before its 2025 amalgamation. Their absence is the survivorship bias that remains in any Yahoo backtest.
- **Partial series:**
  - SCB.BK starts on 2022-04-20. That series is SCB X after the holding-company restructuring; the Siam Commercial Bank series for 2012–2022 is not on Yahoo, which leaves a SET50 bank uncovered in 28 lists.
  - BANPU.BK starts on 2016-10-28.
  - STECON.BK starts on 2024-10-30.
- **A reused symbol.** KK.BK on Yahoo begins in 2020 and is a different company from the KK (Kiatnakin Bank) of 2012–2013. It is never used.

`data/raw/extended/set100_2012_2024_additions/member_price_map_2012_2024.csv` records, for every list and member, which series to use: the own symbol, the current symbol, or none. It is the input for the 2012–2024 session mask.

## Limitations

- **Revision dates.** Revisions apply from their publication date, not an exact exchange-effective date.
- **2012–2014 gaps.** The SET archive offers no between-review update files for 2012–2014, so any replacement inside those periods is not reflected.
- **Thai names in 2014.** Company names for 2014 are in Thai, because the 2014 workbooks have Thai sheets only.
- **Symbols as printed.** Tickers are the symbols printed in each list, and renamed securities appear under each historical symbol. Use `set_symbol_changes_2012_2020.csv` to link them when fetching prices.
- **Acquisition only.** The 227-ticker union is for fetching prices only. Using it as a static model universe would reintroduce survivorship and future-membership bias. Use the dated membership file for every as-of join.
- **Not publication-grade.** The archive is official, but the Yahoo prices are not licensed. Results built on them stay exploratory.

## Use and rebuild

```python
from pathlib import Path
from src.data.ingest_set_constituent_history_extended import (
    load_membership_2012_2024,
    membership_as_of_2012_2024,
)

history = load_membership_2012_2024(Path("data/reference/set50_set100_membership_2012_2024.csv"))
set50_on_date = membership_as_of_2012_2024(history, "2016-03-01", index_name="SET50")
```

```bash
pip install -r requirements-data-extensions.txt   # xlrd, for the 2012-2014 workbooks
python -m src.data.ingest_set_constituent_history_extended
python -m pytest tests/test_ingest_set_constituent_history_extended.py
```

## Next steps

1. **Point-in-time walk-forward.** Prices are fetched (done 7 November). Next, build the 2012–2024 session membership mask from `member_price_map_2012_2024.csv` and run the walk-forward on the point-in-time universe, which is the M3 exit criterion. Report the 38 names with no prices, and the SCB and BANPU gaps, as an explicit survivorship limitation.
2. **Licensed data (Tier A).** The SETSMART package is still needed for delisted names and publication-grade results.
3. **2025–2026 lists.** Parse the archived files only when the sealed holdout is opened for the final evaluation.
