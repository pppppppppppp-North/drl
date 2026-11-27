# The Sealed Holdout Stage: Built and Tested, Not Run (M6)

Date: 2026-11-26
Status: **built, not run.** The real holdout has not been opened. Two draft locks wait for the authors' decisions and approval.
Code: `scripts/run_holdout_evaluation.py`, `src/data/stale_rows.py`, and one optional setting in `scripts/run_pit_walk_forward.py` (`data.end`); tests in `tests/test_run_holdout_evaluation.py` and `tests/test_stale_rows.py`.
Locks (drafts): `config/holdout_protocol_pit.yaml` (headline) and `config/holdout_protocol_five_stock.yaml` (secondary).

## What it is for

The test years (2023–2024) were read once on 2026-11-25 ([results](final_test_results_20261125.md)). The plan's last evaluation is the sealed holdout: Yahoo prices from 2025 on, sealed on 2026-11-06 with SHA-256 manifests ([decision](final_design_decision_20261125.md#stage-2-the-sealed-holdout-after-stage-1)). It evaluates the same locked designs once more, on years nobody has looked at, as two walk-forward folds:

- **2025:** trained on everything through 2024, evaluated on 2025.
- **2026:** trained on everything through 2025, evaluated from January to 4 November 2026.

## What the script does

1. **Checks, before anything is opened:**
   - the lock is `APPROVED`, with names and a date, for a holdout stage;
   - every seal holds (the files' bytes are hashed, not parsed);
   - the design's config matches its hash, and the code and configs are committed;
   - the matching test stage is already in `results/FINAL_TEST_LEDGER.jsonl`, and this stage is not;
   - the lock continues exactly the files the config reads, and each sealed part runs to the holdout end;
   - the output folder does not exist.
2. **Opening:** the ledger records the stage first, so an opened holdout counts as used even if a later step fails. Each sealed part is appended to the development file it continues, refusing any overlap, and written inside the output folder. The development files are not changed. The receipt records each file's rows and hash and the price step where the two files meet.
3. **Stale prices, before training:** the share of stale rows from 2025 on (zero volume, open = high = low = close = previous close) is written to `stale_rows.json` and printed. In 2024 it was 17.5% of stock rows. If the lock sets `stop_if_stale_share_above` and any file is above it, the stage stops there.
4. **The run:** the same runners, folds, seeds, seed ensemble with the band, statistics and figure as the test stage; receipts and one ledger line.

`--dry-run` prints the checks and the plan; it verifies the seals but opens no data. Both drafts are refused today: draft status, no names, not committed.

## Decisions for the authors before approval

1. **SET50 membership after 2024 (point-in-time only).** The official archive ends with the 2024 H2 list.
   - `hold_last_list` (in the draft): keep the 2024 H2 list for the whole holdout. No hindsight, and no download, but it ignores the 2025 and 2026 revisions. Its publication date stays 17 June 2024, so it is never used before it was known.
   - `as_recorded`: download the published 2025–2026 lists (needs approval), add them to the membership file and the price map, and set this.
2. **Stale prices.** The draft reports them and continues, with the same session calendar as the test years: a date is dropped when fewer than half the series traded. Set `stop_if_stale_share_above` (for example 0.25) to stop before training if the holdout is worse than 2024, so a decision can be made before any model is evaluated.
3. **Whether to run the five-stock holdout too.** It is secondary and carries the survivorship caveat, as in the test years.
4. **The design is unchanged** from the test locks: point-in-time cash-capped ensemble with the band, 100k timesteps, seeds 42–44; five-stock A2C at 300k, seeds 42–46. The checks since then (five seeds, 300k; [report](cash_cap_5seeds_300k_20261126.md)) support keeping it. `n_trials` stays 22 and 14, as in the test locks.
5. **Holdout end: 4 November 2026,** the last date every sealed part covers (the ETF's file ends then; the stocks end on 5 November).
6. **Who signs.** The test locks record one name; the co-author and the advisor have not signed.

## Which sealed file continues which

| Development file (read by the config) | Sealed folder | Part |
| --- | --- | --- |
| `data/raw/extended/prices_set100_union_2012_2024.csv.gz` | `data/holdout_sealed/2025_onward_set100_union` | stocks |
| `data/raw/extended/set100_2012_2024_additions/prices_set100_2012_2024_additions.csv.gz` | `data/holdout_sealed/2025_onward_set100_2012_2024_additions` | stocks |
| `data/raw/extended/prices_starter5_2012_2024.csv` (five-stock) | `data/holdout_sealed/2025_onward` | stocks |
| `data/raw/extended/index_set50_etf_2012_2024.csv` (both) | `data/holdout_sealed/2025_onward` | index |

Each download wrote its own copy of the ETF series, and their hashes differ, so Yahoo's dividend adjustment may differ between them. The development ETF file was written by the five-stock download (same time as `prices_starter5_2012_2024.csv`), so that download's sealed ETF rows continue it in both universes. The receipt records the step at the join, so a mismatch would show.

## Smoke test on synthetic data

Both stages ran end to end on 2026-11-26 without opening the real holdout. The script is `holdout_smoke.py` in the session scratchpad; it is not kept in the repository.

- **Inputs:** the real development files cut at 2022-12-30 (validation years only). The "holdout" was synthetic random-walk prices for every series from 2023-01-02 to 2026-11-04, with ten stale sessions planted in one series per file. It was sealed with the same function that sealed the real one.
- **What it exercised:** the checks, opening, continuing the files, the held membership list, the stale-row count (it found the ten planted rows in each file), the derived config, two folds × two seeds through the task pool (2,048 timesteps each), the seed ensemble with the band, and the aggregate tables.
- **What happened:** both stages finished. In the point-in-time 2026 fold, training ran to 2025-12-31 and evaluation to 2026-11-04, with 47 tradable members a session from the held 2024 H2 list. The five-stock stage ran A2C and every baseline for both folds. The synthetic returns mean nothing.
- **Tests:** 21 tests for the gate and 2 for the stale-row count, all passing. Across the whole suite, 677 pass. 54 fail, the same as before this change, because the clone lacks some git-ignored data files.

## How to approve and run

1. Make the decisions above in the lock(s), set `status: APPROVED`, `approved_by` and `approved_on`, and commit.
2. Dry run: `python -m scripts.run_holdout_evaluation --lock config/holdout_protocol_pit.yaml --dry-run`.
3. Run once: `python -m scripts.run_holdout_evaluation --lock config/holdout_protocol_pit.yaml --workers 3`, then the five-stock lock if chosen.

Time on the Mac with three workers: about an hour for the point-in-time stage (6 models at 100k; the test stage took 59 minutes) and about 25 minutes for the five-stock stage (10 A2C models at 300k).
