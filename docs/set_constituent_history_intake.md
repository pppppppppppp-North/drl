# Official SET Constituent-History Intake

Last updated: 2026-07-31

This intake builds the project’s point-in-time SET50/SET100 membership and sector archive for 2021–2024. It is a standalone structural-data milestone and does not modify the locked five-ticker V2 preregistration, environment, campaign specification, or deployed release.

## Source package

- Archive index: [Stock Exchange of Thailand constituent archive](https://www.set.or.th/en/market/information/securities-list/constituents-list-set50-set100)
- Preserved originals: `data/external/original/set_constituent_history/`
- Source manifest: `data/external/set_constituent_source_manifest.csv`
- Records: eight scheduled half-year publications plus two official mid-period revisions
- Effective coverage: 2021-01-01 through 2025-01-01 exclusive

Every source row declares its original URL, local path, retrieval timestamp, byte count, and SHA-256. The importer fails before parsing if a source file differs from that declaration.

## Rebuild

From the project root:

```bash
python -m src.data.ingest_set_constituent_history --project-root .
python -m scripts.build_set_constituent_archive_report --project-root .
python -m src.data.source_readiness --output reports/source_readiness.csv
```

The readiness command intentionally exits nonzero until historical news/sentiment and licensed fundamentals are also present.

Run the focused verification suite:

```bash
python -m pytest -q \
  tests/test_ingest_set_constituent_history.py \
  tests/test_constituent_history.py \
  tests/test_source_readiness.py
```

## Generated outputs

| Artifact | Purpose |
| --- | --- |
| `data/reference/set50_set100_membership_2021_2024.csv` | Dated SET100 membership with SET50 flags, company, sector, publication date, source URL, and source hash |
| `data/reference/set50_or_set100_universe.csv` | 138-ticker acquisition union; never a static model membership mask |
| `data/reference/sector_mapping.csv` | Latest-label compatibility mapping; dated scientific joins must use the membership archive |
| `reports/set_constituent_archive_readiness.csv` | Per-interval counts, sector breadth, additions, removals, and turnover |
| `reports/data_sources/set_constituent_archive_20260731.json` | Source, quality-gate, eligibility, and output-hash receipt |
| `reports/data_sources/set_constituent_archive_20260731.json.sha256` | Receipt checksum |
| `reports/set_constituent_archive_readiness_20260731.md` and `.sha256` | Reader-facing methodology and findings |
| `reports/figures/set_constituent_turnover_2021_2024.svg` and `.sha256` | Reproducible turnover visualization |

## Acceptance contract

The build accepts a record only if:

- its raw bytes and SHA-256 match the source manifest;
- all ranks 1–50 and 1–100 are present exactly once;
- each symbol, company, and sector is nonblank;
- ticker conversion produces a valid `.BK` symbol;
- SET50 is a subset of SET100;
- the two index tables agree on the sector of every shared ticker;
- each normalized interval has exactly 50 SET50 and 100 SET100 members;
- effective intervals are continuous and non-overlapping;
- no record becomes available before its publication date; and
- the generated union, sector file, readiness table, and membership history are bound into the receipt by path, rows, bytes, and SHA-256.

## As-of usage

Use `src.data.constituent_history.membership_as_of` for any historical selection:

```python
from pathlib import Path
from src.data.constituent_history import membership_as_of

members = membership_as_of(
    Path("data/reference/set50_set100_membership_2021_2024.csv"),
    "2023-08-01",
    index_name="SET100",
)
```

The interval convention is half-open:

```text
effective_from <= date < effective_to_exclusive
```

Do not filter a historical panel with all 138 acquisition tickers at every date. The union intentionally includes securities that entered or left the indices at different times.

## Known classification changes

- `MINT.BK`: Food & Beverage through 2023 H1; Tourism & Leisure from 2023 H2.
- `NEX.BK`: Electronic Components in 2023 H1; Automotive from 2023 H2.

These genuine changes are why the compatibility sector map cannot replace an as-of sector join.

## Scientific boundary

The receipt marks the archive as structurally ready and point-in-time eligible for membership and sector lookup. It marks model input and publication-grade performance claims as ineligible because:

- licensed historical OHLCV for the 138-ticker union is still missing;
- the current Yahoo OHLCV covers only five pilot tickers and is private-research-only;
- the isolated V3 membership panel and mask-aware environment mechanics are implemented, but final investability still needs licensed trading-status/corporate-action inputs; and
- the V3 protocol is intentionally an unsigned, training-disabled draft and must be approved and checksum-locked before train/validation execution.

For the two mid-period revisions, the archived PDFs do not state a separate exchange-effective transition date. The archive applies them from their publication date, which prevents future-information leakage but does not assert the exact corporate-action effective instant.

The separate V3 close-to-close development path is more conservative: it keeps the prior record on the publication session and activates each same-day revision on the next observed SET session. See `reports/v3_membership_mask_readiness_20260731.md`.
