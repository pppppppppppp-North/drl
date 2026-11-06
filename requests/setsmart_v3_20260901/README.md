# SETSMART V3 Research Data Request Packet

**Status: DRAFT — UNSENT**  
**Prepared:** 1 September 2026  
**Project:** Deep Reinforcement Learning for Long Term Profit Optimization in Thai Stock Markets Using Multi-Source Data

This packet is ready for advisor and institutional review. It has not been emailed, uploaded, signed, or submitted to SET or any data provider. It requests the smallest licensed market package that the current V3 intake gate can validate.

## Packet contents

| File | Purpose |
| --- | --- |
| `cover_letter_DRAFT_UNSENT.md` | Concise request letter for advisor/institution review |
| `requested_five_file_schema.csv` | Exact 33-field delivery contract used by the current V3 validator |
| `set50_set100_acquisition_union_138.csv` | Byte-for-byte copy of the canonical 138-ticker acquisition union |
| `data_protection_license_checklist.md` | Questions and safeguards that must be resolved before receipt or use |
| `example_market_package_manifest_NONDATA.yaml` | Filled structural example only; deliberately ineligible and not a data receipt |
| `SHA256SUMS` | Deterministically sorted hashes for every payload file above |
| `SHA256SUMS.sha256` | Sidecar binding `SHA256SUMS` itself |

## Requested scope

The preferred delivery window is **2020-10-01 through 2025-03-31**. The core development interval is 2021–2024; the earlier and later buffers support lag construction, exit-path reconciliation, and boundary checks. Any post-2024 records must remain outside model development unless a separately approved protocol designates an untouched holdout.

The five normalized CSV components are:

1. `market_bars.csv` — licensed adjusted daily OHLCV;
2. `trading_status.csv` — explicit daily trading, suspension, halt, no-trade, delisting, or not-yet-listed state;
3. `set_sessions.csv` — authoritative SET session calendar;
4. `corporate_actions.csv` — dividends, splits, rights, mergers, symbol changes, delistings, and related settlement evidence; and
5. `symbol_history.csv` — stable-instrument and effective-dated ticker intervals.

The attached 138 tickers are an **acquisition union**, not a static membership list. The project already maintains point-in-time SET100 membership separately. Provider-native predecessor, successor, ISIN, sign, cause, correction, and event tables may be supplied as supplemental originals, but they do not replace the five normalized files.

## Provider fulfillment requirements

- Supply UTF-8 comma-delimited files with one header row and the exact normalized column names in `requested_five_file_schema.csv`.
- Document the mapping from provider-native SETSMART/SET fields to every normalized field.
- Document the adjusted-price formula, included corporate-action types, correction policy, and availability timestamps.
- Provide a license or written permission reference covering research use, KVIS BistKA storage/processing, retention/deletion, and publication of non-reconstructible derived results.
- State explicitly whether raw redistribution is allowed. Raw redistribution may be forbidden, but the current gate requires research use and derived-result publication to be permitted.
- Treat a schema-only empty corporate-action table as complete only if the provider explicitly attests that no qualifying events exist in the requested scope.

## Local fulfillment and validation

After authorized data are received:

1. Preserve provider originals and the permission/license record without modification.
2. Normalize working copies under `data/external/v3_candidate/`; never overwrite locked V2 inputs or the canonical constituent archive.
3. Create a **new real manifest** from `config/v3/market_package_manifest_template.yaml`. Do not use the non-data example in this packet as a candidate receipt.
4. Replace every example identity, permission, attestation, hash, byte count, and row count with verified facts. Create a matching `.sha256` sidecar for the real manifest.
5. Run:

```bash
python -m scripts.build_v3_market_package_gap \
  --project-root . \
  --candidate-manifest data/external/v3_candidate/market_package.yaml \
  --require-pass

python -m scripts.build_v3_investability_policy_readiness --project-root .
```

A passing market-package receipt certifies intake only. It does not approve training, unlock `test` or `all`, approve either draft policy, create a final holdout, or authorize an HPC submission.

## Verify this unsent packet

From this directory:

```bash
shasum -a 256 -c SHA256SUMS.sha256
shasum -a 256 -c SHA256SUMS
```

`SHA256SUMS` covers all five payload documents plus the acquisition-union CSV. Its sidecar covers the inventory file, avoiding an impossible self-hash.

## Before sending

- Advisor verifies the requested scope and signs or endorses the institutional request.
- KVIS confirms that the named student researchers are eligible for the SET academic support route.
- The checklist’s permission, security, retention, and publication questions are resolved.
- Any required proposal, institutional letter, and contact details are attached.
- The word **DRAFT — UNSENT** is removed only by the authorized sender immediately before submission.

