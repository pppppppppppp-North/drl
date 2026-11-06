# DRAFT — UNSENT

**Date:** 1 September 2026  
**To:** Head of Strategic Planning, The Stock Exchange of Thailand  
**Subject:** Request for SETSMART historical market data support for KVIS student research

Dear Sir or Madam,

Kamnoetvidya Science Academy requests guidance and, if eligible, authorized SETSMART historical data support for the student project **“Deep Reinforcement Learning for Long Term Profit Optimization in Thai Stock Markets Using Multi-Source Data.”**

**Student researchers**  
Nathawat Youngyim  
Kunach Kijsanayothin

**Advisor**  
Prompong Pakawanwong

The project studies long-horizon Thai equity portfolio allocation while accounting for changing SET100 membership, trading suspensions, transaction costs, corporate actions, ticker changes, and out-of-sample evaluation. We have already preserved a checksum-bound official SET50/SET100 membership and sector archive for 2021–2024. Broad-market model training remains disabled until an authorized market package passes our data-quality gate.

We request CSV or database extracts for **1 October 2020 through 31 March 2025**, covering the attached canonical 138-security acquisition union and any supplemental predecessor or successor identity records required for continuity:

1. authoritative SET trading sessions;
2. raw and adjusted daily OHLCV with stable security identity and availability/correction metadata;
3. daily trading, suspension, halt, no-trade, listing, and delisting status;
4. dividends, splits, rights, mergers, symbol changes, delistings, adjustment factors, and explicit terminal settlement evidence; and
5. stable-instrument, ISIN, and effective-dated symbol history.

The attached schema maps this request to `CALENDAR`, `DAILY_SECURITY_TRADING`, `ADJUST_FACTOR`, `RIGHTS_BENEFIT`, `SECURITY`, `SIGN_POSTING`, `SIGN_LIFTING`, and effective-dated symbol-change records. We also request documentation for the price-adjustment method, event coverage, correction policy, and field availability timing.

Raw data would be accessible only to the two student researchers and advisor, stored and processed in access-controlled KVIS project storage, and never redistributed unless expressly permitted. Reports would contain only non-reconstructible aggregate methods, statistics, figures, and model results, subject to written permission. We will follow any required citation, security, retention, deletion, or incident-reporting conditions.

Please confirm:

- whether KVIS and this project are eligible for academic data support;
- whether research use and KVIS BistKA HPC processing are permitted;
- whether non-reconstructible derived findings may be published;
- whether raw redistribution is prohibited;
- the required agreement, fee, security, retention, and deletion terms; and
- the appropriate SETSMART fields or export route for the attached five-file contract.

Thank you for considering this request.

Yours faithfully,

Nathawat Youngyim  
Student researcher

Kunach Kijsanayothin  
Student researcher

Prompong Pakawanwong  
Advisor

Kamnoetvidya Science Academy

## Attachments in this packet

1. Exact five-file schema and field definitions.
2. Canonical 138-ticker acquisition union.
3. Data-protection, license, and permitted-use checklist.
4. Non-data example of the required checksum manifest.
5. SHA-256 packet inventory.

> This letter has not been signed, emailed, uploaded, or submitted. Advisor and institutional approval are required before use.

