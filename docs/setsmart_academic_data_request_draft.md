# Draft Request for SETSMART Research Data Support

> Draft for KVIS/advisor review. This document has not been submitted.

**Date:** [insert date]

**To:** Head of Strategic Planning, The Stock Exchange of Thailand  
**Subject:** Request for SETSMART data support for student research

Dear Sir or Madam,

Kamnoetvidya Science Academy respectfully requests research-data support for the student project:

**Deep Reinforcement Learning for Long Term Profit Optimization in Thai Stock Markets Using Multi-Source Data**

**Student researchers**

Nathawat Youngyim  
Kunach Kijsanayothin

**Advisor**

Prompong Pakawanwong

## Research purpose

The project studies whether deep reinforcement learning can support long-horizon portfolio allocation in the Thai equity market after accounting for changing SET100 membership, transaction costs, suspensions, corporate actions, ticker changes, and realistic out-of-sample evaluation.

The current project has completed a checksum-bound point-in-time SET50/SET100 membership and sector archive for 2021–2024 and a reproducible five-ticker pilot. Broad-universe conclusions remain blocked until an authorized historical market-data package can be validated.

## Requested scope

Please provide or authorize a CSV/database extract for **1 October 2020 through 31 March 2025**, covering the attached 138-security acquisition union and any predecessor or successor symbols needed for continuous instrument history.

The preferred SETSMART/SET fields are:

1. `CALENDAR`, including authoritative trading sessions.
2. `DAILY_SECURITY_TRADING`, including `D_TRADE`, `I_SECURITY`, raw OHLC, total volume, daily status, and adjustment factor.
3. `ADJUST_FACTOR`, including effective dates and related corporate-action identifiers.
4. `RIGHTS_BENEFIT`, including cash/stock dividends, rights, splits, capital return/change, and relevant announcement, ex, record, and payment dates where available.
5. `SECURITY`, including stable `I_SECURITY`, symbol, ISIN, listing, delisting, and market identifiers.
6. `SIGN_POSTING`, `SIGN_LIFTING`, posting details, and cause records sufficient to reconstruct halt and suspension intervals.
7. Effective-dated symbol-change events containing old symbol, new symbol, stable security ID, and effective date.
8. Source publication, availability, and correction timestamps where available.

We would also appreciate documentation for the adjustment-factor formula and confirmation of which corporate-action types are included.

## Proposed access and safeguards

- Raw data will be accessible only to the two student researchers and the advisor.
- The data may be stored and processed in access-controlled project directories on the KVIS BistKA HPC cluster.
- Checksums and an internal manifest will be used to preserve provenance and detect file changes.
- Raw provider data will not be redistributed or published.
- The paper, poster, and presentation will contain only non-reconstructible aggregate statistics, methods, figures, and model results, subject to SET's written permission.
- Any retention, deletion, citation, or security conditions specified by SET will be followed.

Please confirm whether these uses are permitted and whether additional agreements, fees, or security requirements apply.

## Attachments

1. Research proposal.
2. Detailed field and schema request.
3. The 138-ticker acquisition-union list.
4. Advisor-endorsed institution letter.
5. Project data-protection and publication plan, if required.

Thank you for considering this request. We would be grateful for guidance on KVIS eligibility for academic data support and on the most appropriate SETSMART or historical-data extract.

Yours faithfully,

**Nathawat Youngyim**  
Student researcher

**Kunach Kijsanayothin**  
Student researcher

**Prompong Pakawanwong**  
Advisor

**Kamnoetvidya Science Academy**

## Submission references

- [SET academic SETSMART data-support process](https://www.set.or.th/th/education-research/research/database/setsmart/approval-process)
- Academic-support contact: `ResearchDepartment@set.or.th`
- Data-product and licensing contact: `infoproducts@set.or.th`
