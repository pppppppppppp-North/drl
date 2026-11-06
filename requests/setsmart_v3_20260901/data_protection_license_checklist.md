# Data Protection, License, and Permitted-Use Checklist

**Status: DRAFT — UNSENT — NOT YET APPROVED**

Do not mark an item complete without written provider, advisor, or institutional evidence. This checklist does not grant permission.

## A. Provider and license identity

- [ ] Provider legal name and data-product name are recorded.
- [ ] Contract, approval letter, order, or license reference is recorded.
- [ ] Authorized users are explicitly limited to Nathawat Youngyim, Kunach Kijsanayothin, and advisor Prompong Pakawanwong, unless the agreement states otherwise.
- [ ] Authorized institution and project purpose are stated.
- [ ] Requested date range and the attached 138-ticker acquisition scope are approved.
- [ ] Any fee, account, export, API, or expiry conditions are recorded.

## B. Machine-readable license declarations

The real package manifest may use `true` only when written evidence supports it.

- [ ] `license.reference` is nonblank and points to the governing record.
- [ ] `license.research_use_allowed: true` is supported.
- [ ] `license.derived_publication_allowed: true` is supported for non-reconstructible aggregate findings.
- [ ] `license.raw_redistribution_allowed` is explicitly recorded as `true` or `false`.
- [ ] KVIS BistKA HPC storage and processing are expressly allowed.
- [ ] The provider’s required citation and acknowledgement wording is recorded.

Raw redistribution may be false. Research use and derived-result publication must both be true for the current V3 market-package gate to pass.

## C. Required completeness attestations

Each attestation must be supported by provider documentation and the delivered scope.

- [ ] `authoritative_session_calendar_complete`
- [ ] `trading_status_complete`
- [ ] `membership_exit_path_status_complete`
- [ ] `corporate_actions_complete`
- [ ] `symbol_history_complete`
- [ ] `adjustment_method_documented`
- [ ] `point_in_time_feature_safe`

A missing row cannot be interpreted as “no suspension,” “no event,” or “no symbol change.” An empty corporate-action CSV is acceptable only when it retains the complete header and the provider certifies that the requested scope truly contains no qualifying events.

## D. Access and storage safeguards

- [ ] Provider originals will be preserved unchanged with receipt date and checksums.
- [ ] Raw files will be stored only in an access-controlled project location approved by the license.
- [ ] Raw files will not be committed to a public repository, attached to public reports, or sent through unauthorized channels.
- [ ] Working copies and derived data will remain within approved accounts and systems.
- [ ] Access logs, account removal, and credential rules are confirmed with KVIS where required.
- [ ] Backup, encryption, and device-copy restrictions are documented rather than assumed.
- [ ] A contact and procedure exist for loss, unauthorized access, or accidental disclosure.

## E. Retention, deletion, and audit

- [ ] Retention start and end dates are documented.
- [ ] Required deletion method and confirmation evidence are documented.
- [ ] Rules for backups, cached copies, trained models, and reproducibility archives are documented.
- [ ] Provider audit or reporting obligations are documented.
- [ ] Project closeout assigns a named person to verify access removal and deletion.

## F. Publication and scientific boundary

- [ ] Allowed paper, poster, presentation, and competition outputs are listed.
- [ ] Minimum aggregation, suppression, or non-reconstruction requirements are recorded.
- [ ] Raw row examples, identifiers, and excerpts will not be published unless expressly allowed.
- [ ] Post-2024 data are embargoed from development unless a separately approved protocol assigns an untouched holdout.
- [ ] A passing intake receipt is understood not to approve training, model selection, holdout access, investment claims, or an HPC submission.
- [ ] Both V3 policies and the development protocol remain separately author/advisor approved and checksum locked before execution.

## Approval record

| Role | Name | Decision | Date | Evidence reference |
| --- | --- | --- | --- | --- |
| Student researcher | Nathawat Youngyim | Pending | — | — |
| Student researcher | Kunach Kijsanayothin | Pending | — | — |
| Advisor | Prompong Pakawanwong | Pending | — | — |
| KVIS/institutional authority, if required | Pending | Pending | — | — |
| Data provider/licensor | Pending | Pending | — | — |

