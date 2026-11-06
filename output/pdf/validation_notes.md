# Part 3 — Final pass/fail checklist

- **PASS — Title unchanged.** The registered title is identical in the source, visible PDF, and PDF metadata.
- **PASS — Cover-page label removed.** The standalone `BY` label is absent from the first page while both author names remain present.
- **PASS — Abstract terminology defined.** Technical terms are defined before first use in the abstract, including DRL, OHLCV, PPO, A2C, HPC, baseline strategies, cumulative return, the return-to-volatility ratio, parallel simulation copies, fixed random seed, and checkpoints.
- **PASS — No editorial drafting markers.** The manuscript and rendered PDF contain none of the prohibited marker text.
- **PASS — No internal pipeline-version terminology.** None remains in the manuscript or rendered PDF.
- **PASS — No local references.** The bibliography contains only external scholarly references.
- **PASS — Correct chapter titles.** Chapters 3–5 are `REVIEW OF LITERATURE`, `MAIN RESULT`, and `CONCLUSION AND DISCUSSION`, matching the supplied school template.
- **PASS — Two research questions only.** They address the financial comparison and the computational efficiency/replay experiments.
- **PASS — No unsupported stock-selection rationale.** The five equities are treated as a fixed pilot cohort.
- **PASS — No unsupported preprocessing details.** Unreconstructable procedures remain explicitly limited.
- **PASS — No unsupported baseline mechanics.** Missing rebalancing, estimation, constraint, and transaction-cost details are acknowledged.
- **PASS — Drawdown sign not invented.** $D_t$ is called a drawdown term, and its internal sign convention remains unresolved.
- **PASS — Reward/return notation consistent.** $u_t$, $R_t$, and $r_t$ retain distinct definitions.
- **PASS — Baseline wording properly qualified.** The manuscript uses “best-performing tested baseline” where appropriate.
- **PASS — Historical results appropriately caveated.** Summary-only values are identified and limited consistently.
- **PASS — HPC claims appropriately scoped.** They refer to the tested workload and configuration.
- **PASS — Storage claim descriptive.** The report states that the checkpoint-free run returned 49.94% less output.
- **PASS — Multi-source claims appropriately limited.** The extensions are exploratory and do not establish a consistent performance improvement.
- **PASS — Tables, figures, and cross-references consistent.** Every table and figure is referenced; numbering is sequential.
- **PASS — References internally consistent.** All 17 bibliography items are cited, and all citations resolve.
- **PASS — Conclusion answers both research questions.** It states the financial and computational findings within scope.
- **PASS — No numerical results altered.** Comparison with the prior source found no removed or changed numerical token; one additional “42” restates the existing seed in the Figure 4.1 caption.
- **PASS — Finished two-author manuscript.** The prose consistently uses the established authorial voice.

## Build and visual checks

- The document compiled with XeLaTeX until the contents and references stabilized.
- The final PDF contains 34 A4 pages with embedded Times New Roman regular, bold, and italic fonts.
- All 34 pages were rendered at 120 dpi and inspected. The cover, abstract, contents, chapter openings, tables, figures, appendix, references, and pagination are legible and unclipped.
- The build log contains no overflowing boxes, unresolved citations, unresolved references, or missing characters.
- The final deliverable is `25FCS03 Full Report_completed.pdf`; the editable source is `25FCS03 Full Report_completed.tex`.
- The original supplied PDF and KVIS template remain unchanged.
