# Part 2 — Exact changes made in the final submission polish

- Removed the cover-page `BY` label so the two author names appear directly beneath the title.
- Rewrote the abstract so technical terms are defined before first use, including deep reinforcement learning (DRL), simulation environment, OHLCV, PPO, A2C, buy-and-hold, equal-weight, mean-variance, HPC, cumulative return, the return-to-volatility ratio, parallel simulation copies, fixed random seed, and checkpoints.
- Kept the registered title exactly unchanged on the cover, information page, source, and PDF metadata.
- Changed the reward description so $D_t$ is called a drawdown term. The manuscript continues to state that its exact internal sign convention cannot be reconstructed.
- Revised Section 1.2 to say that the reward includes return, turnover, and drawdown terms rather than implying a verified interpretation of how those terms balance.
- Added the missing Section 5.3 limitation: the retained summary does not preserve all principal PPO/A2C training details or the exact baseline rebalancing, estimation, constraint, and transaction-cost procedures.
- Changed the metric provenance statement from “in our implementation” to “in the retained historical metric definition.”
- Replaced the earlier universe terminology with “historical OHLCV cohort” and “stock cohort.”
- Removed “selected” from the survivorship limitation because the stock-selection criterion is not preserved.
- Refined Research Question 1 to refer to the baselines included in the historical pilot, avoiding an exhaustive claim about possible baselines.
- Changed the feature-extension interpretation to state that the negative PPO extension results did not establish a consistent improvement in portfolio performance.
- Expanded Appendix A headers from “Annualized” to “Annualized return” and from “Mean annualized” to “Mean annualized return.”
- Defined $\gamma$ as the discount factor used by the sampled configuration.
- Balanced the abstract ending so it states both the financial result and the effect of computational configuration on workflow efficiency.
- Changed the keyword “Thai stocks” to “Thai equities.”
- Changed the Figure 4.1 caption to describe the three observations as repeated measurements under seed 42.
- Preserved all reported numerical results, the two-question structure, the template-aligned chapter titles, the conservative storage statement, and the workload-specific vectorization interpretation.
- Added no new citations and retained the 17 external scholarly references.

# Part 1 — Full final revised manuscript

The complete section-by-section manuscript is supplied in `25FCS03 Full Report_completed.tex` and its compiled PDF. The source retains the KVIS school layout and all editable TikZ/PGFPlots body figures.
