# V3 Author and Advisor Decision Packet

Prepared: 4 September 2026  
Status: **DRAFT — no decision is approved by this document**

Project: **Deep Reinforcement Learning for Long-Term Profit Optimization in Thai Stock Markets Using Multi-Source Data**

Authors:

- Nathawat Youngyim
- Kunach Kijsanayothin

Advisor: Prompong Pakawanwong

## Purpose

The V3 code now enforces the draft policies, materializes the exact 540-cell future workload, and refuses training. Seven scientific choices still require a recorded human decision before the policies or development protocol may be locked. This packet puts those choices in one place so the authors and advisor can accept a recommendation, select an alternative, or request a revision without editing code first.

Checking a box below records only the review decision. It does not certify the missing licensed data, create a final holdout, enable training, or authorize an HPC submission. After review, an implementation update must encode the approved choices, rebuild every checksum-bound receipt, and rerun the gate.

## Decision 1 — Maximum stale-valuation age

**Question.** How long may an untradeable held security retain its last observed raw close before the fold is invalid?

**Recommended default:** abort the affected fold after **five consecutive authoritative SET sessions** without a fresh valid valuation, while retaining an age flag on every earlier carried valuation.

**Reasoning.** Five sessions is approximately one trading week and prevents a long suspension from being silently represented as a stable zero return. It is conservative enough for a research comparison while still allowing short operational gaps to be represented explicitly.

- [ ] Accept the five-session abort rule.
- [ ] Select another maximum: ______ authoritative sessions.
- [ ] Request analysis before deciding.

Required implementation field: `rules.stale_valuation.maximum_age_sessions`.

## Decision 2 — Dividend credit and reinvestment timing

**Question.** When does dividend cash become portfolio wealth, and when may the agent reinvest it?

**Recommended default:** credit cash once on the provider-authorized **payment date** (or the first authoritative session on or after that date), and make it available to the policy only at the next decision. Never infer cash from adjusted-close returns, and never count both the adjusted return and a separate cash dividend.

**Reasoning.** The payment date represents actual cash availability. Deferring policy use until the next decision gives the model no same-session information or execution advantage.

- [ ] Accept payment-date credit with next-decision reinvestment.
- [ ] Select another observable convention: ________________________________.
- [ ] Request analysis before deciding.

Required implementation fields: `rules.corporate_actions.dividend_credit_event` and `rules.corporate_actions.dividend_reinvestment_start`.

## Decision 3 — Terminal delisting and successor mapping

**Question.** What wealth value may be recorded when a security delists or merges?

**Recommended default:** accept only a provider-documented cash settlement or a provider-documented successor instrument and conversion ratio. Apply it on the authorized effective or settlement date. If neither path is complete, invalidate the fold; do not assume zero, last price, or an average recovery.

**Reasoning.** An assumed recovery can materially change return and risk estimates. Requiring explicit settlement preserves self-financing accounting and instrument continuity.

- [ ] Accept explicit settlement/successor only; unresolved cases abort the fold.
- [ ] Select another evidence-backed mapping: ______________________________.
- [ ] Request analysis before deciding.

Required implementation fields: explicit cash/successor mapping schema, effective timing, and unresolved-case action.

## Decision 4 — Final order-information cutoff

**Question.** Which information may define an allocation intended for the closing auction or next close?

**Recommended default:** freeze the model input at a documented timestamp strictly before the order-submission cutoff; exclude the same session's final close, volume, status correction, news, and corporate-action update unless the provider proves they were available before that cutoff. Treat later fields as execution or audit evidence only.

**Reasoning.** A clock-time rule must follow the licensed source's availability timestamps and the chosen broker/order convention. The project should not invent an exchange time before those records are available.

- [ ] Accept the strict pre-order information rule and determine the clock time from provider/broker documentation.
- [ ] Select another observable cutoff: ________________________________.
- [ ] Request analysis before deciding.

Required implementation fields: timezone, cutoff timestamp rule, execution-price rule, and treatment of late corrections.

## Decision 5 — Six expanding half-year validation folds

**Question.** Should the project keep six expanding folds with half-year validation windows from 2022 H1 through 2024 H2?

**Recommended default:** keep the six-fold design only if rebuilding it from the licensed authoritative calendar preserves at least 200 training decisions per fold, exactly 30 context sessions, non-overlapping validation transitions, and an unused train-to-validation boundary transition.

Current provisional geometry contains 719 validation transitions: 118, 121, 120, 120, 118, and 122 across folds 1–6. These counts are not final until the authoritative calendar arrives.

- [ ] Conditionally approve the six-fold design subject to the authoritative-calendar rebuild checks.
- [ ] Select another predeclared fold design: ______________________________.
- [ ] Request analysis before deciding.

## Decision 6 — Candidate, seed, and compute budget

**Question.** Is the fixed search budget scientifically adequate and operationally acceptable?

**Recommended default:** retain 30 configurations (five algorithms × two parameter profiles × three timestep budgets), six folds, and seeds 42, 43, and 44. This creates 540 fits and 63,000,000 planned timesteps. Keep adaptive expansion forbidden. If authorized later, package the work into at most three array-job submissions with no more than two running concurrently.

**Reasoning.** The fixed grid uses large compute for temporal replication and seed sensitivity rather than an open-ended search. It also satisfies the project's fewer-than-five-submissions constraint.

- [ ] Approve the fixed 30 × 6 × 3 design and three-job maximum.
- [ ] Select another fixed budget before any run: __________________________.
- [ ] Request analysis before deciding.

## Decision 7 — Primary metric, tie order, and aggregate definitions

**Question.** Which exact formulas turn validated transition evidence into candidate ranking?

**Recommended default:**

1. For each candidate/fold/seed, derive net log return from the complete transition-level self-financing ledger.
2. Compute annualized excess net log return as `252 / N × (candidate net log return − active-equal-weight net log return)`, where `N` is that fold's validation-transition count.
3. Average the three seeds within each fold, then use the median of the six fold values as the primary score.
4. Use negative underwater return for maximum drawdown, so values lie in `[-1, 0]` and larger is better.
5. Define reported turnover as mean absolute security turnover per validation transition, excluding the balancing cash leg.
6. Define Sharpe as annualized compound return divided by population volatility of the `N` transition returns, using 252 sessions per year; report zero when volatility is exactly zero.
7. Retain the existing tie order: worst-fold excess return, excess drawdown, turnover, smaller timestep budget, then lexical candidate ID.
8. If the best primary score is not positive, select active equal weight rather than forcing a DRL winner.

- [ ] Approve the complete formula and tie-order recommendation.
- [ ] Select amendments before lock: ______________________________________.
- [ ] Request analysis before deciding.

Required policy addition: `result_ledger.metric_derivation` matching the reviewed Sharpe, turnover, and drawdown definitions.

## Review outcome

All seven decisions must have one selected outcome and any alternatives must be written precisely before implementation.

| Reviewer | Role | All seven reviewed? | Decision | Date | Signature or evidence reference |
| --- | --- | --- | --- | --- | --- |
| Nathawat Youngyim | Author | [ ] | Pending | — | — |
| Kunach Kijsanayothin | Author | [ ] | Pending | — | — |
| Prompong Pakawanwong | Advisor | [ ] | Pending | — | — |

## Post-review implementation checklist

- [ ] Encode only the recorded choices in new policy revisions.
- [ ] Obtain the licensed five-file market package and authoritative calendar.
- [ ] Rebuild folds, dataset manifests, and train-only preprocessing states from the licensed package.
- [ ] Rebuild policy receipts and confirm every embedded descriptor and sidecar.
- [ ] Add the exact aggregate metric-derivation mapping to the evaluation policy.
- [ ] Obtain author/advisor approval and lock the policies and protocol.
- [ ] Build and review a separate V3 scientific runner; do not reuse the V2 trainer.
- [ ] Re-run the gate; `test` and `all` remain forbidden during development.
- [ ] Submit no more than three array jobs only after every gate passes.

Until those steps are complete, the correct status is **structurally prepared, scientifically blocked, zero V3 HPC jobs submitted**.
