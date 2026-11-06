# Scientific Preregistration: Five-Ticker Thai DRL Pilot

## Status

This study is preregistered and locked before the large-scale test campaign. The machine-readable source is `config/research_preregistration.yaml`; its SHA-256 sidecar freezes the exact document. The registration is also bound to the current observed-market readiness receipt, so a changed receipt, raw-data drift, synthetic replacement, or broadened publication claim fails validation.

## Research Question

Across fixed chronological out-of-sample test windows, do validation-selected deep reinforcement-learning policies improve net cumulative return, risk-adjusted return, or maximum drawdown relative to equal-weight and SET-index buy-and-hold baselines in the fixed five-ticker Thai-market pilot?

The unit of analysis is a candidate-policy/baseline/window/seed pair. Differences are always candidate minus baseline on identical test dates.

## Frozen Scope

- Universe: `ADVANC.BK`, `AOT.BK`, `CPALL.BK`, `KBANK.BK`, and `PTT.BK`.
- Requested period: 2021-01-01 through 2024-12-30, represented as an exclusive end date of 2024-12-31.
- Observed stock range: 2021-01-04 through 2024-12-30.
- Market benchmark: `^SET.BK` buy-and-hold.
- Evidence class: restricted private research pilot using observed market data.
- Synthetic data: prohibited from satisfying this scientific registration.
- Publication-grade claim eligibility: explicitly false.

This is not a survivorship-free whole-SET study. The ticker list cannot be expanded or replaced after seeing test outcomes.

## Primary Baselines and Paired Outcomes

The two primary baselines are the equal-weight five-stock portfolio and SET-index buy-and-hold. Both must use the candidate's identical chronological test dates.

The primary paired outcomes are:

1. Excess net cumulative return.
2. Excess net Sharpe ratio.
3. Excess maximum drawdown, where a positive candidate-minus-baseline difference means the candidate drawdown is less negative and therefore better.

The uncertainty interval is fixed at two-sided 95%. Multiple paired comparisons use Holm family-wise correction. Turnover, transaction costs and slippage, finite accounting, and complete window/seed pairing are mandatory guardrails rather than optional diagnostics.

## Selection and Test Firewall

Validation data may select algorithms, hyperparameters, training budget, and nested trials. Test data may be read only for final conclusions. It cannot enter selection, and validation data cannot be substituted into the final conclusion table.

After test access, no algorithm, hyperparameter, training budget, feature group, universe, window, seed, baseline, cost assumption, outcome, or interpretation rule may be tuned. Any changed protocol starts a new study version and requires a new untouched holdout; this registration remains unchanged.

## Fixed Negative-Result Rule

Reliable return outperformance requires the Holm-corrected 95% interval for excess cumulative return to have a lower bound strictly above zero for every candidate comparison against each primary baseline. If any required comparison fails that condition, the fixed conclusion is **no reliable return outperformance**.

All outcomes and guardrails must still be reported when the result is negative. A positive excess-drawdown interval may support only the narrower label **risk-control signal**; it may not be presented as return outperformance.

## Claim Exclusions

This registration excludes claims about:

- generalization to the whole SET universe;
- sentiment effects, because sentiment remains a constant scaffold;
- official macroeconomic effects, because official point-in-time macro history is absent;
- publication-grade performance, because the current public-web market snapshot is restricted private-pilot evidence.

The registered result can establish only what occurred within this fixed private five-ticker pilot.

## Evidence Binding and Validation

The registration binds `reports/data_quality_market/market_data_readiness.json` by its exact SHA-256 digest. Validation also independently checks the receipt's own sidecar, passing status, observed-market origin, restricted evidence class, execution eligibility, publication ineligibility, requested dates, and exact ticker contract.

Run the fail-closed validator from the project root:

```bash
python -m src.campaigns.preregistration \
  --path config/research_preregistration.yaml \
  --project-root .
```

Only an empty issue list is a pass. `preregistration_gate_validator(definition, path)` exposes the definition-aware `(issues, referenced_files)` interface used by campaign gates. Referenced sidecars and the bound receipt can be included directly in the gate evidence fingerprint, making later drift stale rather than silently acceptable.
