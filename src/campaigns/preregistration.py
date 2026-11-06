"""Fail-closed validation for the scientific preregistration.

The public ``preregistration_gate_validator`` adapter matches the
definition-aware validator shape used by ``src.campaigns.gates``: it returns a
pair of ``(issues, referenced_files)`` and never treats malformed input as a
pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


EXPECTED_TICKERS = (
    "ADVANC.BK",
    "AOT.BK",
    "CPALL.BK",
    "KBANK.BK",
    "PTT.BK",
)
EXPECTED_BASELINES = {"equal_weight", "set_index"}
EXPECTED_OUTCOMES = {
    "excess_cumulative_return",
    "excess_sharpe",
    "excess_max_drawdown",
}
EXPECTED_GUARDRAILS = {
    "turnover",
    "transaction_costs_and_slippage",
    "finite_accounting",
    "complete_pairing",
}
REQUIRED_CLAIM_EXCLUSIONS = {
    "whole_set_generalization",
    "sentiment_effect",
    "official_macro_effect",
    "publication_grade_performance",
}
EXPECTED_QUESTION = (
    "Across fixed chronological out-of-sample test windows, do "
    "validation-selected deep reinforcement-learning policies improve net "
    "cumulative return, risk-adjusted return, or maximum drawdown relative to "
    "equal-weight and SET-index buy-and-hold baselines in the fixed five-ticker "
    "Thai-market pilot?"
)


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def _mapping(value: Any, label: str, issues: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        issues.append(f"{label} must be a mapping")
        return {}
    return value


def _ids(value: Any, label: str, issues: list[str]) -> set[str]:
    if not isinstance(value, list):
        issues.append(f"{label} must be a list")
        return set()
    identifiers: list[str] = []
    for entry in value:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
            issues.append(f"{label} contains an entry without a string id")
            continue
        identifiers.append(entry["id"])
    if len(identifiers) != len(set(identifiers)):
        issues.append(f"{label} contains duplicate ids")
    return set(identifiers)


def _entries_by_id(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        return {}
    return {
        str(entry["id"]): entry
        for entry in value
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)
    }


def _check_sidecar(path: Path, label: str, issues: list[str]) -> Path:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        issues.append(f"{label} SHA-256 sidecar is missing: {sidecar}")
        return sidecar
    try:
        tokens = sidecar.read_text(encoding="utf-8").split()
    except OSError as exc:
        issues.append(f"cannot read {label} SHA-256 sidecar: {exc}")
        return sidecar
    if len(tokens) != 1 or not _is_sha256(tokens[0]):
        issues.append(f"{label} SHA-256 sidecar must contain exactly one digest")
        return sidecar
    try:
        actual = sha256_file(path)
    except OSError as exc:
        issues.append(f"cannot hash {label}: {exc}")
        return sidecar
    if tokens[0].lower() != actual:
        issues.append(f"{label} SHA-256 sidecar does not match")
    return sidecar


def _safe_project_path(project_root: Path, raw_path: Any, label: str, issues: list[str]) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        issues.append(f"{label} path must be a nonempty project-relative string")
        return None
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        issues.append(f"{label} path must stay inside the project: {raw_path!r}")
        return None
    resolved = (project_root / relative).resolve()
    if not resolved.is_relative_to(project_root):
        issues.append(f"{label} path escapes the project: {raw_path!r}")
        return None
    return resolved


def _check_registration(value: Mapping[str, Any], issues: list[str]) -> None:
    registration = _mapping(value.get("registration"), "registration", issues)
    if registration.get("id") != "thai_drl_five_ticker_pilot_v1":
        issues.append("registration.id must remain thai_drl_five_ticker_pilot_v1")
    if registration.get("status") != "locked":
        issues.append("registration.status must be locked")
    if registration.get("amendments_after_test_access") != "forbidden":
        issues.append("amendments after test access must be forbidden")
    if not str(registration.get("amendment_rule", "")).strip():
        issues.append("registration.amendment_rule must be declared")

    question = _mapping(value.get("research_question"), "research_question", issues)
    if question.get("text") != EXPECTED_QUESTION:
        issues.append("research question does not match the locked five-ticker pilot question")
    if question.get("unit_of_analysis") != "candidate_policy_baseline_window_seed_pair":
        issues.append("research question must declare paired window/seed units")
    if question.get("estimand") != "mean_candidate_minus_baseline_difference_across_paired_test_runs":
        issues.append("research question must declare the locked paired estimand")


def _check_scope(value: Mapping[str, Any], issues: list[str]) -> None:
    scope = _mapping(value.get("scope"), "scope", issues)
    if scope.get("study_type") != "fixed_observed_market_pilot":
        issues.append("scope.study_type must remain a fixed observed-market pilot")
    if scope.get("data_origin") != "observed_market":
        issues.append("synthetic or unknown-origin data cannot satisfy the preregistration")
    if scope.get("evidence_class") != "restricted_private_research_pilot":
        issues.append("scope.evidence_class must remain restricted_private_research_pilot")
    if scope.get("synthetic_data_allowed") is not False:
        issues.append("scope.synthetic_data_allowed must be false")
    if scope.get("publication_claim_eligible") is not False:
        issues.append("scope.publication_claim_eligible must be false")

    universe = _mapping(scope.get("universe"), "scope.universe", issues)
    tickers = universe.get("tickers")
    if not isinstance(tickers, list) or tuple(tickers) != EXPECTED_TICKERS:
        issues.append(f"scope universe must be exactly {list(EXPECTED_TICKERS)}")
    if universe.get("size") != len(EXPECTED_TICKERS):
        issues.append("scope universe size must be exactly five")
    if universe.get("policy") != "fixed_before_test_and_not_survivorship_free":
        issues.append("scope universe must disclose its fixed, non-survivorship-free design")

    period = _mapping(scope.get("period"), "scope.period", issues)
    expected_period = {
        "requested_start": "2021-01-01",
        "requested_end_exclusive": "2024-12-31",
        "observed_start": "2021-01-04",
        "observed_end": "2024-12-30",
    }
    for field, expected in expected_period.items():
        if str(period.get(field, "")) != expected:
            issues.append(f"scope.period.{field} must be {expected}")

    benchmark = _mapping(scope.get("market_benchmark"), "scope.market_benchmark", issues)
    if (
        benchmark.get("id") != "set_index"
        or benchmark.get("symbol") != "^SET.BK"
        or benchmark.get("policy") != "buy_and_hold"
    ):
        issues.append("market benchmark must remain ^SET.BK buy-and-hold")


def _check_analysis(value: Mapping[str, Any], issues: list[str]) -> None:
    baseline_values = value.get("primary_baselines")
    baseline_ids = _ids(baseline_values, "primary_baselines", issues)
    if baseline_ids != EXPECTED_BASELINES:
        issues.append(f"primary baselines must be exactly {sorted(EXPECTED_BASELINES)}")
    baselines = _entries_by_id(baseline_values)
    expected_baselines = {
        "equal_weight": "equal_weight_buy_and_hold",
        "set_index": "index_buy_and_hold",
    }
    for baseline_id, expected_policy in expected_baselines.items():
        baseline = baselines.get(baseline_id, {})
        if baseline.get("policy") != expected_policy:
            issues.append(f"primary baseline {baseline_id} must use policy {expected_policy}")
        if baseline.get("evaluation_dates") != "identical_to_candidate_test_window":
            issues.append(f"primary baseline {baseline_id} must use identical candidate test dates")

    plan = _mapping(value.get("analysis_plan"), "analysis_plan", issues)
    selection = _mapping(plan.get("selection"), "analysis_plan.selection", issues)
    if selection.get("split") != "validation" or selection.get("test_data_allowed") is not False:
        issues.append("model and budget selection must be validation-only")
    if selection.get("use") != "algorithm_hyperparameter_training_budget_and_nested_trial_selection_only":
        issues.append("validation use must remain limited to pre-test selection")
    conclusions = _mapping(plan.get("conclusions"), "analysis_plan.conclusions", issues)
    if conclusions.get("split") != "test" or conclusions.get("validation_data_allowed") is not False:
        issues.append("final conclusions must be test-only")
    if conclusions.get("use") != "final_conclusions_only":
        issues.append("test use must remain final conclusions only")
    if plan.get("pairing_keys") != ["window", "seed"]:
        issues.append("paired comparisons must use exactly window and seed keys")
    if plan.get("comparison") != "candidate_minus_baseline":
        issues.append("paired comparisons must be candidate-minus-baseline")
    if plan.get("interval") != "two_sided_95_percent":
        issues.append("the uncertainty interval must remain two-sided 95 percent")
    if plan.get("multiplicity_control") != "holm_familywise_correction":
        issues.append("paired tests must retain Holm family-wise correction")
    if plan.get("post_test_tuning") != "forbidden":
        issues.append("post-test tuning must be forbidden")
    if plan.get("test_data_reuse_for_selection") != "forbidden":
        issues.append("test data reuse for selection must be forbidden")
    if plan.get("changed_protocol_requires_new_holdout") is not True:
        issues.append("a changed protocol must require a new untouched holdout")

    outcome_values = value.get("primary_paired_outcomes")
    outcome_ids = _ids(outcome_values, "primary_paired_outcomes", issues)
    if outcome_ids != EXPECTED_OUTCOMES:
        issues.append(f"primary paired outcomes must be exactly {sorted(EXPECTED_OUTCOMES)}")
    outcomes = _entries_by_id(outcome_values)
    expected_outcomes = {
        "excess_cumulative_return": ("cumulative_return", "greater_than_zero_is_better"),
        "excess_sharpe": ("sharpe", "greater_than_zero_is_better"),
        "excess_max_drawdown": (
            "max_drawdown",
            "greater_than_zero_is_better_because_less_negative_drawdown_is_better",
        ),
    }
    for outcome_id, (metric, direction) in expected_outcomes.items():
        outcome = outcomes.get(outcome_id, {})
        if outcome.get("candidate_metric") != metric or outcome.get("direction") != direction:
            issues.append(f"primary paired outcome {outcome_id} has changed metric or direction")
        if outcome.get("net_of_costs") is not True:
            issues.append(f"primary paired outcome {outcome_id} must remain net of costs")

    guardrail_values = value.get("guardrails")
    guardrail_ids = _ids(guardrail_values, "guardrails", issues)
    if guardrail_ids != EXPECTED_GUARDRAILS:
        issues.append(f"guardrails must be exactly {sorted(EXPECTED_GUARDRAILS)}")
    guardrails = _entries_by_id(guardrail_values)
    expected_guardrail_rules = {
        "turnover": "disclose_candidate_level_and_do_not_hide_cost_intensity",
        "transaction_costs_and_slippage": "disclose_configured_costs_and_execution_stress_results",
        "finite_accounting": "reject_nonfinite_metrics_impossible_drawdowns_or_invalid_weights",
        "complete_pairing": "reject_incomplete_candidate_baseline_window_seed_pairs",
    }
    for guardrail_id, rule in expected_guardrail_rules.items():
        if guardrails.get(guardrail_id, {}).get("rule") != rule:
            issues.append(f"guardrail {guardrail_id} has changed its fail-closed rule")

    negative = _mapping(value.get("negative_result_rule"), "negative_result_rule", issues)
    required_negative = {
        "locked": True,
        "target_metric": "excess_cumulative_return",
        "comparison_scope": "every_candidate_against_each_primary_baseline",
        "pass_condition": "corrected_95_percent_interval_lower_bound_strictly_greater_than_zero",
        "if_any_comparison_fails": "no_reliable_return_outperformance",
        "report_all_outcomes_when_negative": True,
    }
    for field, expected in required_negative.items():
        if negative.get(field) != expected:
            issues.append(f"negative_result_rule.{field} must remain {expected!r}")
    risk_only = _mapping(negative.get("risk_only_claim"), "negative_result_rule.risk_only_claim", issues)
    if risk_only.get("allowed_only_if") != "excess_max_drawdown_interval_lower_bound_strictly_greater_than_zero":
        issues.append("a risk-only claim must require a strictly positive excess-drawdown interval")
    if risk_only.get("label") != "narrower_risk_control_signal":
        issues.append("a risk-only claim must retain the narrower risk-control label")
    if risk_only.get("may_not_be_described_as_return_outperformance") is not True:
        issues.append("a risk-only signal must not be described as return outperformance")


def _check_claim_boundary(value: Mapping[str, Any], issues: list[str]) -> None:
    boundary = _mapping(value.get("claim_boundary"), "claim_boundary", issues)
    if boundary.get("permitted_scope") != "private_five_ticker_thai_market_pilot_only":
        issues.append("claim boundary must remain private five-ticker pilot only")
    exclusions = _ids(boundary.get("excluded_claims"), "claim_boundary.excluded_claims", issues)
    missing = REQUIRED_CLAIM_EXCLUSIONS - exclusions
    if missing:
        issues.append(f"claim boundary is missing exclusions: {sorted(missing)}")


def _check_receipt(
    value: Mapping[str, Any],
    *,
    project_root: Path,
    issues: list[str],
    referenced: list[Path],
) -> None:
    binding = _mapping(value.get("evidence_binding"), "evidence_binding", issues)
    receipt_binding = _mapping(
        binding.get("readiness_receipt"),
        "evidence_binding.readiness_receipt",
        issues,
    )
    receipt_path = _safe_project_path(
        project_root,
        receipt_binding.get("path"),
        "readiness receipt",
        issues,
    )
    if receipt_binding.get("path") != "reports/data_quality_market/market_data_readiness.json":
        issues.append("readiness receipt must use the fixed observed-market receipt path")
    bound_hash = receipt_binding.get("sha256")
    if not _is_sha256(bound_hash):
        issues.append("readiness receipt binding must contain a valid SHA-256 digest")
    expected_binding = {
        "require_sha256_sidecar": True,
        "required_status": "pass",
        "required_data_origin": "observed_market",
        "required_evidence_class": "restricted_private_research_pilot",
        "required_execution_eligible": True,
        "required_publication_claim_eligible": False,
    }
    for field, expected in expected_binding.items():
        if receipt_binding.get(field) != expected:
            issues.append(f"readiness receipt binding {field} must remain {expected!r}")

    if receipt_path is None:
        return
    if not receipt_path.is_file():
        issues.append(f"bound readiness receipt is missing: {receipt_path}")
        return
    referenced.append(receipt_path)
    receipt_sidecar = _check_sidecar(receipt_path, "readiness receipt", issues)
    if receipt_sidecar.is_file():
        referenced.append(receipt_sidecar)
    try:
        actual_hash = sha256_file(receipt_path)
    except OSError as exc:
        issues.append(f"cannot hash readiness receipt: {exc}")
        return
    if str(bound_hash).lower() != actual_hash:
        issues.append("bound readiness receipt SHA-256 does not match the current receipt")

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"cannot read bound readiness receipt: {exc}")
        return
    if not isinstance(receipt, Mapping):
        issues.append("bound readiness receipt must be a JSON object")
        return
    receipt_expectations = {
        "status": "pass",
        "data_origin": "observed_market",
        "evidence_class": "restricted_private_research_pilot",
        "execution_eligible": True,
        "publication_claim_eligible": False,
    }
    for field, expected in receipt_expectations.items():
        if receipt.get(field) != expected:
            issues.append(f"bound readiness receipt {field} must be {expected!r}")
    period = _mapping(receipt.get("requested_period"), "bound receipt requested_period", issues)
    if period.get("start") != "2021-01-01" or period.get("end_exclusive") != "2024-12-31":
        issues.append("bound readiness receipt does not cover the locked 2021-2024 requested period")
    checks = receipt.get("checks")
    if not isinstance(checks, list):
        issues.append("bound readiness receipt checks must be a list")
        return
    ticker_checks = [
        check
        for check in checks
        if isinstance(check, Mapping) and check.get("check") == "stock_ticker_contract"
    ]
    if (
        len(ticker_checks) != 1
        or ticker_checks[0].get("status") != "pass"
        or tuple(ticker_checks[0].get("observed", [])) != EXPECTED_TICKERS
    ):
        issues.append("bound readiness receipt does not verify the exact five-ticker universe")


def validate_preregistration(
    path: Path,
    *,
    project_root: Path | None = None,
) -> tuple[list[str], list[Path]]:
    """Validate a preregistration and every bound receipt, failing closed.

    Returns a list of human-readable issues plus files that a campaign gate
    should include in its evidence fingerprint. An empty issue list is the only
    passing state.
    """

    preregistration_path = Path(path).resolve()
    root = Path(project_root).resolve() if project_root is not None else preregistration_path.parents[1]
    issues: list[str] = []
    referenced: list[Path] = []
    if not preregistration_path.is_relative_to(root):
        return [f"preregistration is outside project root: {preregistration_path}"], []
    if not preregistration_path.is_file():
        return [f"preregistration is missing: {preregistration_path}"], []

    preregistration_sidecar = _check_sidecar(preregistration_path, "preregistration", issues)
    if preregistration_sidecar.is_file():
        referenced.append(preregistration_sidecar)
    try:
        loaded = yaml.safe_load(preregistration_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [*issues, f"cannot read preregistration YAML: {exc}"], referenced
    value = _mapping(loaded, "preregistration", issues)
    if value.get("schema_version") != 1:
        issues.append("preregistration schema_version must be 1")

    _check_registration(value, issues)
    _check_scope(value, issues)
    _check_analysis(value, issues)
    _check_claim_boundary(value, issues)
    _check_receipt(value, project_root=root, issues=issues, referenced=referenced)

    return list(dict.fromkeys(issues)), sorted(set(referenced))


def preregistration_gate_validator(definition: Any, path: Path) -> tuple[list[str], list[Path]]:
    """Definition-aware adapter ready to be called by ``gates.py``."""

    root = getattr(definition, "project_root", None)
    if root is None:
        return ["campaign definition has no project_root for preregistration validation"], []
    return validate_preregistration(path, project_root=Path(root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the locked research preregistration")
    parser.add_argument("--path", type=Path, default=Path("config/research_preregistration.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    issues, referenced = validate_preregistration(args.path, project_root=args.project_root)
    result = {
        "status": "pass" if not issues else "fail",
        "path": str(args.path),
        "issues": issues,
        "referenced_files": [str(path) for path in referenced],
    }
    print(json.dumps(result, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
