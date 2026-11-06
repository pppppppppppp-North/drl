from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from src.v3.evaluation_policy import (
    FOLD_IDS,
    build_candidate_ledger,
    build_evaluation_schedule,
    normalize_session_calendar,
    rank_candidates,
    validate_candidate_ledger,
    validate_candidate_metric_ledger,
    validate_evaluation_policy,
    validate_fold_schedule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config/v3/evaluation_policy_draft.yaml"
MEMBERSHIP_PATH = (
    PROJECT_ROOT
    / "data/processed/v3_development/"
    "set100_membership_session_mask_2021_2024.csv.gz"
)


def _policy() -> dict:
    issues, policy = validate_evaluation_policy(POLICY_PATH)
    assert issues == []
    return policy


def _sessions() -> pd.DatetimeIndex:
    frame = pd.read_csv(MEMBERSHIP_PATH, usecols=["date"])
    return pd.DatetimeIndex(pd.to_datetime(frame["date"].drop_duplicates()))


def _complete_metrics(
    candidates: pd.DataFrame,
    policy: dict,
    *,
    excess: float = 0.01,
) -> pd.DataFrame:
    rows = []
    for candidate_id in candidates["candidate_id"]:
        for fold_index, fold_id in enumerate(FOLD_IDS):
            equal_log = 0.01 + fold_index * 0.001
            for seed in policy["candidate_budget"]["model_seeds"]:
                candidate_log = equal_log + excess
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "fold_id": fold_id,
                        "seed": seed,
                        "status": "complete",
                        "net_cumulative_return": np.expm1(candidate_log),
                        "net_log_return": candidate_log,
                        "sharpe": 0.5,
                        "max_drawdown": -0.2,
                        "turnover": 0.3,
                        "equal_weight_net_log_return": equal_log,
                        "equal_weight_max_drawdown": -0.25,
                        "set_index_net_log_return": equal_log - 0.002,
                    }
                )
    return pd.DataFrame(rows)


def test_draft_policy_is_structurally_valid_but_not_lockable() -> None:
    issues, _ = validate_evaluation_policy(POLICY_PATH)
    locked_issues, _ = validate_evaluation_policy(
        POLICY_PATH,
        require_locked=True,
    )

    assert issues == []
    assert "V3 evaluation policy is not locked" in locked_issues
    assert "V3 evaluation policy is not approved by both authors" in locked_issues


def test_policy_rejects_validation_fitted_preprocessing(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    shutil.copy2(POLICY_PATH, policy_path)
    value = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    value["preprocessing"]["validation_fit_allowed"] = True
    policy_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    issues, _ = validate_evaluation_policy(policy_path)

    assert "preprocessing.validation_fit_allowed must be False" in issues


def test_fold_schedule_has_six_disjoint_half_year_validation_blocks() -> None:
    schedule, summary = build_evaluation_schedule(_sessions(), _policy())

    assert tuple(summary["fold_id"]) == FOLD_IDS
    assert summary["validation_decisions"].tolist() == [118, 121, 120, 120, 118, 122]
    assert schedule["holdout_access_allowed"].eq(False).all()
    assert schedule.loc[
        schedule["validation_scoring_allowed"], "transition_owner"
    ].eq("validation").all()
    validation_keys = schedule.loc[
        schedule["validation_decision"], ["fold_id", "date"]
    ]
    assert not validation_keys["date"].duplicated().any()
    boundary = schedule[schedule["unused_boundary_transition_start"]]
    assert len(boundary) == 6
    assert boundary["transition_owner"].eq("").all()
    assert boundary["transition_outcome_date"].eq("").all()
    assert schedule.loc[
        schedule["validation_context"], "validation_scoring_allowed"
    ].eq(False).all()


def test_fold_schedule_is_row_order_and_future_suffix_invariant() -> None:
    policy = _policy()
    sessions = _sessions()
    first, first_summary = build_evaluation_schedule(sessions, policy)
    shuffled, shuffled_summary = build_evaluation_schedule(
        pd.Series(sessions).sample(frac=1.0, random_state=17),
        policy,
    )
    suffix = pd.date_range("2025-01-02", periods=40, freq="B")
    extended, extended_summary = build_evaluation_schedule(
        list(sessions) + list(suffix),
        policy,
    )

    pd.testing.assert_frame_equal(first, shuffled)
    pd.testing.assert_frame_equal(first_summary, shuffled_summary)
    pd.testing.assert_frame_equal(first, extended)
    pd.testing.assert_frame_equal(first_summary, extended_summary)
    assert validate_fold_schedule(first, sessions, policy) == []


def test_session_calendar_rejects_duplicates_intraday_and_insufficient_history() -> None:
    sessions = list(_sessions())
    for invalid in (
        sessions + [sessions[0]],
        [*sessions[:-1], sessions[-1] + pd.Timedelta(hours=1)],
    ):
        try:
            normalize_session_calendar(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid session calendar was accepted")

    try:
        build_evaluation_schedule(sessions[:100], _policy())
    except ValueError as exc:
        assert "insufficient training sessions" in str(exc) or "validation sessions" in str(exc)
    else:
        raise AssertionError("insufficient history was accepted")


def test_candidate_ledger_is_exact_finite_and_deterministic() -> None:
    policy = _policy()
    first = build_candidate_ledger(policy)
    second = build_candidate_ledger(policy)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 30
    assert first["algorithm"].value_counts().sort_index().eq(6).all()
    assert set(first["timestep_budget"]) == {50_000, 100_000, 200_000}
    assert first["candidate_id"].is_unique
    assert first["candidate_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert validate_candidate_ledger(first, policy) == []


def test_candidate_ledger_tampering_is_rejected() -> None:
    policy = _policy()
    ledger = build_candidate_ledger(policy)
    ledger.loc[0, "timestep_budget"] = 300_000

    assert validate_candidate_ledger(ledger, policy) == [
        "candidate ledger differs from the exact preregistered grid"
    ]


def test_metric_ledger_requires_all_540_fold_seed_cells() -> None:
    policy = _policy()
    candidates = build_candidate_ledger(policy)
    metrics = _complete_metrics(candidates, policy)

    assert len(metrics) == 540
    assert validate_candidate_metric_ledger(metrics, policy, candidates) == []
    issues = validate_candidate_metric_ledger(
        metrics.iloc[:-1].copy(),
        policy,
        candidates,
    )
    assert "candidate metric ledger is not the exact preregistered grid" in issues


def test_selection_is_deterministic_and_can_choose_no_drl() -> None:
    policy = _policy()
    candidates = build_candidate_ledger(policy)
    _, summary = build_evaluation_schedule(_sessions(), policy)
    metrics = _complete_metrics(candidates, policy, excess=0.0)

    first, first_decision = rank_candidates(metrics, policy, candidates, summary)
    second, second_decision = rank_candidates(
        metrics.sample(frac=1.0, random_state=9),
        policy,
        candidates,
        summary,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_decision == second_decision
    assert first_decision["drl_selected"] is False
    assert first_decision["selected_policy"] == "active_equal_weight_no_drl"
    assert first.iloc[0]["timestep_budget"] == 50_000
