from __future__ import annotations

import pandas as pd
import pytest

from scripts.walk_forward_agent_diagnostics import (
    curve_diagnostics,
    diagnose,
    holdings_diagnostics,
    slot_change_diagnostics,
)


def _curve(policy: str, gross: list[float], turnover: list[float], cost: float = 0.0015, slip: float = 0.0005,
           cash: float = 0.0, tradable: int = 4) -> pd.DataFrame:
    values, rows = [1_000_000.0], []
    for g, t in zip(gross, turnover):
        values.append(values[-1] * (1 - (cost + slip) * t) * (1 + g))
    dates = pd.bdate_range("2020-01-01", periods=len(values)).date.astype(str)
    for i, (date, value) in enumerate(zip(dates, values)):
        t = turnover[i - 1] if i else 0.0
        rows.append({"date": date, "policy": policy, "portfolio_value": value, "turnover": t,
                     "transaction_cost_paid": cost * t, "slippage_paid": slip * t, "cash_weight": cash if i else 1.0,
                     "forced_exit_turnover": 0.0, "tradable_count": tradable})
    return pd.DataFrame(rows)


def test_return_splits_into_gross_return_and_cost_drag() -> None:
    curve = _curve("a", gross=[0.01, 0.02, -0.01], turnover=[1.0, 0.0, 0.5])
    out = curve_diagnostics(curve, rebalance_every=2)
    assert out["gross_return"] == pytest.approx(1.01 * 1.02 * 0.99 - 1)
    assert out["net_return"] == pytest.approx(curve["portfolio_value"].iloc[-1] / 1e6 - 1)
    assert out["cost_drag"] == pytest.approx(out["gross_return"] - out["net_return"])
    assert out["costs_paid"] == pytest.approx(0.002 * 1.5)
    assert out["turnover_total"] == pytest.approx(1.5) and out["turnover_after_entry"] == pytest.approx(0.5)
    assert out["turnover_per_rebalance"] == pytest.approx(0.5)  # the decision two sessions after entry


def test_distance_from_equal_weight_uses_rebalance_sessions_and_tradable_members() -> None:
    curve = _curve("a", gross=[0.0] * 4, turnover=[1.0, 0.0, 0.0, 0.0])
    dates = list(curve["date"])
    holdings = pd.DataFrame(
        [{"date": dates[1], "ticker": t, "weight": 0.25} for t in "ABCD"]  # rebalance row: equal weight
        + [{"date": dates[2], "ticker": "A", "weight": 1.0}]  # not a rebalance row: ignored
        + [{"date": dates[3], "ticker": "A", "weight": 1.0}]  # second rebalance (every 2): all in one name
    )
    out = holdings_diagnostics(holdings, curve, rebalance_every=2)
    assert out["distance_from_equal_weight"] == pytest.approx((0.0 + 0.75) / 2)
    assert out["effective_holdings"] == pytest.approx((4 + 1) / 2)
    assert out["max_weight"] == pytest.approx((0.25 + 1.0) / 2)


def test_diagnose_compares_each_seed_with_the_reference_on_the_same_fold(tmp_path) -> None:
    fold = tmp_path / "fold_2020"
    for seed, gross in ((42, 0.01), (43, 0.03)):
        part = fold / f"agent_seed{seed}"
        part.mkdir(parents=True)
        curves = pd.concat([_curve(f"agent_seed{seed}", [gross] * 3, [1.0, 0.4, 0.4]),
                            _curve("ref", [0.02] * 3, [1.0, 0.0, 0.0])])
        curves.to_csv(part / "validation_equity_curves.csv", index=False)
    frame = diagnose(tmp_path, "agent", "ref", rebalance_every=1, reference_rebalance_every=None)
    assert set(frame["policy"]) == {"agent_seed42", "agent_seed43", "ref"}
    seed43 = frame.set_index("policy").loc["agent_seed43"]
    assert seed43["gross_return_minus_reference"] == pytest.approx(1.03**3 - 1.02**3)
    assert seed43["cost_drag_minus_reference"] > 0


def test_rebalance_turnover_is_split_by_whether_a_member_kept_its_slot() -> None:
    curve = _curve("a", gross=[0.0] * 4, turnover=[1.0, 0.0, 0.4, 0.0])
    dates = list(curve["date"])
    slots = {dates[0]: {"A": 0, "B": 1, "C": 2}, dates[2]: {"A": 0, "B": 2, "C": 1}}  # B and C swap slots
    holdings = pd.DataFrame(
        [{"date": dates[2], "ticker": t, "weight": w} for t, w in (("A", 0.3), ("B", 0.3), ("C", 0.4))]
        + [{"date": dates[3], "ticker": t, "weight": w} for t, w in (("A", 0.2), ("B", 0.5), ("C", 0.3))]
    )
    out = slot_change_diagnostics(holdings, curve, slots, rebalance_every=2)
    assert out["slot_moved_name_share"] == pytest.approx(2 / 3)
    assert out["slot_moved_turnover_share"] == pytest.approx(0.3 / 0.4)
    assert out["mean_weight_change_slot_moved"] == pytest.approx(0.15)
    assert out["mean_weight_change_slot_kept"] == pytest.approx(0.1)
