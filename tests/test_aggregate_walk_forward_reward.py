"""The walk-forward aggregator's --rename option: two runs of one variant stay apart."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from scripts import aggregate_walk_forward_reward as agg


def _write_run(root, years, variant, seeds, growth, extra=()):
    for year in years:
        dates = pd.bdate_range(f"{year}-01-03", periods=30).strftime("%Y-%m-%d")
        rows = []
        for policy, rate in [("set50_etf_buy_hold", 0.0002), *extra]:
            rows += [{"date": d, "policy": policy, "portfolio_value": 1e6 * (1 + rate) ** i} for i, d in enumerate(dates)]
        for seed in seeds:
            rate = growth + 0.0001 * (seed - 42)
            rows += [{"date": d, "policy": f"{variant}_seed{seed}", "portfolio_value": 1e6 * (1 + rate) ** i}
                     for i, d in enumerate(dates)]
        part = root / f"fold_{year}" / f"{variant}_part"
        part.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(part / "validation_equity_curves.csv", index=False)


def test_rename_variants_keeps_seed_tags_and_leaves_others():
    curves = pd.DataFrame({"policy": ["x_seed42", "x_seed43", "x_ensemble_band", "set50_etf_buy_hold", "xy_seed42"]})
    out = agg.rename_variants(curves, {"x": "x_100k", "x_ensemble_band": "x_ensemble_band_100k"})
    assert out["policy"].tolist() == ["x_100k_seed42", "x_100k_seed43", "x_ensemble_band_100k",
                                      "set50_etf_buy_hold", "xy_seed42"]
    assert curves["policy"].iloc[0] == "x_seed42"  # the input is not changed


def test_parse_renames_rejects_malformed_items(tmp_path):
    parsed = agg.parse_renames([f"{tmp_path}:a=b", f"{tmp_path}:c=d"])
    assert parsed == {tmp_path.resolve(): {"a": "b", "c": "d"}}
    for bad in ["a=b", f"{tmp_path}:a", f"{tmp_path}:=b", f"{tmp_path}:a="]:
        with pytest.raises(SystemExit):
            agg.parse_renames([bad])


def test_main_keeps_both_runs_of_one_variant_apart(tmp_path, monkeypatch):
    new, old, out = tmp_path / "new", tmp_path / "old", tmp_path / "out"
    _write_run(new, [2016, 2017], "v", [42, 43, 44, 45, 46], 0.001)
    _write_run(old, [2016, 2017], "v", [42, 43, 44], 0.0005)
    monkeypatch.setattr(sys, "argv", ["agg", "--run-dir", str(new), "--extra-run-dirs", str(old),
                                      "--rename", f"{old}:v=v_100k", "--references", "v_100k",
                                      "--output-dir", str(out), "--n-trials", "2"])
    agg.main()
    summary = pd.read_csv(out / "walk_forward_summary.csv").set_index("variant")
    assert summary.loc["v", "seeds_per_fold"] == 5
    assert summary.loc["v_100k", "seeds_per_fold"] == 3
    assert summary.loc["v", "mean_annual_return"] > summary.loc["v_100k", "mean_annual_return"]
    metrics = pd.read_csv(out / "walk_forward_metrics.csv")
    expected = (1 + 0.0005) ** 29 - 1  # seed 42 of the old run, unchanged by the merge
    got = metrics.loc[(metrics["policy"] == "v_100k_seed42") & (metrics["fold"] == 2016), "cumulative_return"].iloc[0]
    assert np.isclose(got, expected)
    sig = pd.read_csv(out / "walk_forward_significance_vs_v_100k.csv")
    assert "v" in set(sig.iloc[:, 0])


def test_main_rejects_a_rename_of_a_folder_not_aggregated(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _write_run(run, [2016], "v", [42], 0.001)
    monkeypatch.setattr(sys, "argv", ["agg", "--run-dir", str(run), "--rename", f"{tmp_path / 'other'}:v=w"])
    with pytest.raises(SystemExit):
        agg.main()


def _chained(n_variants: int) -> pd.DataFrame:
    dates = pd.bdate_range("2016-01-04", periods=3)
    rows = [{"date": d, "policy": f"v{i}_seed42", "value": 1.0 + 0.01 * j, "variant": f"v{i}"}
            for i in range(n_variants) for j, d in enumerate(dates)]
    return pd.DataFrame(rows)


def test_plot_draws_every_learned_variant(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    agg.plot(_chained(8), tmp_path / "chained.png")
    assert (tmp_path / "chained.png").is_file()
    plt.close("all")


def test_plot_refuses_to_drop_variants_silently(tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="13 learned variants"):
        agg.plot(_chained(13), tmp_path / "chained.png")
