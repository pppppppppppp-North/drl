from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.ingest_investor_flow import INVESTOR_TYPES, normalize_investor_flow
from src.features.fundamental_flags import merge_fundamental_context_with_flags
from src.features.investor_flow_context import build_investor_flow_features, merge_investor_flow_context
from src.sentiment.score_news import (
    daily_embeddings,
    resolve_label_values,
    score_with_classifier,
)


def _flow(dates) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for d in dates:
        nets = rng.normal(0, 100, 3)
        nets = np.append(nets, -nets.sum())  # every buy has a seller
        for group, net in zip(INVESTOR_TYPES, nets):
            buy = 1000 + max(net, 0)
            rows.append({"date": d, "investor_type": group, "buy_value": buy, "sell_value": buy - net})
    return pd.DataFrame(rows)


def test_normalize_investor_flow_long_wide_and_thai_aliases() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    long = normalize_investor_flow(_flow(dates), source_name="unit")
    assert set(long["investor_type"]) == set(INVESTOR_TYPES)
    assert long.groupby("date")["net_value"].sum().abs().max() < 1e-6

    wide = pd.DataFrame({"date": dates})
    for group in INVESTOR_TYPES:
        wide[f"{group}_buy"] = 1000.0
        wide[f"{group}_sell"] = 1000.0
    assert len(normalize_investor_flow(wide, source_name="wide")) == 12

    thai = _flow(dates[:1])
    thai["investor_type"] = ["นักลงทุนต่างประเทศ", "สถาบันในประเทศ", "บัญชีบริษัทหลักทรัพย์", "นักลงทุนทั่วไปในประเทศ"]
    assert normalize_investor_flow(thai, source_name="th")["investor_type"].tolist() == sorted(INVESTOR_TYPES)


def test_normalize_investor_flow_rejects_bad_inputs() -> None:
    dates = pd.bdate_range("2024-01-01", periods=2)
    flow = _flow(dates)
    with pytest.raises(ValueError, match="all four"):
        normalize_investor_flow(flow.iloc[1:], source_name="x")
    unbalanced = flow.copy()
    unbalanced.loc[0, "buy_value"] += 500
    with pytest.raises(ValueError, match="sum to about zero"):
        normalize_investor_flow(unbalanced, source_name="x")
    bad = flow.copy()
    bad.loc[0, "investor_type"] = "martians"
    with pytest.raises(ValueError, match="unrecognized"):
        normalize_investor_flow(bad, source_name="x")


def test_investor_flow_merge_uses_only_previous_sessions() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    flow = normalize_investor_flow(_flow(dates), source_name="unit")
    stocks = pd.DataFrame([{"date": d, "ticker": t, "close": 1.0} for d in dates for t in ("AAA", "BBB")])
    merged = merge_investor_flow_context(stocks, flow)
    covered = merged[merged["flow_data_available"] > 0]
    assert (covered["flow_source_date"] < covered["date"]).all()
    first = merged.sort_values("date").iloc[0]
    assert first["flow_data_available"] == 0.0
    # Market-level features are identical across tickers on a date.
    by_date = merged.groupby("date")["flow_foreign_net_share"].nunique()
    assert (by_date == 1).all()
    # Changing the last flow day cannot affect any earlier merged row.
    changed = flow.copy()
    last = changed["date"].max()
    changed.loc[changed["date"].eq(last) & changed["investor_type"].eq("foreign"), "buy_value"] += 50
    changed.loc[changed["date"].eq(last) & changed["investor_type"].eq("local_individual"), "sell_value"] += 50
    changed["net_value"] = changed["buy_value"] - changed["sell_value"]
    merged2 = merge_investor_flow_context(stocks, changed)
    cols = [c for c in merged.columns if c.startswith("flow_") and c != "flow_source_date"]
    pd.testing.assert_frame_equal(merged[cols].iloc[:-2], merged2[cols].iloc[:-2])
    with pytest.raises(ValueError):
        merge_investor_flow_context(stocks, flow, availability_lag_sessions=0)


def test_investor_flow_zscore_uses_past_window_only() -> None:
    dates = pd.bdate_range("2024-01-01", periods=70)
    flow = normalize_investor_flow(_flow(dates), source_name="unit")
    feats = build_investor_flow_features(flow, z_window=60).set_index("date")
    share = feats["flow_foreign_net_share"]
    t = dates[-1]
    past = share.loc[: dates[-2]].iloc[-60:]
    expected = (share.loc[t] - past.mean()) / past.std(ddof=0)
    assert feats.loc[t, "flow_foreign_net_z60"] == pytest.approx(expected)


def test_resolve_label_values() -> None:
    assert resolve_label_values(["negative", "neutral", "positive"]) == {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    with pytest.raises(ValueError, match="not recognizable"):
        resolve_label_values(["LABEL_0", "LABEL_1"])
    mapped = resolve_label_values(["LABEL_0", "LABEL_1", "LABEL_2"], {"LABEL_0": -1, "LABEL_2": 1})
    assert mapped == {"LABEL_0": -1.0, "LABEL_1": 0.0, "LABEL_2": 1.0}
    with pytest.raises(ValueError):
        resolve_label_values(["a", "b"], {"c": 1})


def test_score_with_classifier_uses_label_probabilities() -> None:
    def fake(batch):
        return [
            [{"label": "positive", "score": 0.7}, {"label": "negative", "score": 0.2}, {"label": "neutral", "score": 0.1}]
            if "good" in text
            else [{"label": "positive", "score": 0.1}, {"label": "negative", "score": 0.8}, {"label": "neutral", "score": 0.1}]
            for text in batch
        ]

    values = resolve_label_values(["positive", "negative", "neutral"])
    scores = score_with_classifier(["good news", "bad news", ""], fake, values, batch_size=2)
    assert scores.tolist() == pytest.approx([0.5, -0.7, 0.0])


def test_embedding_pca_is_fitted_on_training_dates_only() -> None:
    rng = np.random.default_rng(0)
    news = pd.DataFrame(
        {"date": pd.bdate_range("2021-01-01", periods=20).repeat(2), "ticker": ["AAA", "BBB"] * 20, "text": ["x"] * 40}
    )
    emb = rng.normal(size=(40, 6))
    a = daily_embeddings(news, emb, "2021-01-14", dims=3, model_name="m")
    emb2 = emb.copy()
    emb2[30:] += 100.0  # later (validation) rows change drastically
    b = daily_embeddings(news, emb2, "2021-01-14", dims=3, model_name="m")
    early = a["date"] <= pd.Timestamp("2021-01-14")
    cols = [c for c in a.columns if c.startswith("sentiment_emb_")]
    assert np.allclose(a.loc[early, cols].abs(), b.loc[early, cols].abs())


def test_fundamental_missing_flags() -> None:
    dates = pd.bdate_range("2021-01-01", periods=200)
    features = pd.DataFrame({"date": dates, "ticker": "AAA", "close": 1.0})
    fundamentals = pd.DataFrame(
        {
            "ticker": "AAA",
            "metric": ["Total Revenue", "Net Income", "Total Assets"],
            "period_end": "2021-03-31",
            "value": [100.0, 10.0, 500.0],
        }
    )
    merged = merge_fundamental_context_with_flags(features, fundamentals, reporting_lag_days=60)
    effective = pd.Timestamp("2021-03-31") + pd.Timedelta(days=60)
    before = merged[merged["date"] < effective]
    after = merged[merged["date"] >= effective]
    assert (before["fundamental_available"] == 0).all()
    assert (after["fundamental_available"] == 1).all()
    assert np.allclose(after["fundamental_metric_coverage"], 3 / 6)
    assert (after["fundamental_report_age_years"] >= 0).all()
    assert (before["fundamental_report_age_years"] == 0).all()


def test_download_extended_history_seals_holdout(tmp_path, monkeypatch) -> None:
    import scripts.download_extended_history as dl

    def fake_download(symbol, start, end):
        dates = pd.bdate_range("2024-12-20", "2025-01-10")
        return pd.DataFrame(
            {
                "date": dates,
                "ticker": symbol,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": np.linspace(1, 2, len(dates)),
                "volume": 100.0,
                "dividends": 0.0,
                "stock_splits": 0.0,
            }
        )

    monkeypatch.setattr(dl, "download_one", fake_download)
    argv = [
        "x", "--tickers", "AAA.BK", "--index", "IDX", "--dev-end", "2024-12-31", "--holdout-start", "2025-01-01",
        "--dev-output", str(tmp_path / "dev.csv"), "--index-dev-output", str(tmp_path / "idx.csv"),
        "--holdout-dir", str(tmp_path / "holdout"),
    ]
    monkeypatch.setattr("sys.argv", argv)
    dl.main()
    dev = pd.read_csv(tmp_path / "dev.csv")
    assert pd.to_datetime(dev["date"]).max() <= pd.Timestamp("2024-12-31")
    manifest = json.loads((tmp_path / "holdout" / "HOLDOUT_MANIFEST.json").read_text())
    assert manifest["status"] == "SEALED"
    assert manifest["files"]["stocks"]["start"] >= "2025-01-01"
    with pytest.raises(FileExistsError):
        dl.main()
