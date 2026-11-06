from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import make_synthetic_prices
from src.data.leakage import LeakageCheckConfig, assert_no_leakage, check_split_boundaries, run_leakage_checks
from src.data.ingest_ohlcv import download_ohlcv, write_manifest_row
from src.data.ingest_indices import download_indices
from src.data.ingest_macro import download_macro_series
from src.data.ingest_fundamentals import _statement_to_long
from src.data.ingest_news_yahoo import download_yahoo_news
from src.data.ingest_bot_macro import parse_bot_leading_indicator_table
from src.data.quality_report import coverage_by_ticker, dataframe_to_markdown, missingness_by_column
from src.data.splits import (
    generate_walk_forward_splits,
    split_summary,
    walk_forward_config_from_mapping,
    walk_forward_summary,
    chronological_split,
)
from src.agents.baselines import _run_return_benchmark, run_baselines, run_baselines_on_features
from src.envs.trading_env import ThaiStockTradingEnv, load_feature_table
from src.evaluation.metrics import equity_metrics
from src.evaluation.statistics import holm_adjust, sign_flip_p_value
from src.evaluation.compare_results import load_comparison, summarize_actions
from src.experiments.aggregate_trials import aggregate_trials
from src.experiments.aggregate_ablation import aggregate_ablation
from src.experiments.aggregate_walk_forward import aggregate_walk_forward
from src.experiments.aggregate_robustness import aggregate_robustness
from src.experiments.aggregate_screening import aggregate_screening
from src.experiments.aggregate_feature_ablation import aggregate_feature_ablation
from src.experiments.aggregate_nested_screening import aggregate_nested_screening
from src.experiments.aggregate_budget_sweep import aggregate_budget_sweep
from src.experiments.experiment_manifest import collect_experiment_manifests, tracked_experiment
from src.experiments.run_robustness import robustness_seeds, robustness_task_grid
from src.experiments.run_screening import screening_task_grid
from src.experiments.run_feature_ablation import groups_for_task
from src.experiments.run_execution_stress import apply_execution_scenario, run_execution_stress
from src.experiments.run_trial import apply_trial_config, parse_algorithms, read_objective, select_trial_algorithm, suggest_hyperparameters
from src.experiments.run_budget_sweep import budget_values
from src.experiments.run_walk_forward import make_env_from_features, run_walk_forward_window
from src.experiments.run_regime_tests import configured_regime_slices, high_volatility_slices, run_regime_tests
from src.features.technical import add_technical_features
from src.features.market_context import build_index_context, merge_market_context
from src.features.sector_context import add_sector_context, load_sector_mapping
from src.features.macro_context import build_macro_context, merge_macro_context
from src.features.fundamental_context import build_fundamental_context, merge_fundamental_context
from src.features.sentiment_context import merge_sentiment_context
from src.features.official_macro_context import build_official_macro_context, merge_official_macro_context
from src.sentiment.extract_embeddings import _hash_sentiment
from src.train import (
    _algorithm_kwargs,
    build_training_feature_slices,
    ensure_feature_inputs,
    ensure_pilot_features,
    evaluate_model,
    make_trading_env,
    write_split_summary,
    write_walk_forward_summary,
)
from src.utils.config import load_config
from src.utils.checkpoints import prune_checkpoints
from scripts.build_reproducibility_archive import build_archive, collect_archive_files, verify_archive
from scripts.verify_determinism import verify as verify_determinism


def _config() -> dict:
    return {
        "project": {"seed": 1},
        "data": {
            "source": "synthetic",
            "tickers": ["PTT.BK", "AOT.BK"],
            "start": "2023-01-01",
            "end": "2023-06-30",
            "initial_price": 100.0,
        },
        "features": {
            "lookback": 5,
            "columns": ["return_1d", "return_5d", "volatility_20d", "ma_ratio_10", "rsi_14"],
        },
        "environment": {"initial_cash": 1000000.0, "transaction_cost": 0.001, "reward": {}},
    }


def test_config_inheritance_deep_merges_overrides(tmp_path) -> None:
    (tmp_path / "base.yaml").write_text(
        "project:\n  name: base\ntraining:\n  total_timesteps: 1000\n  num_envs: 2\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nproject:\n  name: child\ntraining:\n  total_timesteps: 5000\n",
        encoding="utf-8",
    )

    loaded = load_config(tmp_path / "child.yaml")

    assert loaded["project"]["name"] == "child"
    assert loaded["training"]["total_timesteps"] == 5000
    assert loaded["training"]["num_envs"] == 2


def test_env_reset_and_step_shapes() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    env = ThaiStockTradingEnv(features, config["features"]["columns"], lookback=5)
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(obs).all()

    obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5]))
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["portfolio_value"] > 0


def test_env_close_remains_callable_after_price_panel_initialization() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    env = ThaiStockTradingEnv(features, config["features"]["columns"], lookback=5)

    env.close()


def test_checkpoint_retention_keeps_complete_latest_step_groups(tmp_path) -> None:
    prefix = "ppo_checkpoint"
    for step in (100, 200, 300):
        (tmp_path / f"{prefix}_{step}_steps.zip").write_bytes(b"model")
        (tmp_path / f"{prefix}_replay_buffer_{step}_steps.pkl").write_bytes(b"buffer")
    unrelated = tmp_path / "other_100_steps.zip"
    unrelated.write_bytes(b"other")

    removed = prune_checkpoints(tmp_path, name_prefix=prefix, keep_last=2)

    assert {path.name for path in removed} == {
        "ppo_checkpoint_100_steps.zip",
        "ppo_checkpoint_replay_buffer_100_steps.pkl",
    }
    assert (tmp_path / "ppo_checkpoint_200_steps.zip").exists()
    assert (tmp_path / "ppo_checkpoint_300_steps.zip").exists()
    assert unrelated.exists()


def test_holm_adjustment_controls_a_family_and_preserves_order() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.02, float("nan")])
    assert adjusted[:3] == pytest.approx([0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])
    assert sign_flip_p_value([1.0, 1.0, 1.0, 1.0], resamples=1_000, seed=7) < 0.2


def test_reproducibility_archive_has_hash_verified_manifest(tmp_path) -> None:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "config").mkdir()
    (project / "results" / "run").mkdir(parents=True)
    (project / "src" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "config" / "experiment.yaml").write_text("seed: 42\n", encoding="utf-8")
    (project / "results" / "run" / "metrics.csv").write_text("score\n1\n", encoding="utf-8")
    (project / "results" / "run" / "checkpoints").mkdir()
    (project / "results" / "run" / "checkpoints" / "old.zip").write_bytes(b"old")

    files = collect_archive_files(
        project,
        result_dirs=[project / "results" / "run"],
        selected_models=[],
    )
    archive = tmp_path / "repro.tar.gz"
    manifest = build_archive(project, archive, files)
    verify_archive(archive)

    assert manifest["file_count"] == 3
    assert all("checkpoints" not in entry["path"] for entry in manifest["files"])


def test_fixed_seed_environment_trajectory_is_deterministic() -> None:
    result = verify_determinism(17)
    assert result["deterministic"] is True
    assert result["first_sha256"] == result["second_sha256"]


def test_action_is_normalized_to_long_only_budget() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    env = ThaiStockTradingEnv(features, config["features"]["columns"], lookback=5)
    env.reset(seed=1)
    env.step(np.array([10.0, 10.0]))
    assert np.isclose(env.weights.sum(), 1.0)
    assert env.cash_weight <= 1e-9


def test_execution_constraints_limit_position_cash_and_turnover() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    env = ThaiStockTradingEnv(
        features,
        config["features"]["columns"],
        lookback=5,
        transaction_cost=0.0,
        slippage=0.01,
        max_turnover=0.5,
        max_position_weight=0.4,
        min_cash_weight=0.2,
    )
    env.reset(seed=1)

    _, _, _, _, info = env.step(np.array([1.0, 1.0]))

    assert info["turnover"] <= 0.5 + 1e-9
    assert info["weights"].max() <= 0.4 + 1e-9
    assert info["cash_weight"] >= 0.2 - 1e-9
    assert np.isclose(info["slippage_paid"], 0.01 * info["turnover"])


def test_evaluate_model_returns_equity_curve() -> None:
    class EqualWeightModel:
        def predict(self, obs, deterministic: bool = True):
            return np.array([0.5, 0.5]), None

    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    env = ThaiStockTradingEnv(features, config["features"]["columns"], lookback=5)
    curve = evaluate_model(EqualWeightModel(), env, seed=1)

    assert list(curve.columns) == [
        "date",
        "portfolio_value",
        "policy",
        "net_return",
        "turnover",
        "transaction_cost_paid",
        "slippage_paid",
        "drawdown",
        "cash_weight",
    ]
    assert curve["policy"].eq("ppo").all()
    assert curve["portfolio_value"].gt(0).all()
    assert len(curve) > 1


def test_evaluate_model_can_return_action_weights() -> None:
    class EqualWeightModel:
        def predict(self, obs, deterministic: bool = True):
            return np.array([0.5, 0.5]), None

    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    env = ThaiStockTradingEnv(features, config["features"]["columns"], lookback=5)
    curve, actions = evaluate_model(EqualWeightModel(), env, seed=1, return_actions=True)

    assert len(actions) == len(curve) * env.num_assets
    assert set(actions.columns) == {"date", "policy", "ticker", "weight"}
    assert actions["weight"].between(0.0, 1.0).all()


def test_equity_metrics_include_trading_risk_fields() -> None:
    curve = np.array([100.0, 102.0, 99.0, 103.0])
    metrics = equity_metrics(
        pd.DataFrame(
            {
                "portfolio_value": curve,
                "turnover": [0.0, 0.5, 0.2, 0.0],
                "cash_weight": [1.0, 0.5, 0.3, 0.3],
            }
        )
    )

    assert "value_at_risk_95" in metrics
    assert "conditional_value_at_risk_95" in metrics
    assert np.isclose(metrics["total_turnover"], 0.7)
    assert metrics["active_trading_steps"] == 2.0


def test_ensure_pilot_features_regenerates_when_config_changes(tmp_path) -> None:
    config = _config()
    config["paths"] = {
        "raw_prices": str(tmp_path / "prices.csv"),
        "features": str(tmp_path / "features.parquet"),
        "fallback_features_csv": str(tmp_path / "features.csv"),
    }
    ensure_pilot_features(config)

    changed = _config()
    changed["paths"] = config["paths"]
    changed["data"] = {
        "tickers": ["PTT.BK", "AOT.BK", "CPALL.BK"],
        "start": "2023-01-01",
        "end": "2023-06-30",
        "initial_price": 100.0,
    }
    ensure_pilot_features(changed)

    regenerated = load_feature_table(changed["paths"]["features"], changed["paths"]["fallback_features_csv"])
    assert set(regenerated["ticker"]) == {"PTT.BK", "AOT.BK", "CPALL.BK"}


def test_non_synthetic_feature_inputs_do_not_regenerate_from_synthetic(tmp_path) -> None:
    config = _config()
    config["data"]["source"] = "yahoo_ohlcv"
    config["paths"] = {
        "raw_prices": str(tmp_path / "prices_real.csv"),
        "features": str(tmp_path / "features_real.parquet"),
        "fallback_features_csv": str(tmp_path / "features_real.csv"),
    }
    features = add_technical_features(make_synthetic_prices(_config()))
    features["close"] = 999.0
    features.to_csv(config["paths"]["fallback_features_csv"], index=False)

    ensure_feature_inputs(config)

    loaded = load_feature_table(config["paths"]["features"], config["paths"]["fallback_features_csv"])
    assert loaded["close"].eq(999.0).all()


def test_compare_results_writes_ranked_policy_table(tmp_path) -> None:
    run_dir = tmp_path
    (run_dir / "baselines").mkdir()
    pd.DataFrame([{"policy": "ppo", "sharpe": 0.5, "cumulative_return": 0.2}]).to_csv(
        run_dir / "ppo_metrics.csv", index=False
    )
    pd.DataFrame([{"policy": "equal_weight", "sharpe": 1.0, "cumulative_return": 0.1}]).to_csv(
        run_dir / "baselines" / "baselines_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {"date": "2024-01-01", "policy": "ppo", "ticker": "AOT.BK", "weight": 0.2},
            {"date": "2024-01-02", "policy": "ppo", "ticker": "AOT.BK", "weight": 0.4},
        ]
    ).to_csv(run_dir / "ppo_actions.csv", index=False)

    comparison = load_comparison(run_dir)
    action_summary = summarize_actions(run_dir)

    assert comparison.iloc[0]["policy"] == "equal_weight"
    assert np.isclose(action_summary.iloc[0]["mean_weight"], 0.3)


def test_run_baselines_includes_buy_hold_and_ma_crossover(tmp_path) -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    feature_path = tmp_path / "features.csv"
    features.to_csv(feature_path, index=False)
    config["paths"] = {
        "raw_prices": str(tmp_path / "prices.csv"),
        "features": str(feature_path),
        "fallback_features_csv": str(feature_path),
    }

    curves, metrics = run_baselines(config)
    policies = set(metrics["policy"])

    assert {"equal_weight", "random", "momentum", "ma_crossover", "mean_variance"}.issubset(policies)
    assert {"buy_hold_AOT.BK", "buy_hold_PTT.BK"}.issubset(policies)
    assert curves["policy"].nunique() == len(policies)
    assert metrics["final_portfolio_value"].gt(0).all()


def test_return_benchmark_aligns_with_env_dates_and_initial_cost() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    features["set_return_1d"] = 0.01

    curve = _run_return_benchmark(
        features,
        return_column="set_return_1d",
        policy_name="set_index",
        lookback=config["features"]["lookback"],
        initial_cash=config["environment"]["initial_cash"],
        transaction_cost=config["environment"]["transaction_cost"],
    )

    unique_dates = sorted(pd.to_datetime(features["date"]).unique())
    assert curve.iloc[0]["date"] == pd.Timestamp(unique_dates[config["features"]["lookback"]]).date().isoformat()
    assert curve.iloc[1]["turnover"] == 2.0
    assert np.isclose(curve.iloc[1]["net_return"], 0.01 - 2 * config["environment"]["transaction_cost"])
    assert curve["policy"].eq("set_index").all()


def test_run_baselines_adds_configured_market_index() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    features["set_return_1d"] = features.groupby("ticker")["close"].pct_change().fillna(0.0)
    config["baselines"] = {"market_index": {"policy_name": "set_index", "return_column": "set_return_1d"}}

    curves, metrics = run_baselines_on_features(config, features)

    assert "set_index" in set(curves["policy"])
    assert "set_index" in set(metrics["policy"])


def test_run_baselines_on_features_uses_supplied_window() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    window = features[pd.to_datetime(features["date"]) <= pd.Timestamp("2023-04-30")]

    curves, metrics = run_baselines_on_features(config, window)

    assert curves["date"].max() <= "2023-04-30"
    assert {"equal_weight", "ma_crossover", "mean_variance"}.issubset(set(metrics["policy"]))
    assert metrics["final_portfolio_value"].gt(0).all()


def test_make_env_from_features_builds_window_specific_env() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    window = features[pd.to_datetime(features["date"]) <= pd.Timestamp("2023-04-30")]

    env = make_env_from_features(config, window)
    obs, info = env.reset(seed=1)

    assert obs.shape == env.observation_space.shape
    assert max(env.dates) <= pd.Timestamp("2023-04-30")
    assert info["portfolio_value"] == config["environment"]["initial_cash"]


def test_training_feature_slices_are_distinct_complete_panels() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))

    slices = build_training_feature_slices(
        features,
        num_envs=4,
        lookback=config["features"]["lookback"],
        seed=42,
        window_dates=60,
    )

    assert len(slices) == 4
    date_ranges = []
    for part in slices:
        dates = pd.to_datetime(part["date"])
        date_ranges.append((dates.min(), dates.max()))
        assert dates.nunique() == 60
        assert set(part["ticker"]) == set(config["data"]["tickers"])
        assert not part.duplicated(["date", "ticker"]).any()
    assert len(set(date_ranges)) == 4


def test_chronological_split_preserves_order_and_panel_rows() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    splits = chronological_split(features)
    summary = split_summary(splits)

    assert list(splits) == ["train", "validation", "test"]
    assert sum(len(split) for split in splits.values()) == len(features)
    assert summary["unique_dates"].gt(0).all()
    assert splits["train"]["date"].max() < splits["validation"]["date"].min()
    assert splits["validation"]["date"].max() < splits["test"]["date"].min()
    assert set(splits["test"]["ticker"]) == set(config["data"]["tickers"])
    assert check_split_boundaries(splits)["status"].eq("pass").all()


def test_make_trading_env_uses_requested_chronological_split(tmp_path) -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    feature_path = tmp_path / "features.csv"
    features.to_csv(feature_path, index=False)
    config["paths"] = {
        "raw_prices": str(tmp_path / "prices.csv"),
        "features": str(feature_path),
        "fallback_features_csv": str(feature_path),
    }
    config["splits"] = {"train_fraction": 0.6, "validation_fraction": 0.2}

    splits = chronological_split(features)
    train_env = make_trading_env(config, split="train")
    validation_env = make_trading_env(config, split="validation")

    assert min(train_env.dates) == pd.to_datetime(splits["train"]["date"]).min()
    assert max(train_env.dates) == pd.to_datetime(splits["train"]["date"]).max()
    assert min(validation_env.dates) == pd.to_datetime(splits["validation"]["date"]).min()
    assert max(train_env.dates) < min(validation_env.dates)


def test_split_summary_is_written_when_config_has_splits(tmp_path) -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    feature_path = tmp_path / "features.csv"
    features.to_csv(feature_path, index=False)
    config["paths"] = {
        "features": str(feature_path),
        "fallback_features_csv": str(feature_path),
    }
    config["splits"] = {"train_fraction": 0.6, "validation_fraction": 0.2}

    write_split_summary(config, tmp_path)
    summary = pd.read_csv(tmp_path / "split_summary.csv")

    assert list(summary["split"]) == ["train", "validation", "test"]
    assert summary["unique_dates"].gt(config["features"]["lookback"]).all()


def test_walk_forward_splits_roll_without_overlap_inside_window() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    wf_config = walk_forward_config_from_mapping(
        {"train_window": 30, "validation_window": 10, "test_window": 10, "step_size": 10}
    )

    windows = generate_walk_forward_splits(features, wf_config)
    summary = walk_forward_summary(windows)

    assert len(windows) >= 2
    assert set(summary["split"]) == {"train", "validation", "test"}
    first = windows[0]
    second = windows[1]
    assert first["train"]["date"].max() < first["validation"]["date"].min()
    assert first["validation"]["date"].max() < first["test"]["date"].min()
    assert second["train"]["date"].min() > first["train"]["date"].min()
    assert first["test"]["date"].min() == second["validation"]["date"].min()


def test_walk_forward_summary_is_written_when_config_has_walk_forward(tmp_path) -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    feature_path = tmp_path / "features.csv"
    features.to_csv(feature_path, index=False)
    config["paths"] = {
        "features": str(feature_path),
        "fallback_features_csv": str(feature_path),
    }
    config["walk_forward"] = {
        "train_window": 30,
        "validation_window": 10,
        "test_window": 10,
        "step_size": 10,
    }

    write_walk_forward_summary(config, tmp_path)
    summary = pd.read_csv(tmp_path / "walk_forward_summary.csv")

    assert set(summary["split"]) == {"train", "validation", "test"}
    assert summary["window"].nunique() >= 2
    assert summary["unique_dates"].min() == 10


def test_walk_forward_runner_rejects_empty_or_unknown_evaluation_splits(tmp_path) -> None:
    config = _config()
    config["data"]["source"] = "unit_test_features"
    features = add_technical_features(make_synthetic_prices(config))
    feature_path = tmp_path / "features.csv"
    features.to_csv(feature_path, index=False)
    config["paths"] = {
        "raw_prices": str(tmp_path / "prices.csv"),
        "features": str(feature_path),
        "fallback_features_csv": str(feature_path),
    }
    config["walk_forward"] = {
        "train_window": 60,
        "validation_window": 20,
        "test_window": 20,
        "step_size": 20,
    }
    config["training"] = {"total_timesteps": 10, "num_envs": 1, "seed": 1}

    with pytest.raises(ValueError, match="evaluation_splits"):
        run_walk_forward_window(config, 0, tmp_path / "empty", evaluation_splits=())
    with pytest.raises(ValueError, match="evaluation_splits"):
        run_walk_forward_window(config, 0, tmp_path / "unknown", evaluation_splits=("test", "future"))


def test_leakage_checks_pass_clean_feature_table() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    features["macro_release_date"] = features["date"]

    results = assert_no_leakage(features)

    assert results["status"].eq("pass").all()
    assert "macro_release_date_not_future" in set(results["check"])


def test_leakage_checks_detect_future_release_dates() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    features["macro_release_date"] = pd.to_datetime(features["date"]) + pd.Timedelta(days=1)

    results = run_leakage_checks(features, LeakageCheckConfig(release_date_columns=("macro_release_date",)))

    failed = results[results["status"].eq("fail")]
    assert "macro_release_date_not_future" in set(failed["check"])


def test_download_ohlcv_normalizes_provider_data_and_context_columns() -> None:
    def provider(ticker: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.date_range(start, periods=3)
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 101.5, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.0, 102.0, 101.0],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.Index(dates, name="Date"),
        )

    prices = download_ohlcv(["PTT.BK", "AOT.BK"], "2024-01-01", "2024-01-04", provider=provider)

    assert set(["date", "ticker", "open", "high", "low", "close", "volume"]).issubset(prices.columns)
    assert set(["market_return_1d", "macro_rate_change", "sentiment_score"]).issubset(prices.columns)
    assert len(prices) == 6
    assert prices["ticker"].nunique() == 2
    assert prices["macro_rate_change"].eq(0.0).all()
    assert prices["sentiment_score"].eq(0.0).all()
    assert int(prices["ohlc_repaired"].sum()) == 2
    assert (prices["high"] >= prices[["open", "close"]].max(axis=1)).all()


def test_download_indices_normalizes_configured_index_data() -> None:
    def provider(ticker: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.date_range(start, periods=3)
        return pd.DataFrame(
            {
                "Open": [1400.0, 1401.0, 1402.0],
                "High": [1410.0, 1411.0, 1412.0],
                "Low": [1390.0, 1391.0, 1392.0],
                "Close": [1400.0, 1402.0, 1401.0],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.Index(dates, name="Date"),
        )

    from src.data import ingest_indices

    original = ingest_indices._yahoo_provider
    ingest_indices._yahoo_provider = provider
    try:
        prices = download_indices({"set": "^SET.BK"}, "2024-01-01", "2024-01-04")
    finally:
        ingest_indices._yahoo_provider = original

    assert set(["date", "ticker", "open", "high", "low", "close", "volume", "index_name"]).issubset(prices.columns)
    assert prices["index_name"].eq("set").all()
    assert prices["ticker"].eq("^SET.BK").all()


def test_download_macro_series_normalizes_configured_macro_data() -> None:
    def provider(ticker: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.date_range(start, periods=3)
        return pd.DataFrame(
            {
                "Open": [35.0, 35.1, 35.2],
                "High": [35.2, 35.3, 35.4],
                "Low": [34.9, 35.0, 35.1],
                "Close": [35.0, 35.2, 35.1],
                "Volume": [0, 0, 0],
            },
            index=pd.Index(dates, name="Date"),
        )

    from src.data import ingest_macro

    original = ingest_macro._yahoo_provider
    ingest_macro._yahoo_provider = provider
    try:
        prices = download_macro_series({"usdthb": "USDTHB=X"}, "2024-01-01", "2024-01-04")
    finally:
        ingest_macro._yahoo_provider = original

    assert set(["date", "ticker", "open", "high", "low", "close", "volume", "macro_name"]).issubset(prices.columns)
    assert prices["macro_name"].eq("usdthb").all()
    assert prices["ticker"].eq("USDTHB=X").all()


def test_download_yahoo_news_normalizes_current_news_schema() -> None:
    def provider(ticker: str) -> list[dict]:
        return [
            {
                "content": {
                    "title": f"{ticker} earnings improve",
                    "summary": "Quarterly margin expanded.",
                    "pubDate": "2024-06-03T02:15:00Z",
                    "provider": {"displayName": "Example Wire"},
                    "canonicalUrl": {"url": "https://example.com/news"},
                }
            }
        ]

    news = download_yahoo_news(["PTT.BK"], provider=provider)

    assert set(["date", "published_at", "ticker", "title", "summary", "publisher", "url", "text"]).issubset(
        news.columns
    )
    assert news.iloc[0]["date"] == "2024-06-03"
    assert news.iloc[0]["ticker"] == "PTT.BK"
    assert "earnings improve" in news.iloc[0]["text"]


def test_parse_bot_leading_indicator_table_assigns_release_dates() -> None:
    table = pd.DataFrame(
        {
            "No.": ["1", "2", "3"],
            "Indicator": [
                "Leading Economic Index",
                "Index Change",
                "Business Sentiment Index 3 Months",
            ],
            "DEC 2023": ["150.0", "1.5", "55.0"],
            "JAN 2024": ["151.0", "-0.2", "54.0"],
        }
    )

    parsed = parse_bot_leading_indicator_table(table)

    assert {"period", "release_date", "metric", "value", "source_table"}.issubset(parsed.columns)
    dec_lei = parsed[(parsed["period"].eq("2023-12-01")) & (parsed["metric"].eq("leading_economic_index"))].iloc[0]
    assert dec_lei["release_date"] == "2024-01-31"
    jan_lei = parsed[(parsed["period"].eq("2024-01-01")) & (parsed["metric"].eq("leading_economic_index"))].iloc[0]
    assert jan_lei["release_date"] == "2024-02-29"


def test_merge_market_context_replaces_placeholder_market_return() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    features["market_return_1d"] = 999.0
    index_prices = pd.DataFrame(
        {
            "date": sorted(features["date"].unique())[:40],
            "ticker": ["^SET.BK"] * 40,
            "open": np.linspace(100.0, 110.0, 40),
            "high": np.linspace(101.0, 111.0, 40),
            "low": np.linspace(99.0, 109.0, 40),
            "close": np.linspace(100.0, 120.0, 40),
            "volume": [1000] * 40,
            "index_name": ["set"] * 40,
        }
    )

    merged = merge_market_context(features, index_prices, "set")
    context = build_index_context(index_prices, "set")

    assert {
        "set_return_1d",
        "set_return_5d",
        "set_volatility_20d",
        "set_ma_ratio_20",
        "set_context_available",
    }.issubset(merged.columns)
    assert not merged["market_return_1d"].eq(999.0).all()
    assert set(merged["set_context_available"].unique()) == {0.0, 1.0}
    sample_date = pd.to_datetime(context.iloc[10]["date"])
    expected = context[context["date"].eq(sample_date)].iloc[0]["set_return_1d"]
    actual = merged[pd.to_datetime(merged["date"]).eq(sample_date)].iloc[0]["market_return_1d"]
    assert np.isclose(actual, expected)


def test_merge_macro_context_forward_fills_to_stock_dates_and_sets_rate_change() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    features["macro_rate_change"] = 999.0
    dates = sorted(pd.to_datetime(features["date"]).unique())
    macro_prices = pd.DataFrame(
        {
            "date": [dates[0], dates[2], dates[4], dates[6]],
            "ticker": ["^TNX"] * 4,
            "open": [4.0, 4.1, 4.2, 4.4],
            "high": [4.1, 4.2, 4.3, 4.5],
            "low": [3.9, 4.0, 4.1, 4.3],
            "close": [4.0, 4.1, 4.2, 4.4],
            "volume": [0] * 4,
            "macro_name": ["us10y"] * 4,
        }
    )

    merged = merge_macro_context(features, macro_prices, primary_rate_column="us10y_change_1d")
    context = build_macro_context(macro_prices)

    assert {"us10y_level", "us10y_change_1d", "us10y_return_1d"}.issubset(merged.columns)
    assert not merged["macro_rate_change"].eq(999.0).all()
    filled_date = dates[3]
    expected = context[context["date"].eq(dates[2])].iloc[0]["us10y_change_1d"]
    actual = merged[merged["date"].eq(filled_date)].iloc[0]["macro_rate_change"]
    assert np.isclose(actual, expected)


def test_merge_official_macro_context_uses_release_dates_without_lookahead() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    features["date"] = pd.to_datetime(features["date"])
    features = features[features["date"].between("2023-01-20", "2023-03-10")].copy()
    macro = pd.DataFrame(
        [
            {
                "period": "2022-12-01",
                "release_date": "2023-01-31",
                "metric": "leading_economic_index",
                "value": 150.0,
            },
            {
                "period": "2022-12-01",
                "release_date": "2023-01-31",
                "metric": "index_change",
                "value": 1.0,
            },
            {
                "period": "2023-01-01",
                "release_date": "2023-02-28",
                "metric": "leading_economic_index",
                "value": 151.0,
            },
            {
                "period": "2023-01-01",
                "release_date": "2023-02-28",
                "metric": "index_change",
                "value": -0.5,
            },
        ]
    )

    merged = merge_official_macro_context(features, macro)
    context = build_official_macro_context(macro)

    assert {"bot_leading_economic_index", "bot_leading_index_change", "bot_macro_release_age_days"}.issubset(
        merged.columns
    )
    before_release = merged[merged["date"].lt(pd.Timestamp("2023-01-31"))]
    assert before_release["bot_leading_economic_index"].eq(0.0).all()
    after_first_release = merged[merged["date"].between(pd.Timestamp("2023-01-31"), pd.Timestamp("2023-02-27"))]
    assert after_first_release["bot_leading_economic_index"].eq(150.0).all()
    after_second_release = merged[merged["date"].ge(pd.Timestamp("2023-02-28"))]
    assert after_second_release["bot_leading_economic_index"].eq(151.0).all()
    expected = context[context["release_date"].eq(pd.Timestamp("2023-02-28"))].iloc[0]["bot_leading_index_change"]
    actual = after_second_release.iloc[0]["bot_leading_index_change"]
    assert np.isclose(actual, expected)


def test_statement_to_long_normalizes_yahoo_fundamentals() -> None:
    statement = pd.DataFrame(
        {
            pd.Timestamp("2024-03-31"): [100.0, 10.0],
            pd.Timestamp("2024-06-30"): [120.0, 12.0],
        },
        index=pd.Index(["Total Revenue", "Net Income"], name="metric"),
    )

    long = _statement_to_long("PTT.BK", "income_quarterly", statement)

    assert set(["ticker", "statement_type", "metric", "period_end", "value"]).issubset(long.columns)
    assert long["ticker"].eq("PTT.BK").all()
    assert long["statement_type"].eq("income_quarterly").all()
    assert len(long) == 4


def test_merge_fundamental_context_respects_reporting_lag() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    features["date"] = pd.to_datetime(features["date"])
    features = features[features["date"].between("2023-02-01", "2023-05-15")].copy()
    fundamentals = pd.DataFrame(
        [
            {"ticker": "PTT.BK", "metric": "Total Revenue", "period_end": "2022-12-31", "value": 100.0},
            {"ticker": "PTT.BK", "metric": "Net Income", "period_end": "2022-12-31", "value": 10.0},
            {"ticker": "PTT.BK", "metric": "Stockholders Equity", "period_end": "2022-12-31", "value": 50.0},
            {"ticker": "PTT.BK", "metric": "Total Debt", "period_end": "2022-12-31", "value": 25.0},
            {"ticker": "PTT.BK", "metric": "Total Assets", "period_end": "2022-12-31", "value": 200.0},
            {"ticker": "PTT.BK", "metric": "Free Cash Flow", "period_end": "2022-12-31", "value": 5.0},
            {"ticker": "AOT.BK", "metric": "Total Revenue", "period_end": "2022-12-31", "value": 80.0},
            {"ticker": "AOT.BK", "metric": "Net Income", "period_end": "2022-12-31", "value": 4.0},
            {"ticker": "AOT.BK", "metric": "Stockholders Equity", "period_end": "2022-12-31", "value": 40.0},
            {"ticker": "AOT.BK", "metric": "Total Debt", "period_end": "2022-12-31", "value": 20.0},
            {"ticker": "AOT.BK", "metric": "Total Assets", "period_end": "2022-12-31", "value": 160.0},
            {"ticker": "AOT.BK", "metric": "Free Cash Flow", "period_end": "2022-12-31", "value": 8.0},
        ]
    )

    merged = merge_fundamental_context(features, fundamentals, reporting_lag_days=60)
    context = build_fundamental_context(fundamentals, reporting_lag_days=60)

    assert {"fundamental_net_margin", "fundamental_debt_to_equity", "fundamental_return_on_equity"}.issubset(
        merged.columns
    )
    before_effective = merged[pd.to_datetime(merged["date"]).lt(pd.Timestamp("2023-03-01"))]
    after_effective = merged[pd.to_datetime(merged["date"]).ge(pd.Timestamp("2023-03-01"))]
    assert before_effective["fundamental_net_margin"].eq(0.0).all()
    assert after_effective["fundamental_net_margin"].gt(0.0).all()
    expected_ptt = context[context["ticker"].eq("PTT.BK")].iloc[0]["fundamental_net_margin"]
    actual_ptt = after_effective[after_effective["ticker"].eq("PTT.BK")].iloc[0]["fundamental_net_margin"]
    assert np.isclose(actual_ptt, expected_ptt)


def test_merge_sentiment_context_uses_backward_only_news_and_expires_old_items() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    features["date"] = pd.to_datetime(features["date"])
    features = features[features["date"].between("2023-01-03", "2023-01-20")].copy()
    features["sentiment_score"] = 999.0
    daily_sentiment = pd.DataFrame(
        [
            {
                "date": "2023-01-05",
                "ticker": "PTT.BK",
                "sentiment_score": _hash_sentiment("PTT earnings improve"),
                "news_count": 2,
            },
            {
                "date": "2023-01-10",
                "ticker": "AOT.BK",
                "sentiment_score": _hash_sentiment("AOT traffic declines"),
                "news_count": 1,
            },
        ]
    )

    merged = merge_sentiment_context(features, daily_sentiment, max_age_days=3)

    assert {"sentiment_score", "news_count", "sentiment_news_age_days", "sentiment_release_date"}.issubset(
        merged.columns
    )
    before_news = merged[merged["date"].lt(pd.Timestamp("2023-01-05"))]
    assert before_news["sentiment_score"].eq(0.0).all()
    same_day_ptt = merged[(merged["ticker"].eq("PTT.BK")) & (merged["date"].eq(pd.Timestamp("2023-01-05")))].iloc[0]
    assert np.isclose(same_day_ptt["sentiment_score"], _hash_sentiment("PTT earnings improve"))
    assert same_day_ptt["news_count"] == 2
    expired_ptt = merged[(merged["ticker"].eq("PTT.BK")) & (merged["date"].gt(pd.Timestamp("2023-01-09")))]
    assert expired_ptt["sentiment_score"].eq(0.0).all()


def test_add_sector_context_adds_mapping_and_numeric_features(tmp_path) -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    mapping_path = tmp_path / "sector_mapping.csv"
    pd.DataFrame(
        [
            {"ticker": "PTT.BK", "sector": "Energy & Utilities", "industry_group": "Resources"},
            {"ticker": "AOT.BK", "sector": "Transportation & Logistics", "industry_group": "Services"},
        ]
    ).to_csv(mapping_path, index=False)

    mapping = load_sector_mapping(mapping_path)
    enriched = add_sector_context(features, mapping)

    expected_columns = {
        "sector",
        "industry_group",
        "sector_equal_weight_return_1d",
        "sector_relative_return_1d",
        "sector_peer_count",
        "sector_energy_utilities",
        "sector_transportation_logistics",
    }
    assert expected_columns.issubset(enriched.columns)
    assert enriched["sector"].notna().all()
    assert enriched["sector_peer_count"].eq(1).all()
    assert enriched["sector_energy_utilities"].isin([0.0, 1.0]).all()


def test_write_manifest_row_upserts_source(tmp_path) -> None:
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2),
            "ticker": ["PTT.BK", "PTT.BK"],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.0, 1.2],
            "volume": [1000, 1100],
            "market_return_1d": [0.0, 0.2],
            "macro_rate_change": [0.0, 0.0],
            "sentiment_score": [0.0, 0.0],
        }
    )
    manifest = tmp_path / "manifest.csv"

    write_manifest_row(
        manifest,
        source_name="yahoo_thai_ohlcv",
        access_method="mock",
        prices=prices,
        raw_file_path=tmp_path / "prices.csv",
        license_note="test",
    )
    write_manifest_row(
        manifest,
        source_name="yahoo_thai_ohlcv",
        access_method="mock",
        prices=prices,
        raw_file_path=tmp_path / "prices.csv",
        license_note="test",
    )

    loaded = pd.read_csv(manifest)
    assert len(loaded) == 1
    assert loaded.iloc[0]["symbols"] == "PTT.BK"


def test_algorithm_kwargs_accept_training_overrides() -> None:
    kwargs = _algorithm_kwargs(
        "ppo",
        1000,
        {
            "learning_rate": 0.001,
            "n_steps": 256,
            "batch_size": 128,
            "clip_range": 0.2,
            "buffer_size": 999,
        },
    )

    assert kwargs["learning_rate"] == 0.001
    assert kwargs["n_steps"] == 256
    assert kwargs["batch_size"] == 128
    assert kwargs["clip_range"] == 0.2
    assert "buffer_size" not in kwargs


def test_optuna_trial_helpers_select_algorithm_and_apply_config() -> None:
    algorithms = parse_algorithms("ppo,a2c")
    algo, trial_number = select_trial_algorithm(3, algorithms)
    params = suggest_hyperparameters(algo, trial_number, seed=42)
    config = _config()
    config["training"] = {"total_timesteps": 1000}

    trial_config = apply_trial_config(
        config,
        algo=algo,
        trial_id=3,
        params=params,
        total_timesteps=500,
    )

    assert algo == "a2c"
    assert trial_number == 1
    assert "learning_rate" in params
    assert "gamma" in params
    assert "gae_lambda" in params
    assert "ent_coef" in params
    assert "n_steps" in params
    assert trial_config["training"]["algo"] == "a2c"
    assert trial_config["training"]["total_timesteps"] == 500
    assert trial_config["training"]["seed"] == config["project"]["seed"] + 3


def test_optuna_trial_sampling_changes_across_trial_numbers() -> None:
    params_0 = suggest_hyperparameters("ppo", trial_number=0, seed=42)
    params_1 = suggest_hyperparameters("ppo", trial_number=1, seed=42)

    assert params_0 != params_1


def test_large_scale_screening_supports_all_continuous_algorithms() -> None:
    config = {"screening": {"algorithms": ["ppo", "a2c", "ddpg", "td3", "sac"]}}
    tasks = screening_task_grid(config)

    assert [task.algorithm for task in tasks] == ["ppo", "a2c", "ddpg", "td3", "sac"]
    for task in tasks:
        params = suggest_hyperparameters(task.algorithm, trial_number=0, seed=42)
        kwargs = _algorithm_kwargs(task.algorithm, 20_000, params)
        assert "learning_rate" in kwargs
        assert "gamma" in kwargs
    assert "policy_delay" in _algorithm_kwargs("td3", 20_000, suggest_hyperparameters("td3", 0, 42))
    assert "ent_coef" in _algorithm_kwargs("sac", 20_000, suggest_hyperparameters("sac", 0, 42))


def test_aggregate_trials_ranks_by_objective(tmp_path) -> None:
    for trial_id, value in [(0, 0.1), (1, 0.3)]:
        trial_dir = tmp_path / f"trial_{trial_id}"
        trial_dir.mkdir()
        (trial_dir / "trial_metadata.json").write_text(
            json.dumps(
                {
                    "trial_id": trial_id,
                    "trial_number": trial_id,
                    "algorithm": "ppo",
                    "seed": 42 + trial_id,
                    "objective_metric": "sharpe",
                    "objective_value": value,
                    "sampled_params": {"learning_rate": 0.001, "gamma": 0.99},
                    "metrics": {"sharpe": value, "cumulative_return": value / 10},
                }
            ),
            encoding="utf-8",
        )

    results = aggregate_trials(tmp_path)

    assert list(results["trial_id"]) == [1, 0]
    assert "param_learning_rate" in results.columns
    assert "metric_cumulative_return" in results.columns


def test_aggregate_screening_requires_complete_algorithm_trial_grid(tmp_path) -> None:
    algorithms = ["ppo", "sac"]
    for algorithm_index, algorithm in enumerate(algorithms):
        for trial_number in range(2):
            trial_id = algorithm_index * 2 + trial_number
            trial_dir = tmp_path / f"trial_{trial_id}"
            trial_dir.mkdir()
            value = float(algorithm_index + trial_number / 10)
            (trial_dir / "trial_metadata.json").write_text(
                json.dumps(
                    {
                        "trial_id": trial_id,
                        "trial_number": trial_number,
                        "algorithm": algorithm,
                        "seed": 42 + trial_id,
                        "objective_metric": "sharpe",
                        "objective_value": value,
                        "sampled_params": {"learning_rate": 0.001},
                        "metrics": {"sharpe": value},
                    }
                ),
                encoding="utf-8",
            )
    config = {
        "screening": {
            "algorithms": algorithms,
            "trials_per_algorithm": 2,
            "objective_metric": "sharpe",
            "top_n_per_algorithm": 1,
        }
    }

    results, top = aggregate_screening(tmp_path, config=config)

    assert len(results) == 4
    assert len(top) == 2
    assert set(top["algorithm"]) == set(algorithms)
    assert (tmp_path / "aggregate" / "screening_completeness.csv").exists()
    ranking = pd.read_csv(tmp_path / "aggregate" / "screening_algorithm_ranking.csv")
    selection = json.loads((tmp_path / "aggregate" / "screening_selected_algorithms.json").read_text())
    assert ranking["selected"].all()
    assert set(selection["selected_algorithms"]) == set(algorithms)


def test_read_objective_rejects_nonfinite_metrics_and_invalid_actions(tmp_path) -> None:
    pd.DataFrame([{"policy": "ppo", "sharpe": float("nan"), "max_drawdown": -0.1}]).to_csv(
        tmp_path / "ppo_metrics.csv", index=False
    )
    with pytest.raises(ValueError, match="NaN or infinite"):
        read_objective(tmp_path, "ppo")

    pd.DataFrame([{"policy": "ppo", "sharpe": 0.2, "max_drawdown": -0.1}]).to_csv(
        tmp_path / "ppo_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "A", "weight": 0.8},
            {"date": "2024-01-01", "ticker": "B", "weight": 0.8},
        ]
    ).to_csv(tmp_path / "ppo_actions.csv", index=False)
    with pytest.raises(ValueError, match="exceed total capital"):
        read_objective(tmp_path, "ppo")


def test_budget_sweep_aggregation_is_complete_validation_only_and_selects_plateau(tmp_path) -> None:
    config = {
        "budget_sweep": {
            "algorithms": ["ppo", "a2c"],
            "budgets": [20, 100],
            "seeds": [1, 2],
            "window_ids": [0],
            "objective_metric": "sharpe",
            "plateau_tolerance": 0.03,
        }
    }
    assert budget_values(config) == [20, 100]
    for budget in (20, 100):
        for algorithm in ("ppo", "a2c"):
            for seed in (1, 2):
                run_dir = tmp_path / f"budget_{budget}" / f"algorithm_{algorithm}" / "window_0" / f"seed_{seed}"
                run_dir.mkdir(parents=True)
                sharpe = 0.10 if budget == 20 else 0.12
                (run_dir / "budget_trial_result.json").write_text(
                    json.dumps(
                        {
                            "algorithm": algorithm,
                            "window": 0,
                            "seed": seed,
                            "training_budget": budget,
                            "selection_split": "validation",
                            "metrics": {"split": "validation", "policy": algorithm, "sharpe": sharpe},
                        }
                    ),
                    encoding="utf-8",
                )

    results, summary, selection = aggregate_budget_sweep(tmp_path, config=config)

    assert set(results["split"]) == {"validation"}
    assert len(summary) == 4
    assert selection["selected_training_budget"] == 20
    assert (tmp_path / "aggregate" / "training_budget_completeness.csv").exists()
    assert (tmp_path / "aggregate" / "selected_training_budget.json").exists()


def test_feature_ablation_task_distribution_and_complete_aggregation(tmp_path) -> None:
    config = {
        "feature_ablation": {
            "task_count": 2,
            "algorithm": "ppo",
            "seeds": [42, 43],
            "window_ids": [0],
            "bootstrap_resamples": 100,
            "groups": [
                {"name": "returns", "config": "unused_returns.yaml"},
                {"name": "technical", "config": "unused_technical.yaml"},
                {"name": "market", "config": "unused_market.yaml"},
            ],
        }
    }
    assert [group.name for group in groups_for_task(config, 0)] == ["returns", "market"]
    assert [group.name for group in groups_for_task(config, 1)] == ["technical"]

    for group_index, group in enumerate(("returns", "technical", "market")):
        for seed in (42, 43):
            run_dir = tmp_path / f"feature_{group}" / "window_0" / f"seed_{seed}"
            identity = {
                "experiment_id": f"{group}_{seed}",
                "algorithm": "ppo",
                "window": 0,
                "seed": seed,
                "feature_group": group,
            }
            with tracked_experiment(run_dir, config={"paths": {}}, identity=identity):
                pd.DataFrame(
                    [
                        {
                            "window": 0,
                            "seed": seed,
                            "split": "test",
                            "policy": "ppo",
                            "cumulative_return": 0.01 * (group_index + 1),
                            "sharpe": 0.1 * (group_index + 1),
                            "max_drawdown": -0.1,
                        }
                    ]
                ).to_csv(run_dir / "walk_forward_metrics.csv", index=False)

    metrics, summary = aggregate_feature_ablation(tmp_path, config=config)

    assert len(metrics) == 6
    assert summary.iloc[0]["feature_group"] == "market"
    assert (tmp_path / "aggregate" / "feature_ablation_completeness.csv").exists()
    assert (tmp_path / "aggregate" / "feature_ablation_confidence_intervals.png").exists()
    comparisons = pd.read_csv(tmp_path / "aggregate" / "feature_ablation_paired_comparisons.csv")
    assert set(comparisons["feature_group"]) == {"returns", "market"}
    assert comparisons["excess_sharpe_holm_p_value"].between(0.0, 1.0).all()


def test_nested_screening_aggregation_selects_validation_best_per_window(tmp_path) -> None:
    algorithms = ["ppo", "sac"]
    config = {
        "nested_screening": {
            "algorithms": algorithms,
            "window_ids": [0],
            "trials_per_algorithm_window": 2,
            "objective_metric": "sharpe",
        }
    }
    for algorithm_index, algorithm in enumerate(algorithms):
        for trial_number in range(2):
            trial_dir = tmp_path / f"algorithm_{algorithm}" / "window_0" / f"trial_{trial_number}"
            value = 0.1 * algorithm_index + 0.01 * trial_number
            identity = {
                "experiment_id": f"{algorithm}_{trial_number}",
                "algorithm": algorithm,
                "window": 0,
                "seed": 42 + trial_number,
            }
            with tracked_experiment(trial_dir, config={"paths": {}}, identity=identity):
                (trial_dir / "nested_trial_result.json").write_text(
                    json.dumps(
                        {
                            "trial_id": algorithm_index * 2 + trial_number,
                            "trial_number": trial_number,
                            "algorithm": algorithm,
                            "window": 0,
                            "seed": 42 + trial_number,
                            "selection_split": "validation",
                            "objective_metric": "sharpe",
                            "metrics": {"sharpe": value, "cumulative_return": value / 10},
                            "sampled_params": {"learning_rate": 0.001},
                        }
                    ),
                    encoding="utf-8",
                )

    results, selected = aggregate_nested_screening(tmp_path, config=config)

    assert len(results) == 4
    assert selected["trial_number"].eq(1).all()
    assert (tmp_path / "aggregate" / "nested_screening_completeness.csv").exists()


def test_aggregate_ablation_writes_best_policy_and_ppo_summary(tmp_path) -> None:
    for feature_group, ppo_sharpe, equal_weight_sharpe in [
        ("ablation_returns_real_ohlcv", -0.2, 0.1),
        ("ablation_technical_real_ohlcv", 0.3, 0.0),
    ]:
        run_dir = tmp_path / feature_group
        run_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "policy": "ppo",
                    "cumulative_return": ppo_sharpe / 10,
                    "annualized_return": ppo_sharpe / 5,
                    "sharpe": ppo_sharpe,
                    "max_drawdown": -0.1,
                    "final_portfolio_value": 1_000_000,
                },
                {
                    "policy": "equal_weight",
                    "cumulative_return": equal_weight_sharpe / 10,
                    "annualized_return": equal_weight_sharpe / 5,
                    "sharpe": equal_weight_sharpe,
                    "max_drawdown": -0.1,
                    "final_portfolio_value": 1_000_000,
                },
            ]
        ).to_csv(run_dir / "comparison_metrics.csv", index=False)

    metrics, summary = aggregate_ablation(tmp_path)

    assert len(metrics) == 4
    assert summary.iloc[0]["feature_group"] == "ablation_technical_real_ohlcv"
    assert (tmp_path / "ablation_metrics.csv").exists()
    assert (tmp_path / "ablation_ppo_summary.csv").exists()
    assert (tmp_path / "ablation_ppo_sharpe.png").exists()


def test_aggregate_walk_forward_combines_window_metrics(tmp_path) -> None:
    for window, sharpe in [(0, 0.1), (1, 0.3)]:
        window_dir = tmp_path / f"window_{window}" / f"walk_forward_window_{window}"
        window_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "window": window,
                    "split": "test",
                    "policy": "ppo",
                    "cumulative_return": sharpe / 10,
                    "annualized_return": sharpe / 5,
                    "sharpe": sharpe,
                    "max_drawdown": -0.1,
                    "final_portfolio_value": 1_000_000 + window,
                }
            ]
        ).to_csv(window_dir / "walk_forward_metrics.csv", index=False)
        pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "portfolio_value": [1_000_000 + window],
                "policy": ["ppo"],
            }
        ).to_csv(window_dir / "ppo_test_equity_curve.csv", index=False)

    metrics, summary = aggregate_walk_forward(tmp_path)

    assert len(metrics) == 2
    assert summary.iloc[0]["policy"] == "ppo"
    assert summary.iloc[0]["windows"] == 2
    assert (tmp_path / "walk_forward_all_metrics.csv").exists()
    assert (tmp_path / "walk_forward_test_equity_curves.csv").exists()


def test_robustness_task_grid_maps_algorithm_and_window_without_duplicate_seeds() -> None:
    config = {
        "robustness": {
            "algorithms": ["ppo", "a2c"],
            "window_ids": [0, 1],
            "seeds": [42, 43, 44],
        }
    }

    tasks = robustness_task_grid(config)

    assert [(task.task_id, task.algorithm, task.window) for task in tasks] == [
        (0, "ppo", 0),
        (1, "ppo", 1),
        (2, "a2c", 0),
        (3, "a2c", 1),
    ]
    assert robustness_seeds(config) == [42, 43, 44]


def test_aggregate_robustness_is_test_only_and_writes_uncertainty_artifacts(tmp_path) -> None:
    for algorithm in ("ppo", "a2c"):
        for window in (0, 1):
            for seed in (42, 43):
                run_dir = tmp_path / f"algorithm_{algorithm}" / f"window_{window}" / f"seed_{seed}"
                run_dir.mkdir(parents=True)
                adjustment = (seed - 42) * 0.01 + window * 0.02
                rows = []
                for split in ("validation", "test"):
                    split_adjustment = 10.0 if split == "validation" else 0.0
                    rows.extend(
                        [
                            {
                                "window": window,
                                "seed": seed,
                                "split": split,
                                "policy": algorithm,
                                "cumulative_return": 0.02 + adjustment + split_adjustment,
                                "sharpe": 0.2 + adjustment + split_adjustment,
                                "max_drawdown": -0.08 + adjustment,
                            },
                            {
                                "window": window,
                                "seed": seed,
                                "split": split,
                                "policy": "equal_weight",
                                "cumulative_return": 0.01 + window * 0.01,
                                "sharpe": 0.1 + window * 0.01,
                                "max_drawdown": -0.10,
                            },
                            {
                                "window": window,
                                "seed": seed,
                                "split": split,
                                "policy": "set_index",
                                "cumulative_return": 0.005 + window * 0.01,
                                "sharpe": 0.05 + window * 0.01,
                                "max_drawdown": -0.12,
                            },
                        ]
                    )
                pd.DataFrame(rows).to_csv(run_dir / "walk_forward_metrics.csv", index=False)

    config = {
        "robustness": {
            "algorithms": ["ppo", "a2c"],
            "seeds": [42, 43],
            "window_ids": [0, 1],
            "primary_baselines": ["equal_weight", "set_index"],
            "confidence": 0.95,
            "bootstrap_resamples": 100,
            "bootstrap_seed": 7,
        }
    }
    output_dir = tmp_path / "aggregate"

    all_metrics, test_metrics, policy_summary, paired_summary = aggregate_robustness(
        tmp_path,
        config=config,
        output_dir=output_dir,
    )

    assert set(test_metrics["split"]) == {"test"}
    assert test_metrics["sharpe"].max() < 1.0
    assert len(all_metrics) == 32
    assert set(policy_summary["policy"]) == {"ppo", "a2c", "equal_weight", "set_index"}
    assert policy_summary.loc[policy_summary["policy"].eq("equal_weight"), "runs"].item() == 2
    assert "sharpe_ci_low" in policy_summary
    assert set(paired_summary["baseline"]) == {"equal_weight", "set_index"}
    assert paired_summary["excess_cumulative_return_positive_rate"].eq(1.0).all()
    assert paired_summary["excess_cumulative_return_holm_p_value"].between(0.0, 1.0).all()
    for name in (
        "robustness_test_metrics.csv",
        "robustness_policy_summary.csv",
        "robustness_paired_summary.csv",
        "robustness_completeness.csv",
        "robustness_confidence_intervals.png",
        "robustness_sharpe_heatmap.png",
        "robustness_summary.md",
    ):
        assert (output_dir / name).exists()


def test_aggregate_robustness_rejects_incomplete_candidate_grid(tmp_path) -> None:
    run_dir = tmp_path / "algorithm_ppo" / "window_0" / "seed_42"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "window": 0,
                "seed": 42,
                "split": "test",
                "policy": "ppo",
                "cumulative_return": 0.01,
                "sharpe": 0.1,
                "max_drawdown": -0.1,
            },
            {
                "window": 0,
                "seed": 42,
                "split": "test",
                "policy": "equal_weight",
                "cumulative_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": -0.1,
            },
        ]
    ).to_csv(run_dir / "walk_forward_metrics.csv", index=False)
    config = {
        "robustness": {
            "algorithms": ["ppo"],
            "seeds": [42, 43],
            "window_ids": [0],
            "primary_baselines": ["equal_weight"],
            "bootstrap_resamples": 10,
        }
    }

    with pytest.raises(ValueError, match="incomplete robustness candidate grid"):
        aggregate_robustness(tmp_path, config=config)


def test_tracked_experiment_records_complete_and_failed_status(tmp_path) -> None:
    config = {"paths": {}, "project": {"name": "test"}}
    complete_dir = tmp_path / "complete"
    with tracked_experiment(
        complete_dir,
        config=config,
        identity={"experiment_id": "complete", "algorithm": "ppo", "window": 0, "seed": 42},
    ):
        pass

    failed_dir = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="expected failure"):
        with tracked_experiment(
            failed_dir,
            config=config,
            identity={"experiment_id": "failed", "algorithm": "a2c", "window": 1, "seed": 43},
        ):
            raise RuntimeError("expected failure")

    manifests = pd.DataFrame(collect_experiment_manifests(tmp_path))
    assert set(manifests["status"]) == {"complete", "failed"}
    assert (complete_dir / "resolved_config.json").exists()
    assert "RuntimeError: expected failure" in (failed_dir / "experiment_manifest.json").read_text()


def test_regime_slice_helpers_report_unavailable_and_high_volatility_windows() -> None:
    config = _config()
    features = add_technical_features(make_synthetic_prices(config))
    config["regimes"] = {
        "slices": [
            {"name": "before_data", "start": "2020-01-01", "end": "2020-02-01", "note": "outside sample"},
            {"name": "in_sample", "start": "2023-01-03", "end": "2023-04-30"},
        ],
        "auto_high_volatility": {"enabled": True, "window": 20, "top_n": 1, "min_gap": 20},
    }

    regimes, summary = configured_regime_slices(features, config)
    high_vol, high_vol_summary = high_volatility_slices(features, config)

    assert regimes["before_data"].empty
    assert summary[summary["regime"].eq("before_data")].iloc[0]["unique_dates"] == 0
    assert pd.to_datetime(regimes["in_sample"]["date"]).nunique() > config["features"]["lookback"]
    assert len(high_vol) == 1
    assert high_vol_summary.iloc[0]["unique_dates"] == 20


def test_execution_stress_runs_scenarios_on_test_split(tmp_path) -> None:
    config = _config()
    config["data"]["source"] = "unit_test_features"
    features = add_technical_features(make_synthetic_prices(config))
    feature_path = tmp_path / "features.csv"
    features.to_csv(feature_path, index=False)
    config["paths"] = {
        "raw_prices": str(tmp_path / "prices.csv"),
        "features": str(feature_path),
        "fallback_features_csv": str(feature_path),
        "results_dir": str(tmp_path / "stress"),
    }
    config["splits"] = {"train_fraction": 0.6, "validation_fraction": 0.2}
    config["execution_stress"] = {
        "model_paths": [],
        "scenarios": [
            {"name": "base", "transaction_cost": 0.001, "slippage": 0.0},
            {"name": "hostile", "transaction_cost": 0.01, "slippage": 0.002, "max_turnover": 0.5},
        ],
    }

    scenario_config = apply_execution_scenario(config, config["execution_stress"]["scenarios"][1])
    metrics, readiness = run_execution_stress(config, tmp_path / "stress")

    assert scenario_config["environment"]["slippage"] == 0.002
    assert set(metrics["scenario"]) == {"base", "hostile"}
    assert readiness.empty
    assert (tmp_path / "stress" / "execution_stress_sensitivity.png").exists()


def test_run_regime_tests_writes_metrics_and_summary(tmp_path) -> None:
    config = _config()
    config["data"]["source"] = "unit_test_features"
    features = add_technical_features(make_synthetic_prices(config))
    feature_path = tmp_path / "features.csv"
    features.to_csv(feature_path, index=False)
    config["paths"] = {
        "raw_prices": str(tmp_path / "prices.csv"),
        "features": str(feature_path),
        "fallback_features_csv": str(feature_path),
    }
    config["regimes"] = {
        "slices": [{"name": "pilot", "start": "2023-01-03", "end": "2023-06-30"}],
        "auto_high_volatility": {"enabled": False},
    }

    metrics, slice_summary = run_regime_tests(config, tmp_path / "regime")

    assert "pilot" in set(metrics["regime"])
    assert "equal_weight" in set(metrics["policy"])
    assert slice_summary.iloc[0]["unique_dates"] > config["features"]["lookback"]
    assert (tmp_path / "regime" / "regime_metrics.csv").exists()
    assert (tmp_path / "regime" / "regime_summary.md").exists()


def test_data_quality_reports_panel_coverage_and_missingness() -> None:
    features = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "ticker": ["AOT.BK", "AOT.BK", "PTT.BK"],
            "close": [1.0, None, 2.0],
        }
    )

    coverage = coverage_by_ticker(features)
    missingness = missingness_by_column(features)

    ptt = coverage[coverage["ticker"].eq("PTT.BK")].iloc[0]
    assert ptt["missing_panel_dates"] == 1
    assert np.isclose(ptt["coverage_ratio"], 0.5)
    assert missingness[missingness["column"].eq("close")].iloc[0]["missing_values"] == 1
    assert "| ticker |" in dataframe_to_markdown(coverage)
