from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from scripts.run_reward_variant_experiment import SB3_ALGORITHMS, train_variant


def _features(n_dates: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for t in ("AAA", "BBB", "CCC"):
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_dates))
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t, "close": close[i], "ret": rng.normal(), "mkt": rng.normal()})
    return pd.DataFrame(rows)


def _config(algo: str) -> dict:
    training = {"algo": algo, "total_timesteps": 64, "num_envs": 1, "normalize_observations": True, "verbose": 0}
    if algo in {"sac", "td3", "ddpg"}:
        training.update(
            {
                "buffer_size": 200,
                "learning_starts": 16,
                "batch_size": 8,
                "train_freq": 4,
                "gradient_steps": 1,
                "optimize_memory_usage": True,
                "replay_buffer_kwargs": {"handle_timeout_termination": False},
            }
        )
        if algo in {"td3", "ddpg"}:
            training["action_noise_sigma"] = 0.3
    else:
        training["n_steps"] = 16
    return {
        "features": {"columns": ["ret", "mkt"], "lookback": 5},
        "environment": {
            "initial_cash": 1_000_000.0,
            "transaction_cost": 0.0015,
            "accounting": "self_financing",
            "action_transform": "softmax",
            "rebalance_every": 5,
            "reward": {"type": "log_return", "turnover_penalty": 0.0, "drawdown_penalty": 0.0},
        },
        "training": training,
    }


@pytest.mark.parametrize("algo", ["a2c", "sac", "td3", "ddpg"])
def test_train_variant_supports_each_algorithm(tmp_path, algo) -> None:
    model, normalizer = train_variant(_config(algo), _features(), seed=0, out_dir=tmp_path)
    assert (tmp_path / f"{algo}_model.zip").exists()
    assert json.loads((tmp_path / "model_info.json").read_text())["algo"] == algo
    assert (tmp_path / "vecnormalize.pkl").exists() and normalizer is not None
    obs = np.zeros(model.observation_space.shape, dtype=np.float32)
    action, _ = model.predict(normalizer(obs), deterministic=True)
    assert action.shape == (4,)  # three assets + cash logit


def test_unknown_algorithm_is_rejected(tmp_path) -> None:
    config = _config("a2c")
    config["training"]["algo"] = "dqn"
    with pytest.raises(ValueError, match="training.algo"):
        train_variant(config, _features(), seed=0, out_dir=tmp_path)
    assert set(SB3_ALGORITHMS) == {"ppo", "a2c", "sac", "td3", "ddpg"}
