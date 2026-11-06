from __future__ import annotations

import builtins

import pytest

from src.experiments.run_trial import suggest_hyperparameters


def test_preregistered_screening_requires_optuna(monkeypatch) -> None:
    original_import = builtins.__import__

    def reject_optuna(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_optuna)

    with pytest.raises(RuntimeError, match="Optuna is required"):
        suggest_hyperparameters(
            "ppo",
            trial_number=0,
            seed=42,
            require_optuna=True,
        )
