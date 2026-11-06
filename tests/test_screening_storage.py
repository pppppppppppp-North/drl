from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from src.experiments import run_screening_storage_safe as storage_safe_runner
from src.experiments.screening_storage import (
    RETENTION_RECEIPT,
    finalize_checkpoint_free_trial,
    validate_checkpoint_retention,
)


@pytest.fixture
def fake_sb3_callbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Install just enough of SB3 for the callback substitution tests."""

    original_writes: list[Path] = []

    class OriginalCheckpointCallback:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = dict(kwargs)
            self.save_path = Path(kwargs.get("save_path", tmp_path / "original_checkpoints"))

        def _on_step(self) -> bool:
            self.save_path.mkdir(parents=True, exist_ok=True)
            checkpoint = self.save_path / "original_callback_wrote.zip"
            checkpoint.write_bytes(b"quota-heavy checkpoint")
            original_writes.append(checkpoint)
            return True

    stable_baselines3 = ModuleType("stable_baselines3")
    stable_baselines3.__path__ = []  # type: ignore[attr-defined]
    common = ModuleType("stable_baselines3.common")
    common.__path__ = []  # type: ignore[attr-defined]
    callbacks = ModuleType("stable_baselines3.common.callbacks")
    callbacks.CheckpointCallback = OriginalCheckpointCallback  # type: ignore[attr-defined]
    common.callbacks = callbacks  # type: ignore[attr-defined]
    stable_baselines3.common = common  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "stable_baselines3", stable_baselines3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.callbacks", callbacks)
    return callbacks, OriginalCheckpointCallback, original_writes


def _write_scientific_outputs(trial_dir: Path, algorithm: str = "ppo") -> dict[Path, bytes]:
    outputs = {
        trial_dir / f"{algorithm}_model.zip": b"final-model-v1",
        trial_dir / f"{algorithm}_metrics.csv": b"policy,sharpe\nppo,1.25\n",
        trial_dir / f"{algorithm}_actions.csv": b"date,ticker,weight\n2026-01-01,A,1.0\n",
        trial_dir / f"{algorithm}_equity_curve.csv": (
            b"date,portfolio_value\n2026-01-01,1.0\n2026-01-02,1.1\n"
        ),
        trial_dir / "tensorboard" / "events.out.tfevents": b"tensorboard-event",
    }
    for path, payload in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return outputs


def _make_valid_retention_trial(trial_dir: Path, algorithm: str = "ppo") -> Path:
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / f"{algorithm}_model.zip").write_bytes(b"trusted-model-v1")
    return finalize_checkpoint_free_trial(
        trial_dir,
        algorithm,
        callback_instances=1,
        callback_on_step_calls=3,
    )


def test_callback_replacement_subclasses_suppresses_writes_and_restores(
    fake_sb3_callbacks,
    tmp_path: Path,
) -> None:
    callbacks, original_callback, original_writes = fake_sb3_callbacks
    save_path = tmp_path / "checkpoints" / "ppo"

    with storage_safe_runner._suppress_checkpoint_writes() as instances:
        replacement = callbacks.CheckpointCallback
        assert replacement is not original_callback
        assert issubclass(replacement, original_callback)

        callback = replacement(save_freq=25, save_path=save_path, name_prefix="ppo")
        assert callback.requested_args == ()
        assert callback.requested_kwargs == {
            "save_freq": 25,
            "save_path": save_path,
            "name_prefix": "ppo",
        }
        assert callback._on_step() is True
        assert callback._on_step() is True
        assert callback.on_step_calls == 2
        assert instances == [callback]

    assert callbacks.CheckpointCallback is original_callback
    assert original_writes == []
    assert not save_path.exists()


def test_callback_replacement_restores_original_on_base_exception(
    fake_sb3_callbacks,
) -> None:
    callbacks, original_callback, _ = fake_sb3_callbacks

    class TrainingAbort(BaseException):
        pass

    failure = TrainingAbort("stop immediately")
    with pytest.raises(TrainingAbort) as caught:
        with storage_safe_runner._suppress_checkpoint_writes():
            assert callbacks.CheckpointCallback is not original_callback
            raise failure

    assert caught.value is failure
    assert callbacks.CheckpointCallback is original_callback


def test_storage_safe_training_retains_scientific_outputs_and_valid_receipt(
    fake_sb3_callbacks,
    tmp_path: Path,
) -> None:
    callbacks, original_callback, original_writes = fake_sb3_callbacks
    trial_dir = tmp_path / "trial_0"
    trial_dir.mkdir()
    config = {
        "project": {"name": "storage_safe_test", "seed": 42},
        "training": {"algo": "PPO", "seed": 42, "total_timesteps": 100},
        "screening": {"objective_metric": "sharpe"},
    }
    original_config = copy.deepcopy(config)
    retained: dict[Path, bytes] = {}

    def train(trial_config: dict[str, Any], output_dir: Path) -> bool:
        assert trial_config is config
        checkpoint_dir = output_dir / "checkpoints" / "ppo"
        checkpoint_dir.mkdir(parents=True)
        callback = callbacks.CheckpointCallback(
            save_freq=50,
            save_path=checkpoint_dir,
            name_prefix="ppo",
        )
        assert callback._on_step()
        assert callback._on_step()
        retained.update(_write_scientific_outputs(output_dir))
        return True

    assert storage_safe_runner._storage_safe_training(train, config, trial_dir) is True

    assert config == original_config
    assert callbacks.CheckpointCallback is original_callback
    assert original_writes == []
    assert not (trial_dir / "checkpoints").exists()
    for path, payload in retained.items():
        assert path.read_bytes() == payload

    receipt = json.loads((trial_dir / RETENTION_RECEIPT).read_text(encoding="utf-8"))
    assert receipt["policy"] == "suppress_checkpoint_writes_retain_final_model"
    assert receipt["removed_file_count"] == 0
    assert receipt["removed_bytes"] == 0
    assert receipt["removed_files"] == []
    assert receipt["callback"] == {
        "implementation": "no_write_checkpoint_callback",
        "instances": 1,
        "on_step_calls": 2,
    }
    assert receipt["final_model"]["path"] == "ppo_model.zip"
    assert receipt["final_model"]["bytes"] == len(b"final-model-v1")
    assert validate_checkpoint_retention(trial_dir, "ppo") == []


def test_unexpected_checkpoint_is_purged_and_successful_training_is_rejected(
    fake_sb3_callbacks,
    tmp_path: Path,
) -> None:
    callbacks, original_callback, original_writes = fake_sb3_callbacks
    trial_dir = tmp_path / "trial_rogue_checkpoint"
    trial_dir.mkdir()
    config = {"training": {"algo": "ppo"}}

    def train(_config: dict[str, Any], output_dir: Path) -> bool:
        checkpoint_dir = output_dir / "checkpoints" / "ppo"
        checkpoint_dir.mkdir(parents=True)
        callback = callbacks.CheckpointCallback(save_freq=1, save_path=checkpoint_dir)
        assert callback._on_step()
        (checkpoint_dir / "unexpected.zip").write_bytes(b"unexpected checkpoint")
        _write_scientific_outputs(output_dir)
        return True

    with pytest.raises(RuntimeError, match="unexpected checkpoint write.*1 file"):
        storage_safe_runner._storage_safe_training(train, config, trial_dir)

    assert callbacks.CheckpointCallback is original_callback
    assert original_writes == []
    assert not (trial_dir / "checkpoints").exists()
    assert not (trial_dir / RETENTION_RECEIPT).exists()
    assert (trial_dir / "ppo_model.zip").read_bytes() == b"final-model-v1"
    assert (trial_dir / "ppo_metrics.csv").is_file()


def test_training_exception_remains_primary_when_checkpoint_cleanup_also_fails(
    fake_sb3_callbacks,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks, original_callback, _ = fake_sb3_callbacks
    trial_dir = tmp_path / "trial_training_failure"
    trial_dir.mkdir()
    config = {"training": {"algo": "ppo"}}

    class TrainingAbort(BaseException):
        pass

    failure = TrainingAbort("primary training failure")

    def train(_config: dict[str, Any], _output_dir: Path) -> bool:
        raise failure

    def failed_cleanup(_trial_dir: Path, _algorithm: str) -> dict[str, Any]:
        raise OSError("secondary quota cleanup failure")

    monkeypatch.setattr(storage_safe_runner, "purge_checkpoint_artifacts", failed_cleanup)
    with pytest.raises(TrainingAbort) as caught:
        storage_safe_runner._storage_safe_training(train, config, trial_dir)

    assert caught.value is failure
    assert callbacks.CheckpointCallback is original_callback
    notes = getattr(caught.value, "__notes__", [])
    assert any(
        "checkpoint cleanup also failed" in note
        and "secondary quota cleanup failure" in note
        for note in notes
    )


def test_missing_model_or_receipt_invalidates_retention(tmp_path: Path) -> None:
    missing_model_dir = tmp_path / "missing_model"
    _make_valid_retention_trial(missing_model_dir)
    (missing_model_dir / "ppo_model.zip").unlink()
    assert any(
        "final model is missing or unsafe" in issue
        for issue in validate_checkpoint_retention(missing_model_dir, "ppo")
    )

    missing_receipt_dir = tmp_path / "missing_receipt"
    receipt_path = _make_valid_retention_trial(missing_receipt_dir)
    receipt_path.unlink()
    assert any(
        "retention receipt is unreadable" in issue
        for issue in validate_checkpoint_retention(missing_receipt_dir, "ppo")
    )


def test_tampered_model_or_receipt_invalidates_retention(tmp_path: Path) -> None:
    tampered_model_dir = tmp_path / "tampered_model"
    _make_valid_retention_trial(tampered_model_dir)
    model_path = tampered_model_dir / "ppo_model.zip"
    original_model = model_path.read_bytes()
    model_path.write_bytes(b"X" * len(original_model))
    assert any(
        "final-model hash does not match" in issue
        for issue in validate_checkpoint_retention(tampered_model_dir, "ppo")
    )

    tampered_receipt_dir = tmp_path / "tampered_receipt"
    receipt_path = _make_valid_retention_trial(tampered_receipt_dir)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["policy"] = "delete_checkpoints_after_writing"
    receipt["callback"]["implementation"] = "ordinary_checkpoint_callback"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    issues = validate_checkpoint_retention(tampered_receipt_dir, "ppo")
    assert "checkpoint retention policy is not the storage-safe policy" in issues
    assert "checkpoint retention callback implementation does not match" in issues


def test_wrapper_reuses_exact_completed_trial_with_valid_retention_and_restores_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial_dir = tmp_path / "trial_0"
    _make_valid_retention_trial(trial_dir)
    config = {
        "project": {"name": "reuse_test", "seed": 42},
        "training": {"algo": "ppo", "seed": 42},
    }
    original_config = copy.deepcopy(config)
    identity = {"algorithm": "ppo", "trial_id": 0}
    reuse_calls: list[dict[str, Any]] = []
    sentinel = object()

    def original_train(_config: dict[str, Any], _output_dir: Path) -> bool:
        raise AssertionError("a valid same-attempt trial must not be retrained")

    def original_reuse(
        candidate: Path,
        *,
        trial_config: dict[str, Any],
        identity: dict[str, Any],
        objective_metric: str,
    ) -> bool:
        reuse_calls.append(
            {
                "candidate": candidate,
                "trial_config": trial_config,
                "identity": identity,
                "objective_metric": objective_metric,
            }
        )
        return True

    def fake_base_run(
        received_config: dict[str, Any], *, task_id: int, output_dir: Path
    ) -> object:
        assert task_id == 0
        assert output_dir == tmp_path / "screening"
        assert storage_safe_runner.base_runner.completed_screening_trial_is_reusable(
            trial_dir,
            trial_config=received_config,
            identity=identity,
            objective_metric="sharpe",
        )
        return sentinel

    monkeypatch.setattr(
        storage_safe_runner.base_runner, "train_with_stable_baselines", original_train
    )
    monkeypatch.setattr(
        storage_safe_runner.base_runner,
        "completed_screening_trial_is_reusable",
        original_reuse,
    )
    monkeypatch.setattr(storage_safe_runner.base_runner, "run_screening_task", fake_base_run)

    result = storage_safe_runner.run_screening_task(
        config,
        task_id=0,
        output_dir=tmp_path / "screening",
    )

    assert result is sentinel
    assert config == original_config
    assert len(reuse_calls) == 1
    assert reuse_calls[0]["candidate"] == trial_dir
    assert reuse_calls[0]["trial_config"] is config
    assert storage_safe_runner.base_runner.train_with_stable_baselines is original_train
    assert (
        storage_safe_runner.base_runner.completed_screening_trial_is_reusable
        is original_reuse
    )


def test_wrapper_restores_base_hooks_when_base_runner_raises_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def original_train(_config: dict[str, Any], _output_dir: Path) -> bool:
        return False

    def original_reuse(*args: Any, **kwargs: Any) -> bool:
        return False

    class SchedulerAbort(BaseException):
        pass

    failure = SchedulerAbort("scheduler interrupted the process")

    def failing_base_run(*args: Any, **kwargs: Any) -> None:
        raise failure

    monkeypatch.setattr(
        storage_safe_runner.base_runner, "train_with_stable_baselines", original_train
    )
    monkeypatch.setattr(
        storage_safe_runner.base_runner,
        "completed_screening_trial_is_reusable",
        original_reuse,
    )
    monkeypatch.setattr(
        storage_safe_runner.base_runner, "run_screening_task", failing_base_run
    )

    with pytest.raises(SchedulerAbort) as caught:
        storage_safe_runner.run_screening_task(
            {"training": {"algo": "ppo"}},
            task_id=0,
            output_dir=tmp_path,
        )

    assert caught.value is failure
    assert storage_safe_runner.base_runner.train_with_stable_baselines is original_train
    assert (
        storage_safe_runner.base_runner.completed_screening_trial_is_reusable
        is original_reuse
    )


def test_storage_safe_sbatch_uses_bounded_resources_and_safe_entrypoint() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sbatch_path = project_root / "slurm" / "large_scale_screening_storage_safe.sbatch"
    text = sbatch_path.read_text(encoding="utf-8")

    assert "-m src.experiments.run_screening_storage_safe" in text
    assert "CONFIG_PATH=\"${CONFIG_PATH:?" in text
    assert re.search(r"^#SBATCH --nodes=1$", text, flags=re.MULTILINE)
    assert re.search(r"^#SBATCH --ntasks=1$", text, flags=re.MULTILINE)
    assert re.search(r"^#SBATCH --cpus-per-task=2$", text, flags=re.MULTILINE)
    assert re.search(r"^#SBATCH --mem=12GB$", text, flags=re.MULTILINE)
    assert re.search(r"^#SBATCH --time=4-00:00:00$", text, flags=re.MULTILINE)
    assert re.search(r"^#SBATCH --partition=compute-long$", text, flags=re.MULTILINE)
    assert re.search(r"^#SBATCH --array=0-4%2$", text, flags=re.MULTILINE)

    array_match = re.search(
        r"^#SBATCH --array=(\d+)-(\d+)%([1-9]\d*)$",
        text,
        flags=re.MULTILINE,
    )
    assert array_match is not None
    first_task, last_task, concurrency = map(int, array_match.groups())
    assert (first_task, last_task) == (0, 4)
    assert concurrency <= 2
