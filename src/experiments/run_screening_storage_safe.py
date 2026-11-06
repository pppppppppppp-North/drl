from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from src.experiments import run_screening as base_runner
from src.experiments.screening_storage import (
    RETENTION_RECEIPT,
    RETENTION_SIDECAR,
    finalize_checkpoint_free_trial,
    purge_checkpoint_artifacts,
    validate_checkpoint_retention,
)
from src.utils.config import load_config


_RUNNER_PATCH_LOCK = threading.Lock()


@contextmanager
def _suppress_checkpoint_writes() -> Iterator[list[Any]]:
    """Replace SB3's checkpoint callback with a verified no-write callback.

    ``src.train.train_with_stable_baselines`` imports ``CheckpointCallback``
    inside the training function.  Replacing that exact module attribute for
    the duration of one single-process screening trial prevents the quota-heavy
    periodic/terminal checkpoint write before it can occur.  The original
    callback is restored even when training raises.
    """

    try:
        from stable_baselines3.common import callbacks
    except ImportError:
        # Preserve the original optional-dependency behavior: the training
        # function will return False when Stable-Baselines3 is unavailable.
        yield []
        return

    original_callback = callbacks.CheckpointCallback
    instances: list[Any] = []

    class NoWriteCheckpointCallback(original_callback):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.requested_args = args
            self.requested_kwargs = dict(kwargs)
            self.on_step_calls = 0
            instances.append(self)

        def _on_step(self) -> bool:
            self.on_step_calls += 1
            return True

    callbacks.CheckpointCallback = NoWriteCheckpointCallback
    try:
        yield instances
    finally:
        callbacks.CheckpointCallback = original_callback


def _storage_safe_training(
    train_fn: Callable[[dict[str, Any], Path], bool],
    config: dict[str, Any],
    output_dir: Path,
) -> bool:
    algorithm = str(config.get("training", {}).get("algo", "")).lower()
    prior_receipt = output_dir / RETENTION_RECEIPT
    prior_sidecar = output_dir / RETENTION_SIDECAR
    for prior_evidence in (prior_receipt, prior_sidecar):
        if prior_evidence.is_symlink():
            raise ValueError("checkpoint retention evidence must not be a symlink")
        prior_evidence.unlink(missing_ok=True)
    try:
        with _suppress_checkpoint_writes() as callbacks:
            trained = train_fn(config, output_dir)
    except BaseException as training_error:
        # Defensive cleanup handles a partial write from an unexpected training
        # path, while the no-write callback prevents the known failure path.
        try:
            purge_checkpoint_artifacts(output_dir, algorithm)
        except BaseException as cleanup_error:
            if hasattr(training_error, "add_note"):
                training_error.add_note(
                    "checkpoint cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise
    if not trained:
        purge_checkpoint_artifacts(output_dir, algorithm)
        return False
    finalize_checkpoint_free_trial(
        output_dir,
        algorithm,
        callback_instances=len(callbacks),
        callback_on_step_calls=(int(callbacks[0].on_step_calls) if len(callbacks) == 1 else 0),
    )
    return True


def run_screening_task(
    config: dict[str, Any],
    *,
    task_id: int,
    output_dir: Path,
):
    """Run the existing strict screening task with storage-retention hooks.

    The temporary callback and runner hooks are process-global.  SLURM gives
    each array task its own process and this guard rejects accidental concurrent
    use inside one process instead of allowing cross-trial contamination.
    """

    if not _RUNNER_PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError(
            "storage-safe screening supports one sequential task per process"
        )

    original_train = base_runner.train_with_stable_baselines
    original_reuse = base_runner.completed_screening_trial_is_reusable

    def storage_safe_train(trial_config: dict[str, Any], trial_dir: Path) -> bool:
        return _storage_safe_training(original_train, trial_config, trial_dir)

    def storage_safe_reuse(
        trial_dir: Path,
        *,
        trial_config: dict[str, Any],
        identity: dict[str, Any],
        objective_metric: str,
    ) -> bool:
        if not original_reuse(
            trial_dir,
            trial_config=trial_config,
            identity=identity,
            objective_metric=objective_metric,
        ):
            return False
        return not validate_checkpoint_retention(trial_dir, str(identity["algorithm"]))

    base_runner.train_with_stable_baselines = storage_safe_train
    base_runner.completed_screening_trial_is_reusable = storage_safe_reuse
    try:
        return base_runner.run_screening_task(config, task_id=task_id, output_dir=output_dir)
    finally:
        base_runner.train_with_stable_baselines = original_train
        base_runner.completed_screening_trial_is_reusable = original_reuse
        _RUNNER_PATCH_LOCK.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full screening while suppressing redundant checkpoint writes"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-task-grid", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.print_task_grid:
        print(
            json.dumps(
                [task.__dict__ for task in base_runner.screening_task_grid(config)],
                indent=2,
            )
        )
        return
    task = run_screening_task(
        config,
        task_id=args.task_id,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(task.__dict__, indent=2))


if __name__ == "__main__":
    main()
