from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def sign_flip_p_value(values: Iterable[float], *, resamples: int = 50_000, seed: int = 42) -> float:
    """Two-sided paired randomization p-value for a zero-mean null hypothesis."""
    clean = np.asarray(list(values), dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return float("nan")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    observed = abs(float(clean.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < resamples:
        count = min(2_000, resamples - completed)
        signs = rng.choice((-1.0, 1.0), size=(count, len(clean)))
        randomized = np.abs((signs * clean).mean(axis=1))
        extreme += int(np.count_nonzero(randomized >= observed))
        completed += count
    return float((extreme + 1) / (resamples + 1))


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down family-wise error correction preserving input order."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if not len(valid_indices):
        return adjusted
    order = valid_indices[np.argsort(values[valid_indices])]
    running_max = 0.0
    family_size = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * float(values[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted
