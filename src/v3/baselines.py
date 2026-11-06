"""Mask-aware baseline actions for the isolated V3 development environment."""

from __future__ import annotations

import numpy as np


def active_equal_weight_action(action_mask: np.ndarray) -> np.ndarray:
    """Allocate equally across currently eligible assets, or remain in cash."""

    mask = np.asarray(action_mask, dtype=bool)
    action = np.zeros(mask.shape, dtype=np.float64)
    count = int(mask.sum())
    if count:
        action[mask] = 1.0 / count
    return action


def mask_full_investment_action(
    action: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    """Zero inactive slots and renormalize a full-investment baseline."""

    values = np.nan_to_num(
        np.asarray(action, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    mask = np.asarray(action_mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("action and action mask must have the same shape")
    values = np.clip(values, 0.0, None)
    values[~mask] = 0.0
    total = float(values.sum())
    if total > 0.0:
        values /= total
    return values

