"""Lightweight baseline forecasters for quick benchmarking.

All baselines operate on sliding windows of shape (hist, N, F) and predict h steps.
"""

from __future__ import annotations

import numpy as np


def persistence(history: np.ndarray, horizon: int) -> np.ndarray:
    """Predict each future step as the last observed value."""
    last = history[-1:]
    return np.repeat(last, horizon, axis=0)


def moving_average(history: np.ndarray, horizon: int, window: int = 3) -> np.ndarray:
    """Predict future as the mean of the last `window` steps."""
    w = min(window, history.shape[0])
    avg = np.mean(history[-w:], axis=0, keepdims=True)
    return np.repeat(avg, horizon, axis=0)


def linear_trend(history: np.ndarray, horizon: int) -> np.ndarray:
    """Simple linear extrapolation using the last two points."""
    if history.shape[0] < 2:
        return persistence(history, horizon)
    y1 = history[-2]
    y2 = history[-1]
    delta = y2 - y1
    preds = []
    for h in range(1, horizon + 1):
        preds.append(y2 + h * delta)
    return np.stack(preds, axis=0)


BASELINES = {
    "persistence": persistence,
    "movavg": moving_average,
    "trend": linear_trend,
}
