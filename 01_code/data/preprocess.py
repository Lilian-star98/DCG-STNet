"""§4.2 preprocessing: missing fill, 6:2:2 split, Z-score, sliding windows."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def time_split(X: np.ndarray, ratios=(0.6, 0.2, 0.2)) -> Dict[str, np.ndarray]:
    """Chronological split into train/val/test (ratios sum to 1)."""
    T = X.shape[0]
    r1, r2, r3 = ratios
    assert abs(r1 + r2 + r3 - 1.0) < 1e-6
    n1 = int(T * r1)
    n2 = int(T * r2)
    return {"train": X[:n1], "val": X[n1 : n1 + n2], "test": X[n1 + n2 :]}


def scale_zscore(train: np.ndarray, x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-node Z-score using train statistics only (paper §4.2.1)."""
    mean = np.nanmean(train, axis=0, keepdims=True)
    std = np.nanstd(train, axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std, mean, std


def inv_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def make_windows(X: np.ndarray, hist: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """X_in: (S, hist, N, F), Y: (S, horizon, N, F)."""
    T = X.shape[0]
    xs, ys = [], []
    for t in range(hist, T - horizon + 1):
        xs.append(X[t - hist : t])
        ys.append(X[t : t + horizon])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


__all__ = ["time_split", "scale_zscore", "inv_zscore", "make_windows"]
