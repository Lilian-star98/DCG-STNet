"""Evaluation metrics for traffic forecasting."""

from __future__ import annotations

import numpy as np


def _flatten(y_true: np.ndarray, y_pred: np.ndarray):
    if np.shape(y_true) != np.shape(y_pred):
        raise ValueError('Prediction and target shapes differ')
    a = y_true.reshape(-1)
    b = y_pred.reshape(-1)
    if not np.isfinite(b[np.isfinite(a)]).all():
        raise ValueError('Nonfinite predictions on valid targets')
    mask = np.isfinite(a)
    if mask.sum() == 0:
        return np.array([np.nan]), np.array([np.nan])
    return a[mask], b[mask][: mask.sum()]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _flatten(y_true, y_pred)
    return float(np.nanmean(np.abs(a - b)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _flatten(y_true, y_pred)
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """MAPE fraction; manuscript §4.3.2 uses denominator floor epsilon=1."""
    a, b = _flatten(y_true, y_pred)
    denom = np.maximum(np.abs(a), eps)
    return float(np.nanmean(np.abs((a - b) / denom)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _flatten(y_true, y_pred)
    var = np.nanvar(a)
    if var < 1e-12:
        return float("nan")
    return float(1.0 - np.nanmean((a - b) ** 2) / var)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
