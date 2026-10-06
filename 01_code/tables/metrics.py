"""Forecast / causal metrics used in Tables 1–4."""

from __future__ import annotations

from typing import Dict

import numpy as np

from dc_ctgcn.metrics import all_metrics, mae, mape, rmse


def pdr(mae_s1: float, mae_s: float) -> float:
    """Performance Degradation Ratio vs Normal (S1)."""
    if mae_s1 == 0:
        return float("nan")
    return 100.0 * (mae_s - mae_s1) / mae_s1


def ted(pred_effect: np.ndarray, true_effect: np.ndarray) -> float:
    """Treatment Effect Deviation: MAE between predicted and true Δ."""
    return mae(true_effect, pred_effect)


def pvr(speed_kmh: np.ndarray, v_free_kmh: float = 80.0, rho=None, rho_jam: float = 0.15) -> float:
    v = np.asarray(speed_kmh)
    bad_v = (v < -1e-6) | (v > v_free_kmh + 0.5)
    if rho is None:
        return float(bad_v.mean() * 100.0)
    bad_rho = (rho < -1e-6) | (rho > rho_jam + 0.02)
    return float((bad_v | bad_rho).mean() * 100.0)


def graph_sparsity(C: np.ndarray, eps: float = 1e-6) -> float:
    C = np.asarray(C)
    return float((np.abs(C) <= eps).mean() * 100.0)


def event_masks(Z_loc: np.ndarray) -> Dict[str, np.ndarray]:
    """S1–S4 masks from scene channels [accident, weather, holiday, weekend]."""
    Z_loc = np.asarray(Z_loc)
    if Z_loc.ndim < 2 or Z_loc.shape[-1] != 4 or not np.isfinite(Z_loc).all():
        raise ValueError('Expected finite scene channels [...,4]')
    acc, weather, holiday, weekend = [Z_loc[..., i] for i in range(4)]
    s2 = acc > 0
    s3 = (weather > 0) & ~s2
    s4 = (holiday > 0) & ~s2 & ~s3
    s1 = ~s2 & ~s3 & ~s4
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4}


__all__ = ["all_metrics", "mae", "rmse", "mape", "pdr", "ted", "pvr", "graph_sparsity", "event_masks"]
