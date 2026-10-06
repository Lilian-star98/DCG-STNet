"""Smooth intervention intensity β(t) (Appendix B 1.6)."""

from __future__ import annotations

import numpy as np


def smooth_beta(T: int, t_start: int, t_peak: int, t_end: int) -> np.ndarray:
    """Rise, hold until t_end, then recover over t_peak-t_start steps.

    The manuscript's printed cosine denominators are ambiguous. Both ramps
    use their positive duration, so intensity remains continuous at joins.
    """
    if any(int(v) != v for v in (T, t_start, t_peak, t_end)) or not 0 <= t_start < t_peak <= t_end < T:
        raise ValueError('Require integer 0 <= t_start < t_peak <= t_end < T')
    t = np.arange(T, dtype=np.float64)
    b = np.zeros(T, dtype=np.float64)
    t_start, t_peak, t_end = int(t_start), int(t_peak), int(t_end)
    t_peak = min(max(t_peak, t_start), t_end)
    rise = t_peak - t_start
    fall = rise
    if rise > 0:
        m = (t >= t_start) & (t <= t_peak)
        u = (t[m] - t_start) / rise
        b[m] = 0.5 * (1.0 - np.cos(np.pi * u))
    b[(t >= t_peak) & (t < t_end)] = 1.0
    if fall > 0:
        m = (t >= t_end) & (t <= t_end + fall)
        u = (t[m] - t_end) / fall
        b[m] = 0.5 * (1.0 + np.cos(np.pi * u))
    b[t >= t_end + fall] = 0.0
    b[t < t_start] = 0.0
    return np.clip(b, 0.0, 1.0)


__all__ = ["smooth_beta"]
