"""Scene / time features (paper §4.2.3)."""

from __future__ import annotations

from typing import Optional

import numpy as np


def build_time_features(T: int, steps_per_day: int = 288, start_dow: int = 0) -> np.ndarray:
    """Sin/cos hour-of-day and day-of-week + holiday placeholder.

    Returns (T, 5): [sin_h, cos_h, sin_d, cos_d, is_weekend].
    METR-LA / PEMS-BAY: steps_per_day=288 (5 min).
    Q-Traffic: steps_per_day=96 (15 min).
    """
    t = np.arange(T)
    tod = (t % steps_per_day) / float(steps_per_day)
    dow = ((t // steps_per_day) + start_dow) % 7
    feat = np.stack(
        [
            np.sin(2 * np.pi * tod),
            np.cos(2 * np.pi * tod),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            (np.isin(dow, [5, 6])).astype(np.float32),
        ],
        axis=1,
    )
    return feat.astype(np.float32)


def build_q_scene_features(
    T: int,
    N: int,
    steps_per_day: int = 96,
    accident_mask: Optional[np.ndarray] = None,
    weather_mask: Optional[np.ndarray] = None,
    holiday_mask: Optional[np.ndarray] = None,
    start_dow: int = 6,
) -> dict:
    """Q-Traffic node-level event labels + global demand proxy.

    April 1 2017 is Saturday -> start_dow=6 if using Python weekday, but
    analyze_school_congestion treats day0 as Saturday, i.e. weekend = {0,1}.
    """
    Zg = build_time_features(T, steps_per_day=steps_per_day, start_dow=0)
    Zloc = np.zeros((T, N, 4), dtype=np.float32)
    if accident_mask is not None:
        Zloc[:, :, 0] = accident_mask
    if weather_mask is not None:
        if weather_mask.ndim == 1:
            Zloc[:, :, 1] = weather_mask[:, None]
        else:
            Zloc[:, :, 1] = weather_mask
    if holiday_mask is not None:
        if holiday_mask.ndim == 1:
            Zloc[:, :, 2] = holiday_mask[:, None]
        else:
            Zloc[:, :, 2] = holiday_mask
    days = np.arange(T) // steps_per_day
    is_weekend = np.isin(days % 7, [0, 1]).astype(np.float32)
    Zloc[:, :, 3] = is_weekend[:, None]
    return {"Z_glob": Zg, "Z_loc": Zloc, "is_weekend": is_weekend}


__all__ = ["build_time_features", "build_q_scene_features"]
