"""Data loading and time-split utilities for dc_ctgcn experiments.

Supports METR-LA, PEMS-BAY, PEMS04, and Q datasets (Beijing 15-min).
Time split follows 6:2:2 (chronological) to avoid leakage.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple

import numpy as np


def _fill_missing(x: np.ndarray) -> np.ndarray:
    """Forward/backward fill NaNs along time for stability."""
    x = np.asarray(x, float)
    # forward fill
    mask = np.isnan(x)
    idx = np.where(~mask, np.arange(x.shape[0])[:, None], 0)
    idx = np.maximum.accumulate(idx, axis=0)
    out = x[idx, np.arange(x.shape[1])[None, :]]
    # backward fill to handle leading NaNs
    mask2 = np.isnan(out)
    idx2 = np.where(~mask2, np.arange(out.shape[0])[:, None], out.shape[0] - 1)
    idx2 = np.minimum.accumulate(idx2[::-1], axis=0)[::-1]
    out = out[idx2, np.arange(out.shape[1])[None, :]]
    return out


def load_metr_la(root: str) -> np.ndarray:
    import h5py
    path = os.path.join(root, "raw_data", "METR-LA", "METR-LA.h5")
    with h5py.File(path, "r") as f:
        if "speed" in f:
            data = f["speed"][:]
        elif "df" in f and "block0_values" in f["df"]:
            data = f["df"]["block0_values"][:]
        else:
            key0 = list(f.keys())[0]
            g = f[key0]
            if hasattr(g, "keys"):
                inner = list(g.keys())[0]
                data = g[inner][:]
            else:
                data = g[:]
    # ensure shape (T, N, F)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data, (1, 0, 2))
    return data.astype(np.float32)


def load_pems_bay(root: str) -> np.ndarray:
    import h5py
    path = os.path.join(root, "raw_data", "PEMS-BAY", "PEMS-BAY.h5")
    with h5py.File(path, "r") as f:
        if "speed" in f:
            obj = f["speed"]
            # Some releases store a pandas HDFStore group at /speed with block0_values
            if hasattr(obj, "keys"):
                if "block0_values" in obj:
                    data = obj["block0_values"][:]
                else:
                    # fallback: take the first dataset inside the group
                    ds_key = next((k for k in obj.keys() if isinstance(obj[k], h5py.Dataset)), None)
                    if ds_key is None:
                        raise ValueError(f"No dataset found under /speed in {path}")
                    data = obj[ds_key][:]
            else:
                data = obj[:]
        elif "df" in f and hasattr(f["df"], "keys") and "block0_values" in f["df"]:
            data = f["df"]["block0_values"][:]
        else:
            key0 = list(f.keys())[0]
            g = f[key0]
            if hasattr(g, "keys"):
                inner = list(g.keys())[0]
                data = g[inner][:]
            else:
                data = g[:]
    if data.ndim == 2:
        data = data[:, :, None]
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data, (1, 0, 2))
    return data.astype(np.float32)


def load_pems04(root: str) -> np.ndarray:
    path = os.path.join(root, "raw_data", "PEMS04", "PEMS04.npz")
    arr = np.load(path)
    data = arr["data"]
    if data.ndim == 2:
        data = data[:, :, None]
    # Common layouts:
    # - (T, N, C) in many STGNN preprocessing pipelines
    # - (N, T, C) in some legacy releases
    # Heuristic: time dimension is typically much larger than number of sensors.
    if data.ndim == 3 and data.shape[0] < data.shape[1]:
        data = np.transpose(data, (1, 0, 2))  # (T, N, C)
    return data.astype(np.float32)


def load_q_beijing(root: str) -> np.ndarray:
    pkl_path = os.path.join(
        root,
        "Q",
        "event_traffic_beijing_1km_mv_avg_15min_completion",
        "event_traffic_beijing_1km_mv_avg_15min_completion.pkl",
    )
    with open(pkl_path, "rb") as f:
        traffic = pickle.load(f, encoding="latin1")
    node_ids = sorted(traffic.keys())
    series_list = [np.asarray(traffic[nid], float) for nid in node_ids]
    # align lengths
    T = min(len(s) for s in series_list)
    data = np.stack([s[:T] for s in series_list], axis=1)  # (T, N)
    data = _fill_missing(data)
    data = data[:, :, None]  # add feature dim
    return data.astype(np.float32)


def load_dataset(name: str, root: str) -> np.ndarray:
    name = name.lower()
    if name == "metr-la":
        return load_metr_la(root)
    if name == "pems-bay":
        return load_pems_bay(root)
    if name == "pems04":
        return load_pems04(root)
    if name in {"q", "beijing-q", "beijing"}:
        return load_q_beijing(root)
    raise ValueError(f"Unknown dataset: {name}")


def time_split(X: np.ndarray, ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2)) -> Dict[str, np.ndarray]:
    """Chronological split into train/val/test (ratios sum to 1)."""
    T = X.shape[0]
    r1, r2, r3 = ratios
    assert abs(r1 + r2 + r3 - 1.0) < 1e-6
    n1 = int(T * r1)
    n2 = int(T * r2)
    train = X[:n1]
    val = X[n1 : n1 + n2]
    test = X[n1 + n2 :]
    return {"train": train, "val": val, "test": test}


def scale_minmax(train: np.ndarray, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Min-max normalize using train statistics; avoid division by zero."""
    x_min = np.nanmin(train, axis=0, keepdims=True)
    x_max = np.nanmax(train, axis=0, keepdims=True)
    denom = np.maximum(x_max - x_min, eps)
    return (x - x_min) / denom
