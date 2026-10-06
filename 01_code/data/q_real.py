"""Load real Q-Traffic subgraph (coords + speed + topology), max_nodes capped."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

from dyca_stgnn.synth.q_scenarios import (
    build_topology_from_coords_grid,
    load_q_link_coords_filtered,
    load_q_traffic_speed_pkl,
    make_flow_proxy_from_speed,
)


def load_real_q_subgraph(
    root: str = ".",
    max_nodes: int = 2000,
    seed: int = 42,
    k: int = 10,
    radius: float = 1000.0,
    T: Optional[int] = None,
    sample: str = "degree",
    cache_dir: Optional[str] = None,
    include_flow: bool = True,
) -> Dict:
    """Load a reproducible Q subgraph for figures / training.

    Parameters
    ----------
    sample:
      - "random": uniform subsample of links
      - "degree": build full-radius candidate graph on a larger pool then keep top-degree
    """
    q_dir = os.path.join(root, "Q")
    cache_dir = cache_dir or os.path.join(root, "out", "paper", "q_cache")
    tag = f"n{max_nodes}_s{seed}_{sample}_k{k}_r{int(radius)}"
    cache_path = os.path.join(cache_dir, f"{tag}.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        return {
            "X": z["X"],
            "A": z["A"],
            "coords": z["coords"],
            "node_ids": z["node_ids"].tolist(),
            "edge_index": z["edge_index"],
            "edge_dist": z["edge_dist"],
            "time_feat": z["time_feat"] if "time_feat" in z.files else None,
        }

    coords_path = os.path.join(q_dir, "event_link_set_all_beijing_1km_filtered")
    traffic_path = os.path.join(
        q_dir,
        "event_traffic_beijing_1km_mv_avg_15min_completion",
        "event_traffic_beijing_1km_mv_avg_15min_completion.pkl",
    )
    if sample == "random":
        node_ids, coords = load_q_link_coords_filtered(coords_path, max_nodes=max_nodes, seed=seed)
    else:
        # degree-aware: first take a larger random pool, build graph, keep top-degree nodes
        pool_n = min(8000, max(max_nodes * 4, max_nodes))
        pool_ids, pool_coords = load_q_link_coords_filtered(coords_path, max_nodes=pool_n, seed=seed)
        A_pool, _, _ = build_topology_from_coords_grid(pool_coords, k=k, radius=radius)
        deg = A_pool.sum(axis=0) + A_pool.sum(axis=1)
        keep = np.argsort(-deg)[: int(max_nodes)]
        keep = np.sort(keep)
        node_ids = [pool_ids[i] for i in keep]
        coords = pool_coords[keep]

    A, edge_index, edge_dist = build_topology_from_coords_grid(coords, k=k, radius=radius)
    X_speed = load_q_traffic_speed_pkl(traffic_path, node_ids=node_ids, T=T)
    # fill nan
    X_speed = _fill_nan(X_speed)
    if include_flow:
        X = np.stack([X_speed, make_flow_proxy_from_speed(X_speed)], axis=-1).astype(np.float32)
    else:
        X = X_speed[:, :, None].astype(np.float32)

    time_feat = None
    tf_path = os.path.join(q_dir, "time_feature_15min.pkl")
    if os.path.exists(tf_path):
        import pickle

        with open(tf_path, "rb") as f:
            try:
                tf = pickle.load(f)
            except Exception:
                f.seek(0)
                tf = pickle.load(f, encoding="latin1")
        tf = np.asarray(tf, dtype=np.float32)
        if tf.ndim == 2 and tf.shape[0] >= X.shape[0]:
            time_feat = tf[: X.shape[0]]

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=X,
        A=A.astype(np.float32),
        coords=coords.astype(np.float32),
        node_ids=np.asarray(node_ids, dtype=object),
        edge_index=edge_index,
        edge_dist=edge_dist.astype(np.float32),
        time_feat=time_feat if time_feat is not None else np.zeros((0, 0), dtype=np.float32),
    )
    meta = {"N": int(X.shape[1]), "T": int(X.shape[0]), "E": int(edge_index.shape[1]), "cache": cache_path}
    with open(os.path.join(cache_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "X": X,
        "A": A.astype(np.float32),
        "coords": coords.astype(np.float32),
        "node_ids": node_ids,
        "edge_index": edge_index,
        "edge_dist": edge_dist.astype(np.float32),
        "time_feat": time_feat,
    }


def pick_bottleneck_node(A: np.ndarray, X_speed: np.ndarray) -> int:
    """High degree × high traffic variability as structural bottleneck."""
    deg = A.sum(axis=0) + A.sum(axis=1)
    vol = np.nanstd(X_speed, axis=0)
    score = deg * (vol / (vol.max() + 1e-6))
    return int(np.argmax(score))


def pick_school_node(X_speed: np.ndarray, steps_per_day: int = 96, start_hour: float = 7.0, end_hour: float = 8.5) -> Tuple[int, np.ndarray]:
    """Weekend–weekday morning speed drop (paper Fig.5 protocol)."""
    T, N = X_speed.shape
    n_days = T // steps_per_day
    data = X_speed[: n_days * steps_per_day].reshape(n_days, steps_per_day, N)
    days = np.arange(n_days)
    # Q starts 2017-04-01 Saturday -> weekend = day%7 in {0,1}
    is_weekend = np.isin(days % 7, [0, 1])
    i0, i1 = int(start_hour * 4), int(end_hour * 4)
    wd = data[~is_weekend, i0:i1].mean(axis=(0, 1))
    we = data[is_weekend, i0:i1].mean(axis=(0, 1))
    drop = we - wd
    return int(np.argmax(drop)), drop


def _fill_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    for j in range(x.shape[1]):
        col = x[:, j]
        nans = np.isnan(col)
        if nans.all():
            col[:] = 0.0
        elif nans.any():
            idx = np.where(~nans, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col[nans] = col[idx[nans]]
            nans = np.isnan(col)
            if nans.any():
                idx = np.where(~nans, np.arange(len(col)), len(col) - 1)
                idx = np.minimum.accumulate(idx[::-1])[::-1]
                col[nans] = col[idx[nans]]
        x[:, j] = col
    return x


__all__ = ["load_real_q_subgraph", "pick_bottleneck_node", "pick_school_node"]
