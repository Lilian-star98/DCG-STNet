"""Dataset loaders (paper §4.1). Delegates to dc_ctgcn.data_utils when present."""

from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

import numpy as np

from dc_ctgcn.data_utils import load_dataset as _load_dataset
from dc_ctgcn.data_utils import load_metr_la, load_pems04, load_pems_bay, load_q_beijing


def load_dataset(name: str, root: str = ".") -> np.ndarray:
    """Return X of shape (T, N, F)."""
    return _load_dataset(name, root)


def load_adjacency(name: str, root: str, n_nodes: Optional[int] = None) -> np.ndarray:
    name = name.lower()
    mapping = {
        "metr-la": os.path.join(root, "raw_data", "METR-LA", "adj_METR-LA.pkl"),
        "pems-bay": os.path.join(root, "raw_data", "PEMS-BAY", "adj_PEMS-BAY.pkl"),
        "pems04": os.path.join(root, "raw_data", "PEMS04", "adj_PEMS04.pkl"),
    }
    if name in mapping and os.path.exists(mapping[name]):
        with open(mapping[name], "rb") as f:
            try:
                adj = pickle.load(f)
            except Exception:
                f.seek(0)
                adj = pickle.load(f, encoding="latin1")
        arr = _extract_square(adj)
        if arr is not None:
            return arr.astype(np.float32)
    if n_nodes is None:
        raise FileNotFoundError(f"No adjacency for {name}")
    return np.eye(n_nodes, dtype=np.float32)


def _extract_square(adj) -> Optional[np.ndarray]:
    if isinstance(adj, (list, tuple)):
        for a in adj:
            a = np.asarray(a)
            if a.ndim == 2 and a.shape[0] == a.shape[1]:
                return a
    if isinstance(adj, dict):
        for k in ("adj", "distance", "weights", "data"):
            if k in adj:
                a = np.asarray(adj[k])
                if a.ndim == 2:
                    return a
    a = np.asarray(adj)
    return a if a.ndim == 2 else None


def load_q_coords(root: str, max_nodes: Optional[int] = None, seed: int = 0) -> Tuple[list, np.ndarray]:
    path = os.path.join(root, "Q", "event_link_set_all_beijing_1km_filtered")
    if not os.path.exists(path):
        # fallback: any similarly named file
        qdir = os.path.join(root, "Q")
        cands = [os.path.join(qdir, n) for n in os.listdir(qdir)] if os.path.isdir(qdir) else []
        path = next((p for p in cands if "filtered" in os.path.basename(p) and os.path.isfile(p)), path)
    node_ids, coords = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            node_ids.append(str(parts[0]))
            coords.append((float(parts[1]), float(parts[2])))
    if max_nodes and len(node_ids) > max_nodes:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(node_ids), size=max_nodes, replace=False))
        node_ids = [node_ids[i] for i in idx]
        coords = [coords[i] for i in idx]
    return node_ids, np.asarray(coords, dtype=np.float32)


__all__ = ["load_dataset", "load_adjacency", "load_q_coords", "load_metr_la", "load_pems_bay", "load_pems04", "load_q_beijing"]
