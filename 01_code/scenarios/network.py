"""Synthetic Chaoyang-like graph used for paper figures (nodes 14/52/83/87/100/...)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


PAPER_NODES = {
    "bottleneck": 100,
    "up1": 83,
    "up2": 87,
    "down": 14,
    "school": 52,
    "nb": [64, 150, 202, 216, 165],
}


def build_demo_network(n: int = 220, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Directed geometric graph with named hubs matching paper Fig.1–8."""
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2), dtype=np.float32) * 4000.0
    # pin special nodes into a corridor
    coords[100] = np.array([2000.0, 2000.0], dtype=np.float32)
    coords[83] = np.array([1600.0, 2200.0], dtype=np.float32)
    coords[87] = np.array([1550.0, 1750.0], dtype=np.float32)
    coords[14] = np.array([2500.0, 2000.0], dtype=np.float32)
    coords[52] = np.array([2100.0, 2600.0], dtype=np.float32)
    for i, nid in enumerate([64, 150, 202, 216, 165]):
        ang = 2 * np.pi * i / 5.0
        coords[nid] = coords[52] + np.array([350.0 * np.cos(ang), 350.0 * np.sin(ang)], dtype=np.float32)

    A = np.zeros((n, n), dtype=np.float32)
    # Sparse directed kNN (out-degree 3) so CTM has exit mass
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1) + 1e-8)
    np.fill_diagonal(d, np.inf)
    k = 3
    for i in range(n):
        nn = np.argsort(d[i])[:k]
        A[i, nn] = 1.0
    # force paper topology: 83,87 -> 100 -> 14 (14 is a sink-ish corridor end)
    A[83, :] = 0.0
    A[87, :] = 0.0
    A[100, :] = 0.0
    A[14, :] = 0.0
    A[83, 100] = 1.0
    A[87, 100] = 1.0
    A[100, 14] = 1.0
    A[52, 100] = 1.0
    for nid in [64, 150, 202, 216, 165]:
        A[52, nid] = 1.0
        A[nid, 52] = 0.5
        # neighbors can exit
        A[nid, nid] = 0.0
    # Explicit sinks near boundary
    for s in (14, 3, 7, 11, 19, 23):
        A[s, :] = 0.0
    np.fill_diagonal(A, 0.0)
    meta = {"coords": coords, "nodes": PAPER_NODES, "dist": d}
    return A, coords, meta


__all__ = ["PAPER_NODES", "build_demo_network"]
