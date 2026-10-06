"""Theorem 3.4 hierarchical graph coarsening (region-level learning)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def coarsen_regions(A: np.ndarray, n_regions: int, seed: int = 0) -> Dict[str, np.ndarray]:
    """Partition nodes by a cheap spectral / degree heuristic.

    Returns region ids (N,), region adjacency (R, R), and assignment matrix.
    Complexity intent: O(N^2) -> O(N n_r + n_r^2) at learning time.
    """
    A = np.asarray(A, dtype=np.float32)
    N = A.shape[0]
    n_regions = max(1, min(int(n_regions), N))
    deg = A.sum(axis=1) + A.sum(axis=0)
    # greedy seed by degree, assign remaining by adjacency affinity
    rng = np.random.default_rng(seed)
    order = np.argsort(-deg)
    seeds = order[:n_regions].copy()
    if seeds.size < n_regions:
        extra = rng.choice(N, size=n_regions - seeds.size, replace=False)
        seeds = np.concatenate([seeds, extra])
    labels = np.full(N, -1, dtype=np.int64)
    labels[seeds[:n_regions]] = np.arange(n_regions)
    for i in order:
        if labels[i] >= 0:
            continue
        aff = A[i] + A[:, i]
        scores = np.zeros(n_regions, dtype=np.float32)
        for r in range(n_regions):
            members = np.where(labels == r)[0]
            if members.size:
                scores[r] = aff[members].mean()
        labels[i] = int(np.argmax(scores)) if scores.max() > 0 else int(i % n_regions)

    R = n_regions
    A_r = np.zeros((R, R), dtype=np.float32)
    for i in range(N):
        for j in np.where(A[i] > 0)[0]:
            A_r[labels[i], labels[j]] += A[i, j]
    np.fill_diagonal(A_r, 0.0)
    assign = np.zeros((N, R), dtype=np.float32)
    assign[np.arange(N), labels] = 1.0
    return {"labels": labels, "A_region": A_r, "assign": assign}


def hierarchical_complexity(N: int, n_regions: int) -> Dict[str, float]:
    """Paper Thm 3.4 complexity comparison."""
    n_r = max(1, int(n_regions))
    region_size = N / n_r
    naive = float(N) ** 2
    hier = n_r * (region_size ** 2) + float(n_r) ** 2
    return {"naive_O_N2": naive, "hier_O": hier, "ratio": hier / max(naive, 1.0)}


__all__ = ["coarsen_regions", "hierarchical_complexity"]
