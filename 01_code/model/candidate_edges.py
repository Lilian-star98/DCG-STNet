"""§3.6.2 Physically constrained candidate edge set E_cand."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def build_candidate_edges(
    coords: Optional[np.ndarray] = None,
    A_static: Optional[np.ndarray] = None,
    dist_thresh: float = 2000.0,
    k: int = 12,
    func_sim: Optional[np.ndarray] = None,
    func_thresh: float = 0.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build candidate mask (N, N) and optional pairwise distances.

    A pair (i, j) is a candidate if it is a static edge, or Euclidean
    distance < dist_thresh, or functional similarity > func_thresh.
    Non-candidates are forced to zero dynamic weight (paper Def. 3.2).
    """
    if A_static is not None:
        N = int(A_static.shape[0])
        mask = (np.asarray(A_static) > 0).astype(np.float32)
    elif coords is not None:
        N = int(coords.shape[0])
        mask = np.zeros((N, N), dtype=np.float32)
    else:
        raise ValueError("Need coords or A_static")

    dist = None
    if coords is not None:
        coords = np.asarray(coords, dtype=np.float32)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
        np.fill_diagonal(dist, np.inf)
        spatial = (dist <= float(dist_thresh)).astype(np.float32)
        # kNN within radius
        knn = np.zeros_like(spatial)
        for i in range(N):
            order = np.argsort(dist[i])
            keep = order[: int(k)]
            knn[i, keep] = 1.0
        mask = np.maximum(mask, spatial * knn)
        np.fill_diagonal(dist, 0.0)

    if func_sim is not None:
        mask = np.maximum(mask, (np.asarray(func_sim) >= func_thresh).astype(np.float32))

    np.fill_diagonal(mask, 1.0)
    return mask.astype(np.float32), dist


__all__ = ["build_candidate_edges"]
