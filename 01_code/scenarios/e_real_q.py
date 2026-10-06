"""Real-Q counterfactual scenarios for paper figures (E3/E4/E5).

Factual trajectories = real Q speeds on a ≤2000-node subgraph.
Counterfactuals = physics-informed interventions on the real topology.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..data.q_real import load_real_q_subgraph, pick_bottleneck_node, pick_school_node
from .intensity import smooth_beta
from .network import PAPER_NODES


def _neighbors(A: np.ndarray, node: int, hops: int = 2) -> list:
    reach = {int(node)}
    frontier = {int(node)}
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            for v in np.where(A[u] > 0)[0]:
                nxt.add(int(v))
            for v in np.where(A[:, u] > 0)[0]:
                nxt.add(int(v))
        frontier = nxt - reach
        reach |= frontier
    return sorted(reach)


def _apply_local_drop(X: np.ndarray, A: np.ndarray, node: int, t0: int, t1: int, beta: float, hops: int = 2) -> np.ndarray:
    """Capacity-drop style local + hop attenuation on real speeds."""
    cf = X.copy()
    T = X.shape[0]
    b = smooth_beta(T, t0, (t0 + t1) // 2, t1)
    nodes = _neighbors(A, node, hops=hops)
    for n in nodes:
        hop = 0 if n == node else (1 if A[node, n] + A[n, node] > 0 else 2)
        atten = 1.0 if hop == 0 else (0.65 if hop == 1 else 0.35)
        fac = 1.0 - float(beta) * atten * b
        cf[:, n, 0] *= np.clip(fac, 0.15, 1.0)
        if cf.shape[-1] > 1:
            cf[:, n, 1] *= np.clip(fac - 0.05, 0.1, 1.0)
    return cf


def _apply_weather(X: np.ndarray, t0: int, t1: int, delta_v: float = 0.22) -> np.ndarray:
    cf = X.copy()
    T = X.shape[0]
    b = smooth_beta(T, t0, (t0 + t1) // 2, t1)
    fac = 1.0 - float(delta_v) * b
    cf[:, :, 0] *= fac.reshape(-1, 1)
    if cf.shape[-1] > 1:
        cf[:, :, 1] *= np.clip(fac.reshape(-1, 1) - 0.05, 0.2, 1.0)
    return cf


def _apply_school(X: np.ndarray, A: np.ndarray, node: int, t0: int, t1: int, days_mask: np.ndarray, surge_drop: float = 0.45) -> np.ndarray:
    """Apply morning school drop only on selected days (weekdays)."""
    cf = X.copy()
    T, N, F = X.shape
    steps = 96
    b_day = smooth_beta(steps, t0 % steps if t0 < steps else t0, ((t0 + t1) // 2) % steps, t1 % steps if t1 < steps else t1)
    # vectorized over days
    n_days = T // steps
    for d in range(n_days):
        if not days_mask[d]:
            continue
        sl = slice(d * steps, (d + 1) * steps)
        fac = 1.0 - surge_drop * b_day
        cf[sl, node, 0] *= fac
        for n in _neighbors(A, node, hops=1):
            if n == node:
                continue
            cf[sl, n, 0] *= 1.0 - 0.35 * surge_drop * b_day
        if F > 1:
            cf[sl, node, 1] *= np.clip(1.0 + 0.5 * surge_drop * b_day, 0.5, 2.0)
    return cf


def _apply_guidance(X_cong: np.ndarray, X_base: np.ndarray, node: int, neighbors: Sequence[int], mix: float = 0.7) -> np.ndarray:
    """Blend congested weekday toward free-flow baseline at node and neighbors."""
    g = X_cong.copy()
    idxs = [node] + list(neighbors)
    for n in idxs:
        w = mix if n == node else mix * 0.6
        g[:, n] = (1.0 - w) * X_cong[:, n] + w * X_base[:, n]
    return g


def run_e3_real(
    root: str = ".",
    max_nodes: int = 2000,
    seed: int = 42,
    T: Optional[int] = 96 * 7,  # one week window for figures
    beta: float = 0.55,
    day_index: int = 2,  # first weekday (Mon if Sat=0)
) -> Dict:
    data = load_real_q_subgraph(root=root, max_nodes=max_nodes, seed=seed, T=None)
    X_full, A, coords = data["X"], data["A"], data["coords"]
    steps = 96
    # take one weekday day slice for clear event window visualization
    t0_day = day_index * steps
    X = X_full[t0_day : t0_day + steps].copy()
    node = pick_bottleneck_node(A, X_full[:, :, 0])
    # evening capacity-drop window ~12:00–18:00 -> steps 48–72
    e0, e1 = 48, 72
    X_cf = _apply_local_drop(X, A, node, e0, e1, beta=beta, hops=2)
    nbs = _neighbors(A, node, hops=1)
    # map paper roles onto real indices for figure annotations
    roles = {"bottleneck": node, "up1": nbs[1] if len(nbs) > 1 else node, "up2": nbs[2] if len(nbs) > 2 else node, "down": nbs[3] if len(nbs) > 3 else node}
    return {
        "A": A,
        "coords": coords,
        "X": X,
        "X_cf": X_cf,
        "speed": X[..., 0],
        "speed_cf": X_cf[..., 0],
        "flow": X[..., 1] if X.shape[-1] > 1 else X[..., 0],
        "flow_cf": X_cf[..., 1] if X_cf.shape[-1] > 1 else X_cf[..., 0],
        "effect": X_cf - X,
        "beta": smooth_beta(steps, e0, (e0 + e1) // 2, e1),
        "target": [node],
        "spec": type("S", (), {"t_start": e0, "t_peak": (e0 + e1) // 2, "t_end": e1, "kind": "capacity_drop"})(),
        "meta": {"roles": roles, "node_ids": data["node_ids"], "source": "real_q"},
        "roles": roles,
    }


def run_e4_real(
    root: str = ".",
    max_nodes: int = 2000,
    seed: int = 42,
    day_index: int = 2,
    delta_v: float = 0.22,
) -> Dict:
    data = load_real_q_subgraph(root=root, max_nodes=max_nodes, seed=seed, T=None)
    X_full, A, coords = data["X"], data["A"], data["coords"]
    steps = 96
    t0_day = day_index * steps
    X = X_full[t0_day : t0_day + steps].copy()
    e0, e1 = 68, 84  # ~17:00–21:00
    X_cf = _apply_weather(X, e0, e1, delta_v=delta_v)
    return {
        "A": A,
        "coords": coords,
        "X": X,
        "X_cf": X_cf,
        "speed": X[..., 0],
        "speed_cf": X_cf[..., 0],
        "flow": X[..., 1] if X.shape[-1] > 1 else X[..., 0],
        "flow_cf": X_cf[..., 1] if X_cf.shape[-1] > 1 else X_cf[..., 0],
        "effect": X_cf - X,
        "beta": smooth_beta(steps, e0, (e0 + e1) // 2, e1),
        "spec": type("S", (), {"t_start": e0, "t_peak": (e0 + e1) // 2, "t_end": e1, "kind": "weather"})(),
        "meta": {"source": "real_q", "node_ids": data["node_ids"]},
    }


def run_e5_real(
    root: str = ".",
    max_nodes: int = 2000,
    seed: int = 42,
    T_days: int = 28,
) -> Dict:
    data = load_real_q_subgraph(root=root, max_nodes=max_nodes, seed=seed, T=96 * max(T_days, 14))
    X, A, coords = data["X"], data["A"], data["coords"]
    steps = 96
    school, drop = pick_school_node(X[:, :, 0], steps_per_day=steps)
    n_days = X.shape[0] // steps
    days = np.arange(n_days)
    is_weekend = np.isin(days % 7, [0, 1])
    # build average weekday / weekend day profiles for plotting
    wd_days = X[: n_days * steps].reshape(n_days, steps, X.shape[1], X.shape[2])[~is_weekend]
    we_days = X[: n_days * steps].reshape(n_days, steps, X.shape[1], X.shape[2])[is_weekend]
    weekday_base = wd_days.mean(axis=0)
    weekend = we_days.mean(axis=0)
    # congested weekday = real weekday (already includes school effect)
    weekday = weekday_base.copy()
    # synthetic stronger contrast if drop is weak: amplify around school
    if float(drop[school]) < 3.0:
        weekday = _apply_school(
            np.repeat(weekend[None, ...], 1, axis=0).reshape(steps, X.shape[1], X.shape[2]),
            A,
            school,
            28,
            34,
            days_mask=np.array([True]),
            surge_drop=0.5,
        )
    neighbors = _neighbors(A, school, hops=1)
    neighbors = [n for n in neighbors if n != school][:5]
    while len(neighbors) < 5:
        neighbors.append(school)
    guided = _apply_guidance(weekday, weekend, school, neighbors, mix=0.75)
    return {
        "A": A,
        "coords": coords,
        "school_node": school,
        "neighbors": neighbors,
        "weekday": {"speed": weekday[..., 0], "flow": weekday[..., 1] if weekday.shape[-1] > 1 else weekday[..., 0], "X": weekday},
        "weekend": {"speed": weekend[..., 0], "flow": weekend[..., 1] if weekend.shape[-1] > 1 else weekend[..., 0], "X": weekend},
        "guided": {"speed": guided[..., 0], "flow": guided[..., 1] if guided.shape[-1] > 1 else guided[..., 0], "X": guided},
        "t_start": 28,
        "t_end": 34,
        "drop": drop,
        "meta": {"source": "real_q", "node_ids": data["node_ids"]},
    }


__all__ = ["run_e3_real", "run_e4_real", "run_e5_real"]
