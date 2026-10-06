"""E3: local structural bottleneck (lane narrowing / capacity drop). Paper §4.6.1."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .generate import ScenarioSpec, generate_pair
from .network import PAPER_NODES, build_demo_network


def run_e3(
    A: np.ndarray = None,
    T: int = 96,
    dt: float = 900.0,
    target: Sequence[int] = None,
    t_start: int = 48,
    t_peak: int = 56,
    t_end: int = 72,
    beta: float = 0.75,
) -> Dict:
    if A is None:
        A, coords, meta = build_demo_network()
    else:
        coords, meta = None, {}
    target = list(target) if target is not None else [PAPER_NODES["bottleneck"]]
    spec = ScenarioSpec(
        kind="capacity_drop",
        nodes=target,
        t_start=t_start,
        t_peak=t_peak,
        t_end=t_end,
        beta=beta,
    )
    out = generate_pair(A, spec, T=T, dt=dt)
    out.update({"A": A, "coords": coords, "meta": meta, "spec": spec, "target": target})
    return out


__all__ = ["run_e3"]
