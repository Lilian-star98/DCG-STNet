"""E4: systemic weather disturbance (rainfall). Paper §4.6.2."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .generate import ScenarioSpec, generate_pair
from .network import build_demo_network


def run_e4(
    A: np.ndarray = None,
    T: int = 96,
    dt: float = 900.0,
    t_start: int = 68,  # ~17:00 if day starts 00:00 at 15-min
    t_peak: int = 72,
    t_end: int = 84,
    beta: float = 0.55,
    delta_v: float = 0.28,
    delta_c: float = 0.25,
) -> Dict:
    if A is None:
        A, coords, meta = build_demo_network()
    else:
        coords, meta = None, {}
    N = A.shape[0]
    spec = ScenarioSpec(
        kind="weather",
        nodes=list(range(N)),
        t_start=t_start,
        t_peak=t_peak,
        t_end=t_end,
        beta=beta,
        region_ratio=1.0,
        extra={"delta_v": delta_v, "delta_c": delta_c},
    )
    # Mild demand so the network stays near free-flow before rainfall (paper Fig.3 ~60–70 km/h)
    t = np.arange(T)
    evening = np.exp(-0.5 * ((t / max(T, 1) - 0.72) / 0.12) ** 2)
    diurnal = 0.45 + 0.55 * evening
    demand = np.zeros((T, A.shape[0]), dtype=np.float32)
    sources = A.sum(axis=0) == 0
    demand[:, sources] = 90.0 + 160.0 * diurnal[:, None]
    out = generate_pair(A, spec, T=T, dt=dt, demand_base=demand)
    out.update({"A": A, "coords": coords, "meta": meta, "spec": spec})
    return out


__all__ = ["run_e4"]
