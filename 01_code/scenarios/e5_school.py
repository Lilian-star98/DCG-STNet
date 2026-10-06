"""E5: school-induced congestion and targeted guidance. Paper §4.7."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .ctm import default_params, simulate
from .generate import ScenarioSpec, generate_pair
from .intensity import smooth_beta
from .network import PAPER_NODES, build_demo_network


def run_e5(
    A: np.ndarray = None,
    T: int = 96,
    dt: float = 900.0,
    school_node: int = None,
    t_start: int = 28,  # 7:00 at 15-min
    t_peak: int = 32,   # 8:00
    t_end: int = 34,    # 8:30
    surge: float = 8.0,
    intervene: bool = True,
) -> Dict:
    if A is None:
        A, coords, meta = build_demo_network()
    else:
        coords, meta = None, {}
    school_node = PAPER_NODES["school"] if school_node is None else int(school_node)
    spec = ScenarioSpec(
        kind="school",
        nodes=[school_node],
        t_start=t_start,
        t_peak=t_peak,
        t_end=t_end,
        beta=0.7,
        extra={"inflow_surge": surge},
    )
    t = np.arange(T)
    morning = np.exp(-0.5 * ((t / max(T, 1) - 0.32) / 0.06) ** 2)
    diurnal = 0.4 + 0.6 * morning
    demand = np.zeros((T, A.shape[0]), dtype=np.float32)
    sources = A.sum(axis=0) == 0
    demand[:, sources] = 70.0 + 120.0 * diurnal[:, None]
    demand[:, school_node] += 500.0 * diurnal  # baseline local activity

    # Stronger local capacity drop during school peak (curbside stop / drop-off)
    weekday = generate_pair(A, spec, T=T, dt=dt, demand_base=demand)

    # weekend baseline: same demand without school surge
    spec_off = ScenarioSpec(kind="school", nodes=[school_node], t_start=t_start, t_peak=t_peak, t_end=t_end, beta=0.0, extra={"inflow_surge": 1.0})
    weekend = generate_pair(A, spec_off, T=T, dt=dt, demand_base=demand)

    guided = None
    if intervene:
        # personnel guidance: cut peak inflow and restore local capacity
        spec_g = ScenarioSpec(
            kind="school",
            nodes=[school_node],
            t_start=t_start,
            t_peak=t_peak,
            t_end=t_end,
            beta=0.15,
            extra={"inflow_surge": 1.15},
        )
        guided = generate_pair(A, spec_g, T=T, dt=dt, demand_base=demand)

    # generate_pair: speed=baseline, speed_cf=intervened world.
    # Weekday school congestion lives in speed_cf; weekend uses the non-surge baseline.
    weekday_view = {
        "speed": weekday["speed_cf"],
        "flow": weekday["flow_cf"],
        "X": weekday["X_cf"],
        "speed_base": weekday["speed"],
        "flow_base": weekday["flow"],
    }
    weekend_view = {
        "speed": weekend["speed"],
        "flow": weekend["flow"],
        "X": weekend["X"],
    }
    guided_view = None
    if guided is not None:
        guided_view = {
            "speed": guided["speed_cf"],
            "flow": guided["flow_cf"],
            "X": guided["X_cf"],
        }

    return {
        "A": A,
        "coords": coords,
        "meta": meta,
        "school_node": school_node,
        "neighbors": PAPER_NODES["nb"],
        "weekday": weekday_view,
        "weekend": weekend_view,
        "guided": guided_view,
        "t_start": t_start,
        "t_end": t_end,
    }


__all__ = ["run_e5"]
