"""Fig.8 Sub-network topology before/after localized guidance."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .style import draw_graph, save, setup
from ..scenarios.e5_school import run_e5


def render(out_path: str, data=None) -> str:
    setup()
    data = data or run_e5()
    A, coords = data["A"], data["coords"]
    school = data["school_node"]
    t_mid = (data["t_start"] + data["t_end"]) // 2
    v0 = data["weekday"]["speed"][t_mid]
    v1 = data["guided"]["speed"][t_mid]
    q0 = data["weekday"]["flow"][t_mid]
    q1 = data["guided"]["flow"][t_mid]
    # local window around school
    d = np.linalg.norm(coords - coords[school], axis=1)
    keep = d < 1200.0
    idx = np.where(keep)[0]
    A_loc = A[np.ix_(idx, idx)]
    c_loc = coords[idx]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    sc = draw_graph(axes[0], c_loc, A_loc, v0[idx], 20 + 50 * q0[idx] / (q0[idx].max() + 1e-6), "Before guidance")
    draw_graph(axes[1], c_loc, A_loc, v1[idx], 20 + 50 * q1[idx] / (q1[idx].max() + 1e-6), "After guidance")
    fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, label="Speed (km/h)")
    fig.suptitle("Fig.8 Sub-network traffic state under localized targeted intervention")
    return save(fig, out_path)
