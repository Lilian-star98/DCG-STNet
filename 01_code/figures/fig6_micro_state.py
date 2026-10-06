"""Fig.6 Microscopic speed/flow of Node 52 and neighbor 202."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .style import save, setup
from ..scenarios.e5_school import run_e5


def render(out_path: str, data=None) -> str:
    setup()
    data = data or run_e5()
    school = data["school_node"]
    nbs = list(data.get("neighbors") or [])
    nb = int(nbs[0]) if nbs else school
    t0, t1 = data["t_start"], data["t_end"]
    hours = np.arange(data["weekday"]["speed"].shape[0]) * 0.25
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex=True)
    series = [
        (axes[0, 0], "speed", school, "Node 52 speed (km/h)"),
        (axes[1, 0], "flow", school, "Node 52 flow (veh/h)"),
        (axes[0, 1], "speed", nb, f"Node {nb} speed (km/h)"),
        (axes[1, 1], "flow", nb, f"Node {nb} flow (veh/h)"),
    ]
    for ax, key, nid, title in series:
        ax.plot(hours, data["weekday"][key][:, nid], color="#cb181d", ls="--", lw=1.6, label="No intervention")
        if data["guided"] is not None:
            ax.plot(hours, data["guided"][key][:, nid], color="#238b45", lw=1.8, label="Guidance")
        ax.axvspan(t0 * 0.25, t1 * 0.25, color="#ffffcc", zorder=0)
        ax.set_title(title)
        ax.set_xlim(6.0, 10.0)
        if key == "speed":
            ax.set_ylim(0, 70)
        ax.legend(fontsize=8)
    axes[1, 0].set_xlabel("Hour of day")
    axes[1, 1].set_xlabel("Hour of day")
    fig.suptitle(f"Fig.6 Microscopic state transitions: Node {school} and neighbor {nb}")
    fig.tight_layout()
    return save(fig, out_path)
