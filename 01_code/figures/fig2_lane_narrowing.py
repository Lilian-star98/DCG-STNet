"""Fig.2 Impact of Node 100 lane narrowing on adjacent nodes."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .style import save, setup
from ..scenarios.e3_capacity_drop import run_e3
from ..scenarios.network import PAPER_NODES
from .fig1_local_flow import _roles


def render(out_path: str, data=None) -> str:
    setup()
    data = data or run_e3()
    spec = data["spec"]
    roles = _roles(data)
    nodes = {
        f"Node {roles['bottleneck']} (bottleneck)": roles["bottleneck"],
        f"Node {roles['up1']} (neighbor)": roles["up1"],
        f"Node {roles['up2']} (neighbor)": roles["up2"],
        f"Node {roles['down']} (neighbor)": roles["down"],
    }
    t = np.arange(data["speed"].shape[0])
    hours = t * 0.25
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = ["#cb181d", "#2171b5", "#238b45", "#6a51a3"]
    for (lab, nid), c in zip(nodes.items(), colors):
        ax.plot(hours, data["speed"][:, nid], color=c, lw=1.2, ls="-", alpha=0.85, label=f"{lab} factual")
        ax.plot(hours, data["speed_cf"][:, nid], color=c, lw=1.6, ls="--", label=f"{lab} capacity-drop")
    ax.axvspan(spec.t_start * 0.25, spec.t_end * 0.25, color="0.85", zorder=0, label="event window")
    ax.set_title("Fig.2 Impact of bottleneck lane narrowing on adjacent nodes")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Speed (km/h)")
    ax.legend(ncol=2, fontsize=7)
    ax.set_xlim(hours[0], hours[-1])
    return save(fig, out_path)
