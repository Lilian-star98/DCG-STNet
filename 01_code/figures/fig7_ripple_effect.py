"""Fig.7 Targeted intervention efficacy and topological ripple effects."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .style import save, setup
from ..scenarios.e5_school import run_e5


def render(out_path: str, data=None) -> str:
    setup()
    data = data or run_e5()
    school = data["school_node"]
    nbs = list(data["neighbors"])
    t0, t1 = data["t_start"], data["t_end"]
    hours = np.arange(data["weekday"]["speed"].shape[0]) * 0.25
    gain = data["guided"]["speed"] - data["weekday"]["speed"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(hours, gain[:, school], color="#238b45", lw=2.0, label=f"Node {school} speed gain")
    axes[0].axvspan(t0 * 0.25, t1 * 0.25, color="#ffffcc", zorder=0)
    axes[0].set_xlim(6.0, 10.0)
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Speed gain (km/h)")
    axes[0].set_title("Direct gain at the bottleneck")
    axes[0].legend()

    mean_gain = gain[t0:t1][:, nbs].mean(axis=0)
    axes[1].bar([str(i) for i in nbs], mean_gain, color="#3182bd", edgecolor="k", linewidth=0.4)
    axes[1].set_xlabel("1-hop neighbor node id")
    axes[1].set_ylabel("Mean speed gain (km/h)")
    axes[1].set_title("Ripple effect on untreated neighbors")
    fig.suptitle("Fig.7 Targeted intervention efficacy and topological ripple effects")
    fig.tight_layout()
    return save(fig, out_path)
