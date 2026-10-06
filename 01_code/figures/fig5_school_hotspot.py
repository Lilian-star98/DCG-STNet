"""Fig.5 School-induced hotspot identification (weekday vs weekend)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .style import save, setup
from ..scenarios.e5_school import run_e5


def render(out_path: str, data=None) -> str:
    setup()
    data = data or run_e5()
    t0, t1 = data["t_start"], data["t_end"]
    wd = data["weekday"]["speed"]
    we = data["weekend"]["speed"]
    drop = we[t0:t1].mean(0) - wd[t0:t1].mean(0)
    school = data["school_node"]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), gridspec_kw={"height_ratios": [1.1, 1.2]})
    axes[0].hist(drop, bins=40, color="#fd8d3c", alpha=0.85, edgecolor="k", linewidth=0.3)
    axes[0].axvline(drop[school], color="#cb181d", lw=2.0, label=f"Node {school} (outlier)")
    axes[0].set_title("Fig.5 Identification of school-induced congestion hotspots")
    axes[0].set_xlabel("Morning speed drop (weekend − weekday, km/h)")
    axes[0].set_ylabel("Number of nodes")
    axes[0].legend()

    hours = np.arange(wd.shape[0]) * 0.25
    top = np.argsort(drop)[-4:][::-1]
    for nid in top:
        axes[1].plot(hours, wd[:, nid], lw=1.5, label=f"Node {nid} weekday")
        axes[1].plot(hours, we[:, nid], lw=1.2, ls="--", label=f"Node {nid} weekend")
    axes[1].axvspan(t0 * 0.25, t1 * 0.25, color="0.9", zorder=0)
    axes[1].set_xlim(5.5, 10.5)
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Speed (km/h)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return save(fig, out_path)
