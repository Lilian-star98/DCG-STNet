"""Shared matplotlib style for paper figures."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


SPEED_CMAP = LinearSegmentedColormap.from_list(
    "traffic",
    ["#7f0000", "#cb181d", "#fc4e2a", "#fd8d3c", "#fed976", "#78c679", "#006837"],
)


def setup():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "axes.grid": False,
        }
    )


def save(fig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def speed_to_color(v, vmin=20.0, vmax=70.0):
    x = (v - vmin) / max(vmax - vmin, 1e-6)
    return SPEED_CMAP(np_clip(x))


def np_clip(x):
    import numpy as np

    return np.clip(x, 0.0, 1.0)


def draw_graph(ax, coords, A, colors, sizes, title=""):
    import numpy as np

    src, dst = np.where(A > 0)
    for i, j in zip(src, dst):
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color="#bbbbbb", lw=0.4, zorder=1)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes, cmap=SPEED_CMAP, vmin=20, vmax=70, zorder=2, edgecolors="k", linewidths=0.2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    return sc
