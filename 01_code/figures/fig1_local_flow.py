"""Fig.1 Local network flow impact under lane narrowing (E3)."""

from __future__ import annotations

import numpy as np

from .style import draw_graph, save, setup
from ..scenarios.e3_capacity_drop import run_e3
from ..scenarios.network import PAPER_NODES


def _roles(data):
    r = data.get("roles") or (data.get("meta") or {}).get("roles")
    if r:
        return r
    return {
        "bottleneck": PAPER_NODES["bottleneck"],
        "up1": PAPER_NODES["up1"],
        "up2": PAPER_NODES["up2"],
        "down": PAPER_NODES["down"],
    }


def render(out_path: str, data=None) -> str:
    setup()
    data = data or run_e3()
    A, coords = data["A"], data["coords"]
    roles = _roles(data)
    target = int(data["target"][0]) if data.get("target") is not None else int(roles["bottleneck"])
    t0, t1 = data["spec"].t_start, data["spec"].t_end
    drop = data["speed"][t0:t1].mean(0) - data["speed_cf"][t0:t1].mean(0)
    colors = 70.0 - np.clip(drop * 4.0, 0, 50)  # more drop -> redder
    sizes = 18 + 80 * (drop / (max(drop.max(), 1e-6)))
    # subsample edges for large graphs
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    A_plot = A
    if A.shape[0] > 800:
        # plot local ego graph around bottleneck for readability
        from ..scenarios.e_real_q import _neighbors

        keep = np.array(_neighbors(A, target, hops=3), dtype=np.int64)
        A_plot = A[np.ix_(keep, keep)]
        coords_plot = coords[keep]
        colors_plot = colors[keep]
        sizes_plot = sizes[keep]
        id_map = {int(k): i for i, k in enumerate(keep)}
        roles_plot = {k: id_map[v] for k, v in roles.items() if v in id_map}
        target_plot = id_map[target]
    else:
        coords_plot, colors_plot, sizes_plot, roles_plot, target_plot = coords, colors, sizes, roles, target
    sc = draw_graph(ax, coords_plot, A_plot, colors_plot, sizes_plot, title="Fig.1 Local network flow impact (lane narrowing)")
    nbs = [roles_plot.get("up1", target_plot), roles_plot.get("up2", target_plot), roles_plot.get("down", target_plot), target_plot]
    ax.scatter(coords_plot[nbs, 0], coords_plot[nbs, 1], s=140, facecolors="none", edgecolors="k", linewidths=1.2, zorder=3)
    for key, lab in [("bottleneck", str(target)), ("up1", "up"), ("up2", "up"), ("down", "dn")]:
        if key in roles_plot:
            ax.annotate(lab if key != "bottleneck" else f"N{target}", coords_plot[roles_plot[key]], textcoords="offset points", xytext=(4, 4), fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.046, label="Proxy speed (km/h)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    return save(fig, out_path)
