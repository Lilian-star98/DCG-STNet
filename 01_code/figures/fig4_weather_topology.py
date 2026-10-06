"""Fig.4 Network-wide vulnerability snapshot under adverse weather."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .style import draw_graph, save, setup
from ..scenarios.e4_weather import run_e4


def render(out_path: str, data=None, t_snap: int = None) -> str:
    setup()
    data = data or run_e4()
    spec = data["spec"]
    t_snap = int(np.clip(spec.t_peak, 0, data["speed"].shape[0] - 1)) if t_snap is None else t_snap
    A, coords = data["A"], data["coords"]
    v = data["speed"][t_snap]
    vcf = data["speed_cf"][t_snap]
    q = data["flow"][t_snap]
    qcf = data["flow_cf"][t_snap]
    # For large real-Q graphs, show a random connected sample for readability
    if A.shape[0] > 600:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(A.shape[0], size=600, replace=False))
        A, coords = A[np.ix_(idx, idx)], coords[idx]
        v, vcf, q, qcf = v[idx], vcf[idx], q[idx], qcf[idx]
    s = 12 + 40 * (q / (q.max() + 1e-6))
    scf = 12 + 40 * (qcf / (qcf.max() + 1e-6))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    c0 = draw_graph(axes[0], coords, A, v, s, title="Normal weather (t ≈ 18:00)")
    draw_graph(axes[1], coords, A, vcf, scf, title="Rainfall +1 h")
    fig.colorbar(c0, ax=axes.ravel().tolist(), fraction=0.025, label="Speed (km/h)")
    fig.suptitle("Fig.4 Network-wide traffic vulnerability to adverse weather")
    return save(fig, out_path)
