"""Fig.3 District-level average speed under normal vs sudden rainfall (E4)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .style import save, setup
from ..scenarios.e4_weather import run_e4


def render(out_path: str, data=None) -> str:
    setup()
    data = data or run_e4()
    spec = data["spec"]
    hours = np.arange(data["speed"].shape[0]) * 0.25
    v = data["speed"].mean(axis=1)
    vcf = data["speed_cf"].mean(axis=1)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(hours, v, color="#2171b5", lw=2.0, label="Normal weather")
    ax.plot(hours, vcf, color="#cb181d", lw=2.0, label="Sudden rainfall")
    ax.axvline(spec.t_start * 0.25, color="0.3", ls=":", lw=1.0)
    ax.axvspan(spec.t_start * 0.25, spec.t_end * 0.25, color="#deebf7", zorder=0)
    ax.set_title("Fig.3 District-level average traffic speed: normal vs rainfall")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean speed (km/h)")
    ax.legend()
    return save(fig, out_path)
