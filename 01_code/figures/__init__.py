"""Render Fig.1–Fig.8; default uses real Q subgraph (max_nodes=2000)."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict


def render_all(out_dir: str = "out/paper_figures", real_q: bool = True, root: str = ".", max_nodes: int = 2000) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    if real_q:
        from ..scenarios.e_real_q import run_e3_real, run_e4_real, run_e5_real

        e3 = run_e3_real(root=root, max_nodes=max_nodes)
        e4 = run_e4_real(root=root, max_nodes=max_nodes)
        e5 = run_e5_real(root=root, max_nodes=max_nodes)
    else:
        from ..scenarios.e3_capacity_drop import run_e3
        from ..scenarios.e4_weather import run_e4
        from ..scenarios.e5_school import run_e5

        e3, e4, e5 = run_e3(), run_e4(), run_e5()

    from . import fig1_local_flow as f1
    from . import fig2_lane_narrowing as f2
    from . import fig3_rainfall_speed as f3
    from . import fig4_weather_topology as f4
    from . import fig5_school_hotspot as f5
    from . import fig6_micro_state as f6
    from . import fig7_ripple_effect as f7
    from . import fig8_subnet_topology as f8

    paths = {
        "fig1": f1.render(os.path.join(out_dir, "fig1_local_flow.png"), e3),
        "fig2": f2.render(os.path.join(out_dir, "fig2_lane_narrowing.png"), e3),
        "fig3": f3.render(os.path.join(out_dir, "fig3_rainfall_speed.png"), e4),
        "fig4": f4.render(os.path.join(out_dir, "fig4_weather_topology.png"), e4),
        "fig5": f5.render(os.path.join(out_dir, "fig5_school_hotspot.png"), e5),
        "fig6": f6.render(os.path.join(out_dir, "fig6_micro_state.png"), e5),
        "fig7": f7.render(os.path.join(out_dir, "fig7_ripple_effect.png"), e5),
        "fig8": f8.render(os.path.join(out_dir, "fig8_subnet_topology.png"), e5),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"real_q": real_q, "max_nodes": max_nodes, "paths": paths}, f, indent=2)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/paper/figures_real_q")
    ap.add_argument("--root", default=".")
    ap.add_argument("--max-nodes", type=int, default=2000)
    ap.add_argument("--demo", action="store_true", help="use synthetic demo network instead of real Q")
    args = ap.parse_args()
    print(json.dumps(render_all(args.out, real_q=not args.demo, root=args.root, max_nodes=args.max_nodes), indent=2))


if __name__ == "__main__":
    main()
