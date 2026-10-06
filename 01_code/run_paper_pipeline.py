"""End-to-end paper pipeline: scenarios -> Fig.1-8 -> Table 1-4 (+ optional model smoke test)."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/paper")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from paper.scenarios import run_e3, run_e4, run_e5
    from paper.scenarios.ctm import default_params, physics_violation_rate
    from paper.figures import render_all as render_figs
    from paper.tables import render_all as render_tabs

    e3 = run_e3()
    e4 = run_e4()
    e5 = run_e5()
    scen_dir = os.path.join(args.out, "scenarios")
    os.makedirs(scen_dir, exist_ok=True)
    np.savez(os.path.join(scen_dir, "e3_capacity_drop.npz"), X=e3["X"], X_cf=e3["X_cf"], A=e3["A"], effect=e3["effect"])
    np.savez(os.path.join(scen_dir, "e4_weather.npz"), X=e4["X"], X_cf=e4["X_cf"], A=e4["A"], effect=e4["effect"])
    np.savez(
        os.path.join(scen_dir, "e5_school.npz"),
        weekday=e5["weekday"]["X"],
        weekend=e5["weekend"]["X"],
        guided=e5["guided"]["X"],
        A=e5["A"],
    )

    fig_dir = os.path.join(args.out, "figures")
    figs = render_figs(fig_dir)
    tab_dir = os.path.join(args.out, "tables")
    tabs = render_tabs(tab_dir)

    params = default_params(e3["A"].shape[0])
    demo_pvr = {
        "e3_cf": physics_violation_rate(e3["speed_cf"], e3["flow_cf"], params),
        "e4_cf": physics_violation_rate(e4["speed_cf"], e4["flow_cf"], params),
    }

    model_ok = None
    if not args.skip_model:
        import torch
        from paper.config import ModelConfig
        from paper.model.stcausalnet import STCausalNet
        from paper.model.candidate_edges import build_candidate_edges

        N, T, F = 12, 8, 1
        cfg = ModelConfig(num_nodes=N, d_x=F, d_y=F, d_hidden=16, d_scenario=8, d_latent=4, k_neighbors=4, d_z_glob=5, d_z_loc=4)
        m = STCausalNet(cfg)
        X = torch.randn(2, T, N, F)
        U = torch.zeros(2, T, N, 1)
        Z = torch.zeros(2, T, N, 4)
        Zg = torch.zeros(2, T, 5)
        A = torch.eye(N)
        cand, _ = build_candidate_edges(A_static=np.eye(N, dtype=np.float32), k=4)
        out = m(X, U, Z, Zg, A, torch.from_numpy(cand), y_true=X[:, 1:])
        model_ok = {k: float(out[k].detach()) for k in ("L_total", "L_pred", "L_sparse", "L_KL") if k in out}

    summary = {"figures": figs, "tables": tabs, "pvr_demo": demo_pvr, "model_smoke": model_ok}
    path = os.path.join(args.out, "pipeline_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
