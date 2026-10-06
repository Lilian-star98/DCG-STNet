"""Train STCausalNet (§3.9 + §4)."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import ModelConfig
from ..data.loaders import load_adjacency, load_dataset
from ..data.preprocess import make_windows, scale_zscore, time_split
from ..data.features import build_time_features
from ..model.candidate_edges import build_candidate_edges
from ..model.stcausalnet import STCausalNet


def grouped_optimizer(model: STCausalNet, cfg: ModelConfig):
    graph, latent, rest = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("graph.") or n.startswith("dynamics.fusion"):
            graph.append(p)
        elif n.startswith("latent."):
            latent.append(p)
        else:
            rest.append(p)
    return torch.optim.Adam(
        [
            {"params": rest, "lr": cfg.lr},
            {"params": graph, "lr": cfg.lr_graph},
            {"params": latent, "lr": cfg.lr_latent},
        ],
        weight_decay=cfg.weight_decay,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metr-la")
    ap.add_argument("--root", default=".")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--hist", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ckpt", default="out/paper/stcausalnet.pt")
    ap.add_argument("--ablation", default="full", help="full | w/o Scene | w/o Macro | w/o Dynamic")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = load_dataset(args.dataset, args.root)
    splits = time_split(X)
    Xtr, mean, std = scale_zscore(splits["train"], splits["train"])
    Xva, _, _ = scale_zscore(splits["train"], splits["val"])
    T, N, F = X.shape
    A = load_adjacency(args.dataset, args.root, n_nodes=N)
    cand, _ = build_candidate_edges(A_static=A, k=args.k)
    steps = 288 if args.dataset != "q" else 96

    def pack(arr):
        xin, y = make_windows(arr, hist=args.hist, horizon=args.horizon)
        return xin, y

    Xin, Y = pack(Xtr)
    Xv, Yv = pack(Xva)
    cfg = ModelConfig(
        num_nodes=N, d_x=F, d_y=F, d_hidden=args.hidden, k_neighbors=args.k,
        hist=args.hist, horizon=args.horizon, d_z_glob=5, d_z_loc=4,
    )
    variant = "full" if args.ablation == "full" else args.ablation
    model = STCausalNet.ablation(cfg, variant) if variant != "full" else STCausalNet(cfg)
    model.to(device)
    opt = grouped_optimizer(model, cfg)
    A_t = torch.from_numpy(A).float().to(device)
    cand_t = torch.from_numpy(cand).float().to(device)

    def extras(B, Tlen, start_idx=0):
        Zg = torch.from_numpy(build_time_features(Tlen, steps_per_day=steps)).float()
        Zg = Zg.unsqueeze(0).expand(B, -1, -1).to(device)
        U = torch.zeros(B, Tlen, N, cfg.d_u, device=device)
        Z = torch.zeros(B, Tlen, N, cfg.d_z_loc, device=device)
        return U, Z, Zg

    loader = DataLoader(TensorDataset(torch.from_numpy(Xin), torch.from_numpy(Y)), batch_size=args.batch_size, shuffle=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        nobs = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            B, Th, _, _ = xb.shape
            U, Z, Zg = extras(B, Th)
            # Teacher-force next-step targets inside the history window: Y_t ≈ X_{t+1}
            out = model(xb, U, Z, Zg, A_t, cand_t, y_true=xb[:, 1:])
            loss = out["L_total"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * B
            nobs += B
        print(f"epoch {ep}: train {tot / max(nobs, 1):.4f}", flush=True)

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": cfg.__dict__, "mean": mean, "std": std, "A": A}, args.ckpt)
    print("saved", args.ckpt)


if __name__ == "__main__":
    main()
