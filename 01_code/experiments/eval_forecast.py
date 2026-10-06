"""Evaluate MAE/RMSE/MAPE on a checkpoint (Table 1 protocol)."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..data.loaders import load_adjacency, load_dataset
from ..data.preprocess import make_windows, scale_zscore, time_split
from ..data.features import build_time_features
from ..model.candidate_edges import build_candidate_edges
from ..model.stcausalnet import STCausalNet
from ..config import ModelConfig
from ..tables.metrics import all_metrics
from ..tables import metrics_from_arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="metr-la")
    ap.add_argument("--root", default=".")
    ap.add_argument("--hist", type=int, default=12)
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**{k: v for k, v in blob["cfg"].items() if k in ModelConfig.__dataclass_fields__})
    X = load_dataset(args.dataset, args.root)
    splits = time_split(X)
    Xte, _, _ = scale_zscore(splits["train"], splits["test"])
    xin, y = make_windows(Xte, hist=args.hist, horizon=1)
    N, F = X.shape[1], X.shape[2]
    A = blob.get("A", load_adjacency(args.dataset, args.root, N))
    cand, _ = build_candidate_edges(A_static=A, k=cfg.k_neighbors)
    model = STCausalNet(cfg)
    model.load_state_dict(blob["state"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    xb = torch.from_numpy(xin[:64]).to(device)
    B, T, _, _ = xb.shape
    U = torch.zeros(B, T, N, cfg.d_u, device=device)
    Z = torch.zeros(B, T, N, cfg.d_z_loc, device=device)
    Zg = torch.from_numpy(build_time_features(T, 288 if args.dataset != "q" else 96)).float().unsqueeze(0).expand(B, -1, -1).to(device)
    with torch.no_grad():
        out = model(xb, U, Z, Zg, torch.from_numpy(A).float().to(device), torch.from_numpy(cand).float().to(device))
    pred = out["Y_F"][:, -1].cpu().numpy()
    true = y[:64, 0]
    print(json.dumps(all_metrics(true, pred), indent=2))


if __name__ == "__main__":
    main()
