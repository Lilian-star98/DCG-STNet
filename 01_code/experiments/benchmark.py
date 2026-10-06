"""Unified multi-horizon benchmark for Table 1 / multi-dataset comparison.

Trains GraphWaveNet / D2STGNN / CaST / PDFormer / STAEformer / STCausalNet
on metr-la, pems-bay, pems04, q and writes computed tables.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from paper.baselines import BASELINES, build_baseline
from paper.baselines.common import normalize_adj
from paper.data.loaders import load_adjacency, load_dataset
from paper.data.preprocess import make_windows, scale_zscore, time_split
from paper.data.q_real import load_real_q_subgraph
from paper.tables.metrics import all_metrics
from paper.tables import table1_in_distribution as t1


class STCausalNetForecast(nn.Module):
    """Multi-horizon wrapper: encode with STCausalNet, project last state to H steps."""

    def __init__(self, num_nodes, in_dim, out_dim, horizon=12, hidden=64, hist=12):
        super().__init__()
        from paper.config import ModelConfig
        from paper.model.stcausalnet import STCausalNet

        self.horizon = horizon
        self.out_dim = out_dim
        self.cfg = ModelConfig(
            num_nodes=num_nodes,
            d_x=in_dim,
            d_y=out_dim,
            d_hidden=hidden,
            d_scenario=16,
            d_latent=8,
            k_neighbors=min(8, num_nodes),
            d_z_glob=5,
            d_z_loc=4,
            hist=hist,
            horizon=horizon,
        )
        self.net = STCausalNet(self.cfg)
        self.multi = nn.Linear(out_dim, horizon * out_dim)

    def forward(self, x, A_static=None):
        B, T, N, F = x.shape
        device = x.device
        U = torch.zeros(B, T, N, self.cfg.d_u, device=device)
        Z = torch.zeros(B, T, N, self.cfg.d_z_loc, device=device)
        Zg = torch.zeros(B, T, self.cfg.d_z_glob, device=device)
        cand = (A_static > 0).float() if A_static is not None else torch.ones(N, N, device=device)
        A = A_static if A_static is not None else torch.eye(N, device=device)
        out = self.net(x, U, Z, Zg, A, cand)
        y1 = out["Y_F"][:, -1]  # (B,N,F)
        y = self.multi(y1).view(B, N, self.horizon, self.out_dim).permute(0, 2, 1, 3)
        return y


def _subsample(Xin, Y, max_n: int, seed: int = 0):
    if max_n <= 0 or Xin.shape[0] <= max_n:
        return Xin, Y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(Xin.shape[0], size=max_n, replace=False))
    return Xin[idx], Y[idx]


def load_xy(dataset: str, root: str, hist: int, horizon: int, q_max_nodes: int = 2000, max_train=8192, max_eval=2048):
    if dataset == "q":
        q = load_real_q_subgraph(root=root, max_nodes=q_max_nodes, seed=42, include_flow=False)
        X = q["X"]  # (T,N,1)
        A = q["A"]
    else:
        X = load_dataset(dataset, root)
        # speed-only for fair Table-1 style
        if X.shape[-1] > 1:
            X = X[..., :1]
        A = load_adjacency(dataset, root, n_nodes=X.shape[1])
    splits = time_split(X)
    Xtr, mean, std = scale_zscore(splits["train"], splits["train"])
    Xva, _, _ = scale_zscore(splits["train"], splits["val"])
    Xte, _, _ = scale_zscore(splits["train"], splits["test"])

    def pack(arr):
        return make_windows(arr, hist=hist, horizon=horizon)

    Xtr, Ytr = pack(Xtr)
    Xva, Yva = pack(Xva)
    Xte, Yte = pack(Xte)
    Xtr, Ytr = _subsample(Xtr, Ytr, max_train, seed=0)
    Xva, Yva = _subsample(Xva, Yva, max_eval, seed=1)
    Xte, Yte = _subsample(Xte, Yte, max_eval, seed=2)
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte), A, {"mean": mean, "std": std, "N": X.shape[1], "F": X.shape[-1]}


def train_one(
    model,
    train_loader,
    val_loader,
    A_t,
    device,
    epochs: int,
    lr: float,
    patience: int = 10,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    best_state = None
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, A_t)
            loss = torch.abs(pred - yb).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * xb.size(0)
            n += xb.size(0)
        tr = tot / max(n, 1)
        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, A_t)
                vloss = torch.abs(pred - yb).mean()
                vtot += float(vloss.detach()) * xb.size(0)
                vn += xb.size(0)
        va = vtot / max(vn, 1)
        print(f"  epoch {ep}: train {tr:.4f} val {va:.4f}", flush=True)
        if va < best - 1e-5:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("  early stop", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


@torch.no_grad()
def evaluate(model, loader, A_t, device, mean, std):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb, A_t).cpu().numpy()
        true = yb.numpy()
        # inverse zscore on feature 0
        pred = pred * std + mean
        true = true * std + mean
        ys.append(true)
        ps.append(pred)
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    # H=12 aggregate metrics on all horizons (paper Table 1 style)
    m = all_metrics(y, p)
    return [m["MAE"], m["RMSE"], 100.0 * m["MAPE"]]


def build_model(name, N, F, horizon, hist, hidden):
    if name == "STCausalNet":
        return STCausalNetForecast(N, F, F, horizon=horizon, hidden=hidden, hist=hist)
    kwargs = {}
    if name == "STAEformer":
        kwargs["hist"] = hist
    return build_baseline(name, num_nodes=N, in_dim=F, out_dim=F, horizon=horizon, hidden=hidden, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--datasets", nargs="+", default=["metr-la", "pems-bay", "pems04", "q"])
    ap.add_argument("--models", nargs="+", default=list(BASELINES.keys()) + ["STCausalNet"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hist", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--q-max-nodes", type=int, default=2000)
    ap.add_argument("--out", default="out/paper/benchmark")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=8192)
    ap.add_argument("--max-eval", type=int, default=2048)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        results: Dict[str, Dict[str, List[float]]] = prev.get("results", {m: {} for m in args.models})
        log = prev.get("log", [])
        print(f"Resuming from {results_path}", flush=True)
    else:
        results = {m: {} for m in args.models}
        log = []

    done = {(r.get("model"), r.get("dataset")) for r in log if "error" not in r}

    for ds in args.datasets:
        print(f"\n=== Dataset {ds} ===", flush=True)
        # Q uses hist/horizon in 15-min units; paper Table1 is H=12 on 5-min data.
        hist = args.hist if ds != "q" else 12
        horizon = args.horizon if ds != "q" else 4  # 60 min
        (Xtr, Ytr), (Xva, Yva), (Xte, Yte), A, meta = load_xy(
            ds, args.root, hist, horizon, args.q_max_nodes, max_train=args.max_train, max_eval=args.max_eval
        )
        # For large N (Q), reduce batch
        bs = args.batch_size if meta["N"] <= 400 else max(1, args.batch_size // 4)
        print(f"  windows train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)}, N={meta['N']}", flush=True)
        # STAEformer on Q with N=2000 and T*N tokens is heavy — subsample nodes for transformer models if needed
        train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)), batch_size=bs, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)), batch_size=bs)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(Yte)), batch_size=bs)
        A_t = normalize_adj(A).to(device)

        for name in args.models:
            key = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY", "pems04": "PEMS04", "q": "Q"}.get(ds, ds)
            if (name, key) in done:
                print(f"\n-- skip {name} on {ds} (cached)", flush=True)
                continue
            print(f"\n-- {name} on {ds}", flush=True)
            t0 = time.time()
            try:
                model = build_model(name, meta["N"], meta["F"], horizon, hist, args.hidden).to(device)
                train_one(model, train_loader, val_loader, A_t, device, args.epochs, args.lr, patience=args.patience)
                metrics = evaluate(model, test_loader, A_t, device, meta["mean"], meta["std"])
                ckpt = os.path.join(args.out, f"{ds}_{name}.pt")
                torch.save({"state": model.state_dict(), "metrics": metrics, "dataset": ds, "model": name}, ckpt)
                results.setdefault(name, {})[key] = metrics
                row = {"model": name, "dataset": key, "MAE": metrics[0], "RMSE": metrics[1], "MAPE": metrics[2], "secs": time.time() - t0}
                log.append(row)
                done.add((name, key))
                print(f"  TEST {row}", flush=True)
            except Exception as e:
                print(f"  FAILED {name}/{ds}: {e}", flush=True)
                log.append({"model": name, "dataset": key, "error": str(e)})

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"results": results, "log": log}, f, indent=2)

            # free memory
            if "model" in locals():
                del model
            if device == "cuda":
                torch.cuda.empty_cache()

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "log": log}, f, indent=2)

    # Render Table 1 style for METR-LA / PEMS-BAY if present
    computed = {}
    for m, dsmap in results.items():
        if "METR-LA" in dsmap or "PEMS-BAY" in dsmap:
            computed[m] = {
                "METR-LA": dsmap.get("METR-LA", ["—", "—", "—"]),
                "PEMS-BAY": dsmap.get("PEMS-BAY", ["—", "—", "—"]),
            }
    if computed:
        paths = t1.render(os.path.join(args.out, "tables"), computed=computed)
        print("Table1:", paths, flush=True)

    # Also dump full markdown
    md = ["## Computed multi-dataset results", ""]
    md.append("| Model | Dataset | MAE | RMSE | MAPE(%) |")
    md.append("|---|---|---:|---:|---:|")
    for r in log:
        if "error" in r:
            md.append(f"| {r['model']} | {r['dataset']} | ERR | ERR | ERR |")
        else:
            md.append(f"| {r['model']} | {r['dataset']} | {r['MAE']:.3f} | {r['RMSE']:.3f} | {r['MAPE']:.2f} |")
    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("Wrote", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
