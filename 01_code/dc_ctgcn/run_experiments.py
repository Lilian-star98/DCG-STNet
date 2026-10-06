"""Experiment runner for baseline forecasting on multiple datasets.

- Chronological 6:2:2 split (train/val/test)
- Horizons: dataset-specific mapping to minutes
- Baselines: persistence, moving average, linear trend
- Metrics: MAE, RMSE, MAPE, R2

Usage examples:
  python -m dc_ctgcn.run_experiments --dataset metr-la
  python -m dc_ctgcn.run_experiments --dataset q --results out/dc_ctgcn_q.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from . import baselines
from .data_utils import load_dataset, scale_minmax, time_split
from .metrics import all_metrics


def make_windows(X: np.ndarray, hist: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding windows X_in, X_out for multi-step prediction."""
    T = X.shape[0]
    X_in = []
    X_out = []
    for t in range(hist, T - horizon + 1):
        X_in.append(X[t - hist : t])
        X_out.append(X[t : t + horizon])
    return np.stack(X_in, axis=0), np.stack(X_out, axis=0)


def evaluate_baselines(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, horizons: List[int], hist: int) -> Dict:
    results = {}
    # normalize using train stats for stability
    X_train_n = scale_minmax(X_train, X_train)
    X_val_n = scale_minmax(X_train, X_val)
    X_test_n = scale_minmax(X_train, X_test)

    for h in horizons:
        Xin, Yout = make_windows(X_test_n, hist=hist, horizon=h)
        res_h = {}
        for name, fn in baselines.BASELINES.items():
            preds = []
            for i in range(Xin.shape[0]):
                preds.append(fn(Xin[i], horizon=h))
            preds = np.stack(preds, axis=0)
            m = all_metrics(Yout, preds)
            res_h[name] = m
        results[h] = res_h
    return results


def dataset_horizons(name: str) -> List[int]:
    name = name.lower()
    if name in {"metr-la", "pems-bay", "pems04"}:
        return [1, 3, 6, 12]  # 5/15/30/60 min if 5-min sampling
    if name in {"q", "beijing-q", "beijing"}:
        return [1, 2, 4]  # 15/30/60 min for 15-min sampling
    return [1, 3, 6, 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="metr-la | pems-bay | pems04 | q")
    ap.add_argument("--root", default=".", help="project root (default: cwd)")
    ap.add_argument("--hist", type=int, default=12, help="history window length")
    ap.add_argument("--results", type=str, default=None, help="optional path to save JSON results")
    args = ap.parse_args()

    X = load_dataset(args.dataset, root=args.root)
    splits = time_split(X, ratios=(0.6, 0.2, 0.2))
    horizons = dataset_horizons(args.dataset)

    res = evaluate_baselines(splits["train"], splits["val"], splits["test"], horizons=horizons, hist=args.hist)

    payload = {
        "dataset": args.dataset,
        "shape": list(X.shape),
        "horizons": horizons,
        "results": res,
    }

    if args.results:
        out_dir = os.path.dirname(args.results)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.results, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
