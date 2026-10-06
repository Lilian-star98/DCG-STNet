"""Ablation runner: train variants and dump Table 3/4 style JSON."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


VARIANTS = ["full", "w/o Scene", "w/o Macro", "w/o Dynamic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metr-la")
    ap.add_argument("--root", default=".")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default="out/paper/ablation.json")
    args = ap.parse_args()
    rows = []
    for v in VARIANTS:
        ckpt = os.path.join("out/paper", f"ablation_{v.replace('/', '_').replace(' ', '')}.pt")
        cmd = [
            sys.executable, "-m", "paper.experiments.train",
            "--dataset", args.dataset, "--root", args.root,
            "--epochs", str(args.epochs), "--ablation", v, "--ckpt", ckpt,
        ]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)
        rows.append({"variant": v, "ckpt": ckpt})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
