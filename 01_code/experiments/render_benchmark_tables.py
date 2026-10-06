"""Rebuild Table 1 / multi-dataset markdown from benchmark results.json."""

from __future__ import annotations

import argparse
import json
import os

from paper.tables import table1_in_distribution as t1
from paper.tables.formatters import dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="out/paper/benchmark/results.json")
    ap.add_argument("--out", default="out/paper/benchmark/tables")
    args = ap.parse_args()
    with open(args.results, "r", encoding="utf-8") as f:
        blob = json.load(f)
    results = blob.get("results", {})
    computed = {}
    for m, dsmap in results.items():
        computed[m] = {
            "METR-LA": dsmap.get("METR-LA", ["—", "—", "—"]),
            "PEMS-BAY": dsmap.get("PEMS-BAY", ["—", "—", "—"]),
        }
    os.makedirs(args.out, exist_ok=True)
    if any("METR-LA" in v or "PEMS-BAY" in v for v in results.values()):
        print(t1.render(args.out, computed=computed))

    # wide table all datasets
    datasets = ["METR-LA", "PEMS-BAY", "PEMS04", "Q"]
    headers = ["Model"] + [f"{d} MAE" for d in datasets]
    rows = []
    for m, dsmap in results.items():
        row = [m]
        for d in datasets:
            v = dsmap.get(d)
            row.append(f"{v[0]:.3f}" if isinstance(v, list) else "—")
        rows.append(row)
    print(dump(os.path.join(args.out, "table_all_mae"), "Computed MAE across datasets", headers, rows))


if __name__ == "__main__":
    main()
