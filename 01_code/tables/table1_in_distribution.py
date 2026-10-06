"""Table 1: in-distribution H=12 metrics (paper §4.5.1)."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from .formatters import dump


def _load_reported() -> dict:
    path = os.path.join(os.path.dirname(__file__), "paper_reported.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["table1_in_distribution"]


def render(out_dir: str, computed: Optional[Dict] = None) -> Dict[str, str]:
    """If `computed` is provided (model -> {dataset: [MAE,RMSE,MAPE]}), use it; else paper numbers."""
    os.makedirs(out_dir, exist_ok=True)
    meta = _load_reported()
    headers = ["Model", "METR-LA MAE", "METR-LA RMSE", "METR-LA MAPE", "PEMS-BAY MAE", "PEMS-BAY RMSE", "PEMS-BAY MAPE"]
    rows = []
    if computed:
        for model, vals in computed.items():
            m = vals.get("METR-LA", ["—"] * 3)
            p = vals.get("PEMS-BAY", ["—"] * 3)
            rows.append([model, *[_fmt(x) for x in m], *[_fmt(x) for x in p]])
        title = meta["caption"] + " (computed)"
    else:
        for r in meta["rows"]:
            rows.append([r["model"], *_fmt_row(r["METR-LA"]), *_fmt_row(r["PEMS-BAY"], mape=True)])
        title = meta["caption"] + " (paper-reported)"
    return dump(os.path.join(out_dir, "table1_in_distribution"), title, headers, rows)


def _fmt(x, mape=False):
    if isinstance(x, str):
        return x
    if mape:
        return f"{x:.2f}%"
    return f"{float(x):.2f}"


def _fmt_row(xs, mape=False):
    return [_fmt(xs[0]), _fmt(xs[1]), _fmt(xs[2], mape=True)]


if __name__ == "__main__":
    print(render("out/paper_tables"))
