"""Table 4: counterfactual validity ablation (PVR / TED / sparsity)."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from .formatters import dump


def render(out_dir: str, computed: Optional[Dict] = None) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "paper_reported.json")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)["table4_ablation_cf"]
    headers = ["Variant", "PVR (%) ↓", "TED (MAE) ↓", "Graph sparsity (%) ↑"]
    rows = []
    items = computed if computed else {r["variant"]: r for r in meta["rows"]}
    for name, r in items.items():
        rows.append([name, f"{float(r['PVR']):.1f}", f"{float(r['TED']):.2f}", f"{float(r['sparsity']):.1f}"])
    title = meta["caption"] + (" (computed)" if computed else " (paper-reported)")
    return dump(os.path.join(out_dir, "table4_ablation_cf"), title, headers, rows)


if __name__ == "__main__":
    print(render("out/paper_tables"))
