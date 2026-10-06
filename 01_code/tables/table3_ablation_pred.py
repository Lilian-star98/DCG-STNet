"""Table 3: prediction ablation (paper §4.8)."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from .formatters import dump


def render(out_dir: str, computed: Optional[Dict] = None) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "paper_reported.json")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)["table3_ablation_pred"]
    headers = ["Variant", "METR-LA MAE/RMSE", "Q-Traffic MAE/RMSE", "Main contribution"]
    rows = []
    items = computed if computed else {r["variant"]: r for r in meta["rows"]}
    for name, r in items.items():
        m = r["METR-LA"]
        q = r["Q-Traffic"]
        rows.append([name, f"{m[0]:.2f} / {m[1]:.2f}", f"{q[0]:.2f} / {q[1]:.2f}", r.get("note", "")])
    title = meta["caption"] + (" (computed)" if computed else " (paper-reported)")
    return dump(os.path.join(out_dir, "table3_ablation_pred"), title, headers, rows)


if __name__ == "__main__":
    print(render("out/paper_tables"))
