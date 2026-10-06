"""Table 2: Q-Traffic event-stratified MAE + PDR (paper §4.5.2)."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from .formatters import dump
from .metrics import pdr


def render(out_dir: str, computed: Optional[Dict] = None) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "paper_reported.json")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)["table2_event_stratified"]
    headers = ["Model", "S1 Normal", "S2 Accident (PDR)", "S3 Weather (PDR)", "S4 Holiday (PDR)"]
    rows = []
    src = computed if computed else {r["model"]: r for r in meta["rows"]}
    for model, r in src.items():
        s1, s2, s3, s4 = float(r["S1"]), float(r["S2"]), float(r["S3"]), float(r["S4"])
        rows.append(
            [
                model,
                f"{s1:.2f}",
                f"{s2:.2f} ({pdr(s1, s2):+.1f}%)",
                f"{s3:.2f} ({pdr(s1, s3):+.1f}%)",
                f"{s4:.2f} ({pdr(s1, s4):+.1f}%)",
            ]
        )
    title = meta["caption"] + (" (computed)" if computed else " (paper-reported)")
    return dump(os.path.join(out_dir, "table2_event_stratified"), title, headers, rows)


if __name__ == "__main__":
    print(render("out/paper_tables"))
