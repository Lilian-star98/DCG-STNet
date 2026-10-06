"""Write CSV / markdown / matplotlib tables from dict rows."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(path: str, headers: Sequence[str], rows: List[Sequence]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return path


def write_md(path: str, title: str, headers: Sequence[str], rows: List[Sequence]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [f"## {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_png(path: str, title: str, headers: Sequence[str], rows: List[Sequence], figsize=(10, 2.8)) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, pad=12)
    cell = [[str(x) for x in r] for r in rows]
    tbl = ax.table(cellText=cell, colLabels=list(headers), loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return path


def dump(out_prefix: str, title: str, headers: Sequence[str], rows: List[Sequence]) -> Dict[str, str]:
    return {
        "csv": write_csv(out_prefix + ".csv", headers, rows),
        "md": write_md(out_prefix + ".md", title, headers, rows),
        "png": write_png(out_prefix + ".png", title, headers, rows),
    }
