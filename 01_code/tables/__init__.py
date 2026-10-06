"""Compute Table 1–4 from predictions when available; otherwise paper-reported numbers."""

from __future__ import annotations

import os
from typing import Dict, Optional

from . import table1_in_distribution as t1
from . import table2_event_stratified as t2
from . import table3_ablation_pred as t3
from . import table4_ablation_cf as t4
from .metrics import all_metrics, graph_sparsity, pvr, ted


def render_all(out_dir: str = "out/paper_tables", computed: Optional[Dict] = None) -> Dict:
    computed = computed or {}
    return {
        "table1": t1.render(out_dir, computed.get("table1")),
        "table2": t2.render(out_dir, computed.get("table2")),
        "table3": t3.render(out_dir, computed.get("table3")),
        "table4": t4.render(out_dir, computed.get("table4")),
    }


def metrics_from_arrays(y_true, y_pred) -> list:
    m = all_metrics(y_true, y_pred)
    return [m["MAE"], m["RMSE"], 100.0 * m["MAPE"]]


def cf_row(pred_cf, true_cf, factual, C, v_free=80.0) -> dict:
    effect_hat = pred_cf - factual
    effect_true = true_cf - factual
    return {
        "PVR": pvr(pred_cf[..., 0] if pred_cf.ndim > 2 else pred_cf, v_free_kmh=v_free),
        "TED": ted(effect_hat, effect_true),
        "sparsity": graph_sparsity(C),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(render_all(), indent=2))
