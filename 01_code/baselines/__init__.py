"""Baseline model registry."""

from __future__ import annotations

from .cast import CaST
from .d2stgnn import D2STGNN
from .graphwavenet import GraphWaveNet
from .pdformer import PDFormer
from .staeformer import STAEformer

BASELINES = {
    "GraphWaveNet": GraphWaveNet,
    "D2STGNN": D2STGNN,
    "CaST": CaST,
    "PDFormer": PDFormer,
    "STAEformer": STAEformer,
}


def build_baseline(name: str, num_nodes: int, in_dim: int, out_dim: int, horizon: int, hidden: int = 64, **kwargs):
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline {name}. Choose from {list(BASELINES)}")
    cls = BASELINES[name]
    return cls(num_nodes=num_nodes, in_dim=in_dim, out_dim=out_dim, horizon=horizon, hidden=hidden, **kwargs)


__all__ = ["BASELINES", "build_baseline", "GraphWaveNet", "D2STGNN", "CaST", "PDFormer", "STAEformer"]
