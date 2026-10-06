"""Hyper-parameters aligned with paper §3.6–§3.9 and §4.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    """DCG-STNet / STCausalNet configuration.

    Paper mapping
    -------------
    d_hidden, d_scenario, d_latent : §3.4 / §3.6 / §3.7
    k_neighbors, d_decay           : §3.6.2 candidate set + LWR decay
    alpha_static_adj               : §3.6.4 fusion coefficient
    lambdas                        : §3.9.1 composite loss
    hist, horizon                  : §4.2.1 sliding windows (H=12, 15/30/60 min)
    """

    num_nodes: int
    d_x: int = 1
    d_u: int = 1
    d_z_loc: int = 4
    d_z_glob: int = 8
    d_w: int = 0
    d_y: int = 1

    d_hidden: int = 128
    d_scenario: int = 32
    d_latent: int = 8
    k_neighbors: int = 12
    d_decay: float = 1500.0
    gumbel_tau: float = 0.5
    alpha_static_adj: float = 0.5
    n_regions: int = 0  # 0 = disable hierarchical coarsening (Thm 3.4)

    lambda_sparse: float = 1e-3
    lambda_smooth: float = 1e-3
    lambda_dual: float = 0.1
    lambda_kl: float = 1e-3
    lambda_res_smooth: float = 1e-4
    lambda_inv: float = 0.0
    prediction_loss: str = "mae"

    hist: int = 12
    horizon: int = 12
    lr: float = 1e-3
    lr_graph: float = 3e-3
    lr_latent: float = 3e-4
    weight_decay: float = 0.0
    dropout: float = 0.1
    use_physics: bool = True  # active only with an explicit, unit-calibrated PhysicsContext


@dataclass
class DataConfig:
    """Dataset paths and split (§4.1–4.2)."""

    root: str = "."
    name: str = "metr-la"  # metr-la | pems-bay | pems04 | q
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    hist: int = 12
    horizon: int = 12
    zscore: bool = True  # paper §4.2.1 uses Z-score (train stats only)


@dataclass
class ScenarioConfig:
    """Appendix B scenario / CTM parameters."""

    dt_sec: float = 300.0  # 5 min; Q-Traffic uses 900 s
    v_free: float = 60.0 / 3.6  # m/s (~60 km/h)
    w_cong: float = 20.0 / 3.6
    length_m: float = 500.0
    n_lanes: int = 3
    sat_flow: float = 1800.0  # veh/h/lane
    rho_jam: float = 0.14  # veh/m/lane
    seed: int = 42


PAPER_DATASETS = {
    "metr-la": {"N": 207, "res_min": 5, "vars": "speed", "range": "2012.03–06"},
    "pems-bay": {"N": 325, "res_min": 5, "vars": "speed", "range": "2017.01–06"},
    "pems04": {"N": 307, "res_min": 5, "vars": "flow/speed/occ", "range": "PEMS04"},
    "q": {"N": 15073, "res_min": 15, "vars": "speed/flow", "range": "2017.04.01–05.31"},
}

__all__ = ["ModelConfig", "DataConfig", "ScenarioConfig", "PAPER_DATASETS"]
