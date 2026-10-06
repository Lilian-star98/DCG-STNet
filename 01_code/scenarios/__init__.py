"""Scenario generation: Appendix B + E3/E4/E5."""

from .ctm import LinkParams, default_params, physics_violation_rate, simulate
from .e3_capacity_drop import run_e3
from .e4_weather import run_e4
from .e5_school import run_e5
from .e_real_q import run_e3_real, run_e4_real, run_e5_real
from .generate import ScenarioSpec, generate_pair
from .intensity import smooth_beta
from .network import PAPER_NODES, build_demo_network

__all__ = [
    "LinkParams",
    "default_params",
    "simulate",
    "physics_violation_rate",
    "smooth_beta",
    "ScenarioSpec",
    "generate_pair",
    "run_e3",
    "run_e4",
    "run_e5",
    "run_e3_real",
    "run_e4_real",
    "run_e5_real",
    "PAPER_NODES",
    "build_demo_network",
]
