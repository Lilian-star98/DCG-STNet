"""§3.8 Structured counterfactual reasoning (Algorithm 3.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class InterventionSpec:
    """do(π) specification for dual-world rollouts."""

    T0: int
    U_prime: Optional[torch.Tensor] = None
    Z_prime: Optional[torch.Tensor] = None
    A_prime: Optional[torch.Tensor] = None
    edge_mask: Optional[torch.Tensor] = None  # (B,T,N,N) or (T,N,N)
    noise_std: float = 0.0
    physics_prime: Optional[object] = None  # optional intervention-world PhysicsContext


def apply_control(U: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
    U_cf = U.clone()
    if spec.U_prime is not None:
        U_cf[:, spec.T0 :] = spec.U_prime[:, spec.T0 :]
    if spec.noise_std > 0:
        U_cf[:, spec.T0 :] = U_cf[:, spec.T0 :] + spec.noise_std * torch.randn_like(U_cf[:, spec.T0 :])
    return U_cf


def apply_exog(Z: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
    Z_cf = Z.clone()
    if spec.Z_prime is not None:
        Z_cf[:, spec.T0 :] = spec.Z_prime[:, spec.T0 :]
    return Z_cf


def apply_edge_mask(C_t: torch.Tensor, spec: InterventionSpec, t: int) -> torch.Tensor:
    if spec.edge_mask is None or t < spec.T0:
        return C_t
    M = spec.edge_mask
    if M.dim() == 4:
        return C_t * M[:, t]
    if M.dim() == 3:
        return C_t * M[t].unsqueeze(0)
    return C_t


def structured_counterfactual(factual_noise: torch.Tensor, L_seq: torch.Tensor) -> tuple:
    """Abduction keeps inferred noise / latent factors (Alg. 3.1 step 1).

    Action/Prediction are performed by the model rollout with InterventionSpec.
    """
    return factual_noise, L_seq


__all__ = [
    "InterventionSpec",
    "apply_control",
    "apply_exog",
    "apply_edge_mask",
    "structured_counterfactual",
]
