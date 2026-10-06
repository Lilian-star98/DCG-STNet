"""§3.6.4 Adaptive fusion adjacency  Ã_t = α A + (1-α) C_t + I."""

from __future__ import annotations

import torch
from torch import nn


class FusionAdjacency(nn.Module):
    """Scene-adaptive mix of static topology and dynamic causal graph."""

    def __init__(self, alpha_init: float = 0.5, learnable: bool = True):
        super().__init__()
        if learnable:
            self.logit = nn.Parameter(torch.tensor(float(_logit(alpha_init))))
        else:
            self.register_buffer("logit", torch.tensor(float(_logit(alpha_init))))

    def forward(self, A: torch.Tensor, C_t: torch.Tensor, add_self: bool = True) -> torch.Tensor:
        alpha = torch.sigmoid(self.logit)
        if A.dim() == 2:
            A = A.unsqueeze(0)
        A_hat = alpha * A + (1.0 - alpha) * C_t
        if add_self:
            n = A_hat.size(-1)
            eye = torch.eye(n, device=A_hat.device, dtype=A_hat.dtype).unsqueeze(0)
            A_hat = A_hat + eye
        # row-normalize for numerical stability
        deg = A_hat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return A_hat / deg


def fuse_adjacency(A: torch.Tensor, C_t: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    if A.dim() == 2:
        A = A.unsqueeze(0)
    A_hat = alpha * A + (1.0 - alpha) * C_t
    n = A_hat.size(-1)
    eye = torch.eye(n, device=A_hat.device, dtype=A_hat.dtype).unsqueeze(0)
    A_hat = A_hat + eye
    deg = A_hat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return A_hat / deg


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    import math

    return math.log(p / (1 - p))


__all__ = ["FusionAdjacency", "fuse_adjacency"]
