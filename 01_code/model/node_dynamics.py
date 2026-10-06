"""§3.3–§3.4 Node structural equations + ST-GNN parameterization."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..config import ModelConfig
from .fusion_adj import FusionAdjacency


class NodeDynamics(nn.Module):
    """GRU time encoder + gated graph convolution (paper §3.4).

    H_{t+1} = GRU(H_t,  Ã_t H_t W + inputs)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.d_x + cfg.d_u + cfg.d_z_loc + cfg.d_z_glob + cfg.d_latent
        self.input_proj = nn.Linear(in_dim, cfg.d_hidden)
        self.msg = nn.Linear(cfg.d_hidden, cfg.d_hidden)
        self.gate = nn.Linear(2 * cfg.d_hidden, cfg.d_hidden)
        self.gru = nn.GRUCell(cfg.d_hidden, cfg.d_hidden)
        self.fusion = FusionAdjacency(alpha_init=cfg.alpha_static_adj, learnable=True)
        self.act = nn.ReLU()

    def forward(
        self,
        H_t: torch.Tensor,
        X_t: torch.Tensor,
        U_t: torch.Tensor,
        Z_loc_t: torch.Tensor,
        Z_glob_t: torch.Tensor,
        L_t: torch.Tensor,
        C_t: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        B, N, d_h = H_t.shape
        if Z_glob_t.dim() == 2:
            Zg = Z_glob_t.unsqueeze(1).expand(-1, N, -1)
        else:
            Zg = Z_glob_t
        L = L_t.unsqueeze(1).expand(-1, N, -1)
        raw = torch.cat([X_t, U_t, Z_loc_t, Zg, L], dim=-1)
        x_emb = self.act(self.input_proj(raw))

        A_tilde = self.fusion(A.to(H_t.dtype), C_t)
        msg = torch.matmul(A_tilde, self.msg(H_t))
        gate = torch.sigmoid(self.gate(torch.cat([H_t, msg], dim=-1)))
        h_in = self.act(x_emb + gate * msg)
        h_flat = H_t.reshape(B * N, d_h)
        h_next = self.gru(h_in.reshape(B * N, d_h), h_flat)
        return h_next.view(B, N, d_h)


class OutputDecoder(nn.Module):
    """Y_t = g_θ(H_t), differentiable readout (§3.4)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_hidden, cfg.d_hidden),
            nn.ReLU(),
            nn.Linear(cfg.d_hidden, cfg.d_y),
        )

    def forward(self, H_t: torch.Tensor) -> torch.Tensor:
        return self.mlp(H_t)


__all__ = ["NodeDynamics", "OutputDecoder"]
