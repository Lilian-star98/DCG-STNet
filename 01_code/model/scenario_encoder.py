"""§3.6.1 Scene encoder  S_t = E_sc(U_t, Z_t, Z^{glob}_t, W_t)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..config import ModelConfig


class ScenarioEncoder(nn.Module):
    """Hierarchical encoder of local + global exogenous variables.

    Local node features (U, Z, W) are encoded then mean-pooled and fused
    with global covariates, matching Lemma 3.4 (scene sufficiency).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        node_in = cfg.d_u + cfg.d_z_loc + (cfg.d_w if cfg.d_w > 0 else 0)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, cfg.d_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, cfg.d_hidden),
            nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(cfg.d_hidden + cfg.d_z_glob, cfg.d_hidden),
            nn.ReLU(),
            nn.Linear(cfg.d_hidden, cfg.d_scenario),
        )

    def forward(
        self,
        U_t: torch.Tensor,
        Z_loc_t: torch.Tensor,
        Z_glob_t: torch.Tensor,
        W_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [U_t, Z_loc_t]
        if W_t is not None and self.cfg.d_w > 0:
            parts.append(W_t)
        node_feat = self.node_mlp(torch.cat(parts, dim=-1))
        node_agg = node_feat.mean(dim=1)
        if Z_glob_t.dim() == 3:
            Z_glob_t = Z_glob_t.squeeze(1)
        S_t = self.global_mlp(torch.cat([node_agg, Z_glob_t], dim=-1))
        return S_t


__all__ = ["ScenarioEncoder"]
