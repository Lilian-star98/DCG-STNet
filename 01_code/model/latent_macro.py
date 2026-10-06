"""§3.7 Latent macro-factor variational inference (amortized BiGRU encoder)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from ..config import ModelConfig


class LatentMacroInferencer(nn.Module):
    """Amortized variational posterior q(L_{1:T} | X, Z^{glob}).

    Prior:    GRU over Z^{glob}
    Posterior: bidirectional GRU over [mean_n(X), Z^{glob}]  (paper §3.7.3)
    ELBO KL is returned as a scalar.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d_l = cfg.d_latent
        self.prior_gru = nn.GRU(cfg.d_z_glob, d_l, batch_first=True)
        self.post_gru = nn.GRU(cfg.d_z_glob + cfg.d_x, d_l, batch_first=True, bidirectional=True)
        self.prior_head = nn.Linear(d_l, 2 * d_l)
        self.post_head = nn.Linear(2 * d_l, 2 * d_l)

    def forward(self, X_seq: torch.Tensor, Zg_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        X_g = X_seq.mean(dim=2)
        prior_h, _ = self.prior_gru(Zg_seq)
        p_mean, p_logvar = torch.chunk(self.prior_head(prior_h), 2, dim=-1)

        post_h, _ = self.post_gru(torch.cat([Zg_seq, X_g], dim=-1))
        q_mean, q_logvar = torch.chunk(self.post_head(post_h), 2, dim=-1)

        eps = torch.randn_like(q_mean) if self.training else torch.zeros_like(q_mean)
        L = q_mean + eps * torch.exp(0.5 * q_logvar.clamp(-8, 8))
        kl = _kl_gauss(q_mean, q_logvar, p_mean, p_logvar).sum(-1).mean()
        return L, {"kl": kl, "q_mean": q_mean, "q_logvar": q_logvar}


def _kl_gauss(mq, lq, mp, lp) -> torch.Tensor:
    vq = torch.exp(lq.clamp(-8, 8))
    vp = torch.exp(lp.clamp(-8, 8))
    return 0.5 * (lp - lq + (vq + (mq - mp) ** 2) / (vp + 1e-8) - 1.0)


__all__ = ["LatentMacroInferencer"]
