"""§3.6.3 Scene-conditioned dynamic causal graph (gate + LWR strength + Top-K)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from ..config import ModelConfig


class SceneAwareGraphGenerator(nn.Module):
    """Generate C_t over candidate edges.

    For each candidate (i, j):
      1) existence gate via Gumbel-Softmax  (§3.6.3 eq. existence)
      2) influence strength with LWR distance decay
      3) immediate vs lagged type mix
      4) differentiable Top-K per destination node
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_hidden + cfg.d_scenario
        self.exist_mlp = nn.Sequential(nn.Linear(2 * d, cfg.d_hidden), nn.ReLU(), nn.Linear(cfg.d_hidden, 1))
        self.strength_mlp = nn.Sequential(nn.Linear(2 * d + 1, cfg.d_hidden), nn.ReLU(), nn.Linear(cfg.d_hidden, 1))
        self.type_mlp = nn.Sequential(nn.Linear(2 * d, cfg.d_hidden), nn.ReLU(), nn.Linear(cfg.d_hidden, 2))
        self.edge_feat_proj = nn.Linear(1, d) if True else None

    def _pair_feat(self, H_t: torch.Tensor, S_t: torch.Tensor) -> torch.Tensor:
        B, N, _ = H_t.shape
        S = S_t.unsqueeze(1).expand(-1, N, -1)
        node = torch.cat([H_t, S], dim=-1)  # (B, N, d)
        return node

    def forward(
        self,
        H_t: torch.Tensor,
        S_t: torch.Tensor,
        cand_mask: torch.Tensor,
        dist: Optional[torch.Tensor] = None,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        C_t : (B, N, N)  dynamic causal weights  (row i <- column j)
        exist : (B, N, N)  relaxed existence probabilities
        """
        B, N, _ = H_t.shape
        node = self._pair_feat(H_t, S_t)
        src = node.unsqueeze(2).expand(-1, -1, N, -1)
        dst = node.unsqueeze(1).expand(-1, N, -1, -1)
        pair = torch.cat([src, dst], dim=-1)

        logits_exist = self.exist_mlp(pair).squeeze(-1)
        mask = cand_mask.to(logits_exist.dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        logits_exist = logits_exist.masked_fill(mask == 0, -1e9)

        exist = (_gumbel_sigmoid(logits_exist, tau=self.cfg.gumbel_tau, hard=hard)
                 if self.training else torch.sigmoid(logits_exist / max(self.cfg.gumbel_tau, 1e-4)))

        if dist is None:
            dist_feat = torch.zeros(B, N, N, 1, device=H_t.device, dtype=H_t.dtype)
            decay = torch.ones(B, N, N, device=H_t.device, dtype=H_t.dtype)
        else:
            D = dist.to(H_t.dtype)
            if D.dim() == 2:
                D = D.unsqueeze(0).expand(B, -1, -1)
            dist_feat = D.unsqueeze(-1)
            decay = torch.exp(-D / max(float(self.cfg.d_decay), 1.0))

        strength_in = torch.cat([pair, dist_feat], dim=-1)
        strength = torch.sigmoid(self.strength_mlp(strength_in).squeeze(-1)) * decay

        type_logits = self.type_mlp(pair)
        type_w = torch.softmax(type_logits, dim=-1)
        immediate = type_w[..., 0]

        raw = exist * strength * immediate
        C_t = _topk_sparsify(raw, k=self.cfg.k_neighbors, mask=mask)
        C_t = C_t * mask
        return C_t, exist


def _gumbel_sigmoid(logits: torch.Tensor, tau: float = 0.5, hard: bool = False) -> torch.Tensor:
    """Binary Gumbel-Softmax (concrete relaxation of Bernoulli)."""
    u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    g = torch.log(u) - torch.log(1 - u)
    y = torch.sigmoid((logits + g) / max(tau, 1e-4))
    if hard:
        y_hard = (y > 0.5).to(y.dtype)
        y = y_hard.detach() - y.detach() + y
    return y


def _topk_sparsify(scores: torch.Tensor, k: int, mask: torch.Tensor) -> torch.Tensor:
    """Keep top-k incoming neighbors per node (dim=-1)."""
    B, N, _ = scores.shape
    k = min(int(k), N)
    if k >= N:
        return scores
    vals, idx = torch.topk(scores, k=k, dim=-1)
    out = torch.zeros_like(scores)
    out.scatter_(-1, idx, vals)
    return out * mask


__all__ = ["SceneAwareGraphGenerator"]
