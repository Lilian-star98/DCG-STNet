"""PDFormer-style progressive spatiotemporal transformer (compact)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PDFormer(nn.Module):
    def __init__(self, num_nodes, in_dim=1, out_dim=1, horizon=12, hidden=64, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.out_dim = out_dim
        self.in_proj = nn.Linear(in_dim, hidden)
        self.t_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=heads, dim_feedforward=hidden * 2, dropout=dropout, batch_first=True
                )
                for _ in range(layers)
            ]
        )
        self.s_blocks = nn.ModuleList([nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True) for _ in range(layers)])
        self.s_ff = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden)) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon * out_dim))
        self.node_emb = nn.Parameter(torch.randn(num_nodes, hidden) * 0.01)

    def forward(self, x, A_static=None):
        B, T, N, _ = x.shape
        h = self.in_proj(x) + self.node_emb.view(1, 1, N, -1)
        ht = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        for blk in self.t_blocks:
            ht = blk(ht)
        h = ht.reshape(B, N, T, -1).permute(0, 2, 1, 3)
        z = 0.5 * (h[:, -1] + h.mean(dim=1))
        for attn, ff, nm in zip(self.s_blocks, self.s_ff, self.norm):
            a, _ = attn(z, z, z, need_weights=False)
            z = nm(z + a)
            z = nm(z + ff(z))
        y = self.head(z).view(B, N, self.horizon, self.out_dim).permute(0, 2, 1, 3)
        return y
