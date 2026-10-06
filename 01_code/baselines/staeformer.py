"""STAEformer-style spatiotemporal adaptive embedding transformer (compact, scalable)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class STAEformer(nn.Module):
    """Adaptive node/time embeddings + temporal Transformer per node + light spatial mix."""

    def __init__(self, num_nodes, in_dim=1, out_dim=1, horizon=12, hist=12, hidden=64, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.out_dim = out_dim
        self.in_proj = nn.Linear(in_dim, hidden)
        self.node_emb = nn.Parameter(torch.randn(num_nodes, hidden) * 0.02)
        self.tod_emb = nn.Parameter(torch.randn(max(hist, 1), hidden) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 8) * 0.1)
        self.nodevec2 = nn.Parameter(torch.randn(8, num_nodes) * 0.1)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon * out_dim))

    def forward(self, x, A_static=None):
        # x: (B,T,N,F)
        B, T, N, _ = x.shape
        h = self.in_proj(x) + self.node_emb.view(1, 1, N, -1)
        te = self.tod_emb
        if te.size(0) >= T:
            te = te[:T]
        else:
            te = torch.cat([te, te[-1:].expand(T - te.size(0), -1)], dim=0)
        h = h + te.view(1, T, 1, -1)
        # temporal tokens per node: (B*N, T, H)
        ht = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        ht = self.encoder(ht)[:, -1].view(B, N, -1)
        A = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=-1)
        if A_static is not None:
            A = 0.5 * (A + A_static)
        ht = torch.einsum("n m, b m h -> b n h", A, ht)
        y = self.head(ht).view(B, N, self.horizon, self.out_dim).permute(0, 2, 1, 3)
        return y
