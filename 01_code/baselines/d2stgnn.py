"""D2STGNN-style decoupled dynamic spatial-temporal GNN (compact)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class D2STGNN(nn.Module):
    def __init__(self, num_nodes, in_dim=1, out_dim=1, horizon=12, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.out_dim = out_dim
        self.embed = nn.Linear(in_dim, hidden)
        self.temp = nn.GRU(hidden, hidden, batch_first=True)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 12))
        self.nodevec2 = nn.Parameter(torch.randn(12, num_nodes))
        self.dyn_mlps = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.fuse = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon * out_dim))
        self.dropout = dropout

    def forward(self, x, A_static=None):
        # x: (B,T,N,F)
        B, T, N, _ = x.shape
        h = self.embed(x)  # (B,T,N,H)
        # temporal per node
        ht = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        ht, _ = self.temp(ht)
        ht = ht.reshape(B, N, T, -1).permute(0, 2, 1, 3)  # (B,T,N,H)
        adp = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=-1)
        A = adp if A_static is None else 0.5 * (A_static + adp)
        hs = ht
        for mlp in self.dyn_mlps:
            msg = torch.einsum("n m, b t m h -> b t n h", A, hs)
            hs = hs + F.dropout(mlp(msg), self.dropout, training=self.training)
        # decouple residual temporal evolution
        z = (ht + hs).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        z, _ = self.fuse(z)
        z = z[:, -1]
        y = self.head(z).view(B, N, self.horizon, self.out_dim).permute(0, 2, 1, 3)
        return y
