"""CaST-style causal / invariant spatiotemporal baseline (compact)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaST(nn.Module):
    """Invariant encoder + environment-specific residual."""

    def __init__(self, num_nodes, in_dim=1, out_dim=1, horizon=12, hidden=64, n_env=4):
        super().__init__()
        self.horizon = horizon
        self.out_dim = out_dim
        self.inv = nn.GRU(in_dim, hidden, batch_first=True)
        self.env = nn.GRU(in_dim, hidden, batch_first=True)
        self.env_proto = nn.Parameter(torch.randn(n_env, hidden))
        self.gcn_w = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, horizon * out_dim))
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 8))
        self.nodevec2 = nn.Parameter(torch.randn(8, num_nodes))

    def forward(self, x, A_static=None):
        B, T, N, Fin = x.shape
        xin = x.permute(0, 2, 1, 3).reshape(B * N, T, Fin)
        h_inv, _ = self.inv(xin)
        h_env, _ = self.env(xin)
        h_inv = h_inv[:, -1]
        h_env = h_env[:, -1]
        # soft environment assignment
        logits = h_env @ self.env_proto.t()
        w = F.softmax(logits, dim=-1)
        e = w @ self.env_proto
        # spatial mix of invariant part
        h_inv = h_inv.view(B, N, -1)
        A = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=-1)
        if A_static is not None:
            A = 0.5 * (A + A_static)
        h_sp = torch.einsum("n m, b m h -> b n h", A, self.gcn_w(h_inv))
        feat = torch.cat([h_sp, e.view(B, N, -1)], dim=-1)
        y = self.head(feat).view(B, N, self.horizon, self.out_dim).permute(0, 2, 1, 3)
        return y
