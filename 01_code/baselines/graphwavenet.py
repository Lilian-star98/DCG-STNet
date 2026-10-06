"""Graph WaveNet (IJCAI 2019) — compact faithful implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import GraphConv, TemporalConv


class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, in_dim=1, out_dim=1, horizon=12, hidden=32, layers=4, dropout=0.3):
        super().__init__()
        self.horizon = horizon
        self.start = nn.Conv2d(in_dim, hidden, kernel_size=(1, 1))
        self.nodes = num_nodes
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10))
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes))
        self.tconvs = nn.ModuleList()
        self.sconvs = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.bn = nn.ModuleList()
        dilation = 1
        for _ in range(layers):
            self.tconvs.append(TemporalConv(hidden, hidden, kernel=2, dilation=dilation))
            self.gates.append(TemporalConv(hidden, hidden, kernel=2, dilation=dilation))
            self.sconvs.append(GraphConv(hidden, hidden))
            self.skips.append(nn.Conv2d(hidden, hidden, kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(hidden))
            dilation *= 2
        self.end1 = nn.Conv2d(hidden, hidden, kernel_size=(1, 1))
        self.end2 = nn.Conv2d(hidden, horizon * out_dim, kernel_size=(1, 1))
        self.dropout = dropout
        self.out_dim = out_dim

    def forward(self, x, A_static=None):
        # x: (B, T, N, F) -> (B, F, N, T)
        x = x.permute(0, 3, 2, 1)
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        if A_static is not None:
            A = 0.5 * (A_static + adp)
        else:
            A = adp
        x = self.start(x)
        skip = 0
        for tconv, gate, sconv, sk, bn in zip(self.tconvs, self.gates, self.sconvs, self.skips, self.bn):
            residual = x
            filter_ = torch.tanh(tconv(x))
            gate_ = torch.sigmoid(gate(x))
            x = filter_ * gate_
            x = F.dropout(x, self.dropout, training=self.training)
            x = sconv(x, A)
            s = sk(x)
            # align temporal length
            if isinstance(skip, int):
                skip = s[..., -s.size(-1) :]
            else:
                skip = skip[..., -s.size(-1) :] + s
            x = x + residual[..., -x.size(-1) :]
            x = bn(x)
        x = F.relu(skip)
        x = F.relu(self.end1(x))
        x = self.end2(x)  # (B, H*F, N, T')
        x = x[..., -1]  # (B, H*F, N)
        B, _, N = x.shape
        x = x.view(B, self.horizon, self.out_dim, N).permute(0, 1, 3, 2)  # (B,H,N,F)
        return x
