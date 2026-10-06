"""Shared utilities for baseline STGNNs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(A: np.ndarray) -> torch.Tensor:
    A = np.asarray(A, dtype=np.float32)
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(axis=1)
    d_inv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    A_hat = d_inv[:, None] * A * d_inv[None, :]
    return torch.from_numpy(A_hat.astype(np.float32))


class TemporalConv(nn.Module):
    def __init__(self, c_in, c_out, kernel=2, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, kernel), dilation=(1, dilation))

    def forward(self, x):
        # x: (B, C, N, T) — causal pad on time
        x = F.pad(x, (self.pad, 0, 0, 0))
        return self.conv(x)


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x, A):
        # x: (B,C,N,T), A: (N,N)
        x = torch.einsum("n m, b c m t -> b c n t", A, x)
        return self.mlp(x)


def masked_mae(pred, true, null_val=0.0):
    mask = (true != null_val).float()
    mask /= torch.clamp(mask.mean(), min=1e-6)
    return (torch.abs(pred - true) * mask).mean()
