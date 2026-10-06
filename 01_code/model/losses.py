"""§3.9 Composite multi-objective loss."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from ..config import ModelConfig


class CompositeLoss(nn.Module):
    """L = L_pred + λ_sp L_sparse + λ_sm L_smooth + λ_dw L_dual + L_KL + λ_rs L_res."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        C_seq: torch.Tensor,
        kl: torch.Tensor,
        residuals: Optional[torch.Tensor] = None,
        A: Optional[torch.Tensor] = None,
        y_cf: Optional[torch.Tensor] = None,
        non_int_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.cfg.prediction_loss == "mse":
            L_pred = ((y_pred - y_true) ** 2).mean()
        else:
            L_pred = (y_pred - y_true).abs().mean()

        L_sparse = C_seq.abs().mean()
        if C_seq.size(1) > 1:
            L_smooth = (C_seq[:, 1:] - C_seq[:, :-1]).abs().mean()
        else:
            L_smooth = y_pred.new_zeros(())

        if y_cf is not None:
            diff = (y_pred - y_cf).abs()
            if non_int_mask is not None:
                diff = diff * non_int_mask
                L_dual = diff.sum() / non_int_mask.sum().clamp_min(1.0)
            else:
                L_dual = diff.mean()
        else:
            L_dual = y_pred.new_zeros(())

        if residuals is not None and A is not None:
            L_res = residual_smooth(residuals, A)
        else:
            L_res = y_pred.new_zeros(())

        cfg = self.cfg
        total = (
            L_pred
            + cfg.lambda_sparse * L_sparse
            + cfg.lambda_smooth * L_smooth
            + cfg.lambda_dual * L_dual
            + cfg.lambda_kl * kl
            + cfg.lambda_res_smooth * L_res
        )
        return {
            "L_total": total,
            "L_pred": L_pred,
            "L_sparse": L_sparse,
            "L_smooth": L_smooth,
            "L_dual": L_dual,
            "L_KL": kl,
            "L_res_smooth": L_res,
        }


def residual_smooth(residuals: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """L_res-smooth = mean_{(i,j) in E} ||r_i - r_j||^2."""
    edge = (A > 0).nonzero(as_tuple=False)
    if edge.numel() == 0:
        return residuals.new_zeros(())
    i, j = edge[:, 0], edge[:, 1]
    d = residuals[..., i, :] - residuals[..., j, :]
    return (d ** 2).mean()


__all__ = ["CompositeLoss", "residual_smooth"]
