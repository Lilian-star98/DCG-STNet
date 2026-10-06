"""STCausalNet / DCG-STNet: wire §3.4–§3.9 into one module."""

from __future__ import annotations

from typing import Dict, Optional
from dataclasses import replace

import torch
from torch import nn

from ..config import ModelConfig
from .counterfactual import InterventionSpec, apply_control, apply_edge_mask, apply_exog
from .dynamic_causal_graph import SceneAwareGraphGenerator
from .latent_macro import LatentMacroInferencer
from .losses import CompositeLoss
from .node_dynamics import NodeDynamics, OutputDecoder
from .scenario_encoder import ScenarioEncoder


class STCausalNet(nn.Module):
    """Scene-conditioned dynamic causal ST-GNN with dual-world rollouts.

    Ablation flags (paper §4.8):
      use_scene, use_macro, use_dynamic, use_causal_reg
    """

    def __init__(self, cfg: ModelConfig, use_scene=True, use_macro=True, use_dynamic=True):
        super().__init__()
        self.cfg = cfg
        self.use_scene = use_scene
        self.use_macro = use_macro
        self.use_dynamic = use_dynamic
        self.use_physics = cfg.use_physics
        self.encoder = ScenarioEncoder(cfg)
        self.latent = LatentMacroInferencer(cfg)
        self.graph = SceneAwareGraphGenerator(cfg)
        self.dynamics = NodeDynamics(cfg)
        self.decoder = OutputDecoder(cfg)
        self.loss_fn = CompositeLoss(cfg)
        self.h0 = nn.Parameter(torch.zeros(1, 1, cfg.d_hidden))

    def _rollout(
        self,
        X, U, Zloc, Zg, A, cand, dist, L_seq, W=None, spec: Optional[InterventionSpec] = None,
        physics=None, factual_noise=None,
    ) -> Dict[str, torch.Tensor]:
        B, T, N, _ = X.shape
        H = self.h0.expand(B, N, -1).contiguous()
        H_list, Y_list, C_list = [], [], []
        steps = T - 1 if T > 1 else T
        projector = None
        physics_rows = []
        if physics is not None and self.use_physics:
            from .physics import CTMProjector
            if self.cfg.d_y != 2:
                raise ValueError('CTM readout requires speed and flow in that order')
            projector = CTMProjector(physics, X[:, 0], steps)
        U_action = apply_control(U, spec) if spec is not None else U
        Z_action = apply_exog(Zloc, spec) if spec is not None else Zloc
        for t in range(steps):
            Xt, Ut, Zt = X[:, t], U[:, t], Zloc[:, t]
            if projector is not None and t > 0:
                Xt = Y_list[-1]
            if spec is not None and t > spec.T0:
                if self.cfg.d_x != self.cfg.d_y:
                    raise ValueError('Autoregressive counterfactual requires matching state/readout dimensions')
                Xt = Y_list[-1]
            Zgt = Zg[:, t]
            Lt = L_seq[:, t] if self.use_macro else torch.zeros(B, self.cfg.d_latent, device=X.device)
            Wt = W[:, t] if W is not None else None
            if spec is not None and t >= spec.T0:
                Ut = U_action[:, t]
                Zt = Z_action[:, t]
                if spec.A_prime is not None:
                    A_use = spec.A_prime
                else:
                    A_use = A
            else:
                A_use = A
            if self.use_scene:
                S = self.encoder(Ut, Zt, Zgt, Wt)
            else:
                S = torch.zeros(B, self.cfg.d_scenario, device=X.device)
            if self.use_dynamic:
                C, _ = self.graph(H, S, cand, dist)
            else:
                C = A_use.to(X.dtype).unsqueeze(0).expand(B, -1, -1)
            if spec is not None:
                C = apply_edge_mask(C, spec, t)
            H = self.dynamics(H, Xt, Ut, Zt, Zgt, Lt, C, A_use)
            Y = self.decoder(H)
            if factual_noise is not None:
                Y = Y + factual_noise[:, t]
            if projector is not None:
                Y, diagnostic = projector(Y, t)
                physics_rows.append(diagnostic)
            H_list.append(H)
            Y_list.append(Y)
            C_list.append(C)
        result = {
            "H": torch.stack(H_list, dim=1),
            "Y": torch.stack(Y_list, dim=1),
            "C": torch.stack(C_list, dim=1),
        }
        if physics_rows:
            result['physics'] = {k: torch.stack([r[k] for r in physics_rows], dim=1) for k in physics_rows[0]}
        return result

    def forward(
        self,
        X, U, Zloc, Zg, A, cand,
        dist=None, W=None, y_true=None, spec: Optional[InterventionSpec] = None, non_int_mask=None, physics=None,
    ) -> Dict[str, torch.Tensor]:
        if self.use_macro:
            L_seq, lat = self.latent(X, Zg)
            kl = lat["kl"]
        else:
            B, T = X.shape[:2]
            L_seq = torch.zeros(B, T, self.cfg.d_latent, device=X.device)
            kl = X.new_zeros(())

        # Common random numbers keep graph sampling identical between worlds.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if X.is_cuda else None
        fact = self._rollout(X, U, Zloc, Zg, A, cand, dist, L_seq, W=W, spec=None, physics=physics)
        out: Dict[str, torch.Tensor] = {"Y_F": fact["Y"], "H_F": fact["H"], "C_F": fact["C"], "L_seq": L_seq}
        if 'physics' in fact:
            out['physics_F'] = fact['physics']

        y_cf = None
        if spec is not None:
            # Retrospective additive-noise abduction. With CTM projection the
            # observed physical states are already constrained; no residual may
            # be added after projection because it would break conservation.
            residual = ((X[:, 1:] - fact['Y']).detach()
                        if (physics is None or not self.use_physics) and X[:, 1:].shape == fact['Y'].shape else None)
            if residual is not None:
                out['Y_F_mean'] = fact['Y']
                out['Y_F'] = fact['Y'] + residual
                out['abducted_noise'] = residual
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if X.is_cuda else []):
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                cf = self._rollout(X, U, Zloc, Zg, A, cand, dist, L_seq, W=W, spec=spec,
                                   physics=spec.physics_prime if spec.physics_prime is not None else physics,
                                   factual_noise=residual)
            out.update({"Y_CF": cf["Y"], "H_CF": cf["H"], "C_CF": cf["C"]})
            y_cf = cf["Y"]
            if residual is not None:
                y_cf = y_cf - residual
            if 'physics' in cf:
                out['physics_CF'] = cf['physics']

        if y_true is not None:
            target = y_true
            if target.shape != fact["Y"].shape:
                raise ValueError('Target shape must match model output; future labels must not be silently replaced')
            residuals = target - fact["Y"] if target.shape == fact["Y"].shape else None
            losses = self.loss_fn(
                fact["Y"], target, fact["C"], kl, residuals=residuals, A=A, y_cf=y_cf, non_int_mask=non_int_mask,
            )
            out.update(losses)
        else:
            out["L_KL"] = kl
        return out

    @classmethod
    def ablation(cls, cfg: ModelConfig, variant: str) -> "STCausalNet":
        cfg = replace(cfg)
        flags = {"use_scene": True, "use_macro": True, "use_dynamic": True}
        if variant == "w/o Scene":
            flags["use_scene"] = False
        elif variant == "w/o Macro":
            flags["use_macro"] = False
        elif variant == "w/o Dynamic":
            flags["use_dynamic"] = False
        elif variant == "w/o Causal-Reg":
            cfg.lambda_sparse = 0.0
            cfg.lambda_smooth = 0.0
            cfg.lambda_dual = 0.0
        elif variant == "w/o Physics":
            cfg.use_physics = False
        elif variant != "full":
            raise ValueError('Unknown ablation: ' + variant)
        return cls(cfg, **flags)


__all__ = ["STCausalNet"]
