"""Differentiable CTM feasibility layer for speed/flow neural proposals.

The neural flow is a desired sending rate, not an unconditionally admissible
flux. Receiving-space allocation, source queues and FD readout are enforced.
This concrete projection is an implementation choice: the paper does not give
an explicit neural-to-CTM interface.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class PhysicsContext:
    # A is upstream -> downstream, unlike neural aggregation adjacency.
    A: torch.Tensor
    length: torch.Tensor
    storage_lanes: torch.Tensor
    rho_jam: torch.Tensor
    v_free: torch.Tensor  # (T,N) or (N,)
    w_cong: torch.Tensor
    capacity: torch.Tensor
    demand: torch.Tensor  # (B,T,N) veh/h
    initial_n: torch.Tensor  # (B,N) veh
    dt_sec: float
    mean: torch.Tensor  # (N,2) speed/flow normalization
    std: torch.Tensor
    green: torch.Tensor | None = None  # effective fraction (B,T,N)
    initial_queue: torch.Tensor | None = None
    sub_dt_sec: float = 30.0
    cycle: torch.Tensor | None = None  # (T,N) seconds, optional phase-resolved signals
    time_origin_sec: float = 0.0

    def validate(self, B, T, N):
        if self.dt_sec<=0 or self.sub_dt_sec<=0:raise ValueError('Invalid physics dt')
        for key in ['length','storage_lanes','rho_jam']:
            a=getattr(self,key)
            if a.shape!=(N,) or not torch.isfinite(a).all() or (a<=0).any():raise ValueError('Invalid physics '+key)
        for key in ['v_free','w_cong','capacity']:
            a=getattr(self,key)
            if a.shape not in [(N,),(T,N)] or not torch.isfinite(a).all() or (a<0).any():raise ValueError('Invalid physics '+key)
        if self.A.shape not in [(N,N),(T,N,N)] or (self.A<0).any() or not torch.isfinite(self.A).all():raise ValueError('Invalid physical A')
        if self.demand.shape!=(B,T,N) or (self.demand<0).any() or not torch.isfinite(self.demand).all():raise ValueError('Invalid external demand')
        if self.initial_n.shape!=(B,N) or (self.initial_n<0).any() or not torch.isfinite(self.initial_n).all():raise ValueError('Invalid initial vehicles')
        if (self.initial_n>self.length*self.storage_lanes*self.rho_jam+1e-6).any():raise ValueError('Initial vehicles exceed storage')
        if self.mean.shape!=(N,2) or self.std.shape!=(N,2) or (self.std<=0).any():raise ValueError('Physical layer requires speed/flow scaler')


class CTMProjector:
    """Stateful within one rollout; create separately for each factual/CF world."""
    def __init__(self, context: PhysicsContext, reference, T):
        self.c=context
        B,N,_=reference.shape
        context.validate(B,T,N)
        self.n=context.initial_n.to(reference)
        self.queue=torch.zeros_like(self.n) if context.initial_queue is None else context.initial_queue.to(reference)
        self.length=context.length.to(reference);self.storage=context.storage_lanes.to(reference)
        self.jam=context.rho_jam.to(reference)
        A=context.A.to(reference)
        if A.ndim==3:A=A[0]
        self.turn=A/A.sum(-1,keepdim=True).clamp_min(1)
        self.exit=(1-self.turn.sum(-1)).clamp_min(0)

    def __call__(self, proposal,t):
        import math
        c=self.c
        def at(name):
            a=getattr(c,name).to(proposal)
            return a[t] if a.ndim==2 else a
        vf,w,cap=at('v_free'),at('w_cong'),at('capacity')
        mean,std=c.mean.to(proposal),c.std.to(proposal)
        physical=proposal*std+mean
        desired=physical[...,1].clamp_min(0)
        demand=c.demand[:,t].to(proposal)
        g=torch.ones_like(demand) if c.green is None else c.green[:,t].to(proposal)
        if ((g<0)|(g>1)).any():raise ValueError('Invalid physical green ratio')
        cfl=float((self.length/torch.maximum(vf,w).clamp_min(1e-8)).min().detach())
        count=max(1,math.ceil(c.dt_sec/min(c.sub_dt_sec,cfl)))
        dt=c.dt_sec/count/3600
        n_before,q_before=self.n,self.queue
        exit_total=torch.zeros_like(self.n);admitted_total=torch.zeros_like(self.n)
        inflow_total=torch.zeros_like(self.n);outflow_total=torch.zeros_like(self.n)
        if c.A.ndim==3:
            A=c.A[t].to(proposal);self.turn=A/A.sum(-1,keepdim=True).clamp_min(1)
            self.exit=(1-self.turn.sum(-1)).clamp_min(0)
        for sub in range(count):
            gate=g
            if c.cycle is not None:
                cycle=c.cycle[t].to(proposal)
                if (cycle<=0).any():raise ValueError('Invalid cycle')
                step_sec=dt*3600
                start=c.time_origin_sec+t*c.dt_sec+sub*step_sec
                def integral(x):
                    k=torch.floor(x/cycle)
                    return k*cycle*g+torch.minimum(x-k*cycle,cycle*g)
                gate=((integral(start+step_sec)-integral(start))/step_sec).clamp(0,1)
            rho=self.n/(self.length*self.storage)
            send=torch.minimum(torch.minimum(vf*rho*3600*self.storage,cap*gate),self.n/dt)
            send=torch.minimum(send,desired)
            space=(self.jam-rho).clamp_min(0)
            recv=torch.minimum(torch.minimum(w*space*3600*self.storage,cap),space*self.length*self.storage/dt)
            req=send[:,:,None]*self.turn
            external=demand+self.queue/dt
            total=req.sum(dim=1)+external
            ratio=torch.minimum(torch.ones_like(total),recv/total.clamp_min(1e-8))
            edge=req*ratio[:,None,:];admit=external*ratio;exits=send*self.exit
            incoming=edge.sum(dim=1)+admit;outgoing=edge.sum(dim=2)+exits
            self.n=self.n+dt*(incoming-outgoing)
            self.queue=self.queue+dt*(demand-admit)
            exit_total=exit_total+exits/count;admitted_total=admitted_total+admit/count
            inflow_total=inflow_total+incoming/count;outflow_total=outflow_total+outgoing/count
        rho=self.n/(self.length*self.storage)
        q=torch.minimum(torch.minimum(vf*rho*3600*self.storage,w*(self.jam-rho).clamp_min(0)*3600*self.storage),cap)
        v=torch.where(rho>1e-8,q/(3600*rho*self.storage).clamp_min(1e-8),vf)*3.6
        y=torch.stack([v,q],dim=-1)
        diagnostics=dict(n=self.n,queue=self.queue,rho=rho,inflow=inflow_total,outflow=outflow_total,
                         exit_flow=exit_total,admitted=admitted_total,
                         conservation=self.n-n_before-c.dt_sec/3600*(inflow_total-outflow_total),
                         queue_conservation=self.queue-q_before-c.dt_sec/3600*(demand-admitted_total))
        return (y-mean)/std,diagnostics
