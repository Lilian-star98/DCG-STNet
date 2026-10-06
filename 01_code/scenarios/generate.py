"""Appendix B interventions modify exogenous parameters, never observed states."""
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Optional, Sequence
import numpy as np
from .ctm import LinkParams, default_params, simulate, turning_matrix
from .intensity import smooth_beta

KINDS=('accident','capacity_drop','signal_timing','distribution_shift','inflow_spike','weather','school')

@dataclass
class ScenarioSpec:
    kind: str
    nodes: Sequence[int]
    t_start: int
    t_peak: int
    t_end: int
    beta: float=.5
    region_ratio: float=1.0
    extra: Optional[dict]=None

    def validate(self,N,T):
        if self.kind not in KINDS: raise ValueError('Unknown scenario: '+self.kind)
        if not 0<=self.beta<=1 or not 0<self.region_ratio<=1: raise ValueError('Invalid intensity/region ratio')
        if len(self.nodes)==0 or len(set(self.nodes))!=len(self.nodes): raise ValueError('Unique target links required')
        if any(int(n)!=n or n<0 or n>=N for n in self.nodes): raise ValueError('Target link out of bounds')
        smooth_beta(T,self.t_start,self.t_peak,self.t_end)

def clone_params(p):
    return LinkParams(**{f.name:None if getattr(p,f.name) is None else np.asarray(getattr(p,f.name)).copy() for f in fields(p)})

def target_region(A,spec):
    """Regional interventions expand deterministically from targets by hop distance."""
    if spec.kind not in ('distribution_shift','weather'): return np.asarray(spec.nodes,dtype=int)
    N=len(A); count=max(1,int(np.ceil(N*spec.region_ratio)))
    if len(spec.nodes)>=count: return np.asarray(list(spec.nodes)[:count],dtype=int)
    order=list(spec.nodes); seen=set(order); frontier=order.copy()
    graph=(np.asarray(A)>0)|(np.asarray(A).T>0)
    while frontier and len(order)<count:
        nxt=[]
        for u in frontier:
            for v in np.flatnonzero(graph[u]):
                if int(v) not in seen:
                    seen.add(int(v)); order.append(int(v)); nxt.append(int(v))
                    if len(order)==count: break
            if len(order)==count: break
        frontier=nxt
    order.extend(i for i in range(N) if i not in seen)
    return np.asarray(order[:count],dtype=int)

def apply_mapping(base,spec,beta_t,N):
    """Apply the appendix's capacity, lane, free-speed and wave-speed equations."""
    if spec.kind not in KINDS: raise ValueError('Unknown scenario')
    p=clone_params(base); nodes=np.asarray(spec.nodes,dtype=int)
    b=float(spec.beta)*float(beta_t); extra=spec.extra or {}
    p.storage_lanes=np.array(base.storage,copy=True)
    p.capacity_limit=np.array(base.capacity,copy=True)
    if spec.kind=='accident':
        p.n_lanes[nodes]=np.maximum(1,base.n_lanes[nodes]*(1-b))
        p.capacity_limit[nodes]=base.capacity[nodes]*(1-b)
        # Avoid multiplying lane loss by capacity loss a second time.
        p.sat_flow[nodes]=base.capacity[nodes]/p.n_lanes[nodes]
        p.v_free[nodes]=base.v_free[nodes]*(1-.3*b)
    elif spec.kind=='capacity_drop':
        p.capacity_limit[nodes]=base.capacity[nodes]*(1-b)
    elif spec.kind=='weather':
        region=np.arange(N) if spec.region_ratio>=1 else nodes
        dv=float(extra.get('beta_v',spec.beta*extra.get('delta_v',.25)))
        dc=float(extra.get('beta_c',spec.beta*extra.get('delta_c',.20)))
        if not 0<=dv<1 or not 0<=dc<=1: raise ValueError('Invalid weather coefficients')
        # beta_v and beta_c are independent weather severity coefficients.
        p.v_free[region]=base.v_free[region]*(1-dv*beta_t)
        p.capacity_limit[region]=base.capacity[region]*(1-dc*beta_t)
        p.w_cong[region]=base.w_cong[region]*(1-.1*b)
    elif spec.kind=='school':
        p.capacity_limit[nodes]=base.capacity[nodes]*(1-.65*b)
        p.n_lanes[nodes]=np.maximum(1,base.n_lanes[nodes]*(1-.55*b))
        p.sat_flow[nodes]=base.capacity[nodes]/p.n_lanes[nodes]
    p.validate(N)
    return p

def generate_pair(A,spec,T,dt,params=None,demand_base=None,n0=None,queue0=None,
                  seed=42,demand_noise=0.0,sub_dt=30.0):
    """Shared initial states and exogenous demand noise for factual/CF worlds.

    distribution_shift scales selected outgoing movements and reallocates removed
    mass to other existing movements (or boundary exits). Supply region and OD
    region coincide unless extra.od_nodes explicitly identifies demand origins.
    """
    A=np.asarray(A,dtype=float); N=len(A); spec.validate(N,T)
    base_turn,base_exit=turning_matrix(A)
    params=params or default_params(N); params.validate(N)
    extra=spec.extra or {}; rng=np.random.default_rng(seed)
    region=target_region(A,spec)
    alpha=smooth_beta(T,spec.t_start,spec.t_peak,spec.t_end)
    beta=spec.beta*alpha
    if demand_base is None:
        t=np.arange(T)/T
        pulse=np.maximum(np.exp(-.5*((t-.3)/.08)**2),np.exp(-.5*((t-.72)/.10)**2))
        demand_base=np.zeros((T,N),dtype=float)
        sources=np.flatnonzero(np.asarray(A).sum(axis=0)==0)
        if len(sources)==0: sources=np.asarray(spec.nodes,dtype=int)
        demand_base[:,sources]=(2500+1500*pulse)[:,None]
        # E3/E5 demo targets have a local entry; this is recorded, not state editing.
        if spec.kind in ('capacity_drop','school'):
            demand_base[:,list(spec.nodes)]+=(600+900*pulse)[:,None]
    demand_base=np.asarray(demand_base,dtype=float)
    if demand_base.shape!=(T,N) or not np.isfinite(demand_base).all() or (demand_base<0).any():
        raise ValueError('Demand must be nonnegative finite (T,N)')
    if demand_noise<0: raise ValueError('demand_noise must be nonnegative')
    noise=rng.normal(0,demand_noise,(T,N))
    demand=np.maximum(demand_base*(1+noise),0)
    dcf=demand.copy()
    g=np.ones((T,N)); gcf=g.copy()
    cycle=None; cyclecf=None
    turns=np.broadcast_to(base_turn,(T,N,N)).copy()
    turns_cf=turns.copy()
    if spec.kind=='inflow_spike':
        targets=np.asarray(spec.nodes,dtype=int)
        if (A[:,targets].sum(axis=0)>0).any() and not extra.get('allow_internal_source',False):
            raise ValueError('inflow_spike targets must be boundary links or explicit on-ramps')
        dcf[:,targets]*=(1+beta[:,None])
    elif spec.kind=='school':
        surge=float(extra.get('inflow_surge',1.8))
        if surge<1: raise ValueError('school inflow_surge must be >=1')
        dcf[:,list(spec.nodes)]*=1+(surge-1)*alpha[:,None]
    elif spec.kind=='distribution_shift':
        od=np.asarray(extra.get('od_nodes',region),dtype=int)
        if od.ndim!=1 or (od<0).any() or (od>=N).any(): raise ValueError('Invalid OD origins')
        dcf[:,od]*=1+beta[:,None]
        edges=extra.get('turn_edges')
        if edges is None:
            edges=[(int(i),int(np.flatnonzero(base_turn[i]>0)[0])) for i in region if (base_turn[i]>0).any()]
        if not edges: raise ValueError('distribution_shift needs at least one existing movement')
        for i,j in edges:
            if not 0<=i<N or not 0<=j<N or base_turn[i,j]<=0: raise ValueError('Invalid movement')
        selected={}
        for i,j in edges: selected.setdefault(i,set()).add(j)
        for i,js in selected.items():
            js=sorted(js)
            removed=beta*base_turn[i,js].sum()
            turns_cf[:,i,js]*=(1-beta[:,None])
            alternatives=[j for j in np.flatnonzero(base_turn[i]>0) if j not in js]
            if alternatives:
                weights=base_turn[i,alternatives]; weights=weights/weights.sum()
                for j,w in zip(alternatives,weights): turns_cf[:,i,j]+=removed*w
            # With no alternative successor, removed mass becomes a boundary exit.
    elif spec.kind=='signal_timing':
        eta=float(extra.get('eta',0.0)); c=float(extra.get('cycle_sec',90.0))
        green0=float(extra.get('green_ratio',.6))
        if not -.2<=eta<=.2 or c<=0 or not 0<green0<=1: raise ValueError('Invalid signal parameters')
        nodes=list(spec.nodes)
        g[:,nodes]=green0; gcf=g.copy(); gcf[:,nodes]=green0*(1-beta[:,None])
        cycle=np.full((T,N),c); cyclecf=cycle.copy()
        cyclecf[:,nodes]*=1+eta*beta[:,None]
    mapped=ScenarioSpec(spec.kind,region.tolist(),spec.t_start,spec.t_peak,spec.t_end,
                        spec.beta,spec.region_ratio,extra)
    overrides=[apply_mapping(params,mapped,float(a),N) for a in alpha]
    common=dict(T=T,dt=dt,n0=n0,queue0=queue0,sub_dt=sub_dt,
                signal_offset=float(extra.get('signal_offset_sec',0)))
    fact=simulate(A,params,demand=demand,green=g,signal_cycle=cycle,turning=turns,**common)
    cf=simulate(A,params,demand=dcf,param_override=overrides,green=gcf,signal_cycle=cyclecf,turning=turns_cf,**common)
    X=np.stack([fact['speed'],fact['flow']],axis=-1)
    Xcf=np.stack([cf['speed'],cf['flow']],axis=-1)
    return dict(X=X,X_cf=Xcf,effect=Xcf-X,beta=alpha,intervention_beta=beta,
                speed=fact['speed'],speed_cf=cf['speed'],flow=fact['flow'],flow_cf=cf['flow'],
                factual=fact,counterfactual=cf,demand=demand,demand_cf=dcf,
                exogenous_noise=noise,region_nodes=region,turning=turns,turning_cf=turns_cf,
                green=g,green_cf=gcf,cycle=cycle,cycle_cf=cyclecf,
                capacity_cf=np.stack([p.capacity for p in overrides]),
                v_free_cf=np.stack([p.v_free for p in overrides]),
                w_cong_cf=np.stack([p.w_cong for p in overrides]),
                initial_n=fact['initial_n'],initial_queue=fact['initial_queue'])

__all__=['ScenarioSpec','KINDS','clone_params','target_region','generate_pair','apply_mapping']

