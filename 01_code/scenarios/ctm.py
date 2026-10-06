"""Conservative Appendix B link CTM: seconds, vehicles/hour and explicit queues."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class LinkParams:
    length: np.ndarray
    v_free: np.ndarray
    w_cong: np.ndarray
    n_lanes: np.ndarray
    sat_flow: np.ndarray
    rho_jam: np.ndarray
    capacity_limit: Optional[np.ndarray] = None
    storage_lanes: Optional[np.ndarray] = None

    @property
    def capacity(self):
        c = self.n_lanes * self.sat_flow
        return c if self.capacity_limit is None else np.minimum(c, self.capacity_limit)

    @property
    def storage(self):
        # Closures restrict discharge without deleting the occupied road space.
        return self.n_lanes if self.storage_lanes is None else self.storage_lanes

    @property
    def rho_crit(self):
        return np.minimum(self.capacity/(3600*self.v_free*self.storage),
                          self.w_cong*self.rho_jam/(self.v_free+self.w_cong))

    def validate(self, N):
        for key in ('length','v_free','w_cong','n_lanes','sat_flow','rho_jam','storage'):
            a=np.asarray(getattr(self,key))
            if a.shape!=(N,) or not np.isfinite(a).all() or (a<=0).any():
                raise ValueError(f'Invalid {key}')
        if not np.isfinite(self.capacity).all() or (self.capacity<0).any():
            raise ValueError('Invalid capacity')

def default_params(N, cfg=None):
    from ..config import ScenarioConfig
    cfg=cfg or ScenarioConfig()
    if N<1: raise ValueError('At least one link is required')
    a=np.ones(N,dtype=float)
    return LinkParams(a*cfg.length_m,a*cfg.v_free,a*cfg.w_cong,a*cfg.n_lanes,a*cfg.sat_flow,a*cfg.rho_jam)

def turning_matrix(A):
    A=np.asarray(A,dtype=float)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or not np.isfinite(A).all() or (A<0).any():
        raise ValueError('A must be finite, square and nonnegative')
    turn=A/np.maximum(A.sum(axis=1),1)[:,None]
    return turn,np.maximum(1-turn.sum(axis=1),0)

def demand_supply(n_veh,params,dt):
    if dt<=0: raise ValueError('dt must be positive')
    n=np.asarray(n_veh,dtype=float)
    rho=n/(params.length*params.storage)
    send=np.minimum.reduce([params.v_free*rho*3600*params.storage,params.capacity,n*3600/dt])
    space=np.maximum(params.rho_jam-rho,0)
    recv=np.minimum.reduce([params.w_cong*space*3600*params.storage,params.capacity,
                            space*params.length*params.storage*3600/dt])
    return send,recv,rho

def triangular_fd(rho,params):
    rho=np.asarray(rho,dtype=float)
    q=np.maximum(0,np.minimum(np.minimum(params.v_free*rho*3600*params.storage,
                   params.w_cong*np.maximum(params.rho_jam-rho,0)*3600*params.storage),params.capacity))
    v=np.divide(q/3600,rho*params.storage,out=np.broadcast_to(params.v_free,rho.shape).copy(),where=rho>1e-12)
    return q,v

def step_ctm(n_veh,demand_in,A,params,dt,green_ratio=None,queue=None):
    """Allocate receiving space proportionally; unadmitted arrivals are queued."""
    n=np.asarray(n_veh,dtype=float); demand=np.asarray(demand_in,dtype=float)
    N=len(n); params.validate(N)
    if dt<=0 or dt>np.min(params.length/np.maximum(params.v_free,params.w_cong))*(1+1e-10):
        raise ValueError('CFL violated; use simulate() for automatic substeps')
    q0=np.zeros(N) if queue is None else np.asarray(queue,dtype=float)
    nmax=params.rho_jam*params.length*params.storage
    if any(a.shape!=(N,) or not np.isfinite(a).all() or (a < -1e-9).any() for a in [n,demand,q0]):
        raise ValueError('Invalid state, queue or demand')
    if (n>nmax+1e-8).any(): raise ValueError('Occupancy exceeds storage; do not clip vehicles')
    send,recv,_=demand_supply(n,params,dt)
    if green_ratio is not None:
        g=np.asarray(green_ratio)
        if g.shape!=(N,) or not np.isfinite(g).all() or ((g<0)|(g>1)).any():
            raise ValueError('Invalid green ratio')
        send=np.minimum(send,params.capacity*g)
    turn,exit_frac=turning_matrix(A)
    if turn.shape!=(N,N): raise ValueError('A/state size mismatch')
    req=send[:,None]*turn
    ext_req=demand+q0*3600/dt
    total=req.sum(axis=0)+ext_req
    ratio=np.divide(recv,total,out=np.ones(N),where=total>recv)
    edge_flow=req*ratio[None,:]
    admitted=ext_req*ratio
    exit_flow=send*exit_frac
    inflow=edge_flow.sum(axis=0)+admitted
    outflow=edge_flow.sum(axis=1)+exit_flow
    nn=n+dt/3600*(inflow-outflow)
    qq=q0+dt/3600*(demand-admitted)
    if (nn < -1e-8).any() or (nn>nmax+1e-8).any() or (qq < -1e-8).any():
        raise RuntimeError('CTM bounds violated')
    nn=np.maximum(nn,0); qq=np.maximum(qq,0)  # floating point roundoff only
    rho=nn/(params.length*params.storage)
    q,v=triangular_fd(rho,params)
    return dict(n=nn,q=q,v=v,rho=rho,inflow=inflow,outflow=outflow,queue=qq,
                external_admitted=admitted,external_arrivals=demand,exit_flow=exit_flow,
                edge_flow=edge_flow,conservation=nn-n-dt/3600*(inflow-outflow))

def signal_open_fraction(start,duration,green,cycle,offset=0.0):
    """Exact fraction of a reporting substep that falls inside a green phase."""
    cycle=np.asarray(cycle,dtype=float); green=np.asarray(green,dtype=float)
    def integral(t):
        x=np.asarray(t)-offset; k=np.floor(x/cycle)
        return k*cycle*green+np.minimum(x-k*cycle,cycle*green)
    return np.clip((integral(start+duration)-integral(start))/duration,0,1)

def simulate(A,params,T,dt,demand,param_override=None,green=None,n0=None,sub_dt=30.0,
             queue0=None,turning=None,signal_cycle=None,signal_offset=0.0):
    """CFL-safe simulation with interval-mean actual fluxes and end-state FD flow.

    A[i,j] is upstream i to downstream j. Rows below one retain boundary exits.
    turning has shape (T,N,N); cycle seconds and green have shape (T,N).
    """
    A=np.asarray(A,dtype=float); turning_matrix(A); N=A.shape[0]; params.validate(N)
    demand=np.asarray(demand,dtype=float)
    if T<1 or dt<=0 or sub_dt<=0 or demand.shape!=(T,N) or not np.isfinite(demand).all() or (demand<0).any():
        raise ValueError('Invalid simulation dimensions/time/demand')
    for val,shape,name in [(green,(T,N),'green'),(turning,(T,N,N),'turning'),(signal_cycle,(T,N),'cycle')]:
        if val is not None and (np.asarray(val).shape!=shape or not np.isfinite(val).all()):
            raise ValueError('Invalid '+name)
    if signal_cycle is not None and ((np.asarray(signal_cycle)<=0).any() or green is None):
        raise ValueError('Positive cycles and green ratios required')
    if param_override is not None and len(param_override)!=T: raise ValueError('Wrong parameter timeline')
    n=params.rho_crit*params.length*params.storage*.15 if n0 is None else np.array(n0,dtype=float)
    queue=np.zeros(N) if queue0 is None else np.array(queue0,dtype=float)
    ninit,qinit=n.copy(),queue.copy()
    names=['speed','flow','rho','n','queue','inflow','outflow','external_admitted','external_arrivals',
           'exit_flow','conservation','queue_conservation']
    result={k:[] for k in names}; substeps=[]
    for t in range(T):
        p=param_override[t] if param_override is not None else params; p.validate(N)
        nsub=max(1,int(np.ceil(dt/min(sub_dt,float(np.min(p.length/np.maximum(p.v_free,p.w_cong)))))))
        step=dt/nsub; nb,qb=n.copy(),queue.copy()
        sums={k:np.zeros(N) for k in ['inflow','outflow','external_admitted','exit_flow']}
        for j in range(nsub):
            g=None if green is None else green[t]
            if signal_cycle is not None: g=signal_open_fraction(t*dt+j*step,step,g,signal_cycle[t],signal_offset)
            state=step_ctm(n,demand[t],A if turning is None else turning[t],p,step,g,queue)
            n,queue=state['n'],state['queue']
            for k in sums: sums[k]+=state[k]/nsub
        values=dict(speed=state['v']*3.6,flow=state['q'],rho=state['rho'],n=n,queue=queue,
                    external_arrivals=demand[t],**sums)
        values['conservation']=n-nb-dt/3600*(sums['inflow']-sums['outflow'])
        values['queue_conservation']=queue-qb-dt/3600*(demand[t]-sums['external_admitted'])
        for k in names: result[k].append(values[k].copy())
        substeps.append(nsub)
    result={k:np.stack(v) for k,v in result.items()}
    result.update(initial_n=ninit,initial_queue=qinit,substeps=np.array(substeps))
    return result

def physics_violation_rate(speed_kmh,flow,params,eps=1e-6,rho=None):
    """Fraction in [0,1]; pass measured CTM density to cover stopped queues."""
    v=np.asarray(speed_kmh)/3.6; flow=np.asarray(flow)
    if rho is None: rho=flow/(3600*np.maximum(v,eps)*params.storage)
    bad=(~np.isfinite(v))|(~np.isfinite(flow))|(~np.isfinite(rho))
    bad|=(v < -eps)|(v>params.v_free+eps)|(rho < -eps)|(rho>params.rho_jam+eps)
    bad|=(flow < -eps)|(flow>params.capacity+eps)
    return float(np.mean(bad))

__all__=['LinkParams','default_params','turning_matrix','demand_supply','triangular_fd',
         'step_ctm','simulate','signal_open_fraction','physics_violation_rate']

