"""Measured fundamental-diagram calibration and held-out CTM validation.

No speed-derived pseudo-flow or guessed topology is accepted as measured data.
Input NPZ uses SI units: density veh/m/lane, speed m/s, flow veh/h total.
Network validation requires boundary arrivals, actual turn fractions and initial
state; a distance adjacency is never silently converted to turning fractions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from paper.scenarios.ctm import LinkParams,simulate
from .common import digest,save


def diagram(k,v,w,jam,lanes):
    return np.maximum(0,np.minimum(v*k,w*(jam-k)))*lanes*3600


def fit_fd(k,q,lanes,bounds):
    good=np.isfinite(k)&np.isfinite(q)&(k>=0)&(q>=0)
    k=k[good];q=q[good]
    if len(k)<40:raise ValueError('Insufficient observed calibration samples')
    low,high=np.array(bounds['lower']),np.array(bounds['upper'])
    # Bounds must be supplied before viewing validation/test performance.
    if low.shape!=(3,) or high.shape!=(3,) or (low<=0).any() or (high<=low).any():raise ValueError('Invalid SI parameter bounds')
    scale=max(float(np.quantile(q,.9)),1.)
    candidates=[]
    for alpha in [.2,.5,.8]:
        r=least_squares(lambda p:(diagram(k,*p,lanes)-q)/scale,low+alpha*(high-low),
                        bounds=(low,high),loss='soft_l1',f_scale=.1,max_nfev=1000)
        candidates.append(r)
    result=min(candidates,key=lambda r:r.cost);v,w,j=result.x;critical=w*j/(v+w)
    rank=int(np.linalg.matrix_rank(result.jac))
    free=int((k<critical).sum());congested=int((k>critical).sum())
    if rank<3 or min(free,congested)<10:
        raise ValueError('Parameters not identifiable: both free/congested regimes and full Jacobian rank required')
    return dict(v_free_mps=float(v),w_cong_mps=float(w),rho_jam_veh_m_lane=float(j),
                capacity_veh_h=float(v*w*j/(v+w)*lanes*3600),jacobian_rank=rank,
                jacobian_condition=float(np.linalg.cond(result.jac)),free_samples=free,congested_samples=congested,
                at_bound=bool(np.any(np.isclose(result.x,low,rtol=.01)|np.isclose(result.x,high,rtol=.01))))


def score(y,p):
    ok=np.isfinite(y)&np.isfinite(p)
    if not ok.any():raise ValueError('No held-out observations')
    return dict(count=int(ok.sum()),MAE=float(abs(y[ok]-p[ok]).mean()),RMSE=float(np.sqrt(np.mean((y[ok]-p[ok])**2))))


def calibrate(npz,metadata,out):
    meta=json.loads(Path(metadata).read_text(encoding='utf-8'))
    needed=['flow_source','density_source','road_geometry_source','split_dates','bounds','units']
    if any(not meta.get(k) for k in needed):raise ValueError('Missing provenance/calibration contract')
    if meta.get('flow_provenance')!='measured' or meta.get('density_provenance')!='measured':
        raise ValueError('This calibrator requires independently measured flow and density; no speed proxy')
    if meta['units']!={'flow':'veh/h','density':'veh/m/lane','speed':'m/s','length':'m'}:
        raise ValueError('Convert and document units explicitly')
    with np.load(npz,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    k=d['density'];q=d['flow'];lanes=d['lanes'];T,N=k.shape
    if q.shape!=k.shape or lanes.shape!=(N,):raise ValueError('Shape mismatch')
    dates=np.asarray(d['dates']).astype(str)
    if dates.shape!=(T,) or len(meta.get('road_ids',[]))!=N or list(map(str,d['road_ids']))!=meta['road_ids']:
        raise ValueError('Explicit aligned dates and ordered road IDs required')
    groups={key:set(meta['split_dates'][key]) for key in ['train','validation','test']}
    if any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Calibration/validation/test dates overlap')
    masks={key:np.isin(dates,list(vals)) for key,vals in groups.items()}
    if any(not m.any() for m in masks.values()):raise ValueError('An independent split is empty')
    params=[];held={}
    for n in range(N):
        params.append(fit_fd(k[masks['train'],n],q[masks['train'],n],lanes[n],meta['bounds']))
    pred=np.stack([diagram(k[:,n],p['v_free_mps'],p['w_cong_mps'],p['rho_jam_veh_m_lane'],lanes[n]) for n,p in enumerate(params)],1)
    for split in ['validation','test']:held[split]=score(q[masks[split]],pred[masks[split]])
    # Day-level block bootstrap preserves within-day dependence.
    rng=np.random.default_rng(meta.get('seed',0));boot=[];train_days=sorted(groups['train'])
    for rep in range(meta.get('bootstrap_replicates',100)):
        sampled=rng.choice(train_days,len(train_days),replace=True)
        indices=np.concatenate([np.where(dates==day)[0] for day in sampled]);row=[]
        try:
            for n in range(N):row.append(fit_fd(k[indices,n],q[indices,n],lanes[n],meta['bounds']))
            boot.append(row)
        except ValueError:continue
    intervals=[]
    for n in range(N):
        intervals.append({key:np.quantile([r[n][key] for r in boot],[.025,.975]).tolist() if len(boot)>=20 else None
                          for key in ['v_free_mps','w_cong_mps','rho_jam_veh_m_lane','capacity_veh_h']})
    result=dict(status='fd_calibrated',network_calibrated=False,parameters=params,intervals_95=intervals,
                bootstrap_success=len(boot),validation=held,input_sha256=digest(npz),contract=meta,
                limitations=['FD residual validation alone does not establish network or intervention validity'])
    save(Path(out)/'calibration.json',result)
    return result


def validate_network(episode,metadata,calibration,out):
    meta=json.loads(Path(metadata).read_text(encoding='utf-8'))
    fit=json.loads(Path(calibration).read_text(encoding='utf-8'))
    for key in ['turn_counts_source','boundary_arrivals_source','initial_state_source','speed_source','flow_source']:
        if not meta.get(key):raise ValueError('Missing '+key)
    if meta.get('episode_date') not in fit['contract']['split_dates']['test']:
        raise ValueError('Network episode must be an untouched held-out date')
    with np.load(episode,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    A=d['turning'];N=len(d['length']);p=fit['parameters']
    if list(map(str,d['road_ids']))!=fit['contract']['road_ids']:
        raise ValueError('Network road order does not match calibrated parameters')
    if (A.sum(-1)>1+1e-8).any() or (A<0).any():raise ValueError('Actual turn fractions required')
    if len(p)!=N:raise ValueError('Road ordering must agree with calibration')
    arr=lambda name:np.array([r[name] for r in p])
    params=LinkParams(d['length'],arr('v_free_mps'),arr('w_cong_mps'),d['lanes'],
                     arr('capacity_veh_h')/d['lanes'],arr('rho_jam_veh_m_lane'))
    r=simulate(A,params,len(d['demand']),meta['dt_sec'],d['demand'],n0=d['initial_n'],queue0=d['initial_queue'],green=d.get('green'))
    result=dict(status='held_out_network_evaluated',episode_id=meta['episode_id'],
                speed_mps=score(d['speed_mps'],r['speed']/3.6),
                measured_interval_outflow=score(d['outflow_veh_h'],r['outflow']),
                max_vehicle_residual=float(abs(r['conservation']).max()),
                max_queue_residual=float(abs(r['queue_conservation']).max()),
                parameters_sha256=digest(calibration),episode_sha256=digest(episode),
                intervention_effect_verified=False)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'simulation.npz',**r)
    save(out/'network_validation.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--metadata',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--calibration',type=Path)
    a=ap.parse_args()
    if a.calibration:validate_network(a.data,a.metadata,a.calibration,a.out)
    else:calibrate(a.data,a.metadata,a.out)

if __name__=='__main__':main()
