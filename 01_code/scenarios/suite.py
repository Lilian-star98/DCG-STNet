"""Generate and validate every Appendix B intervention plus E5 school scenarios.

Example: python -m paper.scenarios.suite --out results/appendix_b_v2
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .ctm import default_params
from .generate import ScenarioSpec, generate_pair


def scenario_catalog(T=72):
    start,peak,end=T//6,T//3,2*T//3
    def spec(kind,nodes,**kw):
        return ScenarioSpec(kind,nodes,start,peak,end,**kw)
    return {
        'B_accident':spec('accident',[3],beta=.6),
        'E3_capacity_drop':spec('capacity_drop',[3],beta=.75),
        'B_signal_short_cycle':spec('signal_timing',[2],beta=.5,extra={'eta':-.2,'cycle_sec':90,'green_ratio':.6}),
        'B_signal_long_cycle':spec('signal_timing',[2],beta=.5,extra={'eta':.2,'cycle_sec':90,'green_ratio':.6}),
        'B_distribution_local':spec('distribution_shift',[2],beta=.5,region_ratio=.125,extra={'od_nodes':[0]}),
        'B_distribution_regional':spec('distribution_shift',[2],beta=.5,region_ratio=.5,extra={'od_nodes':[0,1]}),
        'B_distribution_global':spec('distribution_shift',[2],beta=.5,region_ratio=1,extra={'od_nodes':[0,1]}),
        'B_inflow_spike':spec('inflow_spike',[0],beta=.8),
        'E4_rain':spec('weather',[3],beta=.5,region_ratio=.5,extra={'beta_v':.25,'beta_c':.2}),
        'E4_snow':spec('weather',[3],beta=.8,region_ratio=1,extra={'beta_v':.4,'beta_c':.35}),
        'E5_weekend':spec('school',[3],beta=0,extra={'inflow_surge':1}),
        'E5_weekday':spec('school',[3],beta=.7,extra={'inflow_surge':4}),
        'E5_guided':spec('school',[3],beta=.15,extra={'inflow_surge':1.15}),
    }


def corridor():
    # Physical links are CTM cells; node 2 branches, link 3 is a bottleneck.
    A=np.zeros((8,8)); A[0,2]=1; A[1,2]=1; A[2,3]=.65; A[2,4]=.35
    A[3,5]=1; A[4,6]=1; A[5,7]=1; A[6,7]=1
    return A


def flatten_arrays(result):
    values={}
    def visit(obj,prefix=''):
        for k,v in obj.items():
            if isinstance(v,np.ndarray): values[prefix+k]=v
            elif isinstance(v,dict): visit(v,prefix+k+'_')
    visit(result)
    return values


def validate_pair(r,params,dt):
    checks={}
    for name in ['factual','counterfactual']:
        x=r[name]
        vlim=params.v_free if name=='factual' else r['v_free_cf']
        clim=params.capacity if name=='factual' else r['capacity_cf']
        balance=x['n'].sum(axis=1)+x['queue'].sum(axis=1)-x['initial_n'].sum()-x['initial_queue'].sum()
        expected=np.cumsum((x['external_arrivals']-x['exit_flow']).sum(axis=1))*dt/3600
        checks[name]={
            'finite':all(np.isfinite(v).all().item() for v in x.values()),
            'max_local_conservation_error_veh':float(np.abs(x['conservation']).max()),
            'max_queue_conservation_error_veh':float(np.abs(x['queue_conservation']).max()),
            'max_network_conservation_error_veh':float(np.abs(balance-expected).max()),
            'density_bounds':bool(np.all((x['rho']>=-1e-9)&(x['rho']<=params.rho_jam+1e-9))),
            'speed_bounds':bool(np.all((x['speed']>=-1e-9)&(x['speed']<=vlim*3.6+1e-8))),
            'outflow_capacity':bool(np.all(x['outflow']<=clim+1e-8)),
            'fd_consistency':bool(np.allclose(x['flow'],x['rho']*params.storage*x['speed']/3.6*3600,atol=1e-8)),
        }
        c=checks[name]
        if not all(c[k] for k in ['finite','density_bounds','speed_bounds','outflow_capacity','fd_consistency']) or max(c[k] for k in c if k.startswith('max_'))>1e-7:
            raise AssertionError(f'Physical validation failed: {name}: {c}')
    return checks


def plot_pair(r,dt,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(8,7),sharex=True)
    time=np.arange(len(r['X']))*dt/60
    for key,ax,ylabel in zip(['speed','flow','queue'],axes,['Mean speed (km/h)','Mean FD flow (veh/h)','External queue (veh)']):
        for world,color in [('factual','#2369a1'),('counterfactual','#bd4c36')]:
            y=r[world][key]
            y=y.sum(axis=1) if key=='queue' else y.mean(axis=1)
            ax.plot(time,y,color=color,label=world)
        ax.set_ylabel(ylabel);ax.grid(alpha=.2)
    axes[0].legend();axes[0].set_title(title+' (synthetic CTM)')
    axes[-1].set_xlabel('Time (min)');fig.tight_layout()
    fig.savefig(path.with_suffix('.png'),dpi=220)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    ap.add_argument('--steps',type=int,default=72)
    ap.add_argument('--dt',type=float,default=30)
    ap.add_argument('--no-plots',action='store_true')
    args=ap.parse_args()
    if args.steps<12 or len(set(args.seeds))!=len(args.seeds): raise ValueError('Invalid steps/seeds')
    args.out.mkdir(parents=True,exist_ok=False)
    A=corridor();p=default_params(len(A));t=np.arange(args.steps)/args.steps
    demand=np.zeros((args.steps,len(A)));demand[:,0]=3000+1000*np.sin(np.pi*t)**2
    demand[:,1]=1600;demand[:,3]=600
    index=[]
    for seed in args.seeds:
        school={}
        for name,spec in scenario_catalog(args.steps).items():
            out=generate_pair(A,spec,args.steps,args.dt,p,demand,seed=seed,demand_noise=.03)
            checks=validate_pair(out,p,args.dt)
            folder=args.out/name/f'seed_{seed}';folder.mkdir(parents=True)
            arrays=flatten_arrays(out);arrays['A']=A
            for k in ['length','v_free','w_cong','n_lanes','sat_flow','rho_jam']: arrays['base_'+k]=getattr(p,k)
            np.savez_compressed(folder/'trajectories.npz',**arrays)
            metadata=dict(name=name,seed=seed,dt_sec=args.dt,steps=args.steps,spec=asdict(spec),checks=checks,
                          evidence='synthetic_CTM_not_trained_model_result',units={'speed':'km/h','flow':'veh/h','n':'veh','queue':'veh','rho':'veh/m/storage_lane'})
            (folder/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            if seed==args.seeds[0] and not args.no_plots: plot_pair(out,args.dt,folder/'comparison',name)
            index.append({'scenario':name,'seed':seed,'file':str((folder/'trajectories.npz').relative_to(args.out)),
                          'sha256':hashlib.sha256((folder/'trajectories.npz').read_bytes()).hexdigest(),
                          'max_effect':float(np.abs(out['effect']).max()),'checks_passed':True})
            if name.startswith('E5'): school[name]=out
        np.testing.assert_array_equal(school['E5_weekday']['X'],school['E5_guided']['X'])
        np.savez_compressed(args.out/f'E5_guidance_vs_weekday_seed_{seed}.npz',
            weekday=school['E5_weekday']['X_cf'],guided=school['E5_guided']['X_cf'],
            weekend=school['E5_weekend']['X_cf'],
            guidance_effect=school['E5_guided']['X_cf']-school['E5_weekday']['X_cf'])
        print(f'validated {len(scenario_catalog(args.steps))} scenarios for seed {seed}',flush=True)
    summary=dict(scenarios=len(scenario_catalog(args.steps)),seeds=args.seeds,runs=len(index),all_physical_checks_passed=True,
                 network='synthetic_8_link_corridor_not_Q_Traffic',results=index)
    (args.out/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}))

if __name__=='__main__':main()
