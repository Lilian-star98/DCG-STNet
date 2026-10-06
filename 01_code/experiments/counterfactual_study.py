"""E3-E5/Table-4 execution harness on generated paired CTM data.

Known simulator parameters are used as physical context. This is a synthetic
mechanism test, not real-Q parameter identification or the paper's final table.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from paper.config import ModelConfig
from paper.model.stcausalnet import STCausalNet
from paper.model.counterfactual import InterventionSpec
from paper.model.physics import PhysicsContext

VARIANTS=['full','w/o Scene','w/o Macro','w/o Dynamic','w/o Causal-Reg','w/o Physics']

def load_case(folder):
    with np.load(folder/'trajectories.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    return d,meta


def prepare(d,meta,mean,std):
    tensor=lambda a:torch.as_tensor(np.asarray(a),dtype=torch.float32)
    T,N,F=d['X'].shape
    x=tensor((d['X']-mean)/std)[None]
    u=tensor(d['green'])[None,...,None]
    z=torch.zeros(1,T,N,4);zg=torch.zeros(1,T,5)
    phase=torch.arange(T,dtype=torch.float32)*float(meta['dt_sec'])/86400
    zg[0,:,0]=torch.sin(2*torch.pi*phase);zg[0,:,1]=torch.cos(2*torch.pi*phase)
    zp=z.clone();channel=1 if meta['spec']['kind']=='weather' else 0
    zp[0,:,d['region_nodes'],channel]=tensor(d['intervention_beta'])[:,None]
    def context(cf):
        return PhysicsContext(A=tensor(d['turning_cf'][1:] if cf else d['turning'][1:]),
            length=tensor(d['base_length']),storage_lanes=tensor(d['base_n_lanes']),rho_jam=tensor(d['base_rho_jam']),
            v_free=tensor(d['v_free_cf'][1:] if cf else d['base_v_free']),
            w_cong=tensor(d['w_cong_cf'][1:] if cf else d['base_w_cong']),
            capacity=tensor(d['capacity_cf'][1:] if cf else d['base_n_lanes']*d['base_sat_flow']),
            demand=tensor(d['demand_cf'][1:] if cf else d['demand'][1:])[None],
            initial_n=tensor(d['factual_n'][0])[None],initial_queue=tensor(d['factual_queue'][0])[None],
            dt_sec=float(meta['dt_sec']),mean=tensor(mean),std=tensor(std),
            green=tensor(d['green_cf'][1:] if cf else d['green'][1:])[None],
            cycle=tensor(d['cycle_cf'][1:] if cf else d['cycle'][1:]) if 'cycle' in d else None,
            time_origin_sec=float(meta['dt_sec']))
    spec=InterventionSpec(T0=max(0,meta['spec']['t_start']-1),U_prime=tensor(d['green_cf'])[None,...,None],
                          Z_prime=zp,physics_prime=context(True))
    A=tensor(d['A']).T
    return x,u,z,zg,A,(A>0).float(),context(False),spec


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    ap.add_argument('--variants',nargs='+',default=VARIANTS)
    ap.add_argument('--epochs',type=int,default=1);ap.add_argument('--hidden',type=int,default=8)
    args=ap.parse_args()
    if not set(args.variants)<=set(VARIANTS) or args.epochs<1:raise ValueError('Invalid training settings')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    cases=sorted(p for p in args.suite.iterdir() if p.is_dir())
    rows=[]
    for seed in args.seeds:
        train,tm=load_case(args.suite/'E3_capacity_drop'/f'seed_{seed}')
        # Fit scaler to this synthetic training episode only, never test scenarios.
        mean=train['X'].mean(axis=0);std=np.maximum(train['X'].std(axis=0),1.0)
        inputs=prepare(train,tm,mean,std);x,u,z,zg,A,cand,physics,_=inputs
        for variant in args.variants:
            torch.manual_seed(seed)
            cfg=ModelConfig(num_nodes=x.shape[2],d_x=2,d_y=2,d_hidden=args.hidden,d_scenario=8,d_latent=4,d_z_glob=5)
            model=STCausalNet.ablation(cfg,variant)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            model.train();losses=[]
            for _ in range(args.epochs):
                result=model(x,u,z,zg,A,cand,y_true=x[:,1:],physics=physics)
                loss=result['L_total']
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                losses.append(float(loss.detach()))
            target=args.out/variant.replace('/','_').replace(' ','_')/f'seed_{seed}'
            target.mkdir(parents=True)
            torch.save({'state':model.state_dict(),'config':asdict(model.cfg),'variant':variant,
                        'training_case':'E3_capacity_drop','loss':losses,'mean':mean,'std':std},target/'checkpoint.pt')
            model.eval()
            for case in cases:
                d,meta=load_case(case/f'seed_{seed}')
                xx,uu,zz,gg,aa,cc,pp,intervention=prepare(d,meta,mean,std)
                with torch.no_grad():out=model(xx,uu,zz,gg,aa,cc,physics=pp,spec=intervention)
                pf=out['Y_F'][0].numpy()*std+mean;pcf=out['Y_CF'][0].numpy()*std+mean
                true_effect=d['X_cf'][1:]-d['X'][1:];effect=pcf-pf
                vf=d['v_free_cf'][1:]*3.6;cap=d['capacity_cf'][1:]
                bad=(pcf[...,0]<-1e-3)|(pcf[...,0]>vf+1e-3)|(pcf[...,1]<-1e-3)|(pcf[...,1]>cap+1e-2)
                row=dict(scenario=case.name,seed=seed,variant=variant,TED_speed=float(np.abs(effect[...,0]-true_effect[...,0]).mean()),
                         TED_flow=float(np.abs(effect[...,1]-true_effect[...,1]).mean()),PVR_speed_capacity_percent=float(bad.mean()*100),
                         graph_zero_percent=float((out['C_F'].abs()<=1e-6).float().mean()*100),
                         split='training_episode' if case.name=='E3_capacity_drop' else 'held_out_intervention_same_network',
                         evidence='synthetic_short_training_with_known_simulator_parameters')
                if 'physics_CF' in out:
                    row['max_vehicle_conservation_error']=float(out['physics_CF']['conservation'].abs().max())
                    row['max_queue_conservation_error']=float(out['physics_CF']['queue_conservation'].abs().max())
                np.savez_compressed(target/(case.name+'.npz'),pred_factual=pf,pred_cf=pcf,true_factual=d['X'][1:],true_cf=d['X_cf'][1:],pred_effect=effect,true_effect=true_effect)
                rows.append(row)
            print(f'completed counterfactual {variant}, seed {seed}',flush=True)
    (args.out/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
    (args.out/'README.txt').write_text('Synthetic execution coverage only. E3 is the training episode; remaining interventions share its network and base demand. These are not independent real-Q test results. PVR here reports speed/capacity bounds; conservation diagnostics are available only for CTM-projected variants. Do not copy results into paper tables.',encoding='utf-8')

if __name__=='__main__':main()
