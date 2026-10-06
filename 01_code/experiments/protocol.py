"""Reproducible E1/E2/prediction-ablation runner with explicit future targets.

NPZ inputs: X[T,N,F], A[N,N], optional U[T,N,1], Z_loc[T,N,4],
Z_global[T,5], feature_names[F]. Real Q experiments require measured flow
and observed scene features. No values from paper_reported.json are consumed.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from paper.config import ModelConfig
from paper.model.stcausalnet import STCausalNet
from paper.data.features import build_time_features
from paper.tables.metrics import event_masks
from dc_ctgcn.metrics import all_metrics
from paper.baselines import build_baseline, BASELINES


class Windows(Dataset):
    def __init__(self,data,start,end,hist,horizon,mean,std):
        self.data=data;self.hist=hist;self.horizon=horizon;self.mean=mean;self.std=std
        self.starts=np.arange(start,end-hist-horizon+1,dtype=np.int64)
        if not len(self.starts):raise ValueError('Split too short for selected history/horizon')
    def __len__(self):return len(self.starts)
    def __getitem__(self,index):
        s=int(self.starts[index]); e=s+self.hist; d=self.data
        values=[(d['X'][s:e]-self.mean)/self.std,d['U'][s:e],d['Z_loc'][s:e],d['Z_global'][s:e],
                (d['X'][e:e+self.horizon]-self.mean)/self.std]
        return tuple(torch.from_numpy(np.asarray(a,dtype=np.float32)) for a in values)+(torch.tensor(e),)


class ForecastModel(nn.Module):
    def __init__(self,cfg,variant='full'):
        super().__init__();self.cfg=cfg
        self.core=STCausalNet.ablation(cfg,variant)
        self.head=nn.Linear(cfg.d_hidden,cfg.horizon*cfg.d_y)
    def forward(self,x,u,z,zg,A):
        # Core predicts shifts for T-1 positions. Pad once so every observed
        # history sample is encoded; the padding adds no future observations.
        pad=lambda a:torch.cat([a,a[:,-1:]],dim=1)
        out=self.core(pad(x),pad(u),pad(z),pad(zg),A,(A>0).to(x.dtype))
        h=out['H_F'][:,-1]
        y=self.head(h).view(x.shape[0],x.shape[2],self.cfg.horizon,self.cfg.d_y).permute(0,2,1,3)
        return y,out
    def objective(self,pred,target,out,A):
        return self.core.loss_fn(pred,target,out['C_F'],out['L_KL'],residuals=target-pred,A=A)['L_total']


class BaselineForecast(nn.Module):
    """Adapter to the existing compact implementations; not official replicas."""
    def __init__(self,cfg,name):
        super().__init__()
        extra={'hist':cfg.hist} if name=='STAEformer' else {}
        self.model=build_baseline(name,num_nodes=cfg.num_nodes,in_dim=cfg.d_x+10,out_dim=cfg.d_y,
                                 horizon=cfg.horizon,hidden=cfg.d_hidden,**extra)
    def forward(self,x,u,z,zg,A):
        inputs=torch.cat([x,u,z,zg[:,:,None,:].expand(-1,-1,x.shape[2],-1)],dim=-1)
        return self.model(inputs,A),{}
    def objective(self,pred,target,out,A):return (pred-target).abs().mean()


class HistoricalAverage(nn.Module):
    """Training-only weekly-slot averages with a training-mean fallback."""
    def __init__(self,data,ntr,mean,std,cfg,sample_minutes):
        super().__init__();self.steps=1440//sample_minutes;self.horizon=cfg.horizon
        table=np.zeros((7*self.steps,cfg.num_nodes,cfg.d_y),dtype=np.float32)
        counts=np.zeros(7*self.steps,dtype=int)
        X=(data['X'][:ntr]-mean)/std
        phases=data['Z_global'][:ntr]
        hour=np.mod(np.arctan2(phases[:,0],phases[:,1]),2*np.pi)/(2*np.pi)
        day=np.mod(np.arctan2(phases[:,2],phases[:,3]),2*np.pi)/(2*np.pi)
        slots=(np.rint(day*7).astype(int)%7)*self.steps+(np.rint(hour*self.steps).astype(int)%self.steps)
        np.add.at(table,slots,X);np.add.at(counts,slots,1)
        table=np.divide(table,counts[:,None,None],out=table,where=counts[:,None,None]>0)
        table[counts==0]=X.mean(axis=0)
        self.register_buffer('table',torch.from_numpy(table))
    def forward(self,x,u,z,zg,A):
        p=zg[:,-1];hour=torch.remainder(torch.atan2(p[:,0],p[:,1]),2*torch.pi)/(2*torch.pi)
        day=torch.remainder(torch.atan2(p[:,2],p[:,3]),2*torch.pi)/(2*torch.pi)
        last=(torch.round(day*7).long()%7)*self.steps+(torch.round(hour*self.steps).long()%self.steps)
        slots=(last[:,None]+torch.arange(1,self.horizon+1,device=x.device)[None,:])%(7*self.steps)
        return self.table[slots],{}


def read_data(path,dataset,sample_minutes):
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    X=np.asarray(data['X'],dtype=np.float32); A=np.asarray(data['A'],dtype=np.float32)
    if X.ndim!=3 or A.shape!=(X.shape[1],X.shape[1]) or not np.isfinite(X).all() or not np.isfinite(A).all() or (A<0).any():
        raise ValueError('Require cleaned finite X[T,N,F] and nonnegative A[N,N]')
    T,N,F=X.shape
    if dataset=='q':
        if 'flow_provenance' not in data or str(data['flow_provenance'].item())!='measured':
            raise ValueError('Q speed/flow study requires measured flow; a speed-derived proxy is insufficient')
        if not {'U','Z_loc','Z_global','feature_names'} <= data.keys() or list(data['feature_names'])!=['speed','flow']:
            raise ValueError('Q requires explicit speed/flow channels and observed scene/control inputs')
    metadata={'missing_inputs':[]}
    for key,default in [('U',np.zeros((T,N,1))),('Z_loc',np.zeros((T,N,4))),
                        ('Z_global',build_time_features(T,steps_per_day=int(1440/sample_minutes)))]:
        if key not in data:metadata['missing_inputs'].append(key)
        data[key]=np.asarray(data.get(key,default),dtype=np.float32)
        if data[key].shape!=default.shape or not np.isfinite(data[key]).all():raise ValueError('Invalid '+key)
    data['X']=X;data['A']=A
    names=list(map(str,data.get('feature_names',[f'variable_{i}' for i in range(F)])))
    if len(names)!=F or len(set(names))!=F:raise ValueError('Unique feature_names required')
    return data,names,metadata


def metrics(y,p,names,minutes):
    # Report endpoint horizons separately, in original units.
    result={}
    for m,step in minutes:
        result[str(m)]={name:forecast_metrics(y[:,step-1,:,f],p[:,step-1,:,f]) for f,name in enumerate(names)}
    return result


def forecast_metrics(y,p):
    return {k:v for k,v in all_metrics(y,p).items() if k in ('MAE','RMSE','MAPE')}


def stratified(y,p,z,names,minutes):
    masks=event_masks(z)
    if not np.all(sum(m.astype(int) for m in masks.values())==1):raise AssertionError('Event groups must partition targets')
    result={}
    for minute,step in minutes:
        groups={}
        for label,mask in masks.items():
            selected=mask[:,step-1]
            groups[label]={'count':int(selected.sum()),'metrics':{
                name:forecast_metrics(y[:,step-1,:,f][selected],p[:,step-1,:,f][selected]) if selected.any() else None for f,name in enumerate(names)}}
        for label in ['S2','S3','S4']:
            groups[label]['PDR_percent']={}
            for name in names:
                ref=groups['S1']['metrics'][name]; own=groups[label]['metrics'][name]
                groups[label]['PDR_percent'][name]=100*(own['MAE']-ref['MAE'])/ref['MAE'] if own is not None and ref is not None and ref['MAE']>0 else None
        result[str(minute)]=groups
    return result


def train_run(data,names,metadata,args,seed,variant,source_hash):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    device=torch.device(args.device)
    T,N,F=data['X'].shape
    ntr=int(T*.6);nval=int(T*.2)
    hist=args.history_minutes//args.sample_minutes
    hs=[(m,m//args.sample_minutes) for m in args.forecast_minutes]; horizon=max(h for _,h in hs)
    mean=data['X'][:ntr].mean(axis=0);std=data['X'][:ntr].std(axis=0);std=np.maximum(std,1e-6)
    spans=[(0,ntr),(ntr,ntr+nval),(ntr+nval,T)]
    datasets=[Windows(data,a,b,hist,horizon,mean,std) for a,b in spans]
    loaders=[DataLoader(d,batch_size=args.batch_size,shuffle=i==0,generator=torch.Generator().manual_seed(seed)) for i,d in enumerate(datasets)]
    cfg=ModelConfig(num_nodes=N,d_x=F,d_y=F,d_hidden=args.hidden,d_scenario=16,d_latent=8,d_z_glob=5,
                    d_z_loc=4,hist=hist,horizon=horizon,k_neighbors=min(8,N))
    if variant=='HA':model=HistoricalAverage(data,ntr,mean,std,cfg,args.sample_minutes)
    elif variant in BASELINES:model=BaselineForecast(cfg,variant)
    else:model=ForecastModel(cfg,variant)
    model=model.to(device);A=torch.from_numpy(data['A']).to(device)
    # Neural matrix convention is destination <- source; CTM data A uses source -> destination.
    if args.adjacency_convention=='source_to_destination':A=A.T.contiguous()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr) if variant!='HA' else None
    folder=args.out/variant.replace('/','_').replace(' ','_')/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'preprocessing.npz',mean=mean,std=std,split_boundaries=[0,ntr,ntr+nval,T],
                        train_starts=datasets[0].starts,val_starts=datasets[1].starts,test_starts=datasets[2].starts)
    def evaluate(loader):
        model.eval();ys=[];ps=[];starts=[]
        with torch.no_grad():
            for x,u,z,zg,y,s in loader:
                pred,_=model(x.to(device),u.to(device),z.to(device),zg.to(device),A)
                ps.append(pred.cpu().numpy()*std+mean);ys.append(y.numpy()*std+mean);starts.append(s.numpy())
        return np.concatenate(ys),np.concatenate(ps),np.concatenate(starts)
    best=float('inf');bad=0;best_state=None;history=[]
    for epoch in range(args.epochs):
        if opt is None:
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            break
        model.train();loss_sum=0;count=0
        for x,u,z,zg,y,_ in loaders[0]:
            x,u,z,zg,y=[a.to(device) for a in [x,u,z,zg,y]]
            pred,out=model(x,u,z,zg,A);loss=model.objective(pred,y,out,A)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            loss_sum+=float(loss.detach())*len(x);count+=len(x)
        vy,vp,_=evaluate(loaders[1])
        # Validation selection averages standardized MAE to avoid mixing speed/flow units.
        score=float(np.mean(np.abs(vy-vp)/std))
        history.append({'epoch':epoch+1,'train_objective':loss_sum/count,'val_standardized_MAE':score})
        if score<best:
            best=score;bad=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:bad+=1
        if bad>=args.patience:break
    if best_state is None:raise RuntimeError('No valid checkpoint')
    model.load_state_dict(best_state)
    vy,vp,vs=evaluate(loaders[1]);ty,tp,ts=evaluate(loaders[2])
    np.savez_compressed(folder/'predictions.npz',val_true=vy,val_pred=vp,val_target_starts=vs,test_true=ty,test_pred=tp,test_target_starts=ts)
    torch.save({'state':best_state,'config':asdict(cfg),'variant':variant,'source_sha256':source_hash},folder/'checkpoint.pt')
    events=data['Z_loc'][ts[:,None]+np.arange(horizon)[None,:]]
    result={'seed':seed,'variant':variant,'dataset':args.dataset,'source_sha256':source_hash,
            'evidence':'computed_from_training','config':asdict(cfg),'input_metadata':metadata,'history':history,
            'validation':metrics(vy,vp,names,hs),'test':metrics(ty,tp,names,hs),'events':stratified(ty,tp,events,names,hs)}
    (folder/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    rows=[]
    for minute,_ in hs:
        for name in names:
            rows.append(dict(dataset=args.dataset,target=name,horizon_minutes=minute,split_id=source_hash,
                  model='STCausalNet' if variant=='full' else variant,seed=seed,
                  val_mae=result['validation'][str(minute)][name]['MAE'],test_mae=result['test'][str(minute)][name]['MAE'],provenance='computed'))
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--dataset',choices=['metr-la','pems-bay','q','synthetic'],required=True)
    ap.add_argument('--sample-minutes',type=int,required=True);ap.add_argument('--history-minutes',type=int,default=60)
    ap.add_argument('--forecast-minutes',type=int,nargs='+',default=[15,30,60])
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    ap.add_argument('--variants',nargs='+',default=['full'])
    ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--patience',type=int,default=10)
    ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--hidden',type=int,default=64)
    ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--device',default='cpu')
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--adjacency-convention',choices=['source_to_destination','destination_from_source'],required=True)
    args=ap.parse_args()
    if min(args.sample_minutes,args.history_minutes,args.epochs,args.batch_size,args.patience,args.threads,*args.forecast_minutes)<=0:
        raise ValueError('Positive time and training settings required')
    if any(m%args.sample_minutes for m in [args.history_minutes]+args.forecast_minutes):raise ValueError('Minute horizons must be divisible by sample interval')
    supported={'full','w/o Scene','w/o Macro','w/o Dynamic','w/o Causal-Reg','HA'}|set(BASELINES)
    if not set(args.variants)<=supported:raise ValueError('Unsupported prediction variant; run Physics ablation through counterfactual_study with calibrated physical context')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.variants))!=len(args.variants):raise ValueError('Duplicate seeds or variants')
    torch.set_num_threads(args.threads)
    data,names,meta=read_data(args.data,args.dataset,args.sample_minutes)
    digest=hashlib.sha256(args.data.read_bytes()).hexdigest()
    args.out.mkdir(parents=True,exist_ok=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (args.out/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    rows=[]
    for variant in args.variants:
        for seed in args.seeds:
            rows.extend(train_run(data,names,meta,args,seed,variant,digest))
            print(f'completed {variant} seed={seed}',flush=True)
    (args.out/'computed_runs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':main()
