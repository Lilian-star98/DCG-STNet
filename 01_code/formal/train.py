"""Masked real-data training. Search never evaluates or serializes test targets."""
import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from .common import digest,save,source_snapshot,environment
from .models import build


class MeasuredWindows(Dataset):
    def __init__(self,data,split,history=12,horizon=12):
        self.d=data;self.history=history;self.horizon=horizon
        a,b=data['split'][split:split+2]
        self.starts=np.arange(a,b-history-horizon+1,dtype=np.int64)
        if not len(self.starts):raise ValueError('Insufficient split duration')
    def __len__(self):return len(self.starts)
    def __getitem__(self,i):
        a=int(self.starts[i]);b=a+self.history;c=b+self.horizon;d=self.d
        x=(d['X'][a:b]-d['mean'])/d['std']
        return tuple(torch.from_numpy(np.array(v,copy=True,dtype=np.float32)) for v in
                     [x,d['Z_global'][a:b],d['tod'][a:b],d['dow'][a:b],d['truth'][b:c],d['observed'][b:c]])+(torch.tensor(b),)


def read_data(path):
    path=Path(path);m=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    for name,h in m['files'].items():
        if digest(path/name)!=h:raise ValueError('Prepared data changed: '+name)
    data={p.stem:np.load(p,mmap_mode='r') for p in path.glob('*.npy')}
    data['_path']=str(path)
    return data,m


def masked_mae(pred,truth,mask):
    valid=mask.bool()&torch.isfinite(truth)
    if not valid.any():return None
    return (pred[valid]-truth[valid]).abs().mean()


def metrics(truth,pred,mask):
    valid=mask.astype(bool)&np.isfinite(truth)
    if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
    if not valid.any():return {'count':0,'MAE':None,'RMSE':None,'MAPE_percent':None}
    y=truth[valid].astype('float64');p=pred[valid].astype('float64');d=p-y
    return dict(count=int(len(y)),MAE=float(np.mean(abs(d))),RMSE=float(np.sqrt(np.mean(d*d))),
                MAPE_percent=float(100*np.mean(abs(d)/np.maximum(abs(y),1))))


def run(data_path,out,name,settings,seed=0,phase='search',device='cuda'):
    out=Path(out);completed=out/'result.json'
    data,manifest=read_data(data_path)
    contract=dict(data_hash=digest(Path(data_path)/'manifest.json'),model=name,settings=settings,seed=seed,
                  phase=phase,sources=source_snapshot())
    if completed.exists():
        if json.loads((out/'contract.json').read_text(encoding='utf-8'))!=contract:
            raise ValueError('Existing run belongs to a different contract')
        return json.loads(completed.read_text(encoding='utf-8'))
    if out.exists() and (out/'contract.json').exists():
        if json.loads((out/'contract.json').read_text(encoding='utf-8'))!=contract:raise ValueError('Resume contract mismatch')
    out.mkdir(parents=True,exist_ok=True);save(out/'contract.json',contract)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.set_num_threads(settings.get('threads',4))
    hist=settings.get('history',12);horizon=settings.get('horizon',12);batch=settings.get('batch_size',8)
    datasets=[MeasuredWindows(data,i,hist,horizon) for i in [0,1]]
    generator=torch.Generator().manual_seed(seed)
    loaders=[DataLoader(d,batch_size=batch,shuffle=i==0,generator=generator,num_workers=0) for i,d in enumerate(datasets)]
    A=torch.tensor(np.array(data['A']),device=device)
    model,provenance=build(name,A.shape[0],hist,horizon,settings,A,device,data=data);model=model.to(device)
    save(out/'model.json',dict(provenance=provenance,environment=environment(),parameters=sum(p.numel() for p in model.parameters())))
    mean=torch.tensor(np.array(data['mean']),device=device);std=torch.tensor(np.array(data['std']),device=device)
    # Same base optimization budget. Paper's graph/latent multipliers are explicit.
    if name.startswith('STCausalNet'):
        groups=[]
        for label,mult in [('graph',3.),('latent',.3),('other',1.)]:
            params=[p for n,p in model.named_parameters() if ('graph' if '.graph.' in n else 'latent' if '.latent.' in n else 'other')==label]
            groups.append(dict(params=params,lr=settings['lr']*mult))
        opt=torch.optim.Adam(groups,weight_decay=settings.get('weight_decay',0))
    else:opt=torch.optim.Adam(model.parameters(),lr=settings['lr'],weight_decay=settings.get('weight_decay',0))
    history=[];best=float('inf');bad=0;start=0
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);history=ck['history'];start=len(history)
        best=ck['best'];bad=ck['bad'];torch.set_rng_state(ck['cpu_rng'].cpu());generator.set_state(ck['loader_rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])

    @torch.no_grad()
    def evaluate(loader,collect=False):
        model.eval();total=0.;count=0;ys=[];ps=[];ms=[];ts=[]
        for x,zg,tod,dow,y,m,t in loader:
            p,_=model(*[a.to(device) for a in [x,zg,tod,dow]],A)
            p=p*std+mean;y=y.to(device);m=m.to(device)
            valid=m.bool()&torch.isfinite(y)
            if not torch.isfinite(p).all():raise RuntimeError('Nonfinite validation output')
            total+=float((p[valid]-y[valid]).abs().sum());count+=int(valid.sum())
            if collect:ys.append(y.cpu().numpy());ps.append(p.cpu().numpy());ms.append(m.cpu().numpy());ts.append(t.numpy())
        if not count:raise RuntimeError('No observed evaluation targets')
        return total/count, ([np.concatenate(v) for v in [ys,ps,ms,ts]] if collect else None)

    def regularizer(aux,x):
        if 'author_regularizer' in aux:return aux['author_regularizer']
        if 'C_F' not in aux:return x.new_zeros(())
        cfg=model.model.core.cfg;c=aux['C_F']
        return cfg.lambda_sparse*c.abs().mean()+cfg.lambda_smooth*(c[:,1:]-c[:,:-1]).abs().mean()+cfg.lambda_kl*aux['L_KL']

    for epoch in range(start,settings.get('max_epochs',100)):
        if bad>=settings.get('patience',15):break
        tick=time.monotonic();model.train();tot=0.;cnt=0
        for step,(x,zg,tod,dow,y,m,_) in enumerate(loaders[0]):
            x,zg,tod,dow,y,m=[a.to(device) for a in [x,zg,tod,dow,y,m]]
            p,aux=model(x,zg,tod,dow,A)
            loss=masked_mae(p*std+mean,y,m)
            if loss is None:continue
            loss=loss+regularizer(aux,x)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
            tot+=float(loss.detach())*len(x);cnt+=len(x)
            if step%100==0:save(out/'progress.json',dict(epoch=epoch+1,batch=step,total_batches=len(loaders[0]),phase=phase))
        score,_=evaluate(loaders[1]);improved=score<best
        if improved:
            best=score;bad=0;torch.save(model.state_dict(),out/'best.pt')
        else:bad+=1
        history.append(dict(epoch=epoch+1,train_objective=tot/max(cnt,1),validation_mae=score,seconds=time.monotonic()-tick))
        save(out/'history.json',history)
        torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),history=history,best=best,bad=bad,
                        cpu_rng=torch.get_rng_state(),loader_rng=generator.get_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []),out/'last.pt')
        print(f'{name} seed={seed} epoch={epoch+1} val_mae={score:.5f} seconds={history[-1]["seconds"]:.1f}',flush=True)
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True))
    score,_=evaluate(loaders[1])
    result=dict(model=name,seed=seed,phase=phase,settings=settings,validation_mae=score,epochs=len(history),
                best_epoch=min(history,key=lambda r:r['validation_mae'])['epoch'],data_hash=contract['data_hash'],
                stopped_by='early_stopping' if bad>=settings.get('patience',15) else 'epoch_budget',
                test_evaluated=False,physics_calibrated=False)
    if phase=='final':
        test=MeasuredWindows(data,2,hist,horizon);loader=DataLoader(test,batch_size=batch,num_workers=0)
        _,(y,p,m,t)=evaluate(loader,True)
        result['test']={str(k*5):metrics(y[:,k-1],p[:,k-1],m[:,k-1]) for k in [3,6,12] if k<=horizon}
        result['test_all_horizons']=metrics(y,p,m);result['test_evaluated']=True
        np.savez_compressed(out/'test_predictions.npz',truth=y,prediction=p,observed=m.astype(bool),target_starts=t)
    save(out/'result.json',result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',default='STCausalNet');ap.add_argument('--settings',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--phase',choices=['search','final'],default='search')
    ap.add_argument('--device',default='cuda');a=ap.parse_args()
    run(a.data,a.out,a.model,json.loads(a.settings.read_text()),a.seed,a.phase,a.device)

if __name__=='__main__':main()
