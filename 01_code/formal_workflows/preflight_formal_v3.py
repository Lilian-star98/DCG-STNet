"""Check real-size GPU batches before committing to a long study."""
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from paper.formal.common import ROOT,save
from paper.formal.train import read_data,MeasuredWindows,masked_mae
from paper.formal.models import build

ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--dataset',default='metr-la');a=ap.parse_args()
torch.set_num_threads(4);torch.manual_seed(0)
data,meta=read_data(ROOT/'data/formal_v3'/a.dataset);device='cuda'
A=torch.tensor(np.array(data['A']),device=device)
settings={'hidden':32,'lr':.001}
tick=time.monotonic();model,prov=build(a.model,len(A),12,12,settings,A,device,data=data);model.to(device)
loader=DataLoader(MeasuredWindows(data,0),batch_size=8)
mean=torch.tensor(np.array(data['mean']),device=device);std=torch.tensor(np.array(data['std']),device=device)
torch.cuda.reset_peak_memory_stats();start=time.monotonic()
for i,(x,zg,tod,dow,y,m,t) in enumerate(loader):
    if i==3:break
    x,zg,tod,dow,y,m=[v.to(device) for v in [x,zg,tod,dow,y,m]]
    model.train();p,aux=model(x,zg,tod,dow,A);loss=masked_mae(p*std+mean,y,m)
    if 'author_regularizer' in aux:loss=loss+aux['author_regularizer']
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    model.zero_grad(set_to_none=True);loss.backward()
    model.eval()
    with torch.no_grad():q,_=model(x,zg,tod,dow,A)
    if not torch.isfinite(q).all():raise RuntimeError('Nonfinite evaluation')
torch.cuda.synchronize()
report=dict(model=a.model,dataset=a.dataset,status='passed',nodes=len(A),batch_size=8,
            peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2,batch_checks_seconds=time.monotonic()-start,
            initialization_seconds=start-tick,provenance=prov)
save(ROOT/'results/formal_v3/preflight'/(a.dataset+'_'+a.model+'.json'),report)
print({k:v for k,v in report.items() if k!='provenance'},flush=True)
