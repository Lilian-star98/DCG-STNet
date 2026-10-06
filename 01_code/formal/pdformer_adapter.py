"""Official PDFormer architecture; all learned preprocessing uses train data only."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.sparse.csgraph import shortest_path
from .common import digest,save


def features(data):
    from fastdtw import fastdtw
    from tslearn.clustering import KShape
    directory=Path(data['_path']);cache=directory/'pdformer_train_features.npz'
    provenance=directory/'pdformer_train_features.json'
    contract=digest(directory/'manifest.json')
    if cache.exists() and provenance.exists():
        import json
        m=json.loads(provenance.read_text())
        if m['data_manifest_sha256']!=contract or m['sha256']!=digest(cache):raise ValueError('Stale learned preprocessing')
        with np.load(cache) as z:return {k:z[k] for k in z.files}
    end=int(data['split'][1]);train=np.asarray(data['X'][:end]);N=train.shape[1]
    days=len(train)//288
    if not days:raise ValueError('Need full training days for PDFormer semantic graph')
    profile=train[:days*288].reshape(days,288,N,1).mean(0)
    distances=np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):distances[i,j]=distances[j,i]=fastdtw(profile[:,i],profile[:,j],radius=6)[0]
    z=(train-data['mean'])/data['std'];count=min(14*288,len(z)-23)
    patterns=np.stack([z[i:i+3] for i in range(count)]).swapaxes(1,2).reshape(-1,3,1)
    centers=KShape(n_clusters=16,max_iter=5,random_state=0,n_init=1).fit(patterns).cluster_centers_
    A=np.asarray(data['A']);hops=shortest_path((A>0).astype(float),directed=True,unweighted=True)
    # Match the author's normalized Laplacian and real-part eigendecomposition.
    degree=A.sum(1);inv=np.divide(1,np.sqrt(degree),out=np.zeros_like(degree),where=degree>0)
    L=np.eye(N)-inv[:,None]*A.T*inv[None,:]
    values,vectors=np.linalg.eig(L);order=values.argsort();isolated=int((degree==0).sum())
    selected=order[isolated+1:isolated+9]
    lap=np.zeros((N,8),dtype=np.float32);lap[:,:len(selected)]=np.real(vectors[:,selected])
    # Deterministic eigenvector sign convention.
    for j in range(8):
        if lap[np.argmax(abs(lap[:,j])),j]<0:lap[:,j]*=-1
    result=dict(dtw_matrix=distances,pattern_keys=centers.astype('float32'),sh_mx=hops,lap=lap)
    np.savez_compressed(cache,**result)
    save(provenance,dict(data_manifest_sha256=contract,sha256=digest(cache),training_end=end,
                         method='author FastDTW radius6, KShape and Laplacian; all learned statistics training-only'))
    return result


class PDAdapter(nn.Module):
    def __init__(self,path,A,data,history,horizon,device,prepared=None):
        super().__init__()
        if str(path) not in sys.path:sys.path.insert(0,str(path))
        spec=importlib.util.spec_from_file_location('official_pdformer',path/'libcity/model/traffic_flow_prediction/PDFormer.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        f=features(data) if prepared is None else prepared
        config=dict(device=torch.device(device),dataset='unified',input_window=history,output_window=horizon,
                    lape_dim=8,use_curriculum_learning=False)
        d=dict(num_nodes=len(A),feature_dim=9,ext_dim=8,num_batches=1,scaler=None,
               adj_mx=A.cpu().numpy(),sd_mx=None,sh_mx=f['sh_mx'],dtw_matrix=f['dtw_matrix'],pattern_keys=f['pattern_keys'])
        self.model=module.PDFormer(config,d);self.register_buffer('lap',torch.tensor(f['lap'],dtype=torch.float32))
    def forward(self,x,zg,tod,dow,A):
        N=x.shape[2]
        calendar=torch.cat([tod[...,None],torch.nn.functional.one_hot(dow.long(),7).to(x)],-1)
        features=torch.cat([x,calendar[:,:,None,:].expand(-1,-1,N,-1)],-1)
        return self.model({'X':features},self.lap),{}
