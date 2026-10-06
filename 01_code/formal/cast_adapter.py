"""Author CaST with explicit train-only graph attributes and online past-only DTW."""
import importlib
import sys
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def dtw_edges(x,edge,shift):
    # Exact squared-Euclidean DTW followed by sqrt, matching dtaidistance's
    # default 1D convention. The time shift rolls only the observed history.
    a=x[:,:,edge[0],0].transpose(1,2);b=torch.roll(x[:,:,edge[1],0],-shift,1).transpose(1,2)
    T=a.shape[-1];previous=[a.new_zeros(a.shape[:2])]+[a.new_full(a.shape[:2],float('inf')) for _ in range(T)]
    for i in range(T):
        current=[a.new_full(a.shape[:2],float('inf'))]
        for j in range(T):current.append((a[...,i]-b[...,j]).square()+torch.minimum(torch.minimum(previous[j],previous[j+1]),current[j]))
        previous=current
    return previous[-1].sqrt()


class CastAdapter(nn.Module):
    def __init__(self,path,A,train,settings,history,horizon,device):
        super().__init__()
        if history!=horizon:raise ValueError('Author CaST implementation requires equal history/horizon')
        if str(path) not in sys.path:sys.path.insert(0,str(path))
        pkg=types.ModuleType('src');pkg.__path__=[str(path/'src')];sys.modules['src']=pkg
        cls=importlib.import_module('src.models.cast').CaST
        a=A.detach().cpu().numpy();edge=np.array(np.where(np.triu((a+a.T)>0,1)))
        if not edge.shape[1]:raise ValueError('CaST needs non-self graph edges')
        N=len(a);E=edge.shape[1];inc=np.zeros((N,E),dtype=np.float32)
        inc[edge[0],np.arange(E)]=-1;inc[edge[1],np.arange(E)]=1
        L=inc.T@inc;scale=np.linalg.eigvalsh(inc@inc.T).max();L=2*L/scale
        ei=np.array(np.where(abs(L)>1e-9));ew=L[ei[0],ei[1]]
        correlation=np.nan_to_num(np.corrcoef(train[...,0].T))
        attr=np.stack([a[edge[0],edge[1]],correlation[edge[0],edge[1]]],-1)
        for key,value in [('edge',edge),('edge_hodge',ei),('weight_hodge',ew),('attribute',attr)]:
            self.register_buffer(key,torch.tensor(value,dtype=torch.long if key.startswith('edge') else torch.float32))
        self.N=N;self.E=E;self.history=history
        self.model=cls(name='CaST',dataset='unified_real_speed',device=torch.device(device),num_nodes=N,num_edges=E,
                       seq_len=history,horizon=horizon,input_dim=3,output_dim=1,hid_dim=settings.get('hidden',16),
                       dropout=.2,node_embed_dim=5,K=2,depth=10,n_envs=5,time_delay_scaler=6)

    def forward(self,x,zg,tod,dow,A):
        B,T,N,_=x.shape
        calendar=torch.stack([tod,dow],-1)[:,:,None,:].expand(-1,-1,N,-1)
        features=torch.cat([x,calendar],-1).permute(0,2,1,3).reshape(B*N,T,3)
        with torch.no_grad():delay=torch.stack([dtw_edges(x,self.edge,s) for s in range(0,T,6)],-1)
        attrs=torch.cat([self.attribute[None].expand(B,-1,-1),delay],-1).reshape(B*self.E,-1)
        expand=lambda edge,n:torch.cat([edge+i*n for i in range(B)],-1)
        graph=types.SimpleNamespace(x_s=attrs,x_t=features,edge_index=expand(self.edge,N),
             edge_index_s=expand(self.edge_hodge,self.E),edge_weight_s=self.weight_hodge.repeat(B),
             edge_index_t=expand(self.edge,N),edge_weight_t=x.new_ones(B*self.E))
        pred,h,q,indices,classification=self.model(graph,test_flag=not self.training)
        reg=F.mse_loss(h,q)+F.mse_loss(q,h)-F.cross_entropy(classification,indices) if self.training else x.new_zeros(())
        return pred,{'author_regularizer':reg}
