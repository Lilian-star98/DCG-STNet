"""Original author architectures behind an explicit unified-protocol adapter."""
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
from dataclasses import asdict
import torch
from torch import nn
from paper.config import ModelConfig
from paper.experiments.protocol import ForecastModel
from .common import ROOT, repository, digest
from .sparse_graph import CandidateGraph


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class AuthorModel(nn.Module):
    def __init__(self,model,name):
        super().__init__();self.model=model;self.name=name
    def forward(self,x,zg,tod,dow,A):
        N=x.shape[2]
        calendar=torch.stack([tod,dow],-1)[:,:,None,:].expand(-1,-1,N,-1)
        inputs=torch.cat([x,calendar],-1)
        if self.name=='GraphWaveNet':
            return self.model(inputs.permute(0,3,2,1))[...,-1:],{}
        if self.name=='D2STGNN':
            return self.model(inputs).transpose(1,2)[...,None],{}
        return self.model(inputs),{}


class ProposedModel(nn.Module):
    def __init__(self,cfg,variant):
        super().__init__();self.model=ForecastModel(cfg,variant)
        old=self.model.core.graph
        new=CandidateGraph(cfg);new.load_state_dict(old.state_dict());self.model.core.graph=new
    def forward(self,x,zg,tod,dow,A):
        B,T,N,_=x.shape
        return self.model(x,x.new_zeros(B,T,N,1),x.new_zeros(B,T,N,4),zg,A.T.contiguous())


def build(name,N,history,horizon,settings,A,device,data=None):
    hidden=settings.get('hidden',32)
    if name.startswith('STCausalNet'):
        variant=name.split(':',1)[1] if ':' in name else 'full'
        cfg=ModelConfig(num_nodes=N,d_hidden=hidden,d_scenario=16,d_latent=8,d_z_glob=5,
                        hist=history,horizon=horizon,k_neighbors=8,use_physics=False)
        return ProposedModel(cfg,variant),dict(architecture='STCausalNet forecast adapter',config=asdict(cfg),
                    physics=False,scene_inputs='calendar only; local events and controls unavailable',
                    modifications=['candidate-only edge MLP evaluation','direct multi-horizon forecast head'])
    root=ROOT/'third_party/official'
    if name=='GraphWaveNet':
        path=root/'Graph-WaveNet';module=load_file('official_gwn',path/'model.py')
        support=[]
        for a in [A,A.T]:support.append(a/a.sum(-1,keepdim=True).clamp_min(1e-8))
        model=module.gwnet(device,N,supports=support,in_dim=3,out_dim=horizon,
                           residual_channels=hidden,dilation_channels=hidden)
        # Original legacy Conv1d objects use two-dimensional kernels and receive
        # NCHW tensors. Modern PyTorch requires Conv2d; weights/algebra unchanged.
        fixes=[]
        for field in ['gate_convs','residual_convs','skip_convs']:
            for i,old in enumerate(getattr(model,field)):
                if isinstance(old,nn.Conv1d) and len(old.kernel_size)==2:
                    new=nn.Conv2d(old.in_channels,old.out_channels,old.kernel_size,
                                  dilation=old.dilation,bias=old.bias is not None)
                    new.load_state_dict(old.state_dict());getattr(model,field)[i]=new
                    fixes.append(f'{field}.{i}: Conv1d -> Conv2d for original 2D kernel')
        return AuthorModel(model,name),dict(official=repository(path),compatibility=fixes,
                   protocol='unified 6:2:2, masked targets, speed+time+weekday input; not author-protocol reproduction')
    if name=='STAEformer':
        path=root/'STAEformer';module=load_file('official_stae',path/'model/STAEformer.py')
        model=module.STAEformer(N,in_steps=history,out_steps=horizon,input_dim=3)
        return AuthorModel(model,name),dict(official=repository(path),compatibility=[],
                   protocol='author architecture/default dimensions; unified split/scaler/training protocol')
    if name=='D2STGNN':
        path=root/'D2STGNN'
        if str(path) not in sys.path:sys.path.insert(0,str(path))
        pkg='models'  # Author code mixes relative and absolute models.* imports.
        module=types.ModuleType(pkg);module.__path__=[str(path/'models')];sys.modules[pkg]=module
        cls=importlib.import_module(pkg+'.model').D2STGNN
        support=[a/a.sum(-1,keepdim=True).clamp_min(1e-8) for a in [A,A.T]]
        model=cls(num_feat=1,num_hidden=hidden,node_hidden=10,time_emb_dim=10,dropout=.1,
                  seq_length=horizon,k_t=3,k_s=2,gap=3,num_modalities=2,device=torch.device(device),
                  num_nodes=N,adjs=support,adjs_ori=A,dataset='METR-LA')
        return AuthorModel(model,name),dict(official=repository(path),compatibility=[],
                    protocol='author architecture; unified optimization without author curriculum schedule')
    if name=='CaST':
        if data is None:raise ValueError('CaST requires training-only graph attributes')
        from .cast_adapter import CastAdapter
        path=root/'CaST';end=int(data['split'][1])
        train=(np.asarray(data['X'][:end])-data['mean'])/data['std']
        model=CastAdapter(path,A,train,settings,history,horizon,device)
        return model,dict(official=repository(path),protocol='author architecture and auxiliary losses; unified training',
                  preprocessing='same physical candidate graph; correlation train-only; exact DTW from historical speed',
                  compatibility=['torch_sparse preprocessing replaced by equivalent incidence-matrix algebra'])
    if name=='PDFormer':
        if data is None:raise ValueError('PDFormer requires training-only semantic graph and patterns')
        from .pdformer_adapter import PDAdapter
        path=root/'PDFormer';model=PDAdapter(path,A,data,history,horizon,device)
        return model,dict(official=repository(path),protocol='author architecture; unified optimization without curriculum',
                  preprocessing='author FastDTW/KShape; restricted to training period to prevent leakage',
                  learned_features_sha256=digest(Path(data['_path'])/'pdformer_train_features.npz'))
    raise ValueError('No verified official adapter for '+name)
