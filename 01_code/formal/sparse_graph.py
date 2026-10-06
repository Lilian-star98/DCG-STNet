"""Evaluate edge MLPs only on fixed candidate edges; same evaluation algebra."""
import torch
from paper.model.dynamic_causal_graph import SceneAwareGraphGenerator, _gumbel_sigmoid, _topk_sparsify


class CandidateGraph(SceneAwareGraphGenerator):
    def forward(self,H_t,S_t,cand_mask,dist=None,hard=False):
        if cand_mask.ndim!=2:raise ValueError('Formal runner uses a shared fixed candidate graph')
        B,N,_=H_t.shape
        ij=torch.nonzero(cand_mask>0,as_tuple=False);i,j=ij[:,0],ij[:,1]
        node=self._pair_feat(H_t,S_t)
        pair=torch.cat([node[:,i],node[:,j]],-1)
        logits=self.exist_mlp(pair).squeeze(-1)
        exists=(_gumbel_sigmoid(logits,self.cfg.gumbel_tau,hard) if self.training
                else torch.sigmoid(logits/max(self.cfg.gumbel_tau,1e-4)))
        d=H_t.new_zeros(B,len(i)) if dist is None else dist[i,j].to(H_t).expand(B,-1)
        strength=torch.sigmoid(self.strength_mlp(torch.cat([pair,d[...,None]],-1)).squeeze(-1))
        strength=strength*torch.exp(-d/max(float(self.cfg.d_decay),1.))
        immediate=torch.softmax(self.type_mlp(pair),-1)[...,0]
        raw=H_t.new_zeros(B,N,N);exist=H_t.new_zeros(B,N,N)
        raw[:,i,j]=exists*strength*immediate;exist[:,i,j]=exists
        return _topk_sparsify(raw,self.cfg.k_neighbors,cand_mask),exist
