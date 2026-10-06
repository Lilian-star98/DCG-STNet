import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from paper.config import ModelConfig
from paper.model.dynamic_causal_graph import SceneAwareGraphGenerator
from paper.formal.sparse_graph import CandidateGraph
from paper.formal.prepare import causal_fill
from paper.formal.train import masked_mae,metrics,MeasuredWindows,run
from paper.formal.models import build
from paper.formal.calibration import fit_fd,diagram
from paper.formal.events import event_effect
from paper.formal.common import digest,save
from paper.formal.common import ROOT

torch.set_num_threads(1)

class EvidenceTests(unittest.TestCase):
    def test_train_only_preprocessing(self):
        raw=np.array([10,0,20,0,10000,0],dtype='float32')[:,None,None]
        x,m,mean,std=causal_fill(raw,3)
        self.assertEqual(float(mean.item()),15);self.assertEqual(float(x[3].item()),20)
        self.assertFalse(bool(m[1].item()));self.assertEqual(float(x[5].item()),10000)
    def test_missing_target_gradient(self):
        p=torch.tensor([3.,100.],requires_grad=True);y=torch.tensor([1.,float('nan')])
        loss=masked_mae(p,y,torch.tensor([1,0]));loss.backward()
        self.assertEqual(p.grad.tolist(),[1.,0.])
    def test_sparse_evaluation_and_gradient(self):
        cfg=ModelConfig(num_nodes=6,d_hidden=8,d_scenario=4)
        a=SceneAwareGraphGenerator(cfg).eval();b=CandidateGraph(cfg).eval();b.load_state_dict(a.state_dict())
        h=torch.randn(2,6,8,requires_grad=True);s=torch.randn(2,4);mask=(torch.rand(6,6)>.6).float()
        x=a(h,s,mask)[0];y=b(h,s,mask)[0]
        torch.testing.assert_close(x,y)
        gx=torch.autograd.grad(x.sum(),h,retain_graph=True)[0];gy=torch.autograd.grad(y.sum(),h)[0]
        torch.testing.assert_close(gx,gy,atol=1e-6,rtol=1e-5)
    def test_fd_known_parameters(self):
        k=np.linspace(.001,.139,300);q=diagram(k,20,5,.14,3)
        p=fit_fd(k,q,3,dict(lower=[5,1,.1],upper=[40,15,.3]))
        self.assertAlmostEqual(p['v_free_mps'],20,places=2)
        self.assertAlmostEqual(p['w_cong_mps'],5,places=2)
    def test_fd_rejects_unidentified(self):
        k=np.linspace(.001,.005,100);q=diagram(k,20,5,.14,3)
        with self.assertRaises(ValueError):fit_fd(k,q,3,dict(lower=[5,1,.1],upper=[40,15,.3]))
    def test_event_known_effect(self):
        t=np.arange(30);controls=np.stack([t,t+2],1).astype(float);treated=t.astype(float)+1
        treated[20:]-=4;pre=t<20;post=~pre
        r=event_effect(treated,controls,pre,post)
        self.assertAlmostEqual(r['estimated_ATT'],-4,places=4)
        changed=treated.copy();changed[20:]-=100
        np.testing.assert_allclose(r['donor_weights'],event_effect(changed,controls,pre,post)['donor_weights'])
    def test_official_forward_backward(self):
        for name in ['GraphWaveNet','STAEformer','D2STGNN']:
            with self.subTest(name=name):
                A=torch.eye(12);m,prov=build(name,12,12,12,{'hidden':8},A,'cpu')
                x=torch.randn(2,12,12,1);zg=torch.zeros(2,12,5);tod=torch.zeros(2,12);dow=torch.ones(2,12)
                y,_=m(x,zg,tod,dow,A);self.assertEqual(y.shape,(2,12,12,1))
                y.square().mean().backward();self.assertTrue(any(p.grad is not None for p in m.parameters()))
                self.assertIn('official',prov)
    def test_search_does_not_evaluate_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir();T=200;N=5
            raw=np.ones((T,N,1),dtype='float32')*20;raw[160:]=np.nan
            arrays=dict(X=np.ones_like(raw)*20,truth=raw,observed=np.isfinite(raw),A=np.eye(N,dtype='float32'),
                        mean=np.ones((N,1),dtype='float32')*20,std=np.ones((N,1),dtype='float32'),
                        Z_global=np.zeros((T,5),dtype='float32'),tod=np.zeros(T,dtype='float32'),dow=np.zeros(T),split=np.array([0,120,160,200]))
            for k,v in arrays.items():np.save(data/(k+'.npy'),v)
            save(data/'manifest.json',{'files':{p.name:digest(p) for p in data.glob('*.npy')}})
            settings=dict(hidden=8,lr=.001,max_epochs=1,patience=1,batch_size=8,threads=1)
            r=run(data,root/'run','STCausalNet',settings,device='cpu')
            self.assertFalse(r['test_evaluated']);self.assertFalse((root/'run/test_predictions.npz').exists())
            d={k:np.array(v) for k,v in arrays.items()}
            self.assertLess(MeasuredWindows(d,0).starts[-1]+24,121)
            self.assertGreaterEqual(MeasuredWindows(d,1).starts[0],120)
    def test_cast_author_adapter(self):
        from paper.formal.cast_adapter import CastAdapter,dtw_edges
        from dtaidistance import dtw
        A=torch.eye(12)+torch.diag(torch.ones(11),1)
        model=CastAdapter(ROOT/'third_party/official/CaST',A,np.random.default_rng(0).normal(size=(100,12,1)),
                          {'hidden':8},12,12,'cpu')
        x=torch.randn(2,12,12,1);calendar=torch.zeros(2,12);zg=torch.zeros(2,12,5)
        p,aux=model(x,zg,calendar,calendar,A);self.assertEqual(p.shape,(2,12,12,1))
        (p.square().mean()+aux['author_regularizer']).backward()
        model.eval();p,_=model(x,zg,calendar,calendar,A);self.assertTrue(torch.isfinite(p).all())
        e=torch.tensor([[0],[1]])
        expected=dtw.distance(x[0,:,0,0].numpy().astype('float64'),x[0,:,1,0].numpy().astype('float64'))
        self.assertAlmostEqual(float(dtw_edges(x,e,0)[0,0]),expected,places=5)
    def test_pdformer_author_adapter(self):
        from paper.formal.pdformer_adapter import PDAdapter
        N=12;A=torch.eye(N)
        fixture={'dtw_matrix':np.zeros((N,N)),'sh_mx':np.zeros((N,N)),
                 'pattern_keys':np.ones((16,3,1),dtype='float32'),'lap':np.zeros((N,8),dtype='float32')}
        model=PDAdapter(ROOT/'third_party/official/PDFormer',A,None,12,12,'cpu',prepared=fixture)
        x=torch.randn(2,12,N,1);p,_=model(x,torch.zeros(2,12,5),torch.zeros(2,12),torch.zeros(2,12),A)
        self.assertEqual(p.shape,(2,12,N,1));p.square().mean().backward()

if __name__=='__main__':unittest.main(verbosity=2)
