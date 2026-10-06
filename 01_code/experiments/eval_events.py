"""Event-stratified evaluation retaining all variables and every test sample."""
import argparse
import json
import numpy as np
from .protocol import stratified

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred',required=True)
    ap.add_argument('--true',required=True)
    ap.add_argument('--zloc',required=True)
    ap.add_argument('--feature-names',nargs='+',required=True)
    ap.add_argument('--sample-minutes',type=int,default=5)
    ap.add_argument('--forecast-minutes',nargs='+',type=int,default=[15,30,60])
    args=ap.parse_args()
    y=np.load(args.true,allow_pickle=False); p=np.load(args.pred,allow_pickle=False)
    z=np.load(args.zloc,allow_pickle=False)
    if y.shape!=p.shape:raise ValueError('Prediction/target mismatch')
    if y.ndim==2:y=y[:,None,:,None];p=p[:,None,:,None]
    elif y.ndim==3:y=y[:,None];p=p[:,None]
    if z.ndim==3:z=z[:,None]
    if y.ndim!=4 or z.shape!=y.shape[:-1]+(4,) or len(args.feature_names)!=y.shape[-1]:
        raise ValueError('Expected y[B,H,N,F] and labels[B,H,N,4]')
    hs=[]
    for m in args.forecast_minutes:
        if m<=0 or args.sample_minutes<=0 or m%args.sample_minutes:raise ValueError('Invalid time horizon')
        h=m//args.sample_minutes
        if h>y.shape[1]:raise ValueError('Horizon exceeds predictions')
        hs.append((m,h))
    print(json.dumps(stratified(y,p,z,args.feature_names,hs),indent=2,allow_nan=False))

if __name__=='__main__':main()

